"""Golden reference for the browser's from-scratch TESSERA-base TRAINER.

web/js/lab/trainer.js re-implements TESSERA-base training by hand (manual
backward pass, AdamW, gradient clipping, cosine LR) so the demo's Training Lab
can train a real model live in the browser. This module produces the numbers
that implementation must reproduce, computed by the REAL PyTorch model
(tessera.models.tessera_base._TesseraBaseNet) with autograd, in float64:

* the initial weights (weights.json format, via tessera.models.export),
* a 24-row batch of SYNTHETIC rows at the raw scale of the real features
  (zero-inflated, heavy-tailed, values into the millions; no real AIT row is
  used or shipped), with several availability patterns,
* for (a) eval mode (clip 1.0, the recipe's; inactive on this batch) and
  (b) a fixed, exported dropout keep-mask on the head's hidden layer (clip
  0.05, so the scaling branch is exercised): the loss, every parameter's
  gradient, the total grad norm, the post-clip gradients, and the parameters
  after 3 AdamW steps with clipping,
* the eval-mode scores and gates, and torch's CosineAnnealingLR schedule.

To stay small, every flat parameter-sized vector (gradients, clipped
gradients, parameters after the AdamW steps) is stored as base64 of
little-endian float64 bytes - bit-exact, ~10.7 chars per number - in
``net.parameters()`` order, which is exactly the order of a depth-first walk
of the weights.json structure (encoders[0..3]: linear0.weight row-major,
linear0.bias, groupnorm.weight, groupnorm.bias, linear1.weight, linear1.bias;
fusion.gate_linear0, fusion.gate_linear1; head.linear0, head.linear1).
``param_layout`` lists every tensor's name, shape and offset.

Run: uv run python -m tessera.demo.golden_train
"""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tessera.models.export import export_weights
from tessera.models.tessera_base import SLICES, TesseraBase, _TesseraBaseNet

SEED = 0
N_ROWS = 24
DROPOUT_P = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-2
N_ADAMW_STEPS = 3
MAX_NORM = 1.0  # the training recipe's clip; this batch's grad norm is below it
# Mode (b) uses a tighter clip so the golden also exercises the scaling branch
# of clip_grad_norm_ (this batch's gradient norm is ~0.09, above it).
MAX_NORM_DROPOUT = 0.05
COSINE_EPOCHS = 30

OUT_PATH = Path(__file__).resolve().parents[3] / "web" / "data" / "train_golden.json"

# Availability pattern per row, M1/M2/M3/M4 (M3 is always derived, so always 1).
_PATTERNS = (
    [(1, 1, 1, 1)] * 10  # all present
    + [(0, 1, 1, 1)] * 4  # M1 absent (no log lines)
    + [(1, 1, 1, 0)] * 4  # M4 absent
    + [(0, 1, 1, 0)] * 3  # M1 + M4 absent
    + [(1, 0, 1, 1)] * 3  # M2 absent
)
_LABELS = [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0]
_HOST_BUCKETS = (10, 34, 47)


def _synthetic_batch(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows shaped like the real features (raw, unscaled): each feature is zero
    with some probability, else log-normal with a per-feature log-scale spanning
    ~1 to ~1e6; some columns are integer counts. Attack rows are shifted up."""
    n_features = SLICES[-1].stop
    log_scale = rng.uniform(0.0, 12.0, size=n_features)  # e^12 ~ 1.6e5, tails reach 1e6+
    zero_rate = rng.uniform(0.1, 0.6, size=n_features)
    is_integer = rng.random(n_features) < 0.5
    X = np.zeros((N_ROWS, n_features))
    for i, (pattern, label) in enumerate(zip(_PATTERNS, _LABELS, strict=True)):
        for m, s in enumerate(SLICES):
            if m == 2:  # M3 identity: host bucket + number of active sources
                X[i, s.start] = _HOST_BUCKETS[0] if label else _HOST_BUCKETS[i % 3]
                X[i, s.start + 1] = 3 if label else 1 + (i % 3)
                continue
            if not pattern[m]:
                continue  # absent modality: all zero, like the real pipeline
            for j in range(s.start, s.stop):
                if rng.random() < zero_rate[j]:
                    continue
                v = np.exp(log_scale[j] + 1.5 * label + 1.2 * rng.standard_normal())
                X[i, j] = max(1.0, np.round(v)) if is_integer[j] else v
            if not np.any(X[i, s]):  # present means at least one nonzero value
                X[i, s.start] = 1.0
    avail = np.asarray(_PATTERNS, dtype=np.float64)
    y = np.asarray(_LABELS, dtype=np.float64)
    return X, y, avail


def _f64_b64(t: torch.Tensor | np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(t, dtype="<f8").ravel())
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _flat(tensors) -> torch.Tensor:
    return torch.cat([t.detach().reshape(-1) for t in tensors])


def _forward_logits(
    net: _TesseraBaseNet, x: torch.Tensor, a: torch.Tensor, mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eval mode (mask None) is the module's own forward; with a mask, the head
    is replayed layer by layer with nn.Dropout replaced by an explicit, exportable
    keep-mask: h * mask / (1 - p) - exactly what nn.Dropout computes for that mask."""
    if mask is None:
        return net(x, a)
    embeds = torch.stack([enc(x[:, s]) for enc, s in zip(net.encoders, SLICES, strict=True)], 1)
    fused, gates = net.fusion(embeds, a)
    h = net.head[1](net.head[0](fused))
    h = h * mask / (1.0 - DROPOUT_P)
    return net.head[3](h).squeeze(-1), gates


def _run_mode(
    net0: _TesseraBaseNet,
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    pos_weight: torch.Tensor,
    mask: torch.Tensor | None,
    max_norm: float,
) -> dict:
    net = copy.deepcopy(net0)
    net.eval()  # the mask (if any) is the only dropout; eval() makes nn.Dropout inert
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    out: dict = {"max_norm": max_norm, "steps": []}
    for step in range(N_ADAMW_STEPS):
        opt.zero_grad()
        logit, _ = _forward_logits(net, x, a, mask)
        loss = loss_fn(logit, y)
        loss.backward()
        grads = _flat(p.grad for p in net.parameters())
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm)
        clipped = _flat(p.grad for p in net.parameters())
        if step == 0:
            out["loss"] = loss.item()
            out["logits"] = logit.detach().tolist()
            out["grads_b64"] = _f64_b64(grads)
            out["grad_norm"] = norm.item()
            out["clip_active"] = bool(norm.item() > max_norm)
            out["clipped_grads_b64"] = _f64_b64(clipped)
        opt.step()
        out["steps"].append({"loss": loss.item(), "grad_norm": norm.item()})
    out["params_after_steps_b64"] = _f64_b64(_flat(net.parameters()))
    return out


def _cosine_schedule() -> list[float]:
    p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    opt = torch.optim.AdamW([p], lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=COSINE_EPOCHS)
    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(COSINE_EPOCHS):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    return lrs


def build_golden() -> dict:
    torch.manual_seed(SEED)
    net = _TesseraBaseNet().double()
    rng = np.random.default_rng(SEED)
    X, y_np, avail = _synthetic_batch(rng)
    mask_np = (rng.random((N_ROWS, net.head[0].out_features)) >= DROPOUT_P).astype(np.float64)

    x = torch.as_tensor(X, dtype=torch.float64)
    y = torch.as_tensor(y_np, dtype=torch.float64)
    a = torch.as_tensor(avail, dtype=torch.float64)
    n_pos = float(y_np.sum())
    pos_weight_value = (N_ROWS - n_pos) / max(n_pos, 1.0)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float64)

    holder = TesseraBase(seed=SEED, device=torch.device("cpu"))
    holder.net = net
    weights = export_weights(holder)

    net.eval()
    with torch.no_grad():
        logit, gates = net(x, a)
        scores = torch.sigmoid(logit)

    layout, offset = [], 0
    for name, p in net.named_parameters():
        layout.append({"name": name, "shape": list(p.shape), "offset": offset})
        offset += p.numel()

    return {
        "schema": "tessera-train-golden/v1",
        "generated_by": "uv run python -m tessera.demo.golden_train",
        "note": (
            "Synthetic batch (no real AIT rows). Computed by the real PyTorch "
            "_TesseraBaseNet in float64 with autograd. *_b64 fields are base64 "
            "little-endian float64 in param_layout order."
        ),
        "torch_version": torch.__version__,
        "seed": SEED,
        "n_params": offset,
        "param_layout": layout,
        "weights": weights,
        "batch": {
            "n": N_ROWS,
            "X": X.tolist(),
            "y": y_np.astype(int).tolist(),
            "avail": avail.astype(int).tolist(),
        },
        "pos_weight": pos_weight_value,
        "dropout_p": DROPOUT_P,
        "dropout_mask": mask_np.astype(int).tolist(),
        "adamw": {
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "n_steps": N_ADAMW_STEPS,
        },
        "eval_outputs": {"scores": scores.tolist(), "gates": gates.tolist()},
        "eval": _run_mode(net, x, y, a, pos_weight, None, MAX_NORM),
        "dropout": _run_mode(net, x, y, a, pos_weight, torch.as_tensor(mask_np), MAX_NORM_DROPOUT),
        "cosine": {"base_lr": LR, "t_max": COSINE_EPOCHS, "lrs": _cosine_schedule()},
    }


def main() -> None:
    golden = build_golden()
    OUT_PATH.write_text(json.dumps(golden))
    size_kb = OUT_PATH.stat().st_size / 1024
    print(
        f"wrote {OUT_PATH} ({size_kb:.0f} KB): {golden['n_params']} params, "
        f"eval loss {golden['eval']['loss']:.6f} (grad norm {golden['eval']['grad_norm']:.4g}), "
        f"dropout loss {golden['dropout']['loss']:.6f} "
        f"(grad norm {golden['dropout']['grad_norm']:.4g})"
    )


if __name__ == "__main__":
    main()
