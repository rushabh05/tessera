"""TESSERA-base: the neural model. Four small per-modality encoders, Gated
Multimodal Unit fusion (availability-aware), one MLP head.

Deliberately simple, per the plan's own explicit cut: no evidential/Dempster
head (that is TESSERA-EV, cut from required scope), no per-token sequence
encoders (TCN/autoencoder) over raw events. Every GELU in this model uses the
TANH APPROXIMATION (`approximate='tanh'`), not PyTorch's default exact-erf
form: there is no clean, dependency-free erf in JavaScript, and the whole
credibility of a browser demo rests on the JS forward pass matching this
model's real weights exactly, not approximately - training with the identical
closed-form tanh formula on both sides removes that mismatch entirely rather
than approximating around it - the P3 pipeline already reduces
each modality to a small real feature vector (M1: 8-dim template statistics,
M2: 24-dim Suricata aggregates, M3: 2-dim identity, M4: 8-dim graph structure),
and this model's encoders operate on THOSE vectors, not on raw sequences. This
is a genuine divergence from the fuller architecture design in the plan
(written before the "simple" pivot), recorded here rather than silently:
the model takes the same flat (X, availability) arrays the GBDT baseline
consumes, not a WindowSet, so it fits the real, tested, working P3 output
shape rather than a data contract nothing currently produces.

The gate values from the GMU ARE the per-modality attribution - exactly the
design intent recorded in models/base.py's Prediction contract, satisfied here
even though this model does not literally return a Prediction object (see
above); ``predict_with_attribution`` returns the same (score, attribution)
shape a Prediction would carry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from tessera.features.pipeline import M1_SLICE, M2_SLICE, M3_SLICE, M4_SLICE
from tessera.models.encoders.small_mlp import SmallMLPEncoder
from tessera.models.fusion.gmu import GatedMultimodalUnit
from tessera.train.seed import seed_everything

SLICES = (M1_SLICE, M2_SLICE, M3_SLICE, M4_SLICE)
N_MODALITIES = len(SLICES)


def best_device() -> torch.device:
    """MPS if available (measured on this machine at 5.66 TFLOP/s fp32), else
    CPU. CPU training respects the project-wide single-thread pin
    (seed_everything -> ensure_single_thread) that avoids the LightGBM/torch
    OpenMP deadlock documented in tessera/__init__.py; that pin does not apply
    to MPS, which runs on the GPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _TesseraBaseNet(nn.Module):
    def __init__(self, *, embed_dim: int = 16, hidden_dim: int = 32) -> None:
        super().__init__()
        self.encoders = nn.ModuleList(
            [SmallMLPEncoder(s.stop - s.start, hidden_dim, embed_dim) for s in SLICES]
        )
        self.fusion = GatedMultimodalUnit(N_MODALITIES, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, x: torch.Tensor, availability: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeds = torch.stack(
            [enc(x[:, s]) for enc, s in zip(self.encoders, SLICES, strict=True)], dim=1
        )
        fused, gates = self.fusion(embeds, availability)
        logit = self.head(fused).squeeze(-1)
        return logit, gates


def _availability_4col(availability_3col: np.ndarray) -> np.ndarray:
    """RealDataset.availability is (n, 3): [M1 present, M2 present, M4 present] -
    M3 (identity) is always computable (host_bucket + source count are derived,
    never missing raw data), so it never appears in that array. The GMU needs one
    column per modality slot in SLICES order (M1, M2, M3, M4), so M3's column is
    inserted here as a constant True rather than silently misaligning the other
    three columns against the wrong modality slots."""
    n = availability_3col.shape[0]
    out = np.ones((n, N_MODALITIES), dtype=bool)
    out[:, 0] = availability_3col[:, 0]  # M1
    out[:, 1] = availability_3col[:, 1]  # M2
    # column 2 (M3) stays True
    out[:, 3] = availability_3col[:, 2]  # M4
    return out


@dataclass
class TrainHistory:
    train_loss: list
    val_ap: list
    best_epoch: int
    stopped_early: bool


class TesseraBase:
    """Trains and predicts on the flat (X, availability, y) arrays the P3
    pipeline produces. Not literally a Detector (see module docstring)."""

    name = "tessera_base"

    def __init__(
        self,
        *,
        seed: int = 0,
        embed_dim: int = 16,
        hidden_dim: int = 32,
        epochs: int = 30,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        batch_size: int = 256,
        patience: int = 5,
        device: torch.device | None = None,
    ) -> None:
        self.seed = seed
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self.device = device or best_device()
        self.net: _TesseraBaseNet | None = None
        self.history: TrainHistory | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        availability: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        availability_val: np.ndarray | None = None,
    ) -> TesseraBase:
        seed_everything(self.seed)
        torch.manual_seed(self.seed)

        self.net = _TesseraBaseNet(embed_dim=self.embed_dim, hidden_dim=self.hidden_dim).to(
            self.device
        )
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        # Class-weighted loss: attack windows are the minority class (often
        # 5-25% prevalence). A plain BCE would let the model coast by predicting
        # "benign" and still score low loss.
        pos_weight = torch.tensor(
            [(len(y) - y.sum()) / max(y.sum(), 1)], dtype=torch.float32, device=self.device
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        Xt = torch.as_tensor(X, dtype=torch.float32)
        yt = torch.as_tensor(y, dtype=torch.float32)
        avail4 = torch.as_tensor(_availability_4col(availability), dtype=torch.float32)

        n = len(y)
        g = torch.Generator().manual_seed(self.seed)

        have_val = X_val is not None and y_val is not None
        if have_val:
            Xv = torch.as_tensor(X_val, dtype=torch.float32).to(self.device)
            yv = y_val
            av4v = torch.as_tensor(_availability_4col(availability_val), dtype=torch.float32).to(
                self.device
            )

        train_loss_hist, val_ap_hist = [], []
        best_ap, best_state, best_epoch, bad_epochs = -1.0, None, 0, 0

        for epoch in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n, generator=g)
            epoch_loss, n_batches = 0.0, 0
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                xb = Xt[idx].to(self.device)
                yb = yt[idx].to(self.device)
                ab = avail4[idx].to(self.device)

                opt.zero_grad()
                logit, _gates = self.net(xb, ab)
                loss = loss_fn(logit, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            sched.step()
            train_loss_hist.append(epoch_loss / max(n_batches, 1))

            if have_val:
                self.net.eval()
                with torch.no_grad():
                    logit_v, _ = self.net(Xv, av4v)
                    proba_v = torch.sigmoid(logit_v).cpu().numpy()
                from sklearn.metrics import average_precision_score

                ap = float(average_precision_score(yv, proba_v)) if len(np.unique(yv)) > 1 else 0.0
                val_ap_hist.append(ap)
                if ap > best_ap:
                    best_ap, best_epoch, bad_epochs = ap, epoch, 0
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    bad_epochs += 1
                    if bad_epochs >= self.patience:
                        break

        stopped_early = have_val and bad_epochs >= self.patience
        if have_val and best_state is not None:
            self.net.load_state_dict(best_state)

        self.history = TrainHistory(
            train_loss=train_loss_hist,
            val_ap=val_ap_hist,
            best_epoch=best_epoch if have_val else self.epochs - 1,
            stopped_early=stopped_early,
        )
        return self

    def predict_proba(self, X: np.ndarray, availability: np.ndarray) -> np.ndarray:
        assert self.net is not None, "call fit() before predict_proba()"
        self.net.eval()
        Xt = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        avail4 = torch.as_tensor(_availability_4col(availability), dtype=torch.float32).to(
            self.device
        )
        with torch.no_grad():
            logit, _gates = self.net(Xt, avail4)
        return torch.sigmoid(logit).cpu().numpy()

    def predict_with_attribution(
        self, X: np.ndarray, availability: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (score, attribution) - attribution is the GMU's own gate
        values, (N, 4) in M1/M2/M3/M4 order, summing to 1 per row over the
        modalities that were available for that window."""
        assert self.net is not None, "call fit() before predict_with_attribution()"
        self.net.eval()
        Xt = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        avail4 = torch.as_tensor(_availability_4col(availability), dtype=torch.float32).to(
            self.device
        )
        with torch.no_grad():
            logit, gates = self.net(Xt, avail4)
        return torch.sigmoid(logit).cpu().numpy(), gates.cpu().numpy()

    def n_parameters(self) -> int:
        assert self.net is not None
        return sum(p.numel() for p in self.net.parameters())

    def describe(self) -> dict:
        return {
            "name": self.name,
            "seed": self.seed,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "device": str(self.device),
            "n_parameters": self.n_parameters() if self.net is not None else None,
            "history": None
            if self.history is None
            else {
                "best_epoch": self.history.best_epoch,
                "stopped_early": self.history.stopped_early,
                "final_train_loss": self.history.train_loss[-1]
                if self.history.train_loss
                else None,
                "best_val_ap": max(self.history.val_ap) if self.history.val_ap else None,
            },
        }
