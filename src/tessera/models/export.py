"""Export a trained TESSERA-base to a JSON weight file a hand-written JS forward
pass can load directly, plus a set of golden (input, output) pairs computed by
THIS Python model - the reference the JS parity test proves against.

Every number is exported as a plain nested list (JSON has no typed arrays), and
weight matrices are exported in the exact orientation the JS forward pass expects
(out_features x in_features for a Linear layer's weight, matching torch's own
storage layout) so the JS side never has to silently transpose anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tessera.models.tessera_base import SLICES, TesseraBase, _availability_4col


def _linear_to_dict(layer: torch.nn.Linear) -> dict:
    return {
        "weight": layer.weight.detach().cpu().numpy().tolist(),  # (out, in)
        "bias": layer.bias.detach().cpu().numpy().tolist(),  # (out,)
    }


def _groupnorm_to_dict(layer: torch.nn.GroupNorm) -> dict:
    return {
        "num_groups": layer.num_groups,
        "num_channels": layer.num_channels,
        "eps": layer.eps,
        "weight": layer.weight.detach().cpu().numpy().tolist(),
        "bias": layer.bias.detach().cpu().numpy().tolist(),
    }


def export_weights(model: TesseraBase) -> dict:
    """Walk the exact module structure defined in tessera_base.py's
    _TesseraBaseNet, so a change to that architecture breaks this loudly
    (AttributeError) rather than silently exporting a stale shape."""
    net = model.net
    assert net is not None, "model must be fit() before exporting"

    encoders = []
    for enc in net.encoders:
        lin0, gn, lin1 = enc.net[0], enc.net[1], enc.net[3]
        assert isinstance(lin0, torch.nn.Linear) and isinstance(lin1, torch.nn.Linear)
        assert isinstance(gn, torch.nn.GroupNorm)
        encoders.append(
            {
                "linear0": _linear_to_dict(lin0),
                "groupnorm": _groupnorm_to_dict(gn),
                "linear1": _linear_to_dict(lin1),
            }
        )

    gate0, gate1 = net.fusion.gate_net[0], net.fusion.gate_net[2]
    fusion = {
        "n_modalities": net.fusion.n_modalities,
        "embed_dim": net.fusion.embed_dim,
        "gate_linear0": _linear_to_dict(gate0),
        "gate_linear1": _linear_to_dict(gate1),
    }

    head0, head1 = net.head[0], net.head[3]
    head = {"linear0": _linear_to_dict(head0), "linear1": _linear_to_dict(head1)}

    return {
        "architecture": "tessera_base_v1",
        "gelu": "tanh_approx",  # 0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3))) - exact on both sides
        "modality_slices": [[s.start, s.stop] for s in SLICES],
        "modality_names": ["m1_log", "m2_metrics", "m3_identity", "m4_graph"],
        "n_features": SLICES[-1].stop,
        "encoders": encoders,
        "fusion": fusion,
        "head": head,
    }


def build_golden_vectors(
    model: TesseraBase, X: np.ndarray, availability_3col: np.ndarray, *, n: int = 32, seed: int = 0
) -> list[dict]:
    """Real feature vectors, real model, real outputs - what the JS parity test
    must reproduce. Includes a deliberate spread of availability patterns (not
    just the common all-present case) so the parity test actually exercises the
    masking path, not merely the easy one."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)

    proba, attribution = model.predict_with_attribution(X[idx], availability_3col[idx])
    avail4 = _availability_4col(availability_3col[idx])

    vectors = []
    for i in range(len(idx)):
        vectors.append(
            {
                "input": X[idx[i]].tolist(),
                "availability": avail4[i].tolist(),  # 4-col, M1/M2/M3/M4 order
                "expected_score": float(proba[i]),
                "expected_attribution": attribution[i].tolist(),
            }
        )
    return vectors


def export_all(
    model: TesseraBase, X: np.ndarray, availability_3col: np.ndarray, out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = export_weights(model)
    golden = build_golden_vectors(model, X, availability_3col, n=40)

    (out_dir / "weights.json").write_text(json.dumps(weights))
    (out_dir / "golden.json").write_text(json.dumps({"vectors": golden}))

    n_params = sum(
        np.asarray(v).size
        for enc in weights["encoders"]
        for layer in (enc["linear0"], enc["linear1"])
        for v in (layer["weight"], layer["bias"])
    )
    weights_bytes = (out_dir / "weights.json").stat().st_size
    print(
        f"exported weights.json ({n_params}+ encoder params, {weights_bytes} bytes) "
        f"+ golden.json ({len(golden)} vectors) to {out_dir}"
    )
