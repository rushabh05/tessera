"""CPU/MPS parity, so a locally-produced number matches what CPU-only CI validates.

MPS is not bit-identical to CPU: reduction order differs and some kernels use lower
internal precision. The point is not exact equality but a KNOWN, ASSERTED tolerance,
established before any modelling code exists so a later divergence is attributable.

The module below deliberately mirrors the op mix the real encoders will use -
dilated Conv1d with causal padding, GroupNorm, GELU, sigmoid gating, masked pooling
- so the tolerance measured here transfers to the real model.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

MPS = torch.backends.mps.is_available()
TOL = 2e-4  # asserted, not assumed; tightened or loosened only with a recorded reason


class RepresentativeStack(torch.nn.Module):
    """Same op mix as the planned M1 encoder plus the gated fusion."""

    def __init__(self, c_in: int = 16, c: int = 32) -> None:
        super().__init__()
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(c_in if i == 0 else c, c, 3, dilation=2**i) for i in range(3)]
        )
        self.norms = torch.nn.ModuleList([torch.nn.GroupNorm(4, c) for _ in range(3)])
        self.gate = torch.nn.Linear(2 * c, c)
        self.head = torch.nn.Linear(c, 1)

    def forward(self, x, mask):
        for conv, norm in zip(self.convs, self.norms, strict=True):
            pad = conv.dilation[0] * (conv.kernel_size[0] - 1)
            h = conv(torch.nn.functional.pad(x, (pad, 0)))  # causal
            h = torch.nn.functional.gelu(norm(h))
            x = h if h.shape == x.shape else h
        m = mask.unsqueeze(1).to(x.dtype)
        denom = m.sum(-1).clamp(min=1.0)
        mean = (x * m).sum(-1) / denom
        mx = (x.masked_fill(m == 0, -1e9)).max(-1).values
        g = torch.sigmoid(self.gate(torch.cat([mean, mx], dim=-1)))
        return self.head(g * torch.tanh(mean)).squeeze(-1)


def _inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(8, 16, 64, generator=g)
    mask = torch.zeros(8, 64, dtype=torch.bool)
    for i in range(8):
        mask[i, : 20 + 5 * i] = True  # varying true lengths, as real windows have
    return x, mask


@pytest.mark.skipif(not MPS, reason="MPS unavailable")
def test_forward_parity_within_tolerance():
    torch.manual_seed(0)
    model = RepresentativeStack().eval()
    x, mask = _inputs()

    with torch.no_grad():
        cpu_out = model(x, mask)
        model_mps = model.to("mps")
        mps_out = model_mps(x.to("mps"), mask.to("mps")).cpu()

    delta = (cpu_out - mps_out).abs().max().item()
    assert delta < TOL, (
        f"CPU/MPS divergence {delta:.3e} exceeds asserted tolerance {TOL:.1e}. "
        "Either a kernel changed or the model uses an op that degrades on MPS; "
        "record the cause before loosening this bound."
    )


@pytest.mark.skipif(not MPS, reason="MPS unavailable")
def test_seeded_run_is_reproducible_on_mps():
    """Same seed, same device, same numbers - otherwise no result is replicable."""
    outs = []
    for _ in range(2):
        torch.manual_seed(123)
        torch.mps.manual_seed(123)
        model = RepresentativeStack().to("mps").eval()
        x, mask = _inputs(seed=1)
        with torch.no_grad():
            outs.append(model(x.to("mps"), mask.to("mps")).cpu())
    assert torch.allclose(outs[0], outs[1], atol=0, rtol=0), "MPS run is not reproducible"


def test_cpu_run_is_reproducible():
    outs = []
    for _ in range(2):
        torch.manual_seed(123)
        model = RepresentativeStack().eval()
        x, mask = _inputs(seed=1)
        with torch.no_grad():
            outs.append(model(x, mask))
    assert torch.allclose(outs[0], outs[1], atol=0, rtol=0)
