"""Gated Multimodal Unit fusion, with availability masking.

Absent modalities are excluded from the softmax entirely (masked to -inf before
normalising), not given a small residual weight - the gate values that survive
are a genuine, sum-to-one attribution over the modalities that were actually
present for that window, which is what the demo and the viva need to explain a
verdict ("this alert fired mostly because of M2").
"""

from __future__ import annotations

import torch
from torch import nn


class GatedMultimodalUnit(nn.Module):
    """Fuses N per-modality embeddings into one, with a learned, availability-
    aware attention-style gate."""

    def __init__(self, n_modalities: int, embed_dim: int) -> None:
        super().__init__()
        self.n_modalities = n_modalities
        self.embed_dim = embed_dim
        self.gate_net = nn.Sequential(
            nn.Linear(n_modalities * embed_dim + n_modalities, 2 * n_modalities),
            nn.GELU(),
            nn.Linear(2 * n_modalities, n_modalities),
        )

    def forward(
        self, embeds: torch.Tensor, availability: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``embeds``: (B, n_modalities, embed_dim). ``availability``: (B, n_modalities)
        bool/float, 1 where that modality's data was present for that window.

        Returns ``(fused, gates)`` - ``gates`` is (B, n_modalities), sums to 1 over
        the available modalities per row, and IS the attribution the Prediction
        contract requires.
        """
        b = embeds.shape[0]
        flat = embeds.reshape(b, -1)
        avail_f = availability.float()
        gate_input = torch.cat([flat, avail_f], dim=-1)
        logits = self.gate_net(gate_input)

        # Mask absent modalities to -inf BEFORE softmax, so they receive exactly
        # zero weight rather than a small nonzero residual - "not present" must
        # mean the model literally cannot lean on it, not merely down-weight it.
        neg_inf = torch.finfo(logits.dtype).min
        masked_logits = logits.masked_fill(avail_f < 0.5, neg_inf)

        # A row with EVERY modality absent (should not happen given M3 is always
        # derived and always available, but guarded rather than assumed) would
        # softmax an all -inf row to NaN; fall back to uniform weight over all
        # modalities for that row alone rather than propagating NaN into training.
        all_absent = avail_f.sum(dim=-1, keepdim=True) < 0.5
        safe_logits = torch.where(all_absent, torch.zeros_like(logits), masked_logits)
        gates = torch.softmax(safe_logits, dim=-1)

        fused = (gates.unsqueeze(-1) * torch.tanh(embeds)).sum(dim=1)
        return fused, gates
