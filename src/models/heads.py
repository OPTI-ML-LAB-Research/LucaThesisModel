"""Heads attached on top of the backbone feature vector.

Currently exposes:
    QuantificationHead -- maps backbone features to a composition simplex
                          (one entry per known compound, summing to 1).

The quantification head is intentionally small (~33K params) so the
backbone provides the representational capacity; the head's job is just
to project to the correct output space and enforce the simplex constraint
via softmax. Dropout in the head is also what powers MC-Dropout
uncertainty in T18.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantificationHead(nn.Module):
    """Project backbone feature vector onto a composition simplex.

    Architecture:
        Linear(feature_dim -> hidden_dim)
        ReLU
        Dropout(p)
        Linear(hidden_dim -> n_compounds)
        Softmax  (applied in forward; outputs a valid simplex)

    Args:
        feature_dim: Backbone output dimension (default 256).
        n_compounds: Number of compounds in the canonical order
            (default 6: Alanine, Asparagine, Aspartic Acid,
             Glutamic Acid, Histidine, Glucosamine).
        hidden_dim: Width of the single hidden layer.
        dropout_rate: Dropout probability applied between FC layers.

    Shape:
        Input:  (B, feature_dim)
        Output: (B, n_compounds)   simplex (each row sums to 1, all >= 0)
    """

    def __init__(
        self,
        feature_dim: int = 256,
        n_compounds: int = 6,
        hidden_dim: int = 128,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.n_compounds = n_compounds

        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, n_compounds)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier init -- standard for FC layers feeding softmax."""
        for m in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            z: Feature tensor from backbone, shape (B, feature_dim).

        Returns:
            Composition simplex of shape (B, n_compounds).
            Each row >= 0 elementwise and sums to 1 within float precision.
        """
        if z.ndim != 2 or z.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected (B, {self.feature_dim}), got {tuple(z.shape)}."
            )
        h = self.fc1(z)
        h = self.relu(h)
        h = self.dropout(h)
        logits = self.fc2(h)                # (B, n_compounds)
        composition = F.softmax(logits, dim=-1)   # (B, n_compounds)
        return composition


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    head = QuantificationHead(feature_dim=256, n_compounds=6, dropout_rate=0.2)
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[T11] Trainable parameters: {n_params:,}")

    z = torch.randn(8, 256)
    head.eval()  # deterministic for shape check
    with torch.no_grad():
        alpha = head(z)
    print(f"[T11] Input shape : {tuple(z.shape)}")
    print(f"[T11] Output shape: {tuple(alpha.shape)}")
    assert alpha.shape == (8, 6)

    row_sums = alpha.sum(dim=-1)
    print(f"[T11] Row sums    : {row_sums.tolist()}")
    assert torch.allclose(row_sums, torch.ones(8), atol=1e-6), \
        "Composition rows should sum to 1"
    assert (alpha >= 0).all(), "Composition entries must be non-negative"
    assert (alpha <= 1).all(), "Composition entries must be <= 1"
    print("[T11] Smoke test PASSED")
