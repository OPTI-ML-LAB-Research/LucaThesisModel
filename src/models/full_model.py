"""End-to-end Raman Physics-Informed AI model.

Composition:

    Input spectrum (B, 1, P)
        |
        v
    ResNet1DBackbone           -> feature  (B, 256)
        |
        v
    QuantificationHead         -> composition (B, n_compounds), simplex
        |
        v
    ReconstructionModule       -> reconstruction (B, P)
        (uses pure references)

The forward returns a dict so callers (training loop, inference, MC-Dropout
wrapper) can pull whichever pieces they need without unpacking magic.

A factory ``build_full_model_from_config`` constructs the model from the
top-level YAML config -- callers usually go through it rather than the
class constructor directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from .backbone import ResNet1DBackbone
from .heads import QuantificationHead
from .reconstruction import ReconstructionModule


class RamanPhysicsAI(nn.Module):
    """Full pipeline: backbone -> quantification head -> reconstruction.

    Args:
        reference_spectra: Path / array / tensor of shape (n_compounds, P).
        n_compounds: Number of compounds (default 6).
        spectrum_length: Spectrum length P (default 1024).
        feature_dim: Backbone output dimension (default 256).
        head_hidden_dim: Width of the quantification head's hidden layer.
        backbone_dropout: Dropout in the backbone (after global pool).
        head_dropout: Dropout in the quantification head.
        reconstruction_learnable_scale: If True, per-compound scales are
            trained (recommended).

    Forward output:
        Dict with keys:
            "composition"    -- (B, n_compounds) simplex
            "reconstruction" -- (B, P) reconstructed spectrum
            "feature"        -- (B, feature_dim) backbone feature
    """

    def __init__(
        self,
        reference_spectra: Union[str, Path, np.ndarray, torch.Tensor],
        n_compounds: int = 6,
        spectrum_length: int = 1024,
        feature_dim: int = 256,
        head_hidden_dim: int = 128,
        backbone_dropout: float = 0.2,
        head_dropout: float = 0.2,
        reconstruction_learnable_scale: bool = True,
    ) -> None:
        super().__init__()
        self.n_compounds = n_compounds
        self.spectrum_length = spectrum_length
        self.feature_dim = feature_dim

        self.backbone = ResNet1DBackbone(
            in_channels=1,
            feature_dim=feature_dim,
            dropout_rate=backbone_dropout,
        )
        self.quantification_head = QuantificationHead(
            feature_dim=feature_dim,
            n_compounds=n_compounds,
            hidden_dim=head_hidden_dim,
            dropout_rate=head_dropout,
        )
        self.reconstruction = ReconstructionModule(
            reference_spectra=reference_spectra,
            n_compounds=n_compounds,
            spectrum_length=spectrum_length,
            learnable_scale=reconstruction_learnable_scale,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """End-to-end forward pass.

        Args:
            x: Input spectra. Accepts (B, P) or (B, 1, P); a missing
                channel dim is added automatically.

        Returns:
            Dict with keys ``composition`` (B, n_compounds), ``reconstruction``
            (B, P), and ``feature`` (B, feature_dim).
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)            # (B, P) -> (B, 1, P)
        elif x.ndim != 3:
            raise ValueError(
                f"Expected (B, P) or (B, 1, P), got shape {tuple(x.shape)}."
            )
        if x.shape[-1] != self.spectrum_length:
            raise ValueError(
                f"Expected spectrum_length={self.spectrum_length}, "
                f"got {x.shape[-1]}."
            )

        feature = self.backbone(x)                          # (B, feature_dim)
        composition = self.quantification_head(feature)     # (B, n_compounds)
        recon = self.reconstruction(composition)            # (B, P)

        return {
            "composition": composition,
            "reconstruction": recon,
            "feature": feature,
        }

    # -- Utilities --------------------------------------------------------

    def count_parameters(self, trainable_only: bool = True) -> Dict[str, int]:
        """Per-submodule parameter counts (and total).

        Returns:
            Dict {"backbone", "head", "reconstruction", "total"}.
        """
        def _count(mod: nn.Module) -> int:
            it = mod.parameters()
            if trainable_only:
                return sum(p.numel() for p in it if p.requires_grad)
            return sum(p.numel() for p in it)

        counts = {
            "backbone": _count(self.backbone),
            "head": _count(self.quantification_head),
            "reconstruction": _count(self.reconstruction),
        }
        counts["total"] = sum(counts.values())
        return counts


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_full_model_from_config(
    config: Dict[str, Any],
    reference_spectra_path: Optional[Union[str, Path]] = None,
) -> RamanPhysicsAI:
    """Construct a RamanPhysicsAI from a (parsed) YAML config dict.

    The config is expected to follow the schema of ``configs/default.yaml``
    (Custom Instructions Section 5). Specifically these keys are read::

        data:
            spectrum_length
        model:
            feature_dim
            dropout_rate
        # n_compounds is taken from len(data.label_cols) when present,
        # otherwise defaults to 6.

    Args:
        config: Parsed config dict.
        reference_spectra_path: Path to ``reference_spectra.npy``. If not
            given, falls back to ``config['paths']['reference_spectra']``,
            then to ``engine/reference_spectra.npy``.

    Returns:
        Initialised RamanPhysicsAI on CPU. Caller calls ``.to(device)``.
    """
    spectrum_length = (
        config.get("data", {}).get("spectrum_length", 1024)
    )
    model_cfg = config.get("model", {}) or {}
    feature_dim = model_cfg.get("feature_dim", 256)
    dropout_rate = model_cfg.get("dropout_rate", 0.2)

    label_cols = (
        config.get("data", {}).get("label_cols")
        or config.get("datasets", {}).get("registry", {})
            .get(config.get("datasets", {}).get("primary", ""), {})
            .get("label_cols")
    )
    n_compounds = len(label_cols) if label_cols else 6

    if reference_spectra_path is None:
        reference_spectra_path = (
            config.get("paths", {}).get("reference_spectra")
            or "engine/reference_spectra.npy"
        )

    return RamanPhysicsAI(
        reference_spectra=reference_spectra_path,
        n_compounds=n_compounds,
        spectrum_length=spectrum_length,
        feature_dim=feature_dim,
        backbone_dropout=dropout_rate,
        head_dropout=dropout_rate,
        reconstruction_learnable_scale=True,
    )


# ---------------------------------------------------------------------------
# Smoke test (end-to-end forward + grad flow)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # Synthetic refs (no real engine/reference_spectra.npy needed for the test).
    P = 1024
    n_compounds = 6
    refs = np.random.RandomState(0).randn(n_compounds, P).astype(np.float32)

    model = RamanPhysicsAI(
        reference_spectra=refs,
        n_compounds=n_compounds,
        spectrum_length=P,
        feature_dim=256,
    )

    counts = model.count_parameters()
    print("[T14] Parameter counts:")
    for k, v in counts.items():
        print(f"        {k:>16}: {v:>10,}")
    assert counts["total"] < 5_000_000, f"Model too large: {counts['total']:,}"
    # Reconstruction module should contribute exactly n_compounds learnable
    # params (the scale vector); the pure_ref buffer is excluded.
    assert counts["reconstruction"] == n_compounds, \
        f"Recon module should have only {n_compounds} learnable params"

    # ---- Test 1: 3D input ----
    x_3d = torch.randn(4, 1, P)
    out = model(x_3d)
    print(f"[T14] Input (B,1,P)={tuple(x_3d.shape)}")
    print(f"[T14] composition   shape: {tuple(out['composition'].shape)}")
    print(f"[T14] reconstruction shape: {tuple(out['reconstruction'].shape)}")
    print(f"[T14] feature        shape: {tuple(out['feature'].shape)}")
    assert out["composition"].shape == (4, n_compounds)
    assert out["reconstruction"].shape == (4, P)
    assert out["feature"].shape == (4, 256)
    assert torch.isfinite(out["composition"]).all()
    assert torch.isfinite(out["reconstruction"]).all()

    # Composition is a simplex.
    sums = out["composition"].sum(dim=-1)
    assert torch.allclose(sums, torch.ones(4), atol=1e-5), \
        f"Composition rows must sum to 1, got {sums.tolist()}"

    # ---- Test 2: 2D input is auto-promoted ----
    x_2d = torch.randn(2, P)
    out2 = model(x_2d)
    assert out2["composition"].shape == (2, n_compounds)

    # ---- Test 3: gradient flows back ----
    from src.training.losses import combined_loss
    y_true = torch.rand(4, n_compounds)
    y_true = y_true / y_true.sum(-1, keepdim=True)
    loss = combined_loss(
        y_true, out["composition"], x_3d, out["reconstruction"],
        model_parameters=model.parameters(),
    )
    loss.backward()

    # Verify at least one parameter in each submodule got a gradient.
    def grad_norm(mod): return sum(
        p.grad.abs().sum().item() for p in mod.parameters() if p.grad is not None
    )
    print(f"[T14] grad norm backbone : {grad_norm(model.backbone):.3e}")
    print(f"[T14] grad norm head     : {grad_norm(model.quantification_head):.3e}")
    print(f"[T14] grad norm recon    : {grad_norm(model.reconstruction):.3e}")
    assert grad_norm(model.backbone) > 0
    assert grad_norm(model.quantification_head) > 0
    assert grad_norm(model.reconstruction) > 0

    print("[T14] End-to-end smoke test PASSED")
