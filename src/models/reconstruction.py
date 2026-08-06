"""Physics-informed reconstruction module.

Implements the linear-mixture model from Beer-Lambert:

    s_recon[b, p] = sum_i ( alpha[b, i] * scale[i] * pure_ref[i, p] )

where:
    alpha[b, i]    = predicted composition (from QuantificationHead)
    scale[i]       = learnable per-compound multiplier (init = 1.0)
    pure_ref[i, p] = reference spectrum of compound i at pixel p
                     (loaded from engine/reference_spectra.npy, FROZEN)
    s_recon[b, p]  = reconstructed spectrum

The reconstructed spectrum feeds into the physics loss (T13), giving the
model a concrete, interpretable signal that its composition prediction
must "explain" the input spectrum as a linear mixture of pure components.

Design notes:
    * pure_ref is held as a buffer (not nn.Parameter), so it moves with
      ``model.to(device)`` but is excluded from optimizer.param_groups
      and from ``model.state_dict()``-based weight loading conflicts.
    * scale is the only learnable thing here; it absorbs:
          - small concentration -> intensity miscalibration
          - normalisation differences between ENLIGHTEN refs and the
            preprocessed AA spectra (despite both being SNV'd).
    * Output shape is (B, P), without channel dim. The full model adds
      the channel dim back when feeding into the physics loss alongside
      the original input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


class ReconstructionModule(nn.Module):
    """Linear-mixture spectrum reconstruction from composition.

    Args:
        reference_spectra: One of:
            * Path / str -- ``.npy`` file with shape (n_compounds, P).
            * np.ndarray of shape (n_compounds, P).
            * torch.Tensor of shape (n_compounds, P).
        n_compounds: Expected number of compounds. Sanity-checked against
            the reference tensor.
        spectrum_length: Expected spectrum length P. Sanity-checked.
        learnable_scale: If True, per-compound scale factors are trained
            (default). If False, scales are frozen at 1.0.
        scale_init: Initial value for each scale factor.

    Shape:
        Input alpha: (B, n_compounds)   -- composition simplex
        Output:      (B, P)             -- reconstructed spectrum
    """

    def __init__(
        self,
        reference_spectra: Union[str, Path, np.ndarray, torch.Tensor],
        n_compounds: int = 6,
        spectrum_length: int = 1024,
        learnable_scale: bool = True,
        scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_compounds = n_compounds
        self.spectrum_length = spectrum_length

        ref_tensor = self._coerce_to_tensor(reference_spectra)
        if ref_tensor.shape != (n_compounds, spectrum_length):
            raise ValueError(
                f"reference_spectra has shape {tuple(ref_tensor.shape)} but "
                f"({n_compounds}, {spectrum_length}) was expected."
            )

        # Reference spectra: held as buffer (non-trainable, moves with .to(device)).
        self.register_buffer("pure_ref", ref_tensor)

        # Per-compound scale factors. nn.Parameter so they show up in
        # state_dict and the optimizer; requires_grad toggled by flag.
        scale = torch.full((n_compounds,), float(scale_init))
        self.scale = nn.Parameter(scale, requires_grad=learnable_scale)

    @staticmethod
    def _coerce_to_tensor(
        ref: Union[str, Path, np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """Load reference spectra from any supported source -> float32 tensor."""
        if isinstance(ref, (str, Path)):
            path = Path(ref)
            if not path.exists():
                raise FileNotFoundError(
                    f"Reference spectra file not found: {path}. "
                    f"Build it via scripts/extract_pure_references.py."
                )
            arr = np.load(path)
            return torch.as_tensor(arr, dtype=torch.float32)
        if isinstance(ref, np.ndarray):
            return torch.as_tensor(ref, dtype=torch.float32)
        if isinstance(ref, torch.Tensor):
            return ref.detach().to(dtype=torch.float32)
        raise TypeError(
            f"Unsupported reference_spectra type: {type(ref)}."
        )

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        """Reconstruct spectra from compositions.

        Args:
            alpha: Composition tensor, shape (B, n_compounds). Should be
                a valid simplex but the module does not enforce it
                (caller is responsible).

        Returns:
            Reconstructed spectra, shape (B, spectrum_length).
        """
        if alpha.ndim != 2 or alpha.shape[1] != self.n_compounds:
            raise ValueError(
                f"Expected alpha shape (B, {self.n_compounds}), "
                f"got {tuple(alpha.shape)}."
            )

        # Effective per-compound spectrum: (n_compounds, P)
        # = scale[i] * pure_ref[i, :]
        scaled_refs = self.scale.unsqueeze(-1) * self.pure_ref   # (n_compounds, P)

        # Linear mixture: alpha @ scaled_refs   (B, n_compounds) @ (n_compounds, P)
        s_recon = alpha @ scaled_refs                            # (B, P)
        return s_recon


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # Use a synthetic, easily-recognisable reference set so we can verify
    # the linear mixture algebraically.
    P = 1024
    n_compounds = 6
    refs = np.zeros((n_compounds, P), dtype=np.float32)
    for i in range(n_compounds):
        refs[i, i * 100:(i + 1) * 100] = 1.0    # disjoint plateaux

    recon = ReconstructionModule(
        reference_spectra=refs,
        n_compounds=n_compounds,
        spectrum_length=P,
        learnable_scale=True,
    )
    print(f"[T12] Buffer  shape: {tuple(recon.pure_ref.shape)}")
    print(f"[T12] Scale   shape: {tuple(recon.scale.shape)}")
    print(f"[T12] Scale   init : {recon.scale.tolist()}")

    # Test 1: pure compound 0 -> output equals refs[0]
    alpha_pure = torch.zeros(1, n_compounds)
    alpha_pure[0, 0] = 1.0
    out_pure = recon(alpha_pure)
    print(f"[T12] Output shape (pure A): {tuple(out_pure.shape)}")
    assert out_pure.shape == (1, P)
    assert torch.allclose(out_pure[0], torch.as_tensor(refs[0]), atol=1e-6), \
        "Pure compound output must equal its reference"

    # Test 2: 50/30/20 mix of first three compounds
    alpha_mix = torch.tensor([[0.5, 0.3, 0.2, 0.0, 0.0, 0.0]])
    out_mix = recon(alpha_mix)
    expected = 0.5 * refs[0] + 0.3 * refs[1] + 0.2 * refs[2]
    assert torch.allclose(out_mix[0], torch.as_tensor(expected), atol=1e-6), \
        "Mixture output must equal weighted sum of refs"
    print(f"[T12] Mix (0.5, 0.3, 0.2, 0, 0, 0) -> linear combination verified")

    # Test 3: scale propagates
    with torch.no_grad():
        recon.scale[0] = 2.0
    out_scaled = recon(alpha_pure)
    assert torch.allclose(out_scaled[0], 2.0 * torch.as_tensor(refs[0]), atol=1e-6), \
        "Scale factor must multiply through"
    print(f"[T12] Scale=2.0 on compound 0 doubles the output -- OK")

    print("[T12] Smoke test PASSED")
