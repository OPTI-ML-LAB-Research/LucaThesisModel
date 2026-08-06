"""Raman spectrum augmentation for training.

Three transforms (each independently triggered by its own probability):

    * RandomShift     -- circular / zero-padded pixel shift along wavenumber axis.
                         Default: +/-10 cm-1 -> approx +/-6 pixels on the AA grid.
    * IntensityScale  -- multiplicative scaling of the entire spectrum.
                         Default: factor in [0.9, 1.1].
    * GaussianNoise   -- additive white noise of fixed sigma.
                         Default: sigma=0.005 (small, since SNV-normalised).

Composition labels are NEVER touched -- mixing ratios are invariant under
shift / scaling / noise. The label-spectrum correspondence is preserved.

Usage in a DataLoader pipeline:

    from src.data.augmentation import RamanAugmentation, AugmentedDataset

    aug = RamanAugmentation(
        shift_max_px=6, intensity_range=(0.9, 1.1), noise_sigma=0.005,
        shift_p=0.5, scale_p=0.5, noise_p=0.5, seed=42,
    )
    train_ds_aug = AugmentedDataset(train_ds, aug)        # only train!
    train_loader = DataLoader(train_ds_aug, batch_size=64, shuffle=True)

Notes:
    * Per-sample (not per-batch) randomness: each sample in a batch sees
      an independent draw of (shift, scale, noise).
    * Augmentation runs on CPU inside the DataLoader workers; the GPU is
      not involved. This is fine -- ops are cheap.
    * Determinism: pass an integer ``seed`` to make a run reproducible.
      Each call to ``__call__`` advances the internal RNG, so different
      epochs see different draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shift_1d(spectrum: torch.Tensor, shift_px: int) -> torch.Tensor:
    """Shift a 1D spectrum by ``shift_px`` pixels, zero-padding the gap.

    Positive shift moves the signal to higher pixel indices (right);
    negative shift moves to lower (left). The gap left at the boundary
    is filled with zeros (for SNV-normalised spectra this is consistent
    with "no peak"; for raw spectra one would prefer mode='edge' but we
    operate post-preprocessing).

    Args:
        spectrum: shape (P,)
        shift_px: integer pixel offset.

    Returns:
        Shifted spectrum, shape (P,).
    """
    if shift_px == 0:
        return spectrum
    out = torch.zeros_like(spectrum)
    P = spectrum.shape[0]
    if shift_px > 0:
        out[shift_px:] = spectrum[: P - shift_px]
    else:
        k = -shift_px
        out[: P - k] = spectrum[k:]
    return out


# ---------------------------------------------------------------------------
# Augmentation transform
# ---------------------------------------------------------------------------

@dataclass
class RamanAugmentation:
    """Composable augmentation for 1D Raman spectra.

    All probabilities default to 0.5; transforms are independent (any
    subset can fire on a given call). Set a probability to 0 to disable
    that transform; set to 1 to always apply it.

    Args:
        shift_max_px: Maximum absolute pixel shift (uniform integer in
            [-shift_max_px, +shift_max_px]). Default 6, which corresponds
            to ~10 cm-1 on the AA grid (1.7 cm-1/pixel).
        intensity_range: ``(low, high)`` for multiplicative scale factor.
        noise_sigma: Standard deviation of additive Gaussian noise.
        shift_p / scale_p / noise_p: Per-transform application probability.
        seed: Optional RNG seed for reproducibility.
    """

    shift_max_px: int = 6
    intensity_range: Tuple[float, float] = (0.9, 1.1)
    noise_sigma: float = 0.005
    shift_p: float = 0.5
    scale_p: float = 0.5
    noise_p: float = 0.5
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        # Each instance gets its own RNG -- avoids contention between
        # workers and lets the user fully control determinism.
        self._rng = np.random.default_rng(self.seed)

        # Validate.
        if self.shift_max_px < 0:
            raise ValueError("shift_max_px must be >= 0")
        lo, hi = self.intensity_range
        if not (0 < lo <= hi):
            raise ValueError(
                f"intensity_range must be 0 < lo <= hi, got {self.intensity_range}"
            )
        if self.noise_sigma < 0:
            raise ValueError("noise_sigma must be >= 0")
        for name, p in [("shift_p", self.shift_p),
                        ("scale_p", self.scale_p),
                        ("noise_p", self.noise_p)]:
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {p}")

    def __call__(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to a single spectrum.

        Args:
            spectrum: shape (P,) -- a single sample. (For batched calls,
                wrap in ``AugmentedDataset`` so each sample gets fresh
                random draws.)

        Returns:
            Augmented spectrum, shape (P,).
        """
        if spectrum.ndim != 1:
            raise ValueError(
                f"Expected 1D spectrum (P,), got shape {tuple(spectrum.shape)}."
            )
        out = spectrum

        # 1. Shift.
        if self.shift_max_px > 0 and self._rng.random() < self.shift_p:
            shift = int(self._rng.integers(
                -self.shift_max_px, self.shift_max_px + 1
            ))
            out = _shift_1d(out, shift)

        # 2. Intensity scale.
        if self._rng.random() < self.scale_p:
            lo, hi = self.intensity_range
            scale = float(self._rng.uniform(lo, hi))
            out = out * scale

        # 3. Additive Gaussian noise.
        if self.noise_sigma > 0 and self._rng.random() < self.noise_p:
            noise = torch.from_numpy(
                self._rng.normal(0.0, self.noise_sigma, size=out.shape)
                .astype(np.float32)
            )
            out = out + noise

        return out


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class AugmentedDataset(Dataset):
    """Wraps a base Dataset and applies augmentation to each spectrum.

    The base dataset is expected to return ``(spectrum, label)`` tuples
    with ``spectrum`` of shape ``(P,)`` or ``(1, P)``. Labels are passed
    through unchanged.

    Args:
        base: The underlying Dataset (typically the training split).
        augmentation: A ``RamanAugmentation`` (or any callable taking
            ``(P,)`` -> ``(P,)``).
        keep_channel_dim: If True, output retains its channel dim
            ``(1, P)``; if False, output is ``(P,)``. Auto-detected from
            base dataset's first sample if None (default).
    """

    def __init__(
        self,
        base: Dataset,
        augmentation: RamanAugmentation,
        keep_channel_dim: Optional[bool] = None,
    ) -> None:
        self.base = base
        self.augmentation = augmentation

        if keep_channel_dim is None:
            sample = base[0]
            spec = sample[0] if isinstance(sample, (tuple, list)) else sample
            keep_channel_dim = (spec.ndim == 2 and spec.shape[0] == 1)
        self.keep_channel_dim = keep_channel_dim

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        sample = self.base[idx]
        if isinstance(sample, (tuple, list)):
            spec = sample[0]
            rest = tuple(sample[1:])
        else:
            spec, rest = sample, ()

        had_channel = (spec.ndim == 2 and spec.shape[0] == 1)
        if had_channel:
            spec = spec.squeeze(0)              # (1, P) -> (P,)

        spec = self.augmentation(spec)          # (P,) -> (P,)

        if self.keep_channel_dim:
            spec = spec.unsqueeze(0)            # (P,) -> (1, P)

        if rest:
            return (spec,) + rest
        return spec


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    P = 1024
    # Build a recognisable spectrum: a single peak at index 500, height 1.
    spec = torch.zeros(P)
    spec[500] = 1.0

    # ---- Test 1: shift only ----
    aug_shift = RamanAugmentation(
        shift_max_px=6, shift_p=1.0, scale_p=0.0, noise_p=0.0, seed=0,
    )
    out = aug_shift(spec)
    peak_pos = int(out.argmax().item())
    print(f"[T16] shift only: peak moved from 500 -> {peak_pos} "
          f"(expected within +/-6)")
    assert abs(peak_pos - 500) <= 6
    assert out[peak_pos].item() == 1.0       # value preserved

    # ---- Test 2: scale only ----
    aug_scale = RamanAugmentation(
        shift_p=0.0, scale_p=1.0, intensity_range=(2.0, 2.0),
        noise_sigma=0.0, noise_p=0.0, seed=0,
    )
    out = aug_scale(spec)
    assert abs(out.max().item() - 2.0) < 1e-6, \
        f"Expected scaled peak = 2.0, got {out.max().item()}"
    print(f"[T16] scale only (factor=2.0): peak height = {out.max().item():.4f}")

    # ---- Test 3: noise only ----
    aug_noise = RamanAugmentation(
        shift_p=0.0, scale_p=0.0, noise_sigma=0.1, noise_p=1.0, seed=0,
    )
    out = aug_noise(spec)
    # Estimate empirical std at non-peak positions.
    bg_std = out[torch.arange(P) != 500].std().item()
    print(f"[T16] noise only: empirical bg std = {bg_std:.4f} (target ~0.1)")
    assert 0.07 < bg_std < 0.13

    # ---- Test 4: all three composed ----
    aug_all = RamanAugmentation(
        shift_max_px=6, intensity_range=(0.9, 1.1),
        noise_sigma=0.005, seed=42,
    )
    out = aug_all(spec)
    assert out.shape == (P,)
    assert torch.isfinite(out).all()
    print(f"[T16] composed augmentation: shape OK, no NaN")

    # ---- Test 5: probabilities respected ----
    aug_off = RamanAugmentation(
        shift_p=0.0, scale_p=0.0, noise_p=0.0,
        noise_sigma=0.0, seed=0,
    )
    out = aug_off(spec)
    assert torch.equal(out, spec), "All-off augmentation should be identity"
    print(f"[T16] all probabilities=0 -> identity OK")

    # ---- Test 6: AugmentedDataset wrapper ----
    from torch.utils.data import TensorDataset
    X = torch.randn(20, P)
    Y = torch.rand(20, 6); Y = Y / Y.sum(-1, keepdim=True)
    base = TensorDataset(X, Y)
    aug = RamanAugmentation(seed=0)
    wrapped = AugmentedDataset(base, aug, keep_channel_dim=False)
    s0, y0 = wrapped[0]
    assert s0.shape == (P,), f"Expected (P,), got {tuple(s0.shape)}"
    assert torch.equal(y0, Y[0]), "Label must pass through unchanged"
    print(f"[T16] AugmentedDataset wrapper: spec shape {tuple(s0.shape)}, "
          f"label preserved")

    # ---- Test 7: keep_channel_dim=True path ----
    X3 = torch.randn(20, 1, P)
    base3 = TensorDataset(X3, Y)
    wrapped3 = AugmentedDataset(base3, RamanAugmentation(seed=0))
    s3, _ = wrapped3[0]
    assert s3.shape == (1, P), f"Expected (1, P), got {tuple(s3.shape)}"
    print(f"[T16] AugmentedDataset auto-detected channel dim: "
          f"{tuple(s3.shape)}")

    # ---- Test 8: determinism with seed ----
    aug_a = RamanAugmentation(seed=123)
    aug_b = RamanAugmentation(seed=123)
    out_a = aug_a(spec); out_b = aug_b(spec)
    assert torch.equal(out_a, out_b), "Same seed -> identical augmentation"
    print(f"[T16] seed=123 reproducibility OK")

    print("[T16] All smoke tests PASSED")
