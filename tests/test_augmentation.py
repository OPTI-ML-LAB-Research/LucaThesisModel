"""Pytest suite for T16 augmentation.

Run from project root:
    pytest tests/test_augmentation.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from src.data.augmentation import (
    AugmentedDataset,
    RamanAugmentation,
    _shift_1d,
)


@pytest.fixture
def peak_spectrum() -> torch.Tensor:
    """1024-pixel spectrum with a single peak of height 1.0 at index 500."""
    s = torch.zeros(1024)
    s[500] = 1.0
    return s


# ---------------------------------------------------------------------------
# _shift_1d primitive
# ---------------------------------------------------------------------------

class TestShiftPrimitive:

    def test_zero_shift_identity(self, peak_spectrum):
        assert torch.equal(_shift_1d(peak_spectrum, 0), peak_spectrum)

    def test_positive_shift(self, peak_spectrum):
        out = _shift_1d(peak_spectrum, 5)
        assert out[505].item() == 1.0
        assert out[500].item() == 0.0

    def test_negative_shift(self, peak_spectrum):
        out = _shift_1d(peak_spectrum, -5)
        assert out[495].item() == 1.0
        assert out[500].item() == 0.0

    def test_shift_off_the_end_loses_signal(self, peak_spectrum):
        out = _shift_1d(peak_spectrum, 1024)
        assert out.sum().item() == 0.0


# ---------------------------------------------------------------------------
# RamanAugmentation
# ---------------------------------------------------------------------------

class TestRamanAugmentation:

    def test_shift_only_within_bounds(self, peak_spectrum):
        aug = RamanAugmentation(
            shift_max_px=6, shift_p=1.0, scale_p=0.0, noise_p=0.0, seed=0,
        )
        for _ in range(20):
            out = aug(peak_spectrum)
            pos = int(out.argmax().item())
            assert abs(pos - 500) <= 6
            # The peak's value (or close to it after no scaling) survives.
            assert out[pos].item() == 1.0

    def test_scale_factor_applies(self, peak_spectrum):
        aug = RamanAugmentation(
            shift_p=0.0, scale_p=1.0, intensity_range=(2.0, 2.0),
            noise_sigma=0.0, noise_p=0.0, seed=0,
        )
        out = aug(peak_spectrum)
        assert abs(out.max().item() - 2.0) < 1e-6

    def test_noise_std_in_range(self, peak_spectrum):
        aug = RamanAugmentation(
            shift_p=0.0, scale_p=0.0,
            noise_sigma=0.05, noise_p=1.0, seed=0,
        )
        out = aug(peak_spectrum)
        bg_std = out[torch.arange(1024) != 500].std().item()
        assert 0.04 < bg_std < 0.06   # ~3-sigma tolerance for 1023 samples

    def test_all_off_is_identity(self, peak_spectrum):
        aug = RamanAugmentation(
            shift_p=0.0, scale_p=0.0, noise_p=0.0,
            noise_sigma=0.0, seed=0,
        )
        out = aug(peak_spectrum)
        assert torch.equal(out, peak_spectrum)

    def test_seed_reproducibility(self, peak_spectrum):
        a = RamanAugmentation(seed=42)
        b = RamanAugmentation(seed=42)
        assert torch.equal(a(peak_spectrum), b(peak_spectrum))

    def test_different_seeds_diverge(self, peak_spectrum):
        # Make all probs 1 so we definitely sample.
        a = RamanAugmentation(seed=1, shift_p=1.0, scale_p=1.0, noise_p=1.0,
                              noise_sigma=0.01)
        b = RamanAugmentation(seed=2, shift_p=1.0, scale_p=1.0, noise_p=1.0,
                              noise_sigma=0.01)
        assert not torch.equal(a(peak_spectrum), b(peak_spectrum))

    def test_rng_advances_between_calls(self, peak_spectrum):
        # All probs = 1 + noise on -> two consecutive calls must differ.
        aug = RamanAugmentation(seed=0, shift_p=1.0, scale_p=1.0, noise_p=1.0,
                                noise_sigma=0.01)
        a = aug(peak_spectrum)
        b = aug(peak_spectrum)
        assert not torch.equal(a, b)

    def test_shape_preserved(self, peak_spectrum):
        aug = RamanAugmentation(seed=0)
        assert aug(peak_spectrum).shape == peak_spectrum.shape

    def test_rejects_2d_input(self):
        aug = RamanAugmentation(seed=0)
        with pytest.raises(ValueError, match="1D"):
            aug(torch.randn(4, 1024))

    def test_validates_intensity_range(self):
        with pytest.raises(ValueError):
            RamanAugmentation(intensity_range=(1.5, 0.5))   # lo > hi
        with pytest.raises(ValueError):
            RamanAugmentation(intensity_range=(-0.1, 1.0))  # negative

    def test_validates_probabilities(self):
        with pytest.raises(ValueError):
            RamanAugmentation(shift_p=1.5)
        with pytest.raises(ValueError):
            RamanAugmentation(scale_p=-0.1)


# ---------------------------------------------------------------------------
# AugmentedDataset wrapper
# ---------------------------------------------------------------------------

class TestAugmentedDataset:

    def test_wraps_2d_spectra(self):
        X = torch.randn(20, 1024)
        Y = torch.rand(20, 6); Y = Y / Y.sum(-1, keepdim=True)
        base = TensorDataset(X, Y)
        ds = AugmentedDataset(base, RamanAugmentation(seed=0),
                              keep_channel_dim=False)
        s, y = ds[0]
        assert s.shape == (1024,)
        assert torch.equal(y, Y[0])

    def test_wraps_3d_spectra(self):
        X = torch.randn(20, 1, 1024)
        Y = torch.rand(20, 6); Y = Y / Y.sum(-1, keepdim=True)
        base = TensorDataset(X, Y)
        ds = AugmentedDataset(base, RamanAugmentation(seed=0))
        s, y = ds[0]
        assert s.shape == (1, 1024)

    def test_label_passes_through_unchanged(self):
        X = torch.randn(5, 1024)
        Y = torch.tensor([
            [0.5, 0.3, 0.2, 0, 0, 0],
            [0.1, 0.1, 0.1, 0.7, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
            [0, 0, 0, 0, 0, 1],
        ], dtype=torch.float32)
        base = TensorDataset(X, Y)
        ds = AugmentedDataset(base, RamanAugmentation(seed=0))
        for i in range(5):
            _, y = ds[i]
            assert torch.equal(y, Y[i])

    def test_length_matches_base(self):
        base = TensorDataset(torch.randn(37, 1024), torch.rand(37, 6))
        ds = AugmentedDataset(base, RamanAugmentation(seed=0))
        assert len(ds) == 37

    def test_returns_tuple_when_base_returns_tuple(self):
        X = torch.randn(5, 1024)
        Y = torch.rand(5, 6)
        meta = torch.arange(5)
        # 3-tuple base (e.g. spectra, labels, sample_id).
        base = TensorDataset(X, Y, meta)
        ds = AugmentedDataset(base, RamanAugmentation(seed=0),
                              keep_channel_dim=False)
        out = ds[2]
        assert isinstance(out, tuple) and len(out) == 3
        assert torch.equal(out[2], meta[2])
