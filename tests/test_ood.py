"""Pytest suite for T19 OOD scorer (src/inference/ood.py).

Run from project root:
    pytest tests/test_ood.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.inference.ood import (
    OODCalibration,
    OODScorer,
    _normalise_clipped,
    compute_predictive_variance,
    compute_reconstruction_error,
    make_synthetic_ood,
)
from src.models.full_model import RamanPhysicsAI
from src.models.uncertainty import predict_with_uncertainty


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_refs() -> np.ndarray:
    P, K = 1024, 6
    return np.random.RandomState(0).randn(K, P).astype(np.float32) * 0.5


@pytest.fixture
def model(synthetic_refs) -> RamanPhysicsAI:
    torch.manual_seed(0)
    return RamanPhysicsAI(
        reference_spectra=synthetic_refs, n_compounds=6,
        spectrum_length=1024, feature_dim=256,
    ).eval()


@pytest.fixture
def cal_spectra(synthetic_refs) -> torch.Tensor:
    """50 synthetic ID-like spectra (random simplex * refs + small noise)."""
    rng = np.random.RandomState(1)
    samples = []
    for _ in range(50):
        alpha = np.abs(rng.randn(6)); alpha /= alpha.sum()
        x = (alpha[:, None] * synthetic_refs).sum(0)
        x = x + rng.normal(0, 0.05, 1024).astype(np.float32)
        samples.append(x)
    return torch.from_numpy(np.stack(samples).astype(np.float32))


@pytest.fixture
def cal_loader(cal_spectra):
    """A trivial 1-batch loader."""
    return [(cal_spectra,)]


# ---------------------------------------------------------------------------
# Component primitives
# ---------------------------------------------------------------------------

class TestComponentPrimitives:

    def test_recon_err_zero_for_identical(self):
        s = torch.randn(4, 1024)
        err = compute_reconstruction_error(s, s)
        assert err.shape == (4,)
        assert err.abs().max().item() < 1e-6

    def test_recon_err_two_for_negation(self):
        s = torch.randn(3, 1024)
        err = compute_reconstruction_error(s, -s)
        # cosine(s, -s) = -1, so err = 2.
        assert torch.allclose(err, torch.full((3,), 2.0), atol=1e-4)

    def test_recon_err_handles_3d(self):
        s2 = torch.randn(4, 1024)
        s3 = s2.unsqueeze(1)
        err1 = compute_reconstruction_error(s2, s2)
        err2 = compute_reconstruction_error(s3, s3)
        assert torch.allclose(err1, err2, atol=1e-6)

    def test_predictive_variance_passthrough(self, model):
        mc = predict_with_uncertainty(model, torch.randn(4, 1024), n_samples=5)
        v = compute_predictive_variance(mc)
        assert v.shape == (4,)
        assert torch.allclose(v, mc["mean_compound_std"])

    def test_normalise_clipped_caps_at_one(self):
        x = torch.tensor([0.0, 0.5, 1.0, 2.0, 5.0])
        out = _normalise_clipped(x, p95=1.0)
        assert torch.allclose(out, torch.tensor([0.0, 0.5, 1.0, 1.0, 1.0]))

    def test_normalise_clipped_zero_p95_safe(self):
        # Degenerate p95 -> identity (no div by zero).
        x = torch.tensor([0.0, 0.5, 1.0, 2.0])
        out = _normalise_clipped(x, p95=0.0)
        assert torch.allclose(out, x)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class TestCalibration:

    def test_calibrate_populates_percentiles(self, model, cal_loader, cal_spectra):
        scorer = OODScorer(model, mc_samples=5)
        cal = scorer.calibrate(cal_loader)
        assert isinstance(cal, OODCalibration)
        assert cal.n_calibration_samples == len(cal_spectra)
        assert cal.recon_p95 > 0
        assert cal.var_p95 > 0
        assert 0 < cal.score_p95 <= 1.0

    def test_calibrate_stores_on_self(self, model, cal_loader):
        scorer = OODScorer(model, mc_samples=5)
        assert scorer.calibration is None
        scorer.calibrate(cal_loader)
        assert scorer.calibration is not None

    def test_calibrate_records_weights_and_n(self, model, cal_loader):
        scorer = OODScorer(model, recon_weight=0.7, var_weight=0.3, mc_samples=5)
        cal = scorer.calibrate(cal_loader)
        assert cal.recon_weight == 0.7
        assert cal.var_weight == 0.3
        assert cal.mc_samples == 5

    def test_calibrate_empty_loader_raises(self, model):
        scorer = OODScorer(model, mc_samples=5)
        with pytest.raises(ValueError, match="zero samples"):
            scorer.calibrate([])

    def test_score_before_calibrate_raises(self, model):
        scorer = OODScorer(model, mc_samples=5)
        with pytest.raises(RuntimeError, match="not calibrated"):
            scorer.score(torch.randn(1024))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class TestScoring:

    @pytest.fixture
    def calibrated_scorer(self, model, cal_loader):
        s = OODScorer(model, mc_samples=5)
        s.calibrate(cal_loader)
        return s

    def test_score_single_returns_float(self, calibrated_scorer):
        out = calibrated_scorer.score(torch.randn(1024))
        assert isinstance(out, float)

    def test_score_batch_returns_tensor(self, calibrated_scorer):
        out = calibrated_scorer.score_batch(torch.randn(7, 1024))
        assert out.shape == (7,)

    def test_score_in_zero_to_one_range(self, calibrated_scorer):
        # By construction, normalised components are clipped to [0, 1],
        # weights are 0.6 + 0.4 = 1.0, so score in [0, 1].
        scores = calibrated_scorer.score_batch(torch.randn(20, 1024))
        assert (scores >= 0).all()
        assert (scores <= 1.0 + 1e-6).all()

    def test_components_dict_keys(self, calibrated_scorer):
        out = calibrated_scorer.score_batch(
            torch.randn(3, 1024), return_components=True,
        )
        expected = {"score", "recon_err_raw", "recon_err_norm",
                    "pred_var_raw", "pred_var_norm"}
        assert set(out.keys()) == expected
        for v in out.values():
            assert v.shape == (3,)

    def test_score_single_rejects_batch(self, calibrated_scorer):
        with pytest.raises(ValueError, match="single spectrum"):
            calibrated_scorer.score(torch.randn(4, 1024))

    def test_is_ood_returns_bool(self, calibrated_scorer):
        out = calibrated_scorer.is_ood(torch.randn(1024))
        assert isinstance(out, bool)

    def test_is_ood_batch_returns_bool_tensor(self, calibrated_scorer):
        out = calibrated_scorer.is_ood_batch(torch.randn(5, 1024))
        assert out.dtype == torch.bool
        assert out.shape == (5,)

    def test_score_accepts_numpy(self, calibrated_scorer):
        x = np.random.randn(1024).astype(np.float32)
        s = calibrated_scorer.score(x)
        assert isinstance(s, float)


# ---------------------------------------------------------------------------
# Discriminative power on synthetic OOD
# ---------------------------------------------------------------------------

class TestDiscrimination:
    """The "10 ID + 10 OOD" stretch validation from T19 spec."""

    @pytest.fixture
    def calibrated_scorer(self, model, cal_loader):
        s = OODScorer(model, mc_samples=10)
        s.calibrate(cal_loader)
        return s

    def test_spike_ood_mean_above_id(self, calibrated_scorer, cal_spectra):
        id_scores = calibrated_scorer.score_batch(cal_spectra[:10])
        ood = torch.stack([
            make_synthetic_ood(cal_spectra[i], mode="spike", seed=i)
            for i in range(10)
        ])
        ood_scores = calibrated_scorer.score_batch(ood)
        # The OOD perturbation should raise the mean score.
        assert ood_scores.mean() > id_scores.mean(), (
            f"spike OOD mean ({ood_scores.mean():.3f}) should exceed "
            f"ID mean ({id_scores.mean():.3f})"
        )

    def test_mask_ood_above_id(self, calibrated_scorer, cal_spectra):
        ood = torch.stack([
            make_synthetic_ood(cal_spectra[i], mode="mask", seed=i)
            for i in range(10)
        ])
        id_scores = calibrated_scorer.score_batch(cal_spectra[:10])
        ood_scores = calibrated_scorer.score_batch(ood)
        # mask removes signal -> reconstruction will fail badly.
        assert ood_scores.mean() > id_scores.mean()

    def test_scale_ood_above_id(self, calibrated_scorer, cal_spectra):
        # 10x scale: reconstruction error will spike because pure_ref
        # cannot be scaled that far by the learned scale parameter.
        ood = torch.stack([
            make_synthetic_ood(cal_spectra[i], mode="scale", seed=i)
            for i in range(10)
        ])
        id_scores = calibrated_scorer.score_batch(cal_spectra[:10])
        ood_scores = calibrated_scorer.score_batch(ood)
        assert ood_scores.mean() > id_scores.mean()


# ---------------------------------------------------------------------------
# Synthetic OOD generators
# ---------------------------------------------------------------------------

class TestSyntheticOOD:

    def test_spike_changes_input(self):
        s = torch.randn(1024)
        out = make_synthetic_ood(s, mode="spike", seed=0)
        assert not torch.equal(out, s)
        # The pulses raise the max value substantially.
        assert out.max().item() > s.max().item()

    def test_noise_changes_input(self):
        s = torch.zeros(1024)
        out = make_synthetic_ood(s, mode="noise", seed=0)
        assert out.std().item() > 0.01

    def test_mask_zeros_segment(self):
        s = torch.ones(1024)
        out = make_synthetic_ood(s, mode="mask")
        # make_synthetic_ood uses start=int(0.3*P), stop=int(0.6*P).
        # For P=1024 that is [307, 614).
        start, stop = int(0.3 * 1024), int(0.6 * 1024)
        assert (out[start:stop] == 0).all()
        # outside should still be 1.
        assert (out[:start] == 1).all()
        assert (out[stop:] == 1).all()

    def test_scale_multiplies(self):
        s = torch.ones(1024)
        out = make_synthetic_ood(s, mode="scale")
        assert torch.allclose(out, torch.full((1024,), 10.0))

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            make_synthetic_ood(torch.randn(1024), mode="frobnicate")

    def test_deterministic_with_seed(self):
        s = torch.randn(1024)
        a = make_synthetic_ood(s, mode="noise", seed=42)
        b = make_synthetic_ood(s, mode="noise", seed=42)
        assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_save_load_roundtrip(self, model, cal_loader, tmp_path):
        s = OODScorer(model, mc_samples=5)
        s.calibrate(cal_loader)
        path = tmp_path / "ood.json"
        s.save(path)
        assert path.exists()

        loaded = OODScorer.load_calibration(path)
        assert isinstance(loaded, OODCalibration)
        assert loaded.recon_p95 == s.calibration.recon_p95
        assert loaded.score_p95 == s.calibration.score_p95

    def test_from_file_makes_working_scorer(self, model, cal_loader, tmp_path):
        s = OODScorer(model, mc_samples=5)
        s.calibrate(cal_loader)
        path = tmp_path / "ood.json"
        s.save(path)

        s2 = OODScorer.from_file(model, path)
        x = torch.randn(1024)
        # Same model + same calibration -> deterministic scoring should
        # match exactly (modulo MC randomness, which is seeded by
        # ambient torch RNG; pin it).
        torch.manual_seed(0)
        a = s.score(x)
        torch.manual_seed(0)
        b = s2.score(x)
        assert abs(a - b) < 1e-6

    def test_save_before_calibrate_raises(self, model, tmp_path):
        s = OODScorer(model, mc_samples=5)
        with pytest.raises(RuntimeError, match="not calibrated"):
            s.save(tmp_path / "x.json")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_rejects_negative_weight(self, model):
        with pytest.raises(ValueError, match=">= 0"):
            OODScorer(model, recon_weight=-0.1, var_weight=0.4, mc_samples=5)

    def test_rejects_bad_threshold(self, model):
        with pytest.raises(ValueError, match="threshold_percentile"):
            OODScorer(model, threshold_percentile=0, mc_samples=5)
        with pytest.raises(ValueError, match="threshold_percentile"):
            OODScorer(model, threshold_percentile=100, mc_samples=5)

    def test_rejects_low_mc_samples(self, model):
        with pytest.raises(ValueError, match="mc_samples"):
            OODScorer(model, mc_samples=1)