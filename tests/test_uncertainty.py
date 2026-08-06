"""Pytest suite for T18 MC-Dropout uncertainty (src/models/uncertainty.py).

Run from project root:
    pytest tests/test_uncertainty.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.full_model import RamanPhysicsAI
from src.models.uncertainty import (
    MCDropoutWrapper,
    _coerce_spectrum_input,
    _is_dropout_module,
    enable_mc_dropout,
    mc_dropout_mode,
    predict_with_uncertainty,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_refs() -> np.ndarray:
    """Synthetic disjoint-plateau refs (same as T12 fixture)."""
    n_compounds, P = 6, 1024
    refs = np.zeros((n_compounds, P), dtype=np.float32)
    for i in range(n_compounds):
        refs[i, i * 100:(i + 1) * 100] = 1.0
    return refs


@pytest.fixture
def model(synthetic_refs) -> RamanPhysicsAI:
    torch.manual_seed(0)
    return RamanPhysicsAI(
        reference_spectra=synthetic_refs,
        n_compounds=6, spectrum_length=1024, feature_dim=256,
        backbone_dropout=0.2, head_dropout=0.2,
    ).eval()


# ---------------------------------------------------------------------------
# _is_dropout_module + enable_mc_dropout
# ---------------------------------------------------------------------------

class TestDropoutToggle:

    def test_recognises_dropout_classes(self):
        for cls in (nn.Dropout, nn.Dropout1d, nn.Dropout2d):
            assert _is_dropout_module(cls(p=0.5))

    def test_rejects_non_dropout(self):
        assert not _is_dropout_module(nn.Linear(10, 10))
        assert not _is_dropout_module(nn.BatchNorm1d(32))
        assert not _is_dropout_module(nn.Conv1d(1, 8, 3))
        assert not _is_dropout_module(nn.ReLU())

    def test_enable_counts_dropouts(self, model):
        model.eval()
        n = enable_mc_dropout(model)
        # Full RamanPhysicsAI has 1 in backbone + 1 in head = 2.
        assert n == 2

    def test_enable_does_not_touch_batchnorm(self, model):
        model.eval()
        enable_mc_dropout(model)
        # BatchNorm should NOT be in training mode after enable_mc_dropout.
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                assert not m.training, \
                    "BatchNorm must stay in eval after enable_mc_dropout"


class TestMcDropoutContext:

    def test_yields_dropout_count(self, model):
        with mc_dropout_mode(model) as n:
            assert n == 2

    def test_restores_state(self, model):
        # Force a quirky pre-state: model train, dropouts off.
        model.train()
        for m in model.modules():
            if _is_dropout_module(m):
                m.eval()
        # Enter context, then exit.
        with mc_dropout_mode(model):
            pass
        # Pre-state restored?
        assert model.training is True, "model.train() flag must be restored"
        for m in model.modules():
            if _is_dropout_module(m):
                assert not m.training, "pre-state dropout-off must be restored"

    def test_inside_context_dropouts_train_bn_eval(self, model):
        model.eval()
        with mc_dropout_mode(model):
            for m in model.modules():
                if _is_dropout_module(m):
                    assert m.training, "dropout must be ACTIVE in context"
                if isinstance(m, nn.BatchNorm1d):
                    assert not m.training, "BN must be EVAL in context"


# ---------------------------------------------------------------------------
# Input coercion
# ---------------------------------------------------------------------------

class TestInputCoercion:

    def test_1d_to_3d(self):
        x = torch.randn(1024)
        assert _coerce_spectrum_input(x).shape == (1, 1, 1024)

    def test_2d_to_3d(self):
        x = torch.randn(4, 1024)
        assert _coerce_spectrum_input(x).shape == (4, 1, 1024)

    def test_3d_passthrough(self):
        x = torch.randn(4, 1, 1024)
        assert _coerce_spectrum_input(x).shape == (4, 1, 1024)

    def test_numpy_accepted(self):
        x = np.random.randn(1024).astype(np.float32)
        out = _coerce_spectrum_input(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, 1, 1024)

    def test_rejects_4d(self):
        with pytest.raises(ValueError, match="1D/2D/3D"):
            _coerce_spectrum_input(torch.randn(2, 2, 2, 1024))

    def test_rejects_3d_bad_channel(self):
        with pytest.raises(ValueError, match="channel"):
            _coerce_spectrum_input(torch.randn(4, 3, 1024))

    def test_rejects_non_tensor(self):
        with pytest.raises(TypeError):
            _coerce_spectrum_input([1, 2, 3])


# ---------------------------------------------------------------------------
# predict_with_uncertainty -- output shape + content
# ---------------------------------------------------------------------------

class TestPredictWithUncertaintyShapes:

    def test_single_spectrum_returns_batched(self, model):
        x = torch.randn(1024)
        out = predict_with_uncertainty(model, x, n_samples=5)
        assert out["composition_mean"].shape == (1, 6)
        assert out["composition_std"].shape == (1, 6)
        assert out["reconstruction_mean"].shape == (1, 1024)
        assert out["reconstruction_std"].shape == (1, 1024)
        assert out["predictive_entropy"].shape == (1,)
        assert out["mean_compound_std"].shape == (1,)
        assert out["n_samples"] == 5

    def test_batched_spectrum(self, model):
        x = torch.randn(8, 1024)
        out = predict_with_uncertainty(model, x, n_samples=3)
        assert out["composition_mean"].shape == (8, 6)
        assert out["reconstruction_mean"].shape == (8, 1024)
        assert out["mean_compound_std"].shape == (8,)

    def test_3d_input(self, model):
        out = predict_with_uncertainty(model, torch.randn(2, 1, 1024), n_samples=4)
        assert out["composition_mean"].shape == (2, 6)

    def test_return_samples_shapes(self, model):
        out = predict_with_uncertainty(
            model, torch.randn(3, 1024), n_samples=7, return_samples=True,
        )
        assert out["composition_samples"].shape == (7, 3, 6)
        assert out["reconstruction_samples"].shape == (7, 3, 1024)

    def test_default_no_samples_key(self, model):
        out = predict_with_uncertainty(model, torch.randn(1024), n_samples=3)
        assert "composition_samples" not in out
        assert "reconstruction_samples" not in out


class TestPredictWithUncertaintyContent:

    def test_mean_is_valid_simplex(self, model):
        out = predict_with_uncertainty(model, torch.randn(4, 1024),
                                       n_samples=20)
        m = out["composition_mean"]
        sums = m.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        assert (m >= 0).all()

    def test_std_strictly_positive_with_dropout(self, model):
        # With non-zero dropout and n=20, every compound should have
        # some variance.
        out = predict_with_uncertainty(model, torch.randn(1024), n_samples=20)
        assert out["composition_std"].min().item() > 0

    def test_no_dropout_gives_zero_variance(self, synthetic_refs):
        # Build a model with dropout=0 -> all dropout layers are no-ops.
        # The Dropout modules still exist (so no "no nn.Dropout" warning
        # fires), but with p=0 they don't perturb anything, so every MC
        # sample is identical and std collapses to zero.
        model = RamanPhysicsAI(
            reference_spectra=synthetic_refs, n_compounds=6,
            spectrum_length=1024, feature_dim=256,
            backbone_dropout=0.0, head_dropout=0.0,
        ).eval()
        out = predict_with_uncertainty(model, torch.randn(1024), n_samples=10)
        assert out["composition_std"].abs().max().item() < 1e-6

    def test_n_samples_1_zero_variance(self, model):
        out = predict_with_uncertainty(model, torch.randn(1024), n_samples=1)
        # Single sample -> std is the zero-tensor fallback.
        assert (out["composition_std"] == 0).all()

    def test_seeded_reproducibility(self, model):
        x = torch.randn(1024)
        torch.manual_seed(42)
        a = predict_with_uncertainty(model, x, n_samples=10)
        torch.manual_seed(42)
        b = predict_with_uncertainty(model, x, n_samples=10)
        assert torch.allclose(a["composition_mean"], b["composition_mean"])
        assert torch.allclose(a["composition_std"], b["composition_std"])

    def test_entropy_bounds(self, model):
        out = predict_with_uncertainty(model, torch.randn(4, 1024),
                                       n_samples=10)
        # Entropy of K-class distribution: 0 <= H <= log(K) = log(6) ~ 1.79
        ent = out["predictive_entropy"]
        assert (ent >= 0).all()
        assert (ent <= np.log(6) + 1e-4).all()

    def test_state_restored_on_exit(self, model):
        model.eval()
        predict_with_uncertainty(model, torch.randn(1024), n_samples=5)
        assert not model.training
        for m in model.modules():
            if _is_dropout_module(m):
                assert not m.training

    def test_rejects_bad_n_samples(self, model):
        with pytest.raises(ValueError):
            predict_with_uncertainty(model, torch.randn(1024), n_samples=0)


# ---------------------------------------------------------------------------
# MCDropoutWrapper
# ---------------------------------------------------------------------------

class TestMCDropoutWrapper:

    def test_wrapper_default(self, model):
        w = MCDropoutWrapper(model, n_samples=20)
        out = w(torch.randn(1024))
        assert out["n_samples"] == 20

    def test_wrapper_n_samples_override(self, model):
        w = MCDropoutWrapper(model, n_samples=20)
        out = w(torch.randn(1024), n_samples=3)
        assert out["n_samples"] == 3

    def test_wrapper_n_dropout_count(self, model):
        w = MCDropoutWrapper(model, n_samples=5)
        assert w.n_dropout_modules == 2     # backbone + head

    def test_wrapper_warns_on_dropoutless_model(self):
        # Build a pure linear model with no dropout.
        m = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 6))
        with pytest.warns(RuntimeWarning, match="0 dropout"):
            MCDropoutWrapper(m, n_samples=5)

    def test_wrapper_rejects_bad_n(self, model):
        with pytest.raises(ValueError):
            MCDropoutWrapper(model, n_samples=0)