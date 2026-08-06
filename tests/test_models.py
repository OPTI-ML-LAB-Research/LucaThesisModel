"""Pytest suite for Phase 1 model components (T10-T14).

Run from project root:
    pytest tests/test_models.py -v

Tests cover shapes, value constraints (simplex, non-negativity),
end-to-end forward, and gradient flow. No real reference_spectra.npy is
needed -- a synthetic (n_compounds, P) array is constructed in fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.backbone import ResNet1DBackbone, BasicBlock1D
from src.models.heads import QuantificationHead
from src.models.reconstruction import ReconstructionModule
from src.models.full_model import RamanPhysicsAI
from src.training.losses import (
    quantification_loss,
    physics_loss,
    l2_regularization,
    combined_loss,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_refs() -> np.ndarray:
    """Disjoint-plateau reference spectra so mixtures are easy to verify."""
    n_compounds, P = 6, 1024
    refs = np.zeros((n_compounds, P), dtype=np.float32)
    for i in range(n_compounds):
        refs[i, i * 100:(i + 1) * 100] = 1.0
    return refs


@pytest.fixture
def full_model(synthetic_refs):
    torch.manual_seed(0)
    return RamanPhysicsAI(
        reference_spectra=synthetic_refs,
        n_compounds=6,
        spectrum_length=1024,
        feature_dim=256,
    )


# ---------------------------------------------------------------------------
# T10 -- Backbone
# ---------------------------------------------------------------------------

class TestBackbone:

    def test_forward_shape(self):
        torch.manual_seed(0)
        bb = ResNet1DBackbone(in_channels=1, feature_dim=256)
        bb.eval()
        x = torch.randn(8, 1, 1024)
        z = bb(x)
        assert z.shape == (8, 256)

    def test_no_nan(self):
        torch.manual_seed(0)
        bb = ResNet1DBackbone()
        bb.eval()
        x = torch.randn(4, 1, 1024)
        z = bb(x)
        assert torch.isfinite(z).all()

    def test_param_count_under_5M(self):
        bb = ResNet1DBackbone()
        n = bb.count_parameters()
        assert n < 5_000_000, f"Backbone too large: {n:,}"

    def test_rejects_2D_input(self):
        bb = ResNet1DBackbone()
        with pytest.raises(ValueError, match="3D"):
            bb(torch.randn(8, 1024))

    def test_dropout_applies_in_train_mode(self):
        torch.manual_seed(0)
        bb = ResNet1DBackbone(dropout_rate=0.5)
        x = torch.randn(8, 1, 1024)
        bb.train()
        z1 = bb(x); z2 = bb(x)
        # With dropout 0.5 and same input, two forward passes should differ.
        assert not torch.allclose(z1, z2)

    def test_basic_block_residual_shape(self):
        blk = BasicBlock1D(32, 64, stride=2)
        x = torch.randn(4, 32, 100)
        y = blk(x)
        # stride=2 halves the length (with rounding for kernel=3, pad=1: 50)
        assert y.shape == (4, 64, 50)


# ---------------------------------------------------------------------------
# T11 -- Quantification head
# ---------------------------------------------------------------------------

class TestQuantificationHead:

    def test_output_shape(self):
        h = QuantificationHead(); h.eval()
        z = torch.randn(8, 256)
        a = h(z)
        assert a.shape == (8, 6)

    def test_simplex_constraint(self):
        h = QuantificationHead(); h.eval()
        z = torch.randn(16, 256)
        a = h(z)
        sums = a.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(16), atol=1e-5)
        assert (a >= 0).all()
        assert (a <= 1).all()

    def test_rejects_wrong_feature_dim(self):
        h = QuantificationHead(feature_dim=256)
        with pytest.raises(ValueError):
            h(torch.randn(4, 128))


# ---------------------------------------------------------------------------
# T12 -- Reconstruction module
# ---------------------------------------------------------------------------

class TestReconstructionModule:

    def test_pure_compound_recovery(self, synthetic_refs):
        recon = ReconstructionModule(synthetic_refs, n_compounds=6,
                                     spectrum_length=1024)
        alpha = torch.zeros(1, 6); alpha[0, 0] = 1.0
        out = recon(alpha)
        assert out.shape == (1, 1024)
        assert torch.allclose(out[0], torch.as_tensor(synthetic_refs[0]),
                              atol=1e-6)

    def test_linear_combination(self, synthetic_refs):
        recon = ReconstructionModule(synthetic_refs)
        alpha = torch.tensor([[0.5, 0.3, 0.2, 0.0, 0.0, 0.0]])
        out = recon(alpha)
        expected = (
            0.5 * synthetic_refs[0]
            + 0.3 * synthetic_refs[1]
            + 0.2 * synthetic_refs[2]
        )
        assert torch.allclose(out[0], torch.as_tensor(expected), atol=1e-6)

    def test_scale_propagates(self, synthetic_refs):
        recon = ReconstructionModule(synthetic_refs)
        with torch.no_grad():
            recon.scale[0] = 2.0
        alpha = torch.zeros(1, 6); alpha[0, 0] = 1.0
        out = recon(alpha)
        assert torch.allclose(out[0],
                              2.0 * torch.as_tensor(synthetic_refs[0]),
                              atol=1e-6)

    def test_pure_ref_is_buffer_not_param(self, synthetic_refs):
        recon = ReconstructionModule(synthetic_refs, learnable_scale=True)
        # pure_ref must NOT show up in parameters.
        params = list(recon.parameters())
        assert len(params) == 1                # only the scale vector
        assert params[0].shape == (6,)
        # But it MUST be in buffers (so it moves with .to(device)).
        buffers = dict(recon.named_buffers())
        assert "pure_ref" in buffers

    def test_frozen_scale(self, synthetic_refs):
        recon = ReconstructionModule(synthetic_refs, learnable_scale=False)
        assert not recon.scale.requires_grad

    def test_rejects_bad_alpha_shape(self, synthetic_refs):
        recon = ReconstructionModule(synthetic_refs)
        with pytest.raises(ValueError):
            recon(torch.randn(4, 5))   # wrong n_compounds

    def test_rejects_bad_ref_shape(self):
        bad_refs = np.zeros((5, 1024), dtype=np.float32)   # n=5 not 6
        with pytest.raises(ValueError):
            ReconstructionModule(bad_refs, n_compounds=6, spectrum_length=1024)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ReconstructionModule(
                tmp_path / "nonexistent.npy",
                n_compounds=6, spectrum_length=1024,
            )


# ---------------------------------------------------------------------------
# T13 -- Losses
# ---------------------------------------------------------------------------

class TestLosses:

    def test_quantification_known_value(self):
        y_t = torch.tensor([[0.5, 0.3, 0.2]])
        y_p = torch.tensor([[0.6, 0.2, 0.2]])
        # |0.1| + |0.1| + |0.0| = 0.2; mean = 0.2/3
        assert abs(quantification_loss(y_t, y_p).item() - 0.2 / 3) < 1e-6

    def test_quantification_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            quantification_loss(torch.zeros(4, 6), torch.zeros(4, 5))

    def test_physics_identical_is_zero(self):
        s = torch.randn(4, 1024)
        assert physics_loss(s, s).item() < 1e-6

    def test_physics_handles_3d_input(self):
        s = torch.randn(4, 1, 1024)
        assert physics_loss(s, s).item() < 1e-6
        # Mixed 2D / 3D also OK.
        assert physics_loss(s, s.squeeze(1)).item() < 1e-6

    def test_physics_negation(self):
        s = torch.randn(4, 1024)
        # MSE(s,-s) = 4*mean(s^2); cosine(s,-s) = -1 -> term = 2.
        loss = physics_loss(s, -s, lambda_cosine=0.3).item()
        expected = 4 * s.pow(2).mean().item() + 0.3 * 2.0
        assert abs(loss - expected) < 1e-3

    def test_l2_regularization(self):
        p = nn.Parameter(torch.tensor([3.0, 4.0]))
        assert abs(l2_regularization([p]).item() - 25.0) < 1e-6

    def test_l2_skips_frozen(self):
        p_train = nn.Parameter(torch.tensor([3.0, 4.0]))
        p_frozen = nn.Parameter(torch.tensor([10.0, 10.0]),
                                requires_grad=False)
        # Only p_train should contribute -> 25, not 225.
        assert abs(l2_regularization([p_train, p_frozen]).item() - 25.0) < 1e-6

    def test_combined_returns_components(self):
        torch.manual_seed(0)
        y_t = torch.rand(8, 6); y_t = y_t / y_t.sum(-1, keepdim=True)
        y_p = torch.rand(8, 6); y_p = y_p / y_p.sum(-1, keepdim=True)
        s_in = torch.randn(8, 1, 1024)
        s_rc = torch.randn(8, 1024)
        out = combined_loss(y_t, y_p, s_in, s_rc, return_components=True)
        assert set(out.keys()) == {"total", "quant", "physics", "l2"}
        # When no params passed, l2 must be zero.
        assert out["l2"].item() == 0.0


# ---------------------------------------------------------------------------
# T14 -- Full model
# ---------------------------------------------------------------------------

class TestFullModel:

    def test_forward_shapes_3d(self, full_model):
        full_model.eval()
        x = torch.randn(4, 1, 1024)
        out = full_model(x)
        assert out["composition"].shape == (4, 6)
        assert out["reconstruction"].shape == (4, 1024)
        assert out["feature"].shape == (4, 256)

    def test_forward_shapes_2d(self, full_model):
        full_model.eval()
        x = torch.randn(4, 1024)
        out = full_model(x)
        assert out["composition"].shape == (4, 6)

    def test_composition_is_simplex(self, full_model):
        full_model.eval()
        x = torch.randn(8, 1, 1024)
        out = full_model(x)
        sums = out["composition"].sum(-1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)

    def test_no_nan_in_output(self, full_model):
        full_model.eval()
        x = torch.randn(4, 1, 1024)
        out = full_model(x)
        for k, v in out.items():
            assert torch.isfinite(v).all(), f"NaN/Inf in {k}"

    def test_rejects_wrong_length(self, full_model):
        with pytest.raises(ValueError, match="spectrum_length"):
            full_model(torch.randn(4, 1, 999))

    def test_param_count_per_module(self, full_model):
        c = full_model.count_parameters()
        assert c["reconstruction"] == 6        # only the scale vector
        assert c["backbone"] > c["head"]       # backbone is biggest
        assert c["total"] < 5_000_000

    def test_gradient_flows(self, full_model):
        x = torch.randn(4, 1, 1024)
        y_t = torch.rand(4, 6); y_t = y_t / y_t.sum(-1, keepdim=True)
        out = full_model(x)
        loss = combined_loss(
            y_t, out["composition"], x, out["reconstruction"],
            model_parameters=full_model.parameters(),
        )
        loss.backward()
        # Each submodule must have at least one nonzero gradient.
        for name, mod in [
            ("backbone", full_model.backbone),
            ("head", full_model.quantification_head),
            ("reconstruction", full_model.reconstruction),
        ]:
            grads = [p.grad for p in mod.parameters() if p.grad is not None]
            assert grads, f"No grads in {name}"
            assert any(g.abs().sum().item() > 0 for g in grads), \
                f"All-zero grads in {name}"
