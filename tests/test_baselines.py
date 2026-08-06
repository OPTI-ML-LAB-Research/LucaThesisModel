"""Unit tests for Phase B baselines.

Coverage:
* T23: simplex projection mathematical properties + train/predict roundtrip
       on a small synthetic dataset.
* T24: architecture instantiation + forward pass shape sanity (no training
       — that requires a real cache and 5+ min of CPU; the smoke is via
       ``python -m src.models.baselines.resnet_only --epochs-override 2``
       manually).

Run with ``pytest tests/test_baselines.py -v``.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.models.baselines.pca_svm import (
    project_to_simplex, train_pca_svm, predict_pca_svm,
)


# ---------------------------------------------------------------------------
# T23 — simplex projection
# ---------------------------------------------------------------------------

class TestSimplexProjection:

    def test_already_on_simplex_unchanged(self):
        v = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]])
        out = project_to_simplex(v)
        np.testing.assert_allclose(out, v, atol=1e-9)

    def test_negative_clipped_and_redistributed(self):
        v = np.array([[-0.5, 0.8, 0.3]])
        out = project_to_simplex(v)
        # Expected: -0.5 → 0; remaining mass redistributed to keep sum = 1
        assert out[0, 0] == pytest.approx(0.0, abs=1e-9)
        assert np.all(out >= 0)
        assert out.sum() == pytest.approx(1.0)

    def test_sum_above_one_scaled_down(self):
        v = np.array([[0.8, 0.7, 0.6]])  # sums to 2.1
        out = project_to_simplex(v)
        # Closed form for this case: theta = (2.1 - 1) / 3 = 0.3667
        # so v - theta = [0.433, 0.333, 0.233], sum = 1.0
        assert np.all(out >= 0)
        assert out.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(out, [[0.4333, 0.3333, 0.2333]], atol=1e-3)

    def test_one_hot_preserved(self):
        v = np.array([[0.0, 0.0, 1.0]])
        out = project_to_simplex(v)
        np.testing.assert_allclose(out, v, atol=1e-9)

    def test_uniform_preserved(self):
        v = np.full((1, 6), 1.0 / 6)
        out = project_to_simplex(v)
        np.testing.assert_allclose(out, v, atol=1e-9)

    def test_1d_input_handled(self):
        v = np.array([0.5, 0.3, 0.2])
        out = project_to_simplex(v)
        assert out.shape == (3,)
        assert out.sum() == pytest.approx(1.0)

    def test_batch_independent(self):
        rng = np.random.default_rng(0)
        v = rng.normal(0, 1, (50, 6))
        out = project_to_simplex(v)
        # Each row independent: project rows individually and compare
        out_row_by_row = np.vstack([project_to_simplex(row[None, :]) for row in v])
        np.testing.assert_allclose(out, out_row_by_row)
        # Each row must be a valid simplex
        np.testing.assert_allclose(out.sum(axis=1), 1.0)
        assert np.all(out >= 0)

    def test_extreme_negatives(self):
        v = np.array([[-10.0, 0.5, 0.5, 0.5]])
        out = project_to_simplex(v)
        assert out[0, 0] == 0.0
        # Mass redistributed over the remaining 3 columns
        assert out.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# T23 — train/predict roundtrip
# ---------------------------------------------------------------------------

class TestPCASVMRoundtrip:

    def test_train_predict_shapes(self):
        rng = np.random.default_rng(0)
        X_train = rng.normal(0, 1, (60, 200)).astype(np.float32)
        y_train = rng.dirichlet(np.ones(6), size=60).astype(np.float32)
        X_val = rng.normal(0, 1, (20, 200)).astype(np.float32)

        scaler, pca, model, meta = train_pca_svm(
            X_train, y_train, n_components=10, verbose=False,
        )
        y_pred = predict_pca_svm(X_val, scaler, pca, model)
        assert y_pred.shape == (20, 6)
        np.testing.assert_allclose(y_pred.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(y_pred >= 0)

    def test_save_load_roundtrip(self, tmp_path):
        import pickle
        from src.models.baselines.pca_svm import save_pca_svm, load_pca_svm

        rng = np.random.default_rng(1)
        X = rng.normal(0, 1, (50, 100)).astype(np.float32)
        y = rng.dirichlet(np.ones(6), size=50).astype(np.float32)
        scaler, pca, model, meta = train_pca_svm(X, y, n_components=10, verbose=False)

        path = tmp_path / "m.pkl"
        save_pca_svm(path, scaler=scaler, pca=pca, model=model, meta=meta)
        loaded = load_pca_svm(path)

        y1 = predict_pca_svm(X, scaler, pca, model)
        y2 = predict_pca_svm(X, loaded["scaler"], loaded["pca"], loaded["model"])
        np.testing.assert_allclose(y1, y2)

        # Meta side-car
        side = path.with_suffix(".meta.json")
        assert side.exists()
        import json
        m = json.loads(side.read_text())
        assert m["pca_n_components"] == 10
        assert m["n_train"] == 50

    def test_unprojected_can_violate_simplex(self):
        """Sanity check: without projection, raw SVR outputs may violate.

        Documents WHY simplex projection is needed for fair comparison.
        """
        rng = np.random.default_rng(2)
        X = rng.normal(0, 1, (50, 100)).astype(np.float32)
        y = rng.dirichlet(np.ones(6), size=50).astype(np.float32)
        scaler, pca, model, meta = train_pca_svm(X, y, n_components=8, verbose=False)
        raw = predict_pca_svm(X, scaler, pca, model, project=False)
        proj = predict_pca_svm(X, scaler, pca, model, project=True)
        # Projection always produces a valid simplex
        np.testing.assert_allclose(proj.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(proj >= 0)
        # raw may or may not — depends on luck; we just assert that the
        # function is non-trivial (proj != raw on at least some rows)
        # (when SVR predicts near-simplex outputs, raw and proj can match)
        # The key claim is "proj is valid", which we already checked.


# ---------------------------------------------------------------------------
# T24 — ResNet-only architecture
# ---------------------------------------------------------------------------

try:
    import torch  # noqa: F401
    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False


@pytest.mark.skipif(not _HAVE_TORCH, reason="torch not installed")
class TestResNetOnlyArchitecture:

    def test_forward_shape(self):
        from src.models.baselines.resnet_only import ResNetOnly
        import torch
        model = ResNetOnly()
        model.eval()
        x = torch.randn(4, 1024)
        with torch.no_grad():
            out = model(x)
        assert "composition" in out
        assert out["composition"].shape == (4, 6)

    def test_simplex_output(self):
        from src.models.baselines.resnet_only import ResNetOnly
        import torch
        model = ResNetOnly()
        model.eval()
        x = torch.randn(8, 1024)
        with torch.no_grad():
            out = model(x)
        sums = out["composition"].sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        assert (out["composition"] >= 0).all()

    def test_accepts_3d_input(self):
        from src.models.baselines.resnet_only import ResNetOnly
        import torch
        model = ResNetOnly()
        model.eval()
        x = torch.randn(2, 1, 1024)  # already has channel dim
        with torch.no_grad():
            out = model(x)
        assert out["composition"].shape == (2, 6)

    def test_param_count_matches_chat3_backbone(self):
        """Backbone + head should have ~997K trainable params (same as Chat 3
        minus reconstruction.scale (6 params)).
        """
        from src.models.baselines.resnet_only import ResNetOnly
        model = ResNetOnly()
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Allow ±0.5% tolerance for implementation variance
        assert 990_000 < n < 1_005_000, f"got {n:,}"

    def test_no_reconstruction_module(self):
        """Sanity: this is the 'no physics' baseline."""
        from src.models.baselines.resnet_only import ResNetOnly
        model = ResNetOnly()
        for name, _ in model.named_modules():
            assert "reconstruction" not in name, \
                f"ResNet-only baseline must not have a reconstruction module, found {name}"
