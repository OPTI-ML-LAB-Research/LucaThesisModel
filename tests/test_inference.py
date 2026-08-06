"""Tests for src.inference.predict / report / visualize (Phase C).

These tests use a minimal mock model + mock OOD scorer that satisfy the
contracts predict() expects. They verify:

* predict() returns a dict with all the documented keys (§D.2 schema)
* the cross-check pattern (P4AB-9 mandatory) is computed correctly
* report.generate_report produces non-empty markdown + valid JSON
* visualize plot functions return a Figure and write a PNG when asked

The tests do NOT exercise the real RamanPhysicsAI -- that's the
smoke-test script's job. The point here is to lock the orchestration
contract so a future refactor catches breakage.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# ---- Mock environment installed ONCE per session ----

@pytest.fixture(scope="session", autouse=True)
def _install_mocks(tmp_path_factory):
    """Install fake src.models.uncertainty / src.models.full_model /
    src.inference.ood modules before any tests in this file run."""
    import torch
    import torch.nn as nn

    # ---- Mock RamanPhysicsAI ----

    class _Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(1, 8, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.drop = nn.Dropout(0.2)
            self.fc   = nn.Linear(8, 32)
        def forward(self, x):
            h = torch.relu(self.conv(x))
            h = self.pool(h).squeeze(-1)
            h = self.drop(h)
            return self.fc(h)

    class _QuantHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(32, 16)
            self.drop = nn.Dropout(0.2)
            self.fc2 = nn.Linear(16, 6)
        def forward(self, h):
            h = torch.relu(self.fc1(h))
            h = self.drop(h)
            return torch.softmax(self.fc2(h), dim=-1)

    class _Recon(nn.Module):
        def __init__(self, refs):
            super().__init__()
            self.register_buffer("pure_ref", refs)
            self.scale = nn.Parameter(torch.ones(refs.shape[0]))
        def forward(self, alpha):
            scaled = self.pure_ref * self.scale.unsqueeze(-1)
            return alpha @ scaled

    class _Model(nn.Module):
        def __init__(self, refs):
            super().__init__()
            self.backbone = _Backbone()
            self.quant_head = _QuantHead()
            self.reconstruction = _Recon(refs)
        def forward(self, x):
            f = self.backbone(x)
            a = self.quant_head(f)
            r = self.reconstruction(a)
            return {"composition": a, "reconstruction": r, "feature": f}

    # ---- Mock uncertainty ----

    def _predict_with_uncertainty(model, x, *, n_samples=10, **kwargs):
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim == 1: x = x.view(1, 1, -1)
        elif x.ndim == 2: x = x.unsqueeze(1)

        states = []
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout1d)):
                states.append((m, m.training))
                m.train()
        try:
            with torch.no_grad():
                comps = torch.stack([model(x)["composition"]
                                     for _ in range(n_samples)], 0)
                recns = torch.stack([model(x)["reconstruction"]
                                     for _ in range(n_samples)], 0)
        finally:
            for m, t in states:
                m.train(t)

        cm = comps.mean(0).cpu().numpy()
        cs = comps.std(0).cpu().numpy()
        rm = recns.mean(0).cpu().numpy()
        rs = recns.std(0).cpu().numpy()
        eps = 1e-12
        ent = -np.sum(cm * np.log(cm + eps), axis=-1)
        return {
            "composition_mean": cm, "composition_std": cs,
            "reconstruction_mean": rm, "reconstruction_std": rs,
            "predictive_entropy": ent,
            "mean_compound_std":  cs.mean(axis=-1),
            "n_samples": n_samples,
        }

    # ---- Mock OODScorer ----

    from dataclasses import dataclass

    @dataclass
    class _Cal:
        recon_p95: float = 0.05
        var_p95: float = 0.02
        score_p95: float = 0.92
        n_calibration_samples: int = 540
        recon_weight: float = 0.6
        var_weight: float = 0.4
        mc_samples: int = 10

    class _Scorer:
        def __init__(self, model, **kw):
            self.model = model
            self.calibration = _Cal()
        @classmethod
        def from_file(cls, model, path):
            obj = cls(model)
            data = json.load(open(path))["calibration"]
            obj.calibration = _Cal(**data)
            return obj

    # ---- Install in sys.modules ----
    sm = sys.modules
    if "src.models" not in sm:
        sm["src.models"] = types.ModuleType("src.models")
    mu = types.ModuleType("src.models.uncertainty")
    mu.predict_with_uncertainty = _predict_with_uncertainty
    sm["src.models.uncertainty"] = mu

    mf = types.ModuleType("src.models.full_model")
    refs_global = np.eye(6, 1024, dtype=np.float32) * 0.5 + 0.01
    # Make each compound have a Gaussian-ish marker peak
    wn_for_refs = np.linspace(2004.0, 267.0, 1024)
    marker = [894.0, 1265.0, 950.0, 925.0, 1003.0, 1080.0]
    refs_global = np.zeros((6, 1024), dtype=np.float32)
    for i, m in enumerate(marker):
        refs_global[i] = np.exp(-0.5 * ((wn_for_refs - m) / 8.0) ** 2)
    def _builder(cfg, reference_spectra_path=None):
        r = (np.load(reference_spectra_path) if reference_spectra_path
             else refs_global)
        import torch
        return _Model(torch.as_tensor(r, dtype=torch.float32))
    mf.build_full_model_from_config = _builder
    sm["src.models.full_model"] = mf

    mo = types.ModuleType("src.inference.ood")
    mo.OODScorer = _Scorer
    mo.OODCalibration = _Cal
    sm["src.inference.ood"] = mo


@pytest.fixture
def project_tmp(tmp_path):
    """Build a fake project filesystem at tmp_path; yield it as cwd."""
    import os, shutil, torch
    from src.inference.predict import reset_cache
    reset_cache()

    wn = np.linspace(2004.0, 267.0, 1024)
    (tmp_path / "data/processed").mkdir(parents=True)
    np.save(tmp_path / "data/processed/wavenumbers.npy", wn)

    # Build a fake checkpoint
    refs = np.zeros((6, 1024), dtype=np.float32)
    marker = [894.0, 1265.0, 950.0, 925.0, 1003.0, 1080.0]
    for i, m in enumerate(marker):
        refs[i] = np.exp(-0.5 * ((wn - m) / 8.0) ** 2)
    from src.models.full_model import build_full_model_from_config
    model = build_full_model_from_config({})
    sd = model.state_dict()
    # Replace pure_ref with our deterministic refs
    sd["reconstruction.pure_ref"] = torch.as_tensor(refs)
    ck = {
        "model": sd, "epoch": 12,
        "val_metrics": {"val_mae": 0.05},
        "config": {"model": {"n_pixels": 1024}},
    }
    (tmp_path / "checkpoints").mkdir()
    torch.save(ck, tmp_path / "checkpoints/best.pt")

    # OOD calibration
    cal = {"calibration": {
        "recon_p95": 0.05, "var_p95": 0.02, "score_p95": 0.92,
        "n_calibration_samples": 540, "recon_weight": 0.6,
        "var_weight": 0.4, "mc_samples": 10,
    }, "threshold_percentile": 95.0}
    (tmp_path / "results/ood_demo").mkdir(parents=True)
    (tmp_path / "results/ood_demo/calibration.json").write_text(json.dumps(cal))

    # Copy engine into tmp_path (so bond_mapping.json is discoverable)
    proj_root = Path(__file__).resolve().parent.parent
    shutil.copytree(proj_root / "engine", tmp_path / "engine")

    cwd = os.getcwd()
    sys.path.insert(0, str(tmp_path))
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(cwd)
        sys.path.remove(str(tmp_path))
        reset_cache()


def _make_histidine_spectrum() -> np.ndarray:
    wn = np.linspace(2004.0, 267.0, 1024)
    s = np.zeros_like(wn)
    for p in (1003.0, 1180.0, 1495.0, 1575.0):
        s += np.exp(-0.5 * ((wn - p) / 5.0) ** 2)
    return s


# =============================================================================
# Predict
# =============================================================================

class TestPredict:
    def test_returns_required_keys(self, project_tmp):
        from src.inference.predict import predict
        result = predict(_make_histidine_spectrum(), n_mc_samples=5)
        required = {
            "composition", "composition_std", "composition_mean",
            "composition_std_arr", "predictive_entropy", "mean_compound_std",
            "reconstructed_spectrum", "recon_cosine_sim",
            "ood_score", "is_ood", "ood_threshold", "ood_components",
            "peaks", "likely_compounds_symbolic", "compound_votes",
            "unsupported_compounds",
            "unknown_peaks", "novelty_clusters", "novelty_hints",
            "metadata", "input_spectrum", "wavenumbers",
        }
        assert required <= set(result), f"missing keys: {required - set(result)}"

    def test_composition_is_simplex(self, project_tmp):
        from src.inference.predict import predict
        result = predict(_make_histidine_spectrum(), n_mc_samples=5)
        total = sum(result["composition"].values())
        assert abs(total - 1.0) < 1e-4, f"composition not simplex (sum={total})"

    def test_six_compounds(self, project_tmp):
        from src.inference.predict import predict
        result = predict(_make_histidine_spectrum(), n_mc_samples=5)
        canonical = {"Alanine", "Asparagine", "Aspartic Acid",
                     "Glutamic Acid", "Histidine", "Glucosamine"}
        assert set(result["composition"]) == canonical
        assert set(result["composition_std"]) == canonical

    def test_peaks_detected(self, project_tmp):
        from src.inference.predict import predict
        result = predict(_make_histidine_spectrum(), n_mc_samples=5)
        # 4 Histidine peaks should all be detected and matched
        assert len(result["peaks"]) == 4
        matched = [p["matched_to"] for p in result["peaks"]]
        # All 4 should match Histidine entries (P004, P008, P013, P014)
        assert all(m in {"P004", "P008", "P013", "P014"} for m in matched), matched

    def test_likely_compounds_includes_histidine(self, project_tmp):
        from src.inference.predict import predict
        result = predict(_make_histidine_spectrum(), n_mc_samples=5)
        assert "Histidine" in result["likely_compounds_symbolic"]

    def test_recon_cosine_in_range(self, project_tmp):
        from src.inference.predict import predict
        result = predict(_make_histidine_spectrum(), n_mc_samples=5)
        cos = result["recon_cosine_sim"]
        assert -1.0 <= cos <= 1.0

    def test_skip_ood_flag(self, project_tmp):
        from src.inference.predict import predict, reset_cache
        reset_cache()
        result = predict(_make_histidine_spectrum(), n_mc_samples=5, skip_ood=True)
        assert result["ood_score"] is None
        assert result["is_ood"] is None

    def test_ood_present_when_calibration_loaded(self, project_tmp):
        from src.inference.predict import predict
        result = predict(_make_histidine_spectrum(), n_mc_samples=5)
        assert result["ood_score"] is not None
        assert 0.0 <= result["ood_score"] <= 1.0
        assert isinstance(result["is_ood"], bool)

    def test_rejects_2d_input(self, project_tmp):
        from src.inference.predict import predict
        with pytest.raises(ValueError, match="1-D"):
            predict(np.zeros((10, 1024)))

    def test_rejects_wrong_length(self, project_tmp):
        from src.inference.predict import predict
        with pytest.raises(ValueError, match="length"):
            predict(np.zeros(500))

    def test_missing_checkpoint_raises(self, project_tmp):
        from src.inference.predict import predict, reset_cache
        reset_cache()
        with pytest.raises(FileNotFoundError):
            predict(_make_histidine_spectrum(),
                    model_path="checkpoints/does_not_exist.pt")

    def test_predict_batch_runs(self, project_tmp):
        from src.inference.predict import predict_batch
        spectra = np.stack([_make_histidine_spectrum() for _ in range(3)])
        results = predict_batch(spectra, n_mc_samples=3)
        assert len(results) == 3
        for r in results:
            assert "composition" in r


# =============================================================================
# Report
# =============================================================================

class TestReport:
    def _result(self, project_tmp):
        from src.inference.predict import predict
        return predict(_make_histidine_spectrum(), n_mc_samples=5)

    def test_returns_three_views(self, project_tmp):
        from src.inference.report import generate_report
        r = self._result(project_tmp)
        rep = generate_report(r, sample_id="testA")
        assert set(rep) == {"markdown", "json", "plain_text"}
        assert isinstance(rep["markdown"], str) and len(rep["markdown"]) > 100
        assert isinstance(rep["json"], dict)
        assert isinstance(rep["plain_text"], str) and len(rep["plain_text"]) > 20

    def test_markdown_contains_required_sections(self, project_tmp):
        from src.inference.report import generate_report
        rep = generate_report(self._result(project_tmp), sample_id="testB")
        md = rep["markdown"]
        for section in ("Composition Analysis", "Peak Analysis",
                        "Physics Validation", "OOD Assessment"):
            assert section in md, f"section '{section}' missing from markdown"

    def test_markdown_has_cross_check_column(self, project_tmp):
        from src.inference.report import generate_report
        rep = generate_report(self._result(project_tmp), sample_id="testC")
        assert "Cross-check" in rep["markdown"], (
            "P4AB-9 mandatory: markdown must include the cross-check column"
        )

    def test_cross_check_tags(self, project_tmp):
        from src.inference.report import _compute_cross_check
        # Build a controlled result
        result = {
            "composition": {
                "Alanine": 0.10, "Asparagine": 0.01, "Aspartic Acid": 0.50,
                "Glutamic Acid": 0.01, "Histidine": 0.30, "Glucosamine": 0.08,
            },
            "likely_compounds_symbolic": ["Histidine", "Glutamic Acid"],
        }
        cc = _compute_cross_check(result)
        assert cc["Histidine"] == "agreement: present"     # both yes
        assert cc["Asparagine"] == "agreement: absent"     # both no
        assert cc["Aspartic Acid"] == "learned-only"       # learned yes, symbolic no
        assert cc["Glutamic Acid"] == "symbolic-only"      # symbolic yes, learned no

    def test_json_is_serialisable(self, project_tmp, tmp_path):
        from src.inference.report import generate_report, save_report
        rep = generate_report(self._result(project_tmp), sample_id="testD")
        paths = save_report(rep, tmp_path / "rpt", base_name="rpt")
        # Should round-trip without error
        data = json.loads(paths["json"].read_text())
        assert data["sample_id"] == "testD"

    def test_ground_truth_included(self, project_tmp):
        from src.inference.report import generate_report
        gt = {"Histidine": 1.0, "Alanine": 0.0, "Asparagine": 0.0,
              "Aspartic Acid": 0.0, "Glutamic Acid": 0.0, "Glucosamine": 0.0}
        rep = generate_report(self._result(project_tmp),
                              sample_id="testE", ground_truth=gt)
        assert "Ground truth" in rep["markdown"]
        assert "Error" in rep["markdown"]

    def test_benchmark_context_included(self, project_tmp):
        from src.inference.report import generate_report
        ctx = {"pca_svm_mae": 0.048, "resnet_only_mae": 0.046,
               "ours_mae": 0.055, "test_n": 540}
        rep = generate_report(self._result(project_tmp),
                              sample_id="testF", benchmark_context=ctx)
        assert "Benchmark Context" in rep["markdown"]
        assert "0.0480" in rep["markdown"] or "0.048" in rep["markdown"]


# =============================================================================
# Visualize
# =============================================================================

class TestVisualize:
    def _result(self, project_tmp):
        from src.inference.predict import predict
        return predict(_make_histidine_spectrum(), n_mc_samples=5)

    def test_recon_overlay_returns_figure(self, project_tmp, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from src.inference.visualize import plot_reconstruction_overlay
        out = tmp_path / "recon.png"
        fig = plot_reconstruction_overlay(self._result(project_tmp),
                                          save_path=out)
        assert fig is not None
        assert out.exists()

    def test_peak_annotations_returns_figure(self, project_tmp, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from src.inference.visualize import plot_peak_annotations
        out = tmp_path / "peaks.png"
        fig = plot_peak_annotations(self._result(project_tmp), save_path=out)
        assert fig is not None
        assert out.exists()

    def test_ood_summary_returns_figure(self, project_tmp, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from src.inference.visualize import plot_ood_summary
        out = tmp_path / "ood.png"
        fig = plot_ood_summary(self._result(project_tmp), save_path=out)
        assert fig is not None
        assert out.exists()

    def test_plot_all_writes_three_files(self, project_tmp, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from src.inference.visualize import plot_all
        paths = plot_all(self._result(project_tmp),
                         output_dir=tmp_path / "all", prefix="sample")
        assert set(paths) == {"reconstruction", "peaks", "ood"}
        for p in paths.values():
            assert p.exists()
