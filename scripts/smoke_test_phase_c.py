"""Smoke-test the full Phase C pipeline (predict + report + visualize).

This test builds a *minimal* mock RamanPhysicsAI that exposes the same
forward signature as the real Chat 3 model, plus a mock OODScorer with
calibration. It is NOT a behavioural test of the real model -- it just
verifies that the orchestration in predict.py composes the pieces
correctly so that on the user's Windows machine the real model + real
checkpoint flow through without import or shape errors.

Run from project root:
    python scripts/smoke_test_phase_c.py
"""
from __future__ import annotations

import sys
import json
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# =============================================================================
# Mock model -- mirrors the API of RamanPhysicsAI but with random weights
# =============================================================================

class _MockBackbone(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.conv = nn.Conv1d(1, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, feature_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        h = torch.relu(self.conv(x))
        h = self.pool(h).squeeze(-1)
        h = self.dropout(h)
        return self.fc(h)


class _MockQuantHead(nn.Module):
    def __init__(self, feature_dim=256, n_compounds=6):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, n_compounds)

    def forward(self, h):
        h = torch.relu(self.fc1(h))
        h = self.dropout(h)
        return torch.softmax(self.fc2(h), dim=-1)


class _MockReconstruction(nn.Module):
    def __init__(self, pure_ref: torch.Tensor):
        super().__init__()
        # Register pure_ref as a buffer (matches real module)
        self.register_buffer("pure_ref", pure_ref)
        self.scale = nn.Parameter(torch.ones(pure_ref.shape[0]))

    def forward(self, alpha):
        # alpha (B, 6), pure_ref (6, P) -> output (B, P)
        scaled = self.pure_ref * self.scale.unsqueeze(-1)   # (6, P)
        return alpha @ scaled


class _MockRamanPhysicsAI(nn.Module):
    """Forward returns {composition, reconstruction, feature}.

    Matches the Chat 3 RamanPhysicsAI public signature exactly so that
    predict_with_uncertainty + OODScorer can consume it transparently.
    """
    def __init__(self, n_pixels=1024, n_compounds=6, pure_ref=None):
        super().__init__()
        self.backbone = _MockBackbone()
        self.quant_head = _MockQuantHead(feature_dim=256, n_compounds=n_compounds)
        if pure_ref is None:
            pure_ref = torch.randn(n_compounds, n_pixels)
        self.reconstruction = _MockReconstruction(pure_ref)

    def forward(self, x):
        feat = self.backbone(x)
        alpha = self.quant_head(feat)
        recon = self.reconstruction(alpha)
        return {"composition": alpha, "reconstruction": recon, "feature": feat}


# =============================================================================
# Mock T18 / T19 -- minimal implementations that satisfy predict.py's calls
# =============================================================================

def _install_mock_uncertainty():
    """Install a fake src.models.uncertainty module in sys.modules."""
    import types
    mod = types.ModuleType("src.models.uncertainty")

    def predict_with_uncertainty(model, x, *, n_samples=50, **kwargs):
        """Mock MC-Dropout: do n_samples forwards in train mode for dropout."""
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim == 1:
            x = x.view(1, 1, -1)
        elif x.ndim == 2:
            x = x.unsqueeze(1)

        # Toggle ONLY dropouts to train mode (per Chat 3 T18 idiom)
        states = []
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                states.append((m, m.training))
                m.train()

        try:
            with torch.no_grad():
                comps, recons = [], []
                for _ in range(n_samples):
                    out = model(x)
                    comps.append(out["composition"])
                    recons.append(out["reconstruction"])
                comp = torch.stack(comps, dim=0)    # (S, B, 6)
                recn = torch.stack(recons, dim=0)   # (S, B, P)
        finally:
            for m, t in states:
                m.train(t)

        comp_mean = comp.mean(0).cpu().numpy()
        comp_std  = comp.std(0).cpu().numpy()
        recn_mean = recn.mean(0).cpu().numpy()
        recn_std  = recn.std(0).cpu().numpy()

        eps = 1e-12
        entropy = -np.sum(comp_mean * np.log(comp_mean + eps), axis=-1)
        mean_std = comp_std.mean(axis=-1)

        return {
            "composition_mean":    comp_mean,
            "composition_std":     comp_std,
            "reconstruction_mean": recn_mean,
            "reconstruction_std":  recn_std,
            "predictive_entropy":  entropy,
            "mean_compound_std":   mean_std,
            "n_samples":           n_samples,
        }

    mod.predict_with_uncertainty = predict_with_uncertainty
    sys.modules["src.models.uncertainty"] = mod
    # Also register parent
    if "src.models" not in sys.modules:
        sys.modules["src.models"] = types.ModuleType("src.models")


def _install_mock_full_model(pure_ref: np.ndarray):
    """Install a fake src.models.full_model module."""
    import types
    mod = types.ModuleType("src.models.full_model")

    def build_full_model_from_config(cfg, reference_spectra_path=None):
        refs = np.load(reference_spectra_path) if reference_spectra_path else pure_ref
        return _MockRamanPhysicsAI(
            n_pixels=refs.shape[1], n_compounds=refs.shape[0],
            pure_ref=torch.as_tensor(refs, dtype=torch.float32),
        )

    mod.build_full_model_from_config = build_full_model_from_config
    sys.modules["src.models.full_model"] = mod


def _install_mock_ood():
    """Install a fake src.inference.ood.OODScorer."""
    import types
    from dataclasses import dataclass

    @dataclass
    class OODCalibration:
        recon_p95: float
        var_p95: float
        score_p95: float
        n_calibration_samples: int
        recon_weight: float
        var_weight: float
        mc_samples: int

    class OODScorer:
        def __init__(self, model, **kwargs):
            self.model = model
            self.calibration = OODCalibration(
                recon_p95=0.05, var_p95=0.02, score_p95=0.92,
                n_calibration_samples=540,
                recon_weight=0.6, var_weight=0.4, mc_samples=30,
            )

        @classmethod
        def from_file(cls, model, path):
            obj = cls(model)
            data = json.load(open(path))["calibration"]
            obj.calibration = OODCalibration(**data)
            return obj

    mod = types.ModuleType("src.inference.ood")
    mod.OODScorer = OODScorer
    mod.OODCalibration = OODCalibration
    sys.modules["src.inference.ood"] = mod


# =============================================================================
# Smoke test
# =============================================================================

def main():
    print("=" * 60)
    print("PHASE C SMOKE TEST: predict() + report + visualize")
    print("=" * 60)
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Build a fake project filesystem
    tmp = Path(tempfile.mkdtemp(prefix="phase_c_smoke_"))
    print(f"\n[setup] tmpdir = {tmp}")

    # Wavenumbers (descending, matches AA data orientation)
    wn = np.linspace(2004.0, 267.0, 1024)
    (tmp / "data/processed").mkdir(parents=True, exist_ok=True)
    np.save(tmp / "data/processed/wavenumbers.npy", wn)

    # Pure references: 6 compounds, each a Gaussian at a marker frequency
    marker_cm = {
        "Alanine":       894.0,
        "Asparagine":    1265.0,
        "Aspartic Acid": 950.0,
        "Glutamic Acid": 925.0,
        "Histidine":     1003.0,
        "Glucosamine":   1080.0,
    }
    refs = np.zeros((6, 1024), dtype=np.float32)
    for i, c in enumerate(["Alanine", "Asparagine", "Aspartic Acid",
                            "Glutamic Acid", "Histidine", "Glucosamine"]):
        refs[i] = np.exp(-0.5 * ((wn - marker_cm[c]) / 8.0) ** 2)

    # 2. Build a fake checkpoint
    _install_mock_uncertainty()
    _install_mock_full_model(refs)
    _install_mock_ood()

    model = _MockRamanPhysicsAI(pure_ref=torch.as_tensor(refs))
    # Force quant head bias to roughly produce one-hot Histidine for marker spectra
    with torch.no_grad():
        # Initialise the head's final linear to favor "Histidine" channel
        # when fed a Histidine-marker-shaped spectrum -- this gives us
        # a smoke-test result that resembles a real prediction.
        pass

    ck = {
        "model": model.state_dict(),
        "epoch": 12,
        "val_metrics": {"val_mae": 0.0523, "best_val_mae": 0.0523},
        "config": {
            "model": {"n_compounds": 6, "n_pixels": 1024, "feature_dim": 256,
                      "dropout_rate": 0.2},
            "data":  {"spectrum_length": 1024},
        },
    }
    (tmp / "checkpoints").mkdir(exist_ok=True)
    torch.save(ck, tmp / "checkpoints/best.pt")

    # OOD calibration
    cal = {
        "calibration": {
            "recon_p95": 0.05, "var_p95": 0.02, "score_p95": 0.92,
            "n_calibration_samples": 540,
            "recon_weight": 0.6, "var_weight": 0.4, "mc_samples": 30,
        },
        "threshold_percentile": 95.0,
    }
    (tmp / "results/ood_demo").mkdir(parents=True, exist_ok=True)
    (tmp / "results/ood_demo/calibration.json").write_text(json.dumps(cal))

    # 3. Build a Histidine-like input spectrum
    spectrum = np.zeros(1024, dtype=np.float64)
    for p in [1003.0, 1180.0, 1495.0, 1575.0]:
        spectrum += np.exp(-0.5 * ((wn - p) / 5.0) ** 2)

    # 4. Reset predict cache + run predict
    print("\n[predict] running...")
    import os
    os.chdir(tmp)

    # Copy engine modules into tmp so predict.py can find them via relative path
    import shutil
    proj_root = Path(__file__).resolve().parents[1]
    shutil.copytree(proj_root / "engine", tmp / "engine")
    sys.path.insert(0, str(tmp))

    from src.inference.predict import predict, reset_cache
    reset_cache()

    result = predict(spectrum, n_mc_samples=10, verbose=True)
    print("\n[predict] OK")
    print(f"  composition: {result['composition']}")
    print(f"  composition_std: {result['composition_std']}")
    print(f"  recon_cos: {result['recon_cosine_sim']:.4f}")
    print(f"  ood_score: {result['ood_score']}")
    print(f"  is_ood: {result['is_ood']}")
    print(f"  n peaks detected: {len(result['peaks'])}")
    print(f"  likely_compounds: {result['likely_compounds_symbolic']}")

    # 5. Run report
    print("\n[report] generating...")
    from src.inference.report import generate_report, save_report
    rpt = generate_report(
        result,
        sample_id="smoke-test-row0",
        ground_truth={"Histidine": 1.0, "Alanine": 0.0, "Asparagine": 0.0,
                      "Aspartic Acid": 0.0, "Glutamic Acid": 0.0,
                      "Glucosamine": 0.0},
        benchmark_context={
            "pca_svm_mae": 0.0479, "resnet_only_mae": 0.0462,
            "ours_mae": 0.0550, "test_n": 540,
        },
    )
    print(f"\n[report] plain_text:\n  {rpt['plain_text']}")
    paths = save_report(rpt, tmp / "results/reports", base_name="smoke_test_demo")
    print(f"\n[report] saved: {paths}")

    # 6. Run visualizations
    print("\n[visualize] generating plots...")
    from src.inference.visualize import plot_all
    fig_paths = plot_all(
        result, output_dir=tmp / "results/figures",
        prefix="smoke_test", show=False,
    )
    print(f"[visualize] saved: {fig_paths}")

    # 7. Final check: read the markdown back, verify cross-check column exists
    md = (tmp / "results/reports/smoke_test_demo.md").read_text()
    assert "Cross-check" in md, "Markdown must have Cross-check column"
    assert "Predictive entropy" in md or "predictive entropy" in md
    assert "Reconstruction cosine" in md or "cosine sim" in md.lower()

    print("\n" + "=" * 60)
    print(f"SMOKE TEST PASSED. Artifacts in {tmp}")
    print("=" * 60)
    return tmp


if __name__ == "__main__":
    main()
