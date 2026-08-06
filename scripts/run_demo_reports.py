"""Generate 3 thesis-defense demo reports from real cache data (Phase C).

Selects three samples programmatically to illustrate the full range of
behaviours the MVP can exhibit:

1. **ID demo** -- a test-set row that the learned head predicts close to
   ground truth. Shows the system working as intended.

2. **Mild OOD demo** -- a test-set row dominated by Glucosamine, the
   compound on which Chat 2 T17 measured highest MAE / lowest correlation
   (per T17 diagnosis). Should show wide composition_std and possible
   cross-check disagreement, telling the honest "model is uncertain here"
   story.

3. **Hard OOD demo** -- a synthetic injection (spike at 380 cm-1 in the
   middle of an otherwise normal test row). Mimics the MoS2 case so we
   can demonstrate the novelty locator without needing the real MoS2
   data file. The OOD score should rise above threshold.

Outputs land in ``results/reports/demo_*/`` with both Markdown report
and the three plots.

Usage (Windows PowerShell):
    python scripts/run_demo_reports.py
    python scripts/run_demo_reports.py --no-ood       # skip OOD scoring
    python scripts/run_demo_reports.py --mc 30        # fewer MC samples

Author: Chat 4 Phase C.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path so 'src.inference.predict' resolves
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.predict import predict, reset_cache
from src.inference.report import generate_report, save_report
from src.inference.visualize import plot_all


COMPOUND_ORDER = [
    "Alanine", "Asparagine", "Aspartic Acid",
    "Glutamic Acid", "Histidine", "Glucosamine",
]


def load_benchmark_context() -> dict:
    """Pull benchmark numbers from results/benchmark_table.json if present.

    Returns a dict containing only keys whose values were successfully
    parsed. If T23/T24/T25 haven't run yet (file missing or contains
    nulls), the returned dict will simply lack those keys -- the report
    renderer must check ``is not None`` before formatting.
    """
    p = Path("results/benchmark_table.json")
    if not p.exists():
        return {"test_n": 540}
    try:
        data = json.loads(p.read_text())
        ctx: dict = {"test_n": 540}
        # Likely structure: list of model dicts with 'name' and 'quant_mae'
        if isinstance(data, list):
            for row in data:
                name = row.get("name", "").lower()
                mae = row.get("quant_mae")
                if mae is None:
                    continue
                if "pca" in name or "svm" in name:
                    ctx["pca_svm_mae"] = mae
                elif "resnet" in name and "physics" not in name:
                    ctx["resnet_only_mae"] = mae
                elif "ours" in name or "physics" in name:
                    ctx["ours_mae"] = mae
        elif isinstance(data, dict):
            for k in data:
                kl = k.lower()
                v = data[k]
                mae = v.get("quant_mae") if isinstance(v, dict) else v
                if mae is None:
                    continue
                if "pca" in kl:
                    ctx["pca_svm_mae"] = mae
                elif "resnet" in kl and "physics" not in kl:
                    ctx["resnet_only_mae"] = mae
                elif "ours" in kl or "physics" in kl:
                    ctx["ours_mae"] = mae
        return ctx
    except Exception as e:
        print(f"  [warn] could not parse benchmark_table.json: {e}", file=sys.stderr)
        return {"test_n": 540}


def find_id_sample(spectra: np.ndarray, labels: np.ndarray,
                   test_idx: list[int]) -> int:
    """Pick a test row dominated by Histidine (easiest compound to identify).

    Histidine has 4 distinct imidazole peaks (1003/1180/1495/1575) so the
    ID demo will show the engine layer working at its best.
    """
    test_array = np.asarray(test_idx)
    his_in_test = labels[test_array, 4]  # Histidine column
    # Pick the test row with highest Histidine ratio
    j = int(np.argmax(his_in_test))
    return int(test_array[j])


def find_mild_ood_sample(spectra: np.ndarray, labels: np.ndarray,
                         test_idx: list[int]) -> int:
    """Pick a test row dominated by Glucosamine (worst per-compound MAE).

    Chat 2 T17 measured Glucosamine MAE=0.066 (worst of 6) -- this should
    surface wide composition_std and likely cross-check disagreement.
    """
    test_array = np.asarray(test_idx)
    glc_in_test = labels[test_array, 5]  # Glucosamine column
    j = int(np.argmax(glc_in_test))
    return int(test_array[j])


def inject_synthetic_ood(spectrum: np.ndarray, wn: np.ndarray) -> np.ndarray:
    """Inject a Gaussian spike at 380 cm-1 (MoS2 E2g phonon mode).

    The spike is large enough to:
    1. Be picked up by the peak extractor.
    2. Not match any DB entry (380 is in the 'lattice modes' region,
       outside our amino-acid panel).
    3. Significantly hurt the reconstruction error (input has a feature
       at 380 that the 6 pure references can't reproduce).

    Returns a copy; does not modify the input.
    """
    s = spectrum.copy()
    pmax = float(np.max(np.abs(s)))
    amplitude = max(0.5 * pmax, 0.05)
    sigma = 6.0  # cm-1
    s = s + amplitude * np.exp(-0.5 * ((wn - 380.0) / sigma) ** 2)
    return s


def make_demo(
    *,
    spectrum: np.ndarray,
    ground_truth: dict,
    sample_id: str,
    output_dir: Path,
    n_mc: int,
    skip_ood: bool,
    benchmark_context: dict,
) -> None:
    """Run predict + report + visualize for one demo sample."""
    print(f"\n=== {sample_id} ===")
    result = predict(spectrum, n_mc_samples=n_mc, skip_ood=skip_ood, verbose=False)

    # Generate plots first so we can embed them in the markdown
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_paths = plot_all(result, output_dir=output_dir, prefix=sample_id, show=False)
    # Relative paths for Markdown embeds
    image_rel = {
        k: v.name for k, v in fig_paths.items()
    }
    image_paths = {
        "reconstruction": image_rel["reconstruction"],
        "peaks":          image_rel["peaks"],
        "ood":            image_rel["ood"],
    }

    report = generate_report(
        result,
        sample_id=sample_id,
        ground_truth=ground_truth,
        benchmark_context=benchmark_context,
        image_paths=image_paths,
    )
    paths = save_report(report, output_dir, base_name=sample_id)

    print(f"  plain_text: {report['plain_text']}")
    print(f"  saved:      {paths['markdown']}")
    print(f"  plots:      {list(fig_paths.values())}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mc", type=int, default=50, help="MC dropout samples (default 50)")
    p.add_argument("--no-ood", action="store_true",
                   help="Skip OOD scoring (faster; useful if calibration missing)")
    p.add_argument("--output-dir", default="results/reports",
                   help="Where to write demo_*/ subfolders")
    p.add_argument("--skip-id",  action="store_true", help="Skip demo 1 (ID).")
    p.add_argument("--skip-mild", action="store_true", help="Skip demo 2 (mild OOD).")
    p.add_argument("--skip-hard", action="store_true", help="Skip demo 3 (synthetic OOD).")
    args = p.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load test split + cache
    print("[setup] loading data + split ...")
    wn = np.load("data/processed/wavenumbers.npy")
    spectra = torch.load("data/processed/spectra_full.pt", weights_only=True).numpy()
    labels  = torch.load("data/processed/labels.pt", weights_only=True).numpy()
    split = json.loads(Path("data/splits/split_A_composition_ood.json").read_text())
    test_idx = split["test"]
    print(f"  spectra={spectra.shape}, labels={labels.shape}, test={len(test_idx)} rows")

    benchmark_context = load_benchmark_context()
    if benchmark_context:
        print(f"  benchmark context loaded: {benchmark_context}")

    # ---- Demo 1: ID (Histidine-dominant test row) ----
    if not args.skip_id:
        idx = find_id_sample(spectra, labels, test_idx)
        gt = {c: float(labels[idx, i]) for i, c in enumerate(COMPOUND_ORDER)}
        print(f"\n[demo 1] ID -- row {idx}, labels = {[round(v,3) for v in labels[idx]]}")
        make_demo(
            spectrum=spectra[idx], ground_truth=gt,
            sample_id="demo_1_id_histidine",
            output_dir=out_root / "demo_1_id_histidine",
            n_mc=args.mc, skip_ood=args.no_ood,
            benchmark_context=benchmark_context,
        )

    # ---- Demo 2: Mild OOD (Glucosamine-dominant test row) ----
    if not args.skip_mild:
        idx = find_mild_ood_sample(spectra, labels, test_idx)
        gt = {c: float(labels[idx, i]) for i, c in enumerate(COMPOUND_ORDER)}
        print(f"\n[demo 2] mild OOD -- row {idx}, labels = {[round(v,3) for v in labels[idx]]}")
        make_demo(
            spectrum=spectra[idx], ground_truth=gt,
            sample_id="demo_2_mild_glucosamine_heavy",
            output_dir=out_root / "demo_2_mild_glucosamine_heavy",
            n_mc=args.mc, skip_ood=args.no_ood,
            benchmark_context=benchmark_context,
        )

    # ---- Demo 3: Hard OOD (synthetic MoS2-like spike) ----
    if not args.skip_hard:
        # Base sample: any test row (we use the ID demo's row for repeatability)
        idx = find_id_sample(spectra, labels, test_idx)
        contaminated = inject_synthetic_ood(spectra[idx], wn)
        # Ground truth is meaningless for hard OOD (the input is no longer
        # representable by 6 amino acids), so pass None.
        gt = None
        print(f"\n[demo 3] hard OOD -- row {idx} + 380 cm-1 spike (MoS2-like)")
        make_demo(
            spectrum=contaminated, ground_truth=gt,
            sample_id="demo_3_hard_synthetic_mos2",
            output_dir=out_root / "demo_3_hard_synthetic_mos2",
            n_mc=args.mc, skip_ood=args.no_ood,
            benchmark_context=benchmark_context,
        )

    print("\n[done] all demos written under " + str(out_root))


if __name__ == "__main__":
    main()