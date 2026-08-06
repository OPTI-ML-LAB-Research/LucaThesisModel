"""Run the Demo-3 (Beryl, hard far-domain OOD) case end-to-end and dump the
numbers + 3 figures the report needs. Reproducible replacement for the old
MoS2 demo-3 artefacts.

Run from the project root, e.g.:

    python scripts/run_demo3_beryl.py --file "data/raw/ood_demo/beryl/Beryl_01 (6).txt" --mc 50

Outputs (mirrors results/reports/demo_3_real_mos2/):
    results/reports/demo_3_beryl/
        reconstruction.png   peaks.png   ood.png
        report.md   report.json
    + prints the key numbers (composition, recon cosine, OOD score/verdict,
      peaks matched/total) to stdout so you can paste them back for the docx.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.inference.predict import predict
from src.inference.report import generate_report
from src.inference.visualize import (
    plot_reconstruction_overlay,
    plot_peak_annotations,
    plot_ood_summary,
)
from src.data.preprocess import preprocess_batch

WN_PATH = "data/processed/wavenumbers.npy"
MODEL_PATH = "checkpoints/best.pt"
COMPOUND_ORDER = [
    "Alanine", "Asparagine", "Aspartic Acid",
    "Glutamic Acid", "Histidine", "Glucosamine",
]


def read_spectrum_text(path: str):
    """Delimiter/orientation-robust reader (same logic as dashboard/app.py).

    Returns (wavenumber, intensity) float arrays, or (None, vector) for a bare
    single-column file.
    """
    lines = [ln.strip() for ln in Path(path).read_text(
        encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty file")
    if len(lines) == 2 and len(lines[0].split()) > 4:        # transposed 2-row
        return (np.asarray(lines[0].split(), dtype=np.float64),
                np.asarray(lines[1].split(), dtype=np.float64))
    sample = lines[len(lines) // 2]
    delim = "," if "," in sample else ("\t" if "\t" in sample else None)
    wn, inten, single = [], [], []
    for ln in lines:
        try:
            vals = [float(x) for x in ln.split(delim)]
        except ValueError:
            continue
        if len(vals) >= 2:
            wn.append(vals[0]); inten.append(vals[1])
        elif len(vals) == 1:
            single.append(vals[0])
    if wn:
        return np.asarray(wn, np.float64), np.asarray(inten, np.float64)
    if single:
        return None, np.asarray(single, np.float64)
    raise ValueError("no numeric data parsed")


def resample_and_preprocess(wn_src: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """Linear-interp onto the model grid, then training preprocessing."""
    target_wn = np.load(WN_PATH).astype(np.float64)
    flipped = target_wn[0] > target_wn[-1]
    target_asc = target_wn[::-1] if flipped else target_wn
    wn_src = np.asarray(wn_src, np.float64)
    intensity = np.asarray(intensity, np.float64)
    if wn_src[0] > wn_src[-1]:
        wn_src, intensity = wn_src[::-1], intensity[::-1]
    v = np.interp(target_asc, wn_src, intensity, left=0.0, right=0.0)
    if flipped:
        v = v[::-1]
    Xp = preprocess_batch(v[None, :].astype(np.float32)).astype(np.float32)
    return np.nan_to_num(Xp, nan=0.0, posinf=0.0, neginf=0.0)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to a beryl .txt spectrum.")
    ap.add_argument("--mc", type=int, default=50, help="MC-Dropout samples.")
    ap.add_argument("--out", default="results/reports/demo_3_beryl",
                    help="Output directory.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wn, inten = read_spectrum_text(args.file)
    if wn is None:
        raise SystemExit("File is a bare vector; need a 2-column raw spectrum.")
    print(f"Loaded {Path(args.file).name}: {inten.size} points, "
          f"{wn.min():.0f}-{wn.max():.0f} cm-1")

    spec = resample_and_preprocess(wn, inten)
    result = predict(spec, model_path=MODEL_PATH, n_mc_samples=args.mc, skip_ood=False)

    # ---- figures ----
    plot_reconstruction_overlay(result, save_path=out / "reconstruction.png")
    plot_peak_annotations(result, save_path=out / "peaks.png")
    plot_ood_summary(result, save_path=out / "ood.png")

    # ---- report ----
    rep = generate_report(result, sample_id=f"Beryl ({Path(args.file).name})",
                          ground_truth=None)
    (out / "report.md").write_text(rep["markdown"], encoding="utf-8")
    (out / "report.json").write_text(
        json.dumps(rep["json"], indent=2,
                   default=lambda o: float(o) if hasattr(o, "item") else str(o)),
        encoding="utf-8")

    # ---- key numbers for the docx ----
    cos = result.get("recon_cosine_sim", result.get("recon_cosine"))
    peaks = result.get("peaks", [])
    matched = sum(1 for p in peaks if p.get("matched_to"))
    print("\n================ DEMO-3 BERYL — KEY NUMBERS ================")
    print(f"  Reconstruction cosine : {cos}")
    print(f"  OOD score             : {result.get('ood_score')}")
    print(f"  OOD threshold         : {result.get('ood_threshold')}")
    print(f"  OOD verdict           : {'OOD' if result.get('is_ood') else 'ID (false negative)'}")
    print(f"  Mean compound std     : {result.get('mean_compound_std')}")
    print(f"  Peaks matched / total : {matched} / {len(peaks)}")
    print("  Composition (mean +/- std):")
    for c in COMPOUND_ORDER:
        m = result["composition"][c]
        s = result["composition_std"][c]
        print(f"    {c:<15} {m:.3f} +/- {s:.3f}")
    print("  Detected peaks (cm-1 | intensity | FWHM | matched_to):")
    for p in sorted(peaks, key=lambda x: x["position"]):
        print(f"    {p['position']:7.1f}  {p['intensity']:.3f}  "
              f"{p['fwhm']:5.1f}  {p.get('matched_to') or '-'}")
    print("===========================================================")
    print(f"\nSaved figures + report -> {out}")


if __name__ == "__main__":
    main()
