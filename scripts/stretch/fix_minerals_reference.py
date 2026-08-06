"""Fix the minerals reference row in engine/reference_spectra_aam.npy.

T29A built a 7-component reference (6 AA + 1 minerals). But the minerals
row fell back to "AAM mineral-rich mean" because the real
50_50_quartz_calcite.csv (an ENLIGHTEN wide-format export) could not be
parsed by T29A's CSV reader. That fallback is *circular*: the reference
used to reconstruct mineral-rich test spectra is itself the mean of the
mineral-rich data, which artificially lowers reconstruction error and
weakens the OOD signal.

This script parses the ENLIGHTEN file properly (via enlighten_loader),
resamples it onto the AA wavenumber grid, applies the SAME preprocessing
the rest of the pipeline uses, and overwrites ONLY row 6 (minerals) of
the existing reference array. Rows 0-5 (the 6 amino acids) are left
untouched, so nothing T29A produced for the AA references changes.

Run this AFTER T29A and BEFORE T29D (retrain), then re-run T29B so the
zero-shot numbers reflect the real mineral reference.

Usage:
    python scripts/stretch/fix_minerals_reference.py
    python scripts/stretch/fix_minerals_reference.py --no-preprocess
    python scripts/stretch/fix_minerals_reference.py --dry-run

Author: stretch fix.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make project root importable AND this script's own dir (for enlighten_loader).
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR.parents[1]))

from enlighten_loader import load_enlighten_csv, resample_to_grid

MINERALS_ROW = 6   # index of the minerals component in the 7-row reference


def _apply_preprocessing(spectrum: np.ndarray) -> np.ndarray:
    """Apply the project preprocessing pipeline if available.

    Tries src.data.preprocess.preprocess_spectrum (the canonical entry
    point). Falls back to a minimal SNV if the import is unavailable, so
    the script still runs in a bare environment.
    """
    try:
        from src.data.preprocess import preprocess_spectrum  # type: ignore
        out = preprocess_spectrum(spectrum)
        return np.asarray(out, dtype=np.float64).ravel()
    except Exception as e:
        print(f"  [warn] could not import project preprocessing ({e}); "
              f"falling back to SNV-only.")
        x = np.asarray(spectrum, dtype=np.float64).ravel()
        mu, sd = x.mean(), x.std()
        return (x - mu) / sd if sd > 0 else x - mu


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--enlighten-csv",
                   default="data/raw/50_50_quartz_calcite.csv")
    p.add_argument("--reference-npy",
                   default="engine/reference_spectra_aam.npy")
    p.add_argument("--wavenumbers",
                   default="data/processed/wavenumbers.npy",
                   help="AA wavenumber grid to resample onto")
    p.add_argument("--intensity-row", default=None,
                   help="ENLIGHTEN row to use (default: auto Processed→Raw)")
    p.add_argument("--no-preprocess", action="store_true",
                   help="Skip preprocessing (use raw resampled spectrum)")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute and report but do not overwrite the .npy")
    args = p.parse_args()

    csv_path = Path(args.enlighten_csv)
    ref_path = Path(args.reference_npy)
    wn_path = Path(args.wavenumbers)

    if not csv_path.exists():
        print(f"  ERROR: {csv_path} not found."); sys.exit(1)
    if not ref_path.exists():
        print(f"  ERROR: {ref_path} not found. Run T29A first."); sys.exit(1)
    if not wn_path.exists():
        print(f"  ERROR: {wn_path} not found."); sys.exit(1)

    # ---- Load existing reference; keep AA rows intact ----
    ref = np.load(ref_path)
    print(f"[fix-min] loaded reference {ref_path}, shape={ref.shape}")
    if ref.ndim != 2 or ref.shape[0] < 7:
        print(f"  ERROR: expected (>=7, P) reference, got {ref.shape}.")
        sys.exit(1)
    P = ref.shape[1]

    aa_grid = np.load(wn_path).astype(np.float64)
    if aa_grid.size != P:
        print(f"  [warn] wavenumber grid size {aa_grid.size} != reference "
              f"width {P}; will resample to the {P}-point reference width "
              f"using a synthetic linear grid.")
        aa_grid = np.linspace(aa_grid.min(), aa_grid.max(), P)

    # ---- Parse ENLIGHTEN, average all measurements ----
    print(f"[fix-min] parsing ENLIGHTEN {csv_path} ...")
    wn_src, inten_src = load_enlighten_csv(
        csv_path, intensity_row=args.intensity_row, aggregate="mean"
    )
    print(f"  parsed: {wn_src.size} points, "
          f"wn [{wn_src.min():.1f}, {wn_src.max():.1f}], "
          f"intensity [{inten_src.min():.3f}, {inten_src.max():.3f}]")

    # ---- Resample onto AA grid ----
    mineral_resampled = resample_to_grid(wn_src, inten_src, aa_grid)
    print(f"  resampled to AA grid: {mineral_resampled.shape}")

    # ---- Preprocess to match the rest of the references ----
    if args.no_preprocess:
        mineral_final = mineral_resampled
        print("  [skip] preprocessing (--no-preprocess)")
    else:
        mineral_final = _apply_preprocessing(mineral_resampled)
        print(f"  preprocessed: range "
              f"[{mineral_final.min():.3f}, {mineral_final.max():.3f}]")

    # ---- Sanity: how different is the real reference from the old fallback? ----
    old_minerals = ref[MINERALS_ROW].astype(np.float64)
    a, b = old_minerals.ravel(), mineral_final.ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    cos = float(np.dot(a, b) / denom) if denom > 0 else 0.0
    print(f"\n[fix-min] cosine(old_fallback_minerals, real_minerals) = {cos:.4f}")
    if cos > 0.98:
        print("  NOTE: real reference is nearly identical to the old fallback. "
              "The circularity concern may be mild for this dataset.")
    else:
        print("  NOTE: real reference differs substantially from the fallback. "
              "Replacing it should change recon/OOD numbers (as intended).")

    if args.dry_run:
        print("\n[dry-run] not writing. Re-run without --dry-run to apply.")
        return

    # ---- Overwrite ONLY the minerals row ----
    ref[MINERALS_ROW] = mineral_final.astype(ref.dtype)
    # Back up the old file once. Use shutil.copy2 (not np.save) because
    # np.save auto-appends '.npy', which would mangle a '.npy.bak' name.
    import shutil
    backup = Path(str(ref_path) + ".bak")
    if not backup.exists():
        shutil.copy2(ref_path, backup)
        print(f"\n[fix-min] backed up original → {backup}")
    np.save(ref_path, ref)
    print(f"[fix-min] wrote updated reference → {ref_path} "
          f"(row {MINERALS_ROW} replaced, AA rows 0-5 untouched)")
    print("\nNext steps:")
    print("  1. Re-run T29B  (zero-shot numbers now use the real reference)")
    print("  2. Then T29D retrain, T29E post-train test")


if __name__ == "__main__":
    main()