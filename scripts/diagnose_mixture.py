"""Diagnostic for the 50-50 His-Glc mixture symbolic-disambig failure.

Run from project root:

    python scripts/diagnose_mixture.py
    # or
    python -m scripts.diagnose_mixture

Prints, for the chosen mixture row:
  1. Ground truth composition
  2. All peaks extracted by T20 (position / intensity / FWHM / R^2)
  3. For each peak: matched DB entry, compound, discriminative_for, confidence, delta_cm
  4. Vote breakdown from disambiguate_compound
  5. SPECIFIC CHECK for the 4 Histidine imidazole peaks (1003, 1180, 1495, 1575)
     and 2 Glucosamine peaks (1080, 1100):
       - Was a peak extracted within +-15 cm-1 of each expected position?
       - If yes, did it match a DB entry?
       - If no match, why? (delta exceeds tolerance, or fit_quality too low)

This is read-only -- doesn't modify any cache or DB.
"""
from __future__ import annotations

# --- sys.path bootstrap ----------------------------------------------------
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ---------------------------------------------------------------------------

import numpy as np
import torch

from engine.peak_extractor import PeakExtractor
from engine.symbolic_mapper import BondMapper


COMPOUND_ORDER = [
    "Alanine", "Asparagine", "Aspartic Acid",
    "Glutamic Acid", "Histidine", "Glucosamine",
]

# Expected diagnostic peaks per project spec §8
HIS_EXPECTED = [1003.0, 1180.0, 1495.0, 1575.0]   # imidazole ring
GLC_EXPECTED = [1080.0, 1100.0]                   # pyranose


def find_nearest_peak(peaks, target, max_dev=15.0):
    """Return (peak, abs_delta) for the peak closest to target within max_dev,
    or (None, None) if none found within that window.
    """
    best = None
    best_d = None
    for p in peaks:
        d = abs(p.position - target)
        if d <= max_dev and (best_d is None or d < best_d):
            best = p
            best_d = d
    return best, best_d


def main() -> int:
    # ---- Load cache ----
    wn = np.load("data/processed/wavenumbers.npy")
    X = torch.load("data/processed/spectra_full.pt", weights_only=True).numpy()
    Y = torch.load("data/processed/labels.pt", weights_only=True).numpy()
    vials = np.load("data/processed/vial_ids.npy", allow_pickle=True)

    # Pick the same row as test_mixture.py
    target = np.array([0., 0., 0., 0., 0.5, 0.5])
    dists = np.linalg.norm(Y - target, axis=1)
    i = int(np.argmin(dists))
    print("=" * 78)
    print(f"DIAGNOSING mixture row {i}, vial {vials[i]}")
    print(f"Ground truth labels: {dict(zip(COMPOUND_ORDER, Y[i].round(3)))}")
    print("=" * 78)

    ext = PeakExtractor(wn)
    mapper = BondMapper.from_json("engine/bond_mapping.json")

    # ---- Extract + annotate ----
    peaks = ext.extract_full(X[i])
    ann = mapper.annotate_peaks(peaks)
    print(f"\n[T20] Extracted {len(peaks)} peaks. Detail per peak:\n")
    print(f"  {'#':>2}  {'position':>9}  {'I':>6}  {'FWHM':>5}  {'R^2':>5}  "
          f"{'matched':>8}  {'confid':>7}  {'delta':>6}  compounds")
    print("  " + "-" * 96)
    for j, (p, a) in enumerate(zip(peaks, ann)):
        matched = a.matched_to or "-"
        conf = a.match_confidence
        delta = f"{a.delta_cm:+.2f}" if a.delta_cm is not None else "-"
        compounds = ",".join(a.compounds) if a.compounds else "-"
        print(f"  {j:>2}  {p.position:>9.2f}  {p.intensity:>6.3f}  "
              f"{p.fwhm:>5.2f}  {p.fit_quality:>5.3f}  "
              f"{matched:>8}  {conf:>7}  {delta:>6}  {compounds}")

    # ---- Disambiguation ----
    d = mapper.disambiguate_compound(ann)
    print(f"\n[T21] disambiguate_compound output:")
    print(f"  likely_compounds   : {d['likely_compounds']}")
    print(f"  votes (non-zero)   : "
          f"{ {k: round(v, 2) for k, v in d['votes'].items() if v > 0} }")

    # ---- Targeted check: Histidine imidazole peaks ----
    print(f"\n[CHECK] Histidine imidazole peaks (expected per spec §8):")
    print(f"  {'expected':>9}  {'extracted':>10}  {'abs_delta':>10}  "
          f"{'R^2':>5}  matched_to  status")
    print("  " + "-" * 78)
    his_peaks_matched = 0
    for tp in HIS_EXPECTED:
        peak, delta = find_nearest_peak(peaks, tp)
        if peak is None:
            status = "MISS-EXTRACT"
            print(f"  {tp:>9.1f}  {'-':>10}  {'-':>10}  {'-':>5}  {'-':>10}  {status}")
            continue
        # Find this peak's annotation
        idx = peaks.index(peak)
        a = ann[idx]
        matched = a.matched_to or "-"
        is_his = "Histidine" in (a.discriminative_for or [])
        if is_his and a.match_confidence == "high":
            status = "FULL-MATCH"
            his_peaks_matched += 1
        elif is_his and a.match_confidence == "medium":
            status = "MEDIUM (0.5 vote)"
            his_peaks_matched += 0.5
        elif matched != "-":
            status = f"MATCHED-WRONG -> {matched}"
        else:
            status = "EXTRACTED BUT NO MATCH"
        print(f"  {tp:>9.1f}  {peak.position:>10.2f}  {delta:>10.2f}  "
              f"{peak.fit_quality:>5.3f}  {matched:>10}  {status}")
    print(f"  -> Histidine equivalent-high votes: {his_peaks_matched:.1f}")

    # ---- Targeted check: Glucosamine pyranose peaks ----
    print(f"\n[CHECK] Glucosamine pyranose peaks (expected per spec §8):")
    print(f"  {'expected':>9}  {'extracted':>10}  {'abs_delta':>10}  "
          f"{'R^2':>5}  matched_to  status")
    print("  " + "-" * 78)
    glc_peaks_matched = 0
    for tp in GLC_EXPECTED:
        peak, delta = find_nearest_peak(peaks, tp)
        if peak is None:
            status = "MISS-EXTRACT"
            print(f"  {tp:>9.1f}  {'-':>10}  {'-':>10}  {'-':>5}  {'-':>10}  {status}")
            continue
        idx = peaks.index(peak)
        a = ann[idx]
        matched = a.matched_to or "-"
        is_glc = "Glucosamine" in (a.discriminative_for or [])
        if is_glc and a.match_confidence == "high":
            status = "FULL-MATCH"
            glc_peaks_matched += 1
        elif is_glc and a.match_confidence == "medium":
            status = "MEDIUM (0.5 vote)"
            glc_peaks_matched += 0.5
        elif matched != "-":
            status = f"MATCHED-WRONG -> {matched}"
        else:
            status = "EXTRACTED BUT NO MATCH"
        print(f"  {tp:>9.1f}  {peak.position:>10.2f}  {delta:>10.2f}  "
              f"{peak.fit_quality:>5.3f}  {matched:>10}  {status}")
    print(f"  -> Glucosamine equivalent-high votes: {glc_peaks_matched:.1f}")

    # ---- Verdict diagnosis ----
    print(f"\n[VERDICT] Failure-mode classification:")
    if his_peaks_matched == 0:
        # Check if extraction or matching
        all_miss_extract = all(
            find_nearest_peak(peaks, tp)[0] is None for tp in HIS_EXPECTED
        )
        if all_miss_extract:
            print("  -> Scenario 1: PeakExtractor MISSED all Histidine peaks.")
            print("     Fix: lower min_prominence and/or fit_quality_threshold "
                  "in PeakExtractor.")
        else:
            print("  -> Scenario 2: Peaks extracted but BondMapper failed to match.")
            print("     Fix: widen P004/P008/P013/P014 tolerance_cm_inv "
                  "in bond_mapping.json (currently 4-6, try 8-10).")
    elif his_peaks_matched < 1:
        print(f"  -> Scenario 3: Partial matches (vote {his_peaks_matched:.1f} < 1.0).")
        print("     Fix: lower disambiguate threshold "
              "(min_discriminative_hits=0.5) or raise medium-vote weight.")
    else:
        print(f"  -> Histidine WAS identified ({his_peaks_matched:.1f} votes). "
              "Issue must be elsewhere.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())