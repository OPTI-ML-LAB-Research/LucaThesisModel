"""T20+T21+T22 sanity demo on a REAL pure-Histidine spectrum from cache.

Run from project root:

    python -m scripts.real_histidine_demo
    # or
    python scripts/real_histidine_demo.py

Outputs:
    - stdout: peak list, symbolic disambiguation, novelty hints
    - results/sanity/peak_demo_histidine_real.png
"""
from __future__ import annotations

# --- sys.path bootstrap so 'python scripts\...' works too ----------------
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# -------------------------------------------------------------------------

import os
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine.peak_extractor import PeakExtractor
from engine.symbolic_mapper import BondMapper
from engine.novelty_locator import NoveltyLocator


# ---- Compound order is LOCKED (see CHAT2_PHASE_A_HANDOVER C1) ----
COMPOUND_ORDER = [
    "Alanine", "Asparagine", "Aspartic Acid",
    "Glutamic Acid", "Histidine", "Glucosamine",
]
HIS_IDX = COMPOUND_ORDER.index("Histidine")    # = 4

# Expected Histidine imidazole peaks (cm-1) per project spec §8
HIS_IMIDAZOLE_TARGETS = [1003.0, 1180.0, 1495.0, 1575.0]


def main() -> int:
    # ---- Load cache ----
    wn = np.load("data/processed/wavenumbers.npy")
    X = torch.load("data/processed/spectra_full.pt", weights_only=True).numpy()
    Y = torch.load("data/processed/labels.pt", weights_only=True).numpy()
    vials = np.load("data/processed/vial_ids.npy", allow_pickle=True)

    # ---- Pick a pure-Histidine row ----
    pure_his_mask = (Y[:, HIS_IDX] > 0.95) & (Y.sum(axis=1) > 0.99)
    candidates = np.where(pure_his_mask)[0]
    if len(candidates) == 0:
        print("ERROR: No pure-Histidine sample found in cache")
        return 1
    i = int(candidates[0])
    print(f"Using row {i}, vial {vials[i]}, labels {Y[i].round(3)}")

    # ---- T20: extract peaks ----
    ext = PeakExtractor(wn)
    peaks = ext.extract_full(X[i])
    print(f"\nExtracted {len(peaks)} peaks:")
    for p in peaks:
        print(f"  {p.position:7.1f} cm-1   "
              f"I={p.intensity:.3f}   "
              f"FWHM={p.fwhm:5.2f}   "
              f"R^2={p.fit_quality:.3f}")

    # ---- T21: annotate + disambiguate ----
    mapper = BondMapper.from_json("engine/bond_mapping.json")
    ann = mapper.annotate_peaks(peaks)
    disambig = mapper.disambiguate_compound(ann)
    print(f"\nLikely compounds (symbolic): {disambig['likely_compounds']}")
    print(f"Votes: {dict((k, round(v, 2)) for k, v in disambig['votes'].items())}")

    # ---- T22: locate novelty ----
    loc = NoveltyLocator(mapper)
    novelty = loc.locate(peaks)
    print(f"\nUnknown peaks: {len(novelty['unknown_peaks'])}")
    for h in novelty["hints"]:
        print(f"  HINT: {h}")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(wn, X[i], color="steelblue", lw=1.0,
            label=f"real pure Histidine (vial {vials[i]})")
    for j, tp in enumerate(HIS_IMIDAZOLE_TARGETS):
        ax.axvline(tp, color="gray", lw=0.5, ls=":", alpha=0.6,
                   label="expected imidazole" if j == 0 else None)
    for p, a in zip(peaks, ann):
        color = "crimson" if a.matched_to else "darkgray"
        ax.axvline(p.position, color=color, lw=0.8, ls="--", alpha=0.7)
        label = a.matched_to or "unmatched"
        ax.annotate(
            f"{p.position:.1f}\n{label}\nR2={p.fit_quality:.2f}",
            xy=(p.position, p.intensity),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec=color, alpha=0.8),
        )
    ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity (preprocessed, a.u.)")
    ax.set_title(
        f"T20+T21+T22 demo on REAL pure Histidine "
        f"(cache row {i}, vial {vials[i]})"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs("results/sanity", exist_ok=True)
    out = "results/sanity/peak_demo_histidine_real.png"
    plt.savefig(out, dpi=120)
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())