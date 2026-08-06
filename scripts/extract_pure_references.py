"""Extract pure reference spectra from ENLIGHTEN exports → engine/reference_spectra.npy.

Pipeline
--------
1. Read 6 ENLIGHTEN CSVs (one per pure compound) with `enlighten_parser`.
2. Compute the per-compound mean spectrum (with median fallback if outliers
   detected).
3. Resample each compound's spectrum onto the AA dataset's wavenumber grid
   (1024 channels, ~267 → ~2004 cm^-1) so the reconstruction module can do
   ``s_recon = Σ α_i · scale_i · pure_i`` element-wise.
4. Optionally apply the same preprocessing pipeline used for training data
   (AsLS → cosmic ray → SG → SNV) so the references live in the same scale.
5. Save:
   * ``engine/reference_spectra.npy``  shape (6, 1024)
   * ``data/reference/<compound>_mean.npy``  one file per compound (debug)
   * ``data/reference/wavenumbers.npy``  shared wavenumber axis
   * ``results/sanity/pure_spectra.png``  visual proof

Reference: PROJECT_REVISION_v2.md §2.7 + project T06 spec.

CLI
---
::

    python scripts/extract_pure_references.py \\
        --pure-dir data/raw/pure \\
        --target-wavenumbers data/processed/wavenumbers.npy \\
        --out engine/reference_spectra.npy \\
        --apply-preprocessing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

# Make `src` importable when run as a script from project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.enlighten_parser import load_enlighten_csv  # noqa: E402


# Canonical compound order — MUST match the column order of `AA_Data.csv`
# label cols: [Alanine, Asparagine, Aspartic Acid, Glutamic Acid, Histidine, Glucosamine].
# This is also the order rows of `engine/reference_spectra.npy` will follow.
COMPOUND_ORDER: list[str] = [
    "Alanine",
    "Asparagine",
    "Aspartic Acid",
    "Glutamic Acid",
    "Histidine",
    "Glucosamine",
]

# Mapping: canonical compound name → ENLIGHTEN export filename.
# Verified against PROJECT_REVISION_v2.md §2.7 file list.
DEFAULT_FILE_MAP: dict[str, str] = {
    "Alanine":       "DL-alanine.csv",
    "Asparagine":    "L-asparagine.csv",
    "Aspartic Acid": "L-aspartic-acid.csv",
    "Glutamic Acid": "L-glutamic-acid.csv",
    "Histidine":     "L-histidine.csv",
    "Glucosamine":   "D-glucosamine-HCl.csv",
}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_spectra(
    spectra: np.ndarray,
    *,
    outlier_z_threshold: float = 3.0,
    verbose: bool = True,
    label: str = "",
) -> tuple[np.ndarray, str]:
    """Compute a robust per-compound spectrum from ``(N, P)`` raw spectra.

    Strategy: take the mean. If any single spectrum's mean Z-score against
    its peers exceeds ``outlier_z_threshold``, switch to the median (more
    robust, less sensitive to one bad measurement).

    Parameters
    ----------
    spectra : np.ndarray
        Shape ``(n_spectra, n_pixels)``.
    outlier_z_threshold : float
        Per-spectrum Z-score (computed on its mean intensity) above which
        an outlier is declared.
    verbose : bool
        Print which strategy was used.
    label : str
        Compound name for log messages.

    Returns
    -------
    aggregated : np.ndarray
        Shape ``(n_pixels,)``.
    method : str
        Either ``"mean"`` or ``"median"`` — useful for logging / sanity.
    """
    if spectra.ndim != 2:
        raise ValueError(f"Expected (N, P) array, got shape {spectra.shape}")
    if spectra.shape[0] == 0:
        raise ValueError(f"No spectra to aggregate for {label!r}")

    per_spec_means = np.nanmean(spectra, axis=1)
    mu = float(np.nanmean(per_spec_means))
    sigma = float(np.nanstd(per_spec_means)) or 1.0
    z_scores = np.abs(per_spec_means - mu) / sigma
    n_outliers = int(np.sum(z_scores > outlier_z_threshold))

    if n_outliers > 0:
        method = "median"
        agg = np.nanmedian(spectra, axis=0)
        if verbose:
            print(
                f"  [{label}] {n_outliers}/{spectra.shape[0]} outlier(s) "
                f"(|Z| > {outlier_z_threshold}); using MEDIAN."
            )
    else:
        method = "mean"
        agg = np.nanmean(spectra, axis=0)
        if verbose:
            print(f"  [{label}] no outliers; using MEAN of {spectra.shape[0]}.")
    return agg, method


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_to_grid(
    source_wn: np.ndarray,
    source_intensity: np.ndarray,
    target_wn: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate ``source_intensity`` onto ``target_wn``.

    Both axes are sorted internally (``np.interp`` requires ascending x).
    Out-of-range values use the boundary intensity (no extrapolation).

    Parameters
    ----------
    source_wn : np.ndarray
        Source wavenumber axis, shape ``(n_src,)``.
    source_intensity : np.ndarray
        Source intensities, shape ``(n_src,)``.
    target_wn : np.ndarray
        Target wavenumber axis, shape ``(n_tgt,)``.

    Returns
    -------
    np.ndarray
        Resampled intensities, shape ``(n_tgt,)``.
    """
    src_order = np.argsort(source_wn)
    src_x = source_wn[src_order]
    src_y = source_intensity[src_order]
    return np.interp(target_wn, src_x, src_y)


# ---------------------------------------------------------------------------
# Total-intensity sanity check
# ---------------------------------------------------------------------------

def report_intensity_balance(refs: np.ndarray, names: Sequence[str]) -> dict:
    """Per-compound total integrated intensity. Used for the T06 sanity print.

    The project spec calls out: *"kiểm tra tổng intensity của mỗi pure
    spectrum — chuẩn hóa nếu cần"*. We do NOT auto-normalize here (that
    would silently change the scale of ``s_recon`` vs. training data); we
    just report so the human can decide.
    """
    sums = refs.sum(axis=1)
    mx = float(sums.max())
    mn = float(sums.min())
    ratio = mx / mn if mn > 0 else float("inf")
    summary = {
        "per_compound_total_intensity": {n: float(s) for n, s in zip(names, sums)},
        "max_min_ratio": ratio,
        "warning": (ratio > 3.0),
    }
    print("\nTotal-intensity balance check:")
    for n, s in zip(names, sums):
        print(f"  {n:<14s}  total = {s:>12.3f}")
    if ratio > 3.0:
        print(
            f"  ! WARNING: max/min ratio = {ratio:.2f} (>3.0). The per-compound "
            "scale_i in the reconstruction module will absorb this, so it is "
            "fine, but flagging in case you want to normalize upstream."
        )
    else:
        print(f"  OK: max/min ratio = {ratio:.2f}")
    return summary


# ---------------------------------------------------------------------------
# Preprocessing alignment (optional)
# ---------------------------------------------------------------------------

def maybe_apply_preprocessing(refs: np.ndarray, apply: bool) -> np.ndarray:
    """Apply the training-time preprocessing pipeline to references.

    PROJECT_REVISION_v2.md §2.7 emphasises that references must live on the
    same scale as the training spectra (else the reconstruction loss is
    silently weighted by per-compound calibration differences). When the
    full T04 preprocessing module is available, we forward through it.

    Falls back to identity (with a warning) if the import fails. This
    keeps the script useful in environments where T04 hasn't been built
    yet (e.g. unit testing).
    """
    if not apply:
        return refs
    try:
        # T04 entry point. Signature expected: preprocess_batch(np.ndarray (N,P)) → np.ndarray (N,P).
        from src.data.preprocess import preprocess_batch
    except Exception as exc:  # pragma: no cover -- env-dependent
        print(
            f"\n  ! Could not import src.data.preprocess.preprocess_batch ({exc}); "
            "skipping preprocessing alignment. References saved RAW.",
            file=sys.stderr,
        )
        return refs
    print("\nApplying training-time preprocessing pipeline to references ...")
    out = preprocess_batch(refs.astype(np.float64))
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def save_overlay_plot(
    refs: np.ndarray,
    wn: np.ndarray,
    names: Sequence[str],
    out_path: Path,
) -> None:
    """Save a 6-spectrum overlay plot for visual sanity (T06 deliverable)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(names):
        # Offset each spectrum vertically for readability.
        offset = i * 0.5 * float(np.nanmax(np.abs(refs)))
        ax.plot(wn, refs[i] + offset, color=cmap(i), lw=1.0, label=name)

    ax.set_xlabel(r"Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity (offset for clarity)")
    ax.set_title("Pure reference spectra (6 amino acids / sugars)")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\nSaved overlay plot → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build engine/reference_spectra.npy from ENLIGHTEN exports."
    )
    parser.add_argument(
        "--pure-dir",
        type=Path,
        default=Path("data/raw/pure"),
        help="Directory containing the 6 ENLIGHTEN CSV files.",
    )
    parser.add_argument(
        "--target-wavenumbers",
        type=Path,
        default=Path("data/processed/wavenumbers.npy"),
        help="Path to .npy file with the target wavenumber grid (e.g. AA dataset's).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("engine/reference_spectra.npy"),
        help="Output path for the (6, P) reference tensor.",
    )
    parser.add_argument(
        "--per-compound-dir",
        type=Path,
        default=Path("data/reference"),
        help="Directory for per-compound debug .npy files.",
    )
    parser.add_argument(
        "--sanity-plot",
        type=Path,
        default=Path("results/sanity/pure_spectra.png"),
        help="Where to save the overlay plot.",
    )
    parser.add_argument(
        "--apply-preprocessing",
        action="store_true",
        help="Run T04 preprocessing pipeline on each reference (recommended).",
    )
    parser.add_argument(
        "--outlier-z",
        type=float,
        default=3.0,
        help="Per-spectrum mean-intensity Z-score for outlier detection.",
    )
    parser.add_argument(
        "--no-resample",
        action="store_true",
        help="Skip wavenumber resampling. Only use if ENLIGHTEN files already "
        "match the target grid.",
    )
    args = parser.parse_args(argv)

    # ---- Resolve target grid ----
    if args.no_resample:
        target_wn = None  # will be inferred from first file
        print("Resampling DISABLED (--no-resample).")
    else:
        if not args.target_wavenumbers.exists():
            raise FileNotFoundError(
                f"Target wavenumber file not found: {args.target_wavenumbers}\n"
                f"Hint: run scripts/prepare_data.py first to build "
                f"data/processed/wavenumbers.npy, OR pass --no-resample to keep "
                f"the ENLIGHTEN native grid."
            )
        target_wn = np.load(args.target_wavenumbers).astype(np.float64)
        print(
            f"Loaded target grid from {args.target_wavenumbers}: "
            f"{target_wn.size} channels, "
            f"{target_wn.min():.2f} → {target_wn.max():.2f} cm^-1."
        )

    # ---- Iterate compounds in canonical order ----
    refs = []
    aggregation_methods: dict[str, str] = {}
    n_inputs: dict[str, int] = {}
    native_wavenumbers: np.ndarray | None = None

    print("\nLoading ENLIGHTEN files:")
    for compound in COMPOUND_ORDER:
        fname = DEFAULT_FILE_MAP[compound]
        fpath = args.pure_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Missing pure-compound file for {compound!r}: {fpath}. "
                f"Expected one of {list(DEFAULT_FILE_MAP.values())} in {args.pure_dir}."
            )
        parsed = load_enlighten_csv(fpath)
        n_inputs[compound] = parsed.n_spectra
        print(
            f"  {compound:<14s}  ← {fname:<28s} "
            f"({parsed.n_spectra} spectra, {parsed.spectra.shape[1]} pixels, "
            f"note='{parsed.note}')"
        )

        agg, method = aggregate_spectra(
            parsed.spectra,
            outlier_z_threshold=args.outlier_z,
            verbose=True,
            label=compound,
        )
        aggregation_methods[compound] = method

        if args.no_resample:
            if native_wavenumbers is None:
                native_wavenumbers = parsed.wavenumbers
            else:
                if not np.allclose(native_wavenumbers, parsed.wavenumbers, atol=1e-6):
                    raise ValueError(
                        f"--no-resample but {compound} has a different "
                        "wavenumber axis from the first file."
                    )
            ref_on_grid = agg
        else:
            ref_on_grid = resample_to_grid(parsed.wavenumbers, agg, target_wn)

        refs.append(ref_on_grid)

    refs_arr = np.stack(refs, axis=0).astype(np.float64)  # (6, P)
    used_wn = native_wavenumbers if args.no_resample else target_wn
    print(f"\nStacked reference tensor: shape {refs_arr.shape}")

    # ---- Optional preprocessing alignment ----
    refs_arr = maybe_apply_preprocessing(refs_arr, apply=args.apply_preprocessing)

    # ---- Sanity report ----
    intensity_summary = report_intensity_balance(refs_arr, COMPOUND_ORDER)

    # ---- Save outputs ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, refs_arr.astype(np.float32))
    print(f"\nSaved {args.out}  (shape={refs_arr.shape}, dtype=float32)")

    args.per_compound_dir.mkdir(parents=True, exist_ok=True)
    safe_name = lambda s: s.lower().replace(" ", "_")
    for i, name in enumerate(COMPOUND_ORDER):
        fp = args.per_compound_dir / f"{safe_name(name)}_mean.npy"
        np.save(fp, refs_arr[i].astype(np.float32))
    print(f"Saved 6 per-compound files in {args.per_compound_dir}/")

    if used_wn is not None:
        wn_path = args.per_compound_dir / "wavenumbers.npy"
        np.save(wn_path, used_wn.astype(np.float64))
        print(f"Saved wavenumber axis → {wn_path}")

    # ---- Manifest (for traceability) ----
    manifest = {
        "compound_order": COMPOUND_ORDER,
        "source_files": {c: DEFAULT_FILE_MAP[c] for c in COMPOUND_ORDER},
        "n_spectra_per_compound": n_inputs,
        "aggregation_method_per_compound": aggregation_methods,
        "preprocessing_applied": bool(args.apply_preprocessing),
        "resampled_to_target_grid": (not args.no_resample),
        "target_grid_path": str(args.target_wavenumbers) if not args.no_resample else None,
        "n_pixels_output": int(refs_arr.shape[1]),
        "intensity_balance": intensity_summary,
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved manifest → {manifest_path}")

    # ---- Plot ----
    if used_wn is not None:
        save_overlay_plot(refs_arr, used_wn, COMPOUND_ORDER, args.sanity_plot)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
