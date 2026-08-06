"""AAM (AA + Minerals) dataset loader for zero-shot OOD evaluation.

AAM dataset (Zarei 2023, 12956 spectra) extends AA with quartz + calcite
minerals. The MVP model trained on AA-only has 6 outputs; AAM has 8
components (6 AA + quartz + calcite).

This loader handles ZERO-SHOT OOD use: feed AAM spectra to the
AA-trained model. Samples with non-zero mineral content should trigger
high reconstruction error (the 6 AA pure refs cannot represent quartz
peak at 464 cm-1 or calcite at 1086 cm-1).

Input file: data/raw/AAM_Data.csv (12956 x 1033)
- 1024 spectral columns (wavelength headers, same range as AA)
- 1 metadata column ('names' or similar)
- 8 ratio columns (sum to 1, including quartz/calcite)

Usage:
    from src.data.aam_loader import load_aam_subset
    specs, has_minerals = load_aam_subset(n=500, only_with_minerals=False)
    # specs: (500, 1024) resampled to AA grid
    # has_minerals: (500,) bool -- True if quartz+calcite > some_threshold

Author: Chat 5 stretch T29.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _wavelength_to_wavenumber(wl_nm: np.ndarray,
                               laser_nm: float = 784.815734863281) -> np.ndarray:
    """Convert wavelength (nm) → Raman shift (cm-1)."""
    return 1e7 * (1.0 / laser_nm - 1.0 / wl_nm)


def load_aam_subset(
    n: int = 500,
    only_with_minerals: bool = False,
    mineral_threshold: float = 0.05,
    seed: int = 42,
    target_wavenumbers: np.ndarray | None = None,
    apply_preprocessing: bool = True,
    aam_csv: str | Path = "data/raw/AAM_Data.csv",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a subset of AAM spectra, resampled to AA wavenumber grid.

    Args:
        n: number of samples to subsample (random).
        only_with_minerals: if True, only return samples where
            quartz + calcite > mineral_threshold. Useful for proving
            OOD detection works on mineral-containing samples.
        mineral_threshold: cut-off for "has minerals" flag (default 0.05).
        seed: RNG seed.
        target_wavenumbers: 1D array; if None, load from AA cache.
        apply_preprocessing: apply same 4-step pipeline as AA.
        aam_csv: path to AAM_Data.csv.

    Returns:
        spectra: (n, 1024) float32, AA grid, preprocessed.
        labels_8d: (n, 8) float32, full 8-component composition
            [Ala, Asn, Asp, Glu, His, Glc, Quartz, Calcite] (canonical
            order assumed; verify against CSV header at runtime).
        has_minerals: (n,) bool.
    """
    aam_path = Path(aam_csv)
    if not aam_path.exists():
        raise FileNotFoundError(
            f"AAM_Data.csv not found at {aam_path}. "
            f"Place it under data/raw/ first."
        )

    print(f"  loading AAM CSV (~91 MB, may take 10-30 s) ...")
    df = pd.read_csv(aam_path)
    print(f"  CSV shape: {df.shape}")

    # The first 1024 cols are spectra (header = wavelength in nm).
    # Find metadata + ratio cols by scanning column names.
    cols = list(df.columns)
    # Heuristic: spectra cols are those that look like floats (wavelengths)
    spec_cols, meta_cols, ratio_cols = [], [], []
    for c in cols:
        s = str(c)
        try:
            val = float(s)
            if 700 <= val <= 1000:  # AA wavelength range 801-931 nm
                spec_cols.append(c)
                continue
        except ValueError:
            pass
        # If it's not a wavelength, check if it's a known ratio name
        lower = s.lower().strip().replace(" ", "_").replace("-", "_")
        if any(k in lower for k in ["alanine", "asparagine", "aspartic",
                                      "glutamic", "histidine", "glucosamine",
                                      "quartz", "calcite"]):
            ratio_cols.append(c)
        else:
            meta_cols.append(c)
    print(f"  detected: {len(spec_cols)} spectra cols, "
          f"{len(meta_cols)} meta cols, {len(ratio_cols)} ratio cols")

    if len(spec_cols) != 1024:
        raise ValueError(f"Expected 1024 spectral cols, got {len(spec_cols)}")
    if len(ratio_cols) < 6:
        raise ValueError(
            f"Expected at least 6 ratio columns (AA), got {len(ratio_cols)}. "
            f"CSV may use unexpected column naming."
        )

    # Extract arrays
    X_raw = df[spec_cols].values.astype(np.float32)  # (N, 1024)
    Y_raw = df[ratio_cols].values.astype(np.float32)  # (N, 6 or 8)

    # Wavelength from headers
    wl_nm = np.array([float(c) for c in spec_cols], dtype=np.float64)
    wn_aam = _wavelength_to_wavenumber(wl_nm)
    # Ensure ascending for interp
    if wn_aam[0] > wn_aam[-1]:
        wn_aam_asc = wn_aam[::-1]
        X_raw_asc = X_raw[:, ::-1]
    else:
        wn_aam_asc = wn_aam
        X_raw_asc = X_raw

    # Mineral mask
    if Y_raw.shape[1] >= 8:
        # Assume last 2 cols are quartz + calcite (adjust if order differs)
        minerals_total = Y_raw[:, -2:].sum(axis=1)
    else:
        minerals_total = np.zeros(Y_raw.shape[0])
    has_minerals_full = minerals_total > mineral_threshold
    print(f"  AAM samples with minerals > {mineral_threshold}: "
          f"{has_minerals_full.sum()} / {len(has_minerals_full)}")

    # Subsample
    rng = np.random.default_rng(seed)
    if only_with_minerals:
        candidates = np.where(has_minerals_full)[0]
        if len(candidates) == 0:
            raise ValueError("No samples with minerals found")
        n_take = min(n, len(candidates))
        idx = rng.choice(candidates, size=n_take, replace=False)
    else:
        n_take = min(n, X_raw.shape[0])
        idx = rng.choice(X_raw.shape[0], size=n_take, replace=False)
    X_sub = X_raw_asc[idx]
    Y_sub = Y_raw[idx]
    minerals_sub = has_minerals_full[idx]

    # Resample to AA target grid
    if target_wavenumbers is None:
        target_wavenumbers = np.load("data/processed/wavenumbers.npy")
    target = np.asarray(target_wavenumbers, dtype=np.float64)
    if target[0] > target[-1]:
        target_asc = target[::-1]
        flipped_out = True
    else:
        target_asc = target
        flipped_out = False

    out = np.empty((X_sub.shape[0], target.shape[0]), dtype=np.float32)
    for i in range(X_sub.shape[0]):
        resampled = np.interp(target_asc, wn_aam_asc, X_sub[i],
                               left=0.0, right=0.0)
        if flipped_out:
            resampled = resampled[::-1]
        out[i] = resampled

    if apply_preprocessing:
        from src.data.preprocess import preprocess_batch
        out = preprocess_batch(out).astype(np.float32)

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out, Y_sub, minerals_sub
