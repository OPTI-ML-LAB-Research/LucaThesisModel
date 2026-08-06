"""API (Active Pharmaceutical Ingredients) dataset loader.

The API dataset is a cross-domain OOD probe: 33 pharmaceutical chemicals,
~3511 spectra, wavenumber-axis CSV. Used to demonstrate that the
AA-trained model correctly flags pharmaceutical compounds (entirely
different chemistry) as OOD.

Input file: data/raw/API/API_data.csv (185 MB, 3511 x 3278)
- Wavenumber columns (not wavelength)
- May or may not have a class/label column

Usage:
    from src.data.api_loader import load_api_subset
    specs = load_api_subset(n=500)
    # specs: (500, 1024) resampled to AA grid

Author: Chat 5 stretch T31.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_api_subset(
    n: int = 500,
    seed: int = 42,
    target_wavenumbers: np.ndarray | None = None,
    apply_preprocessing: bool = True,
    api_csv: str | Path = "data/raw/API/API_data.csv",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load a subset of API spectra, resampled to AA wavenumber grid.

    Args:
        n: number of samples.
        seed: RNG seed.
        target_wavenumbers: 1D array; if None, load from cache.
        apply_preprocessing: apply AA's 4-step pipeline.
        api_csv: path to API_data.csv.

    Returns:
        spectra: (n, 1024) float32.
        class_ids: (n,) int or None (if no label column found).
    """
    api_path = Path(api_csv)
    if not api_path.exists():
        raise FileNotFoundError(f"API_data.csv not found at {api_path}")

    print(f"  loading API CSV (~185 MB, may take 30-60 s) ...")
    df = pd.read_csv(api_path)
    print(f"  CSV shape: {df.shape}")

    cols = list(df.columns)
    spec_cols, meta_cols = [], []
    label_col = None
    # API axis is wavenumber → header values 100-3500
    for c in cols:
        s = str(c)
        try:
            val = float(s)
            if 50 <= val <= 4000:
                spec_cols.append(c)
                continue
        except ValueError:
            pass
        if any(k in s.lower() for k in ["class", "label", "compound", "name", "id"]):
            label_col = c
        else:
            meta_cols.append(c)
    print(f"  detected: {len(spec_cols)} spectra cols, "
          f"label col: {label_col}, {len(meta_cols)} other meta")

    if len(spec_cols) < 1000:
        raise ValueError(f"Too few spectral cols detected: {len(spec_cols)}")

    X_raw = df[spec_cols].values.astype(np.float32)
    wn_api = np.array([float(c) for c in spec_cols], dtype=np.float64)
    # Ensure ascending
    if wn_api[0] > wn_api[-1]:
        wn_api_asc = wn_api[::-1]
        X_raw_asc = X_raw[:, ::-1]
    else:
        wn_api_asc = wn_api
        X_raw_asc = X_raw

    # Extract labels if available
    y_full = None
    if label_col is not None:
        labels_raw = df[label_col].values
        # Try to encode as int (categorical)
        try:
            from pandas import factorize
            y_full, _ = factorize(labels_raw)
        except Exception:
            y_full = None

    # Subsample
    rng = np.random.default_rng(seed)
    n_take = min(n, X_raw.shape[0])
    idx = rng.choice(X_raw.shape[0], size=n_take, replace=False)
    X_sub = X_raw_asc[idx]
    y_sub = y_full[idx] if y_full is not None else None

    # Resample to AA grid
    if target_wavenumbers is None:
        target_wavenumbers = np.load("data/processed/wavenumbers.npy")
    target = np.asarray(target_wavenumbers, dtype=np.float64)
    if target[0] > target[-1]:
        target_asc = target[::-1]
        flipped_out = True
    else:
        target_asc = target
        flipped_out = False

    # API range likely covers wider than AA (267-2004) -- np.interp will
    # extrapolate with zero outside, which is fine for the OOD score
    out = np.empty((X_sub.shape[0], target.shape[0]), dtype=np.float32)
    for i in range(X_sub.shape[0]):
        resampled = np.interp(target_asc, wn_api_asc, X_sub[i],
                               left=0.0, right=0.0)
        if flipped_out:
            resampled = resampled[::-1]
        out[i] = resampled

    if apply_preprocessing:
        from src.data.preprocess import preprocess_batch
        out = preprocess_batch(out).astype(np.float32)

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out, y_sub
