"""Bacteria-ID dataset loader for cross-domain OOD evaluation.

Loads the Ho et al. 2019 Nature Comms dataset (30 bacteria species,
1000-channel Raman spectra preprocessed by the authors) and returns
a torch-friendly tensor resampled onto the AA wavenumber grid for
OOD scoring.

Input files (under data/raw/bacteria_ID/):
    X_reference.npy  -- ~60K spectra, 1000 channels (large reference set)
    X_test.npy       -- ~3K spectra, 1000 channels (test set, our default)
    y_test.npy       -- corresponding class labels (0-29)
    wavenumbers.npy  -- 1000-element wavenumber axis

Usage:
    from src.data.bacteria_id_loader import load_bacteria_id_subset
    specs = load_bacteria_id_subset(n=1000, split="test")
    # specs.shape == (1000, 1024) -- resampled onto AA grid

Author: Chat 5 stretch T30B.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np


def load_bacteria_id_subset(
    n: int = 1000,
    split: Literal["test", "reference", "2018clinical"] = "test",
    seed: int = 42,
    target_wavenumbers: np.ndarray | None = None,
    apply_preprocessing: bool = True,
    bacteria_dir: str | Path = "data/raw/bacteria_ID",
) -> np.ndarray:
    """Load a random subset of bacteria spectra, resampled to AA grid.

    Args:
        n: number of spectra to subsample. Default 1000.
        split: which file to load -- "test" (3K), "reference" (60K),
            "2018clinical" (10K). Default "test".
        seed: RNG seed for subsampling.
        target_wavenumbers: 1D array; if None, load from cache.
        apply_preprocessing: if True, apply the AA preprocessing pipeline
            so bacteria spectra are on the same scale as training data.
        bacteria_dir: directory containing X_*.npy and wavenumbers.npy.

    Returns:
        ndarray (n, 1024) float32, free of NaN/Inf.
    """
    bdir = Path(bacteria_dir)
    if not bdir.exists():
        raise FileNotFoundError(
            f"Bacteria_ID directory not found: {bdir}. "
            f"Download from Ho 2019 paper Github."
        )

    fname = f"X_{split}.npy"
    x_path = bdir / fname
    if not x_path.exists():
        raise FileNotFoundError(f"{x_path} not found")
    X_full = np.load(x_path)  # (N, 1000)
    wn_bact = np.load(bdir / "wavenumbers.npy")  # (1000,)
    if wn_bact.shape[0] != X_full.shape[1]:
        raise ValueError(
            f"bacteria wn shape {wn_bact.shape} mismatch X cols {X_full.shape[1]}"
        )

    # Subsample
    rng = np.random.default_rng(seed)
    N_total = X_full.shape[0]
    if n >= N_total:
        idx = np.arange(N_total)
    else:
        idx = rng.choice(N_total, size=n, replace=False)
    X_sub = X_full[idx]  # (n, 1000)

    # Load AA target grid
    if target_wavenumbers is None:
        target_wavenumbers = np.load("data/processed/wavenumbers.npy")
    target = np.asarray(target_wavenumbers, dtype=np.float64)

    # Both axes may need orientation alignment
    if wn_bact[0] > wn_bact[-1]:
        wn_bact_asc = wn_bact[::-1]
        X_sub_asc = X_sub[:, ::-1]
    else:
        wn_bact_asc = wn_bact
        X_sub_asc = X_sub
    if target[0] > target[-1]:
        target_asc = target[::-1]
        flipped_out = True
    else:
        target_asc = target
        flipped_out = False

    # Vectorised linear interpolation
    out = np.empty((X_sub.shape[0], target.shape[0]), dtype=np.float32)
    for i in range(X_sub.shape[0]):
        resampled = np.interp(target_asc, wn_bact_asc, X_sub_asc[i],
                               left=0.0, right=0.0)
        if flipped_out:
            resampled = resampled[::-1]
        out[i] = resampled

    if apply_preprocessing:
        from src.data.preprocess import preprocess_batch
        out = preprocess_batch(out).astype(np.float32)

    # Replace any NaN/Inf with 0 (safety)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out
