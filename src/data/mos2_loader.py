"""MoS2 reference spectrum loader for OOD demo.

Parses the tab-separated MoS2 spectrum file and resamples it to the
AA wavenumber grid so it can be fed through the standard predict()
pipeline.

Input file format (data/raw/ood_demo/MoS2-160o-12h-ph5.txt):
    Tab-separated, 2 columns: wavenumber (cm-1) and intensity.
    Around 525 rows. May have a header line we need to skip.

Usage:
    from src.data.mos2_loader import load_mos2_spectrum
    spec = load_mos2_spectrum("data/raw/ood_demo/MoS2-160o-12h-ph5.txt")
    # spec.shape == (1024,) -- resampled onto AA wavenumber grid

Author: Chat 5 stretch T30A.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def _parse_raw_mos2(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse the raw MoS2 file. Returns (wavenumber, intensity).

    Robust to a header line (commonly "Pixel\\tWavenumber\\tIntensity"
    or similar) by skipping non-numeric lines.
    """
    wn_list: list[float] = []
    int_list: list[float] = []
    with open(path) as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            # Split on tab or whitespace
            parts = raw.split()
            if len(parts) < 2:
                continue
            try:
                wn = float(parts[0])
                # If first col is pixel index (small ints from 0/1), use 2nd as wn
                if len(parts) >= 3 and 0 <= wn <= 2000 and wn != int(wn):
                    # First col is wavenumber
                    intensity = float(parts[1])
                elif len(parts) >= 3:
                    # 3-col format: pixel, wn, intensity
                    wn = float(parts[1])
                    intensity = float(parts[2])
                else:
                    intensity = float(parts[1])
            except ValueError:
                # Header row -- skip
                continue
            wn_list.append(wn)
            int_list.append(intensity)
    if not wn_list:
        raise ValueError(f"No numeric rows parsed from {path}")
    wn_arr = np.array(wn_list, dtype=np.float64)
    int_arr = np.array(int_list, dtype=np.float64)
    # Sort ascending by wavenumber for safe interpolation
    order = np.argsort(wn_arr)
    return wn_arr[order], int_arr[order]


def load_mos2_spectrum(
    path: str | Path,
    target_wavenumbers: np.ndarray | None = None,
    apply_preprocessing: bool = True,
) -> np.ndarray:
    """Load MoS2 spectrum and resample to the AA wavenumber grid.

    Args:
        path: path to MoS2-160o-12h-ph5.txt (or any 2-col Raman file).
        target_wavenumbers: 1D array of wavenumbers to resample onto.
            If None, loads from data/processed/wavenumbers.npy.
        apply_preprocessing: if True, apply the same 4-step pipeline
            (cosmic + AsLS + Savitzky-Golay + SNV) used for AA spectra.
            This keeps MoS2 on the same scale as training data.

    Returns:
        ndarray of shape (1024,), float32. NaN/Inf-free.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MoS2 spectrum not found at {path}")

    wn_raw, int_raw = _parse_raw_mos2(path)

    # Load target grid
    if target_wavenumbers is None:
        target_wavenumbers = np.load("data/processed/wavenumbers.npy")
    target = np.asarray(target_wavenumbers, dtype=np.float64)

    # AA grid may be descending; resample needs ascending interp
    if target[0] > target[-1]:
        target_asc = target[::-1]
        flipped = True
    else:
        target_asc = target
        flipped = False

    # Linear interpolation -- safe for grids that overlap
    # MoS2 file typically covers ~200-2000 cm-1, AA covers 267-2004, so overlap is ~complete
    overlap_min = max(wn_raw.min(), target_asc.min())
    overlap_max = min(wn_raw.max(), target_asc.max())
    if overlap_max < overlap_min:
        raise ValueError(
            f"No overlap between MoS2 range [{wn_raw.min():.1f}, {wn_raw.max():.1f}] "
            f"and target range [{target_asc.min():.1f}, {target_asc.max():.1f}]"
        )

    resampled = np.interp(target_asc, wn_raw, int_raw,
                           left=0.0, right=0.0)
    if flipped:
        resampled = resampled[::-1]

    if apply_preprocessing:
        # Import lazily to avoid circular deps
        from src.data.preprocess import preprocess_pipeline
        resampled = preprocess_pipeline(resampled.astype(np.float32))

    return resampled.astype(np.float32)
