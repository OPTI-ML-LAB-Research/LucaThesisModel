"""Loader for Wasatch ENLIGHTEN wide-format Raman CSV files.

Real layout (confirmed against an actual ENLIGHTEN 3.2.6 export):

    Row 0        : ENLIGHTEN Version,3.2.6
    Rows 1..42   : metadata (Measurement ID, Serial Number, CCD C0-C4,
                   Laser ..., Pixel Count, ...).
    Row 43       : blank
    Row 44       : <serial>,,<timestamp1>,,<timestamp2>,, ...
    Row 45       : HEADER -> Pixel,Wavelength,Processed,Raw,Processed,Raw,...
                   (column 0 = "Pixel", column 1 = "Wavelength", then a
                   repeating (Processed, Raw) pair per measurement)
    Rows 46..N   : DATA, one PIXEL per row:
                   <pixel_index>,<wavenumber>,<proc_m0>,<raw_m0>,<proc_m1>,...

The wavenumber axis lives in column 1 ("Wavelength", i.e. Raman shift in
cm-1). Each measurement contributes a "Processed" column (dark/baseline
corrected) and a "Raw" column.

We locate the header row by finding the row whose first cell is "Pixel"
and which contains a "Wavelength" column, then read every subsequent row
as pixel data. Robust to the exact length of the metadata block.

Standalone (no project imports) so it can be unit-tested and reused by
the reference builder without touching the AAM pipeline.

Author: stretch fix (quartz/calcite reference).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np

_PIXEL_LABELS = ("pixel",)
_WAVELENGTH_LABELS = ("wavelength", "wavenumber", "wavenumbers", "raman shift")
_INTENSITY_LABELS_PREFERENCE = ("processed", "raw", "reference")


def _to_float(s: str) -> Optional[float]:
    s = s.strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _find_header(rows: list[list[str]]) -> int:
    """Return the index of the data-table header row.

    The header is the row whose first cell == 'Pixel' and which also
    contains a wavelength/wavenumber column.
    """
    for i, row in enumerate(rows):
        if not row:
            continue
        first = row[0].strip().lower()
        if first in _PIXEL_LABELS:
            lowered = [c.strip().lower() for c in row]
            if any(w in lowered for w in _WAVELENGTH_LABELS):
                return i
    raise ValueError(
        "No data-table header row found (expected a row starting with "
        "'Pixel' that also has a 'Wavelength'/'Wavenumber' column). "
        "Is this really an ENLIGHTEN tabular export?"
    )


def load_enlighten_csv(
    path: str | Path,
    *,
    intensity_row: Optional[str] = None,   # selects the header series label
    aggregate: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Load an ENLIGHTEN tabular CSV.

    Parameters
    ----------
    path : str or Path
        Path to the ENLIGHTEN ``.csv`` export.
    intensity_row : str, optional
        Which intensity series column to use ("Processed" or "Raw").
        Case-insensitive. If None, auto-detects Processed -> Raw ->
        Reference. (Named ``intensity_row`` for backward compatibility;
        it selects the header column label.)
    aggregate : {"mean", "median", "none"}
        How to combine the multiple measurements into a single spectrum.
        ``"none"`` returns all spectra stacked (N, P).

    Returns
    -------
    wavenumbers : np.ndarray, shape (P,)
        The Raman-shift axis (the "Wavelength" column, in cm-1).
    intensities : np.ndarray
        Aggregated (P,) if mean/median, else (N, P).
    """
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.reader(fh))

    hdr_idx = _find_header(rows)
    header = rows[hdr_idx]
    lowered = [c.strip().lower() for c in header]

    wl_col = next((j for j, c in enumerate(lowered)
                   if c in _WAVELENGTH_LABELS), None)
    if wl_col is None:
        raise ValueError(f"No wavelength column in header of {path}.")

    if intensity_row is not None:
        series = intensity_row.strip().lower()
        if series not in lowered:
            raise ValueError(
                f"Requested series {intensity_row!r} not in header of "
                f"{path}. Header labels: {sorted(set(lowered))}"
            )
    else:
        series = next((s for s in _INTENSITY_LABELS_PREFERENCE
                       if s in lowered), None)
        if series is None:
            raise ValueError(
                f"No Processed/Raw/Reference column in header of {path}."
            )

    intensity_cols = [j for j, c in enumerate(lowered) if c == series]
    if not intensity_cols:
        raise ValueError(f"No '{series}' columns found in {path}.")

    wn_list: list[float] = []
    meas: list[list[float]] = [[] for _ in intensity_cols]

    for row in rows[hdr_idx + 1:]:
        if not row:
            continue
        first = row[0].strip()
        if first == "" or not first.lstrip("-").isdigit():
            continue
        wl = _to_float(row[wl_col]) if wl_col < len(row) else None
        if wl is None:
            continue
        wn_list.append(wl)
        for k, col in enumerate(intensity_cols):
            v = _to_float(row[col]) if col < len(row) else None
            meas[k].append(v if v is not None else np.nan)

    if not wn_list:
        raise ValueError(f"No pixel data rows parsed from {path}.")

    wn = np.array(wn_list, dtype=np.float64)
    P = wn.size
    stack = [np.array(m, dtype=np.float64) for m in meas if len(m) == P]
    if not stack:
        raise ValueError(
            f"No intensity series matched the {P}-pixel axis in {path}."
        )
    arr = np.vstack(stack)  # (N, P)
    if aggregate == "none":
        return wn, arr
    if aggregate == "median":
        return wn, np.nanmedian(arr, axis=0)
    return wn, np.nanmean(arr, axis=0)


def resample_to_grid(
    wn_src: np.ndarray,
    inten_src: np.ndarray,
    wn_target: np.ndarray,
) -> np.ndarray:
    """Linearly resample a spectrum onto a target wavenumber grid.

    Out-of-range targets are clamped to the nearest edge (np.interp
    default). Source axis flipped automatically if descending.
    """
    wn_src = np.asarray(wn_src, dtype=np.float64)
    inten_src = np.asarray(inten_src, dtype=np.float64)
    if wn_src[0] > wn_src[-1]:
        wn_src = wn_src[::-1]
        inten_src = inten_src[::-1]
    return np.interp(wn_target, wn_src, inten_src)