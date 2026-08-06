"""ENLIGHTEN spectrometer CSV parser.

The Wasatch ENLIGHTEN software (used by the WP-785-R-SR-LMMF spectrometer
that produced the AA/AAM/pure-compound datasets) exports spectra in a
proprietary CSV format with these characteristics:

* ~30 rows of metadata (Measurement ID, Serial Number, Note, Laser
  Wavelength, etc.), each laid out as a comma-separated row with the value
  duplicated across columns (one column per spectrum in the file).
* A header row starting with ``Pixel,Wavelength,Wavenumber,`` followed by
  alternating ``Processed,,Processed,,...`` columns — one ``Processed``
  per spectrum, separated by blank columns.
* ~1024 data rows, each starting with ``pixel,wavelength,wavenumber,``
  followed by intensity values for each spectrum (interleaved with blanks).

This module locates the header row programmatically and extracts only the
``Processed`` columns plus the ``Wavenumber`` axis.

Reference: PROJECT_REVISION_v2.md §2.4.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


HEADER_TOKEN = "Pixel"
WAVENUMBER_TOKEN = "Wavenumber"
WAVELENGTH_TOKEN = "Wavelength"
PROCESSED_TOKEN = "Processed"


@dataclass
class EnlightenSpectraFile:
    """Parsed contents of one ENLIGHTEN export file.

    Attributes
    ----------
    spectra : np.ndarray
        Shape ``(n_spectra, n_pixels)``. Each row is one ``Processed``
        intensity vector.
    wavenumbers : np.ndarray
        Shape ``(n_pixels,)``. Raman shift axis in cm^-1 (ascending or
        descending depending on the file; not reordered here).
    wavelengths : np.ndarray
        Shape ``(n_pixels,)``. Wavelength axis in nm.
    note : str
        Contents of the ``Note`` metadata row (typically the compound name).
    laser_wavelength_nm : float | None
        Excitation laser wavelength if present in the metadata.
    n_spectra : int
        Convenience: number of spectra in the file.
    """

    spectra: np.ndarray
    wavenumbers: np.ndarray
    wavelengths: np.ndarray
    note: str = ""
    laser_wavelength_nm: float | None = None

    @property
    def n_spectra(self) -> int:
        return self.spectra.shape[0]


def _safe_float(token: str) -> float | None:
    """Return float(token) or None if it is empty / non-numeric."""
    token = token.strip()
    if not token:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _find_header_row(rows: list[list[str]]) -> int:
    """Locate the index of the row beginning with ``Pixel,Wavelength,...``.

    Raises
    ------
    ValueError
        If no such row is found in the file.
    """
    for idx, row in enumerate(rows):
        if not row:
            continue
        first = row[0].strip()
        if first == HEADER_TOKEN and len(row) > 2:
            # Defensive: confirm the next two cells are wavelength + wavenumber
            if (row[1].strip() == WAVELENGTH_TOKEN
                    and row[2].strip() == WAVENUMBER_TOKEN):
                return idx
    raise ValueError(
        f"Could not find ENLIGHTEN header row "
        f"({HEADER_TOKEN},{WAVELENGTH_TOKEN},{WAVENUMBER_TOKEN},...). "
        "File may not be an ENLIGHTEN export."
    )


def _processed_column_indices(header_row: list[str]) -> list[int]:
    """Return the column indices in the header row marked ``Processed``."""
    return [i for i, cell in enumerate(header_row) if cell.strip() == PROCESSED_TOKEN]


def _extract_metadata(rows: list[list[str]], header_idx: int) -> tuple[str, float | None]:
    """Pull the ``Note`` line and ``Laser Wavelength`` from the metadata block.

    Parameters
    ----------
    rows : list[list[str]]
        All CSV rows.
    header_idx : int
        Row index of the data header (metadata is everything before it).
    """
    note = ""
    laser_nm: float | None = None
    for row in rows[:header_idx]:
        if not row:
            continue
        key = row[0].strip().lower()
        if key == "note":
            # Note is typically duplicated across all spectrum columns; take
            # the first non-empty token after the key.
            for cell in row[1:]:
                cell = cell.strip()
                if cell:
                    note = cell
                    break
        elif key.startswith("laser wavelength"):
            for cell in row[1:]:
                val = _safe_float(cell)
                if val is not None:
                    laser_nm = val
                    break
    return note, laser_nm


def load_enlighten_csv(path: str | Path) -> EnlightenSpectraFile:
    """Parse an ENLIGHTEN export CSV.

    Parameters
    ----------
    path : str or Path
        Path to the ``.csv`` file.

    Returns
    -------
    EnlightenSpectraFile
        Spectra with shape ``(n_spectra, n_pixels)`` and axis information.

    Notes
    -----
    The parser is tolerant of trailing blank columns and of variable
    numbers of spectra per file. It assumes the data block contains only
    numeric cells (or empty cells), and that all ``Processed`` columns
    have the same length.

    Examples
    --------
    >>> result = load_enlighten_csv("data/raw/pure/DL-alanine.csv")
    >>> result.spectra.shape  # doctest: +SKIP
    (10, 1024)
    >>> result.note  # doctest: +SKIP
    'DL-alanine'
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ENLIGHTEN file not found: {path}")

    # Read all rows up-front. Files are small (~hundreds of KB), so this is
    # fine and dramatically simplifies the two-pass logic below.
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    header_idx = _find_header_row(rows)
    note, laser_nm = _extract_metadata(rows, header_idx)

    header_row = rows[header_idx]
    processed_cols = _processed_column_indices(header_row)
    if not processed_cols:
        raise ValueError(
            f"No '{PROCESSED_TOKEN}' columns found in header of {path.name}. "
            "Expected at least one spectrum per file."
        )

    wavelengths: list[float] = []
    wavenumbers: list[float] = []
    spectra_cols: list[list[float]] = [[] for _ in processed_cols]

    for row in rows[header_idx + 1:]:
        if not row or all((cell.strip() == "") for cell in row):
            continue
        # Data row format: pixel, wavelength, wavenumber, processed, , processed, ...
        if len(row) < 3:
            continue
        wl = _safe_float(row[1])
        wn = _safe_float(row[2])
        if wl is None or wn is None:
            # Skip non-numeric trailing rows (some exports add a footer)
            continue
        wavelengths.append(wl)
        wavenumbers.append(wn)
        for slot, col_idx in enumerate(processed_cols):
            cell = row[col_idx] if col_idx < len(row) else ""
            val = _safe_float(cell)
            # If a single cell is missing, fall back to NaN; caller can
            # decide to drop / interpolate. With clean ENLIGHTEN files
            # this branch rarely fires.
            spectra_cols[slot].append(val if val is not None else np.nan)

    spectra = np.array(spectra_cols, dtype=np.float64)  # (n_spectra, n_pixels)
    wavenumbers_arr = np.array(wavenumbers, dtype=np.float64)
    wavelengths_arr = np.array(wavelengths, dtype=np.float64)

    if spectra.size == 0:
        raise ValueError(f"No data rows parsed from {path.name}")

    return EnlightenSpectraFile(
        spectra=spectra,
        wavenumbers=wavenumbers_arr,
        wavelengths=wavelengths_arr,
        note=note,
        laser_wavelength_nm=laser_nm,
    )


def load_many(paths: Iterable[str | Path]) -> list[EnlightenSpectraFile]:
    """Convenience: parse a list of ENLIGHTEN files in order."""
    return [load_enlighten_csv(p) for p in paths]


__all__ = [
    "EnlightenSpectraFile",
    "load_enlighten_csv",
    "load_many",
]
