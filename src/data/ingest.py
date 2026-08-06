"""Robust ingestion of a single Raman spectrum onto a wide analysis canvas.

This module decouples two grids that the rest of the pipeline conflates:

  * **Canvas grid** -- a wide, fixed Raman-shift axis (default 0..4000 cm-1)
    used for *display* and *post-hoc peak detection*. Regions the instrument
    never measured are zero-filled and tracked by a boolean ``measured_mask``.
  * **Model grid** -- the 1024-pt, ~267..2004 cm-1 axis the 1D-ResNet, the
    pure-reference spectra and ``bond_mapping.json`` were built on. The learned
    heads (composition / reconstruction / OOD) MUST consume this grid; nothing
    outside it carries information the model can use.

Widening the canvas therefore improves ingestion robustness, display fidelity
and out-of-fingerprint peak reporting -- it does NOT give the quantification
model more chemistry. ``to_model_grid`` exists precisely to route any upload
back onto the model grid unchanged, keeping the existing demo intact.

The dataset was acquired on a 785 nm laser (wavelength headers 801.62..931.28
nm map to ~267..2004 cm-1), so ``LASER_NM_DEFAULT`` is 785.0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- Constants ---------------------------------------------------------------

LASER_NM_DEFAULT: float = 785.0
"""Excitation wavelength (nm). Confirmed from the 801.62..931.28 nm headers
mapping onto 267..2004 cm-1."""

CANVAS_LO: float = 0.0
CANVAS_HI: float = 4000.0
CANVAS_STEP: float = 2.0
"""Default canvas: 0..4000 cm-1 at 2 cm-1 spacing (2001 points). 2 cm-1 is
close to the model grid spacing (~1.7 cm-1) and to typical Raman resolution."""

FINGERPRINT_RANGE: tuple[float, float] = (267.0, 2004.0)
"""The model / bond-DB trained domain. Peaks inside it are assignable; peaks
outside it are reported as out-of-domain (see engine/region_labels.py)."""


@dataclass
class IngestResult:
    """Outcome of ingesting one raw spectrum.

    Attributes:
        wn_src: Source axis in cm-1, sorted ascending (after any nm conversion).
        intensity_src: Source intensities aligned to ``wn_src``.
        canvas_wn: Wide canvas axis in cm-1 (ascending, fixed grid).
        canvas_intensity: Intensities on the canvas; ``0.0`` outside the
            measured span.
        measured_mask: Boolean over ``canvas_wn``; ``True`` where the source
            actually covered the axis (used to hide zero-fill on display).
        measured_lo: Lowest measured wavenumber (cm-1).
        measured_hi: Highest measured wavenumber (cm-1).
        unit_detected: ``"nm"`` or ``"cm-1"`` -- the unit inferred for the input.
    """

    wn_src: np.ndarray
    intensity_src: np.ndarray
    canvas_wn: np.ndarray
    canvas_intensity: np.ndarray
    measured_mask: np.ndarray
    measured_lo: float
    measured_hi: float
    unit_detected: str


# --- Unit handling -----------------------------------------------------------

def nm_to_wavenumber(wavelength_nm: np.ndarray,
                     laser_nm: float = LASER_NM_DEFAULT) -> np.ndarray:
    """Convert scattered wavelengths (nm) to Raman shift (cm-1).

    Stokes shift = (1/laser - 1/scattered) * 1e7, with wavelengths in nm.

    Args:
        wavelength_nm: Scattered-light wavelengths in nm.
        laser_nm: Excitation wavelength in nm.

    Returns:
        Raman shift in cm-1 (same shape as input).
    """
    wl = np.asarray(wavelength_nm, dtype=np.float64)
    return (1.0 / laser_nm - 1.0 / wl) * 1.0e7


def detect_axis_unit(axis: np.ndarray) -> str:
    """Heuristically decide whether an axis is in nm or cm-1.

    For a 785 nm laser, Stokes-scattered light sits roughly in 785..1050 nm,
    whereas a Raman-shift axis spans ~0..4000 cm-1. The cases:

      * ``max > 1200``                       -> ``"cm-1"`` (too large for NIR nm)
      * ``600 <= min`` and ``max <= 1100``   -> ``"nm"``   (tight NIR window)
      * otherwise                            -> ``"cm-1"`` (safer default for a
        fingerprint-only axis such as 400..1100)

    This is a heuristic; callers can override via ``axis_unit`` in
    :func:`to_wavenumber` when the unit is known.

    Args:
        axis: 1-D numeric axis.

    Returns:
        ``"nm"`` or ``"cm-1"``.
    """
    a = np.asarray(axis, dtype=np.float64)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if hi > 1200.0:
        return "cm-1"
    if lo >= 600.0 and hi <= 1100.0:
        return "nm"
    return "cm-1"


def to_wavenumber(axis: np.ndarray,
                  axis_unit: str = "auto",
                  laser_nm: float = LASER_NM_DEFAULT) -> tuple[np.ndarray, str]:
    """Return ``(wavenumber_cm-1, unit_detected)`` for an arbitrary axis.

    Args:
        axis: Raw axis values.
        axis_unit: ``"nm"``, ``"cm-1"`` or ``"auto"`` (detect).
        laser_nm: Excitation wavelength used for nm -> cm-1 conversion.

    Returns:
        Tuple of the cm-1 axis and the unit that was used (``"nm"``/``"cm-1"``).
    """
    a = np.asarray(axis, dtype=np.float64)
    unit = detect_axis_unit(a) if axis_unit == "auto" else axis_unit
    if unit == "nm":
        return nm_to_wavenumber(a, laser_nm), "nm"
    return a, "cm-1"


# --- Canvas construction -----------------------------------------------------

def build_canvas(axis: np.ndarray,
                 intensity: np.ndarray,
                 axis_unit: str = "auto",
                 laser_nm: float = LASER_NM_DEFAULT,
                 lo: float = CANVAS_LO,
                 hi: float = CANVAS_HI,
                 step: float = CANVAS_STEP) -> IngestResult:
    """Ingest a raw spectrum onto the wide canvas.

    Steps (mirrors the requested ingestion contract):
      1. Detect / convert the axis to cm-1.
      2. Sort the source ascending and de-duplicate axis ties.
      3. Linear-interpolate intensities onto ``[lo, hi]`` at ``step`` spacing.
      4. Zero-fill canvas points outside the measured span and record a
         ``measured_mask``.

    Args:
        axis: Raw axis (nm or cm-1).
        intensity: Raw intensities, same length as ``axis``.
        axis_unit: ``"nm"``, ``"cm-1"`` or ``"auto"``.
        laser_nm: Excitation wavelength for nm conversion.
        lo: Canvas lower bound (cm-1).
        hi: Canvas upper bound (cm-1).
        step: Canvas spacing (cm-1).

    Returns:
        An :class:`IngestResult`.

    Raises:
        ValueError: if ``axis`` and ``intensity`` lengths differ or are empty.
    """
    axis = np.asarray(axis, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if axis.shape != intensity.shape:
        raise ValueError(
            f"axis {axis.shape} and intensity {intensity.shape} must match")
    if axis.size == 0:
        raise ValueError("empty spectrum")

    wn, unit = to_wavenumber(axis, axis_unit=axis_unit, laser_nm=laser_nm)

    # Sort ascending; average duplicate axis values so np.interp stays monotone.
    order = np.argsort(wn, kind="mergesort")
    wn, intensity = wn[order], intensity[order]
    if np.any(np.diff(wn) == 0.0):
        uniq, inv = np.unique(wn, return_inverse=True)
        summed = np.zeros_like(uniq)
        counts = np.zeros_like(uniq)
        np.add.at(summed, inv, intensity)
        np.add.at(counts, inv, 1.0)
        wn, intensity = uniq, summed / counts

    measured_lo, measured_hi = float(wn[0]), float(wn[-1])

    n = int(round((hi - lo) / step)) + 1
    canvas_wn = lo + step * np.arange(n, dtype=np.float64)
    canvas_intensity = np.interp(canvas_wn, wn, intensity, left=0.0, right=0.0)
    measured_mask = (canvas_wn >= measured_lo) & (canvas_wn <= measured_hi)
    canvas_intensity[~measured_mask] = 0.0

    return IngestResult(
        wn_src=wn,
        intensity_src=intensity,
        canvas_wn=canvas_wn,
        canvas_intensity=canvas_intensity,
        measured_mask=measured_mask,
        measured_lo=measured_lo,
        measured_hi=measured_hi,
        unit_detected=unit,
    )


def measured_slice(result: IngestResult) -> tuple[np.ndarray, np.ndarray]:
    """Return the canvas restricted to the measured region.

    Use this for display ("only show measured data") and for running the
    classical preprocessing / peak detection on real data rather than on the
    zero-filled tails.

    Args:
        result: An :class:`IngestResult`.

    Returns:
        ``(wavenumber, intensity)`` over the measured span only.
    """
    m = result.measured_mask
    return result.canvas_wn[m], result.canvas_intensity[m]


def to_model_grid(result: IngestResult,
                  model_wn: np.ndarray) -> np.ndarray:
    """Project the source spectrum onto the model's wavenumber grid.

    Interpolates from the original (de-duplicated, ascending) source so the
    learned heads see exactly what the legacy resampler produced. Returns an
    array ordered to match ``model_wn`` (ascending or descending), zero outside
    the measured span.

    Args:
        result: An :class:`IngestResult`.
        model_wn: The model grid (e.g. ``data/processed/wavenumbers.npy``),
            ascending or descending.

    Returns:
        Intensities aligned to ``model_wn``.
    """
    model_wn = np.asarray(model_wn, dtype=np.float64)
    flipped = model_wn[0] > model_wn[-1]
    target_asc = model_wn[::-1] if flipped else model_wn
    v = np.interp(target_asc, result.wn_src, result.intensity_src,
                  left=0.0, right=0.0)
    return v[::-1] if flipped else v
