"""Peak extraction with Voigt profile fitting (T20).

Provides :class:`PeakExtractor` -- the deterministic, non-learned post-hoc
analysis step that takes a (preprocessed) Raman spectrum and returns a
list of physically-meaningful peaks. Each peak comes with sub-pixel
refined position (cm-1), amplitude, FWHM, and a goodness-of-fit score so
downstream modules (symbolic mapper, novelty locator) can prune
unreliable detections.

Pipeline
--------
1. ``find_peaks_basic`` -- coarse detection on normalised intensities
   via :func:`scipy.signal.find_peaks`. Operates in pixel-index space
   but returns wavenumber coordinates for downstream consumers.
2. ``fit_voigt`` -- per-peak refinement using :class:`lmfit.models.VoigtModel`
   inside a +/- (fit_window_cm / 2) window. Yields sub-pixel position,
   amplitude, FWHM and R^2 fit quality.
3. ``extract_full`` -- glue of the above two; applies a fit-quality
   threshold (default 0.8) and de-duplicates peaks that landed in the
   same Voigt-fit window.

Conventions
-----------
- ``spectrum``: 1-D numpy array (N,) of intensities. Length must match
  the configured ``wavenumbers`` array.
- ``wavenumbers``: 1-D numpy array (N,) of Raman shifts in cm-1.
  Either monotonically ascending or descending; the extractor canonicalises
  internally and remembers the orientation so pixel indices it returns
  are always in the caller's orientation.
- All public *position* outputs use cm-1 (not pixel index).

Author: Chat 4 Phase A, Task T20.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from scipy.signal import find_peaks, savgol_filter

try:
    from lmfit.models import VoigtModel
    _HAS_LMFIT = True
except ImportError:  # pragma: no cover -- lmfit is in requirements.txt
    _HAS_LMFIT = False


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass
class Peak:
    """One detected peak after Voigt refinement.

    Attributes
    ----------
    position : float
        Refined peak centre, in cm-1 (sub-pixel from Voigt fit).
    intensity : float
        Peak height in the spectrum's units (height, not Voigt amplitude).
    fwhm : float
        Full width at half maximum, in cm-1.
    fit_quality : float
        R^2 of the local Voigt fit in [0, 1]. ``0.5`` if fit failed and
        the fallback Gaussian estimate was used.
    position_pixel : int
        Coarse pixel index from ``find_peaks_basic`` (before refinement).
        In the caller's original wavenumber orientation.
    """

    position: float
    intensity: float
    fwhm: float
    fit_quality: float
    position_pixel: int

    def to_dict(self) -> dict:
        return asdict(self)


# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------

class PeakExtractor:
    """Extract physical peaks from a Raman spectrum.

    Parameters
    ----------
    wavenumbers : np.ndarray
        1-D array of shape (N,) giving the cm-1 value for each pixel of
        the spectra this extractor will be applied to. May be ascending
        or descending; internally the extractor builds a strictly
        ascending working copy for fitting.
    height : float, default 0.05
        Minimum peak height as a fraction of the spectrum's max.
        Passed to :func:`scipy.signal.find_peaks` after normalising the
        spectrum to peak=1.
    prominence : float, default 0.03
        Minimum peak prominence in normalised units.
    distance : int, default 10
        Minimum separation between peaks, in pixel indices.
    fit_window_cm : float, default 30.0
        Total width (cm-1) of the local window used for Voigt fitting.
        The window extends +/- fit_window_cm/2 around the coarse peak.
    fit_quality_threshold : float, default 0.8
        Minimum R^2 to keep a peak in ``extract_full``. Peaks below this
        are discarded as unreliable.
    use_lmfit : bool, default True
        Whether to attempt lmfit Voigt fitting. If False or lmfit is
        unavailable, falls back to a simple Gaussian FWHM estimate.

    Example
    -------
    >>> import numpy as np
    >>> wn = np.linspace(800, 1700, 1024)
    >>> ext = PeakExtractor(wn)
    >>> # Synthetic single peak at 1003 cm-1
    >>> s = np.exp(-0.5 * ((wn - 1003.0) / 5.0) ** 2)
    >>> peaks = ext.extract_full(s)
    >>> len(peaks) == 1 and abs(peaks[0].position - 1003.0) < 0.5
    True
    """

    def __init__(
        self,
        wavenumbers: np.ndarray,
        *,
        height: float = 0.05,
        prominence: float = 0.03,
        distance: int = 10,
        fit_window_cm: float = 30.0,
        fit_quality_threshold: float = 0.8,
        use_lmfit: bool = True,
    ) -> None:
        wn = np.asarray(wavenumbers, dtype=np.float64)
        if wn.ndim != 1:
            raise ValueError(f"wavenumbers must be 1-D, got shape {wn.shape}")
        if wn.size < 8:
            raise ValueError(
                f"wavenumbers array too short ({wn.size}); need at least 8 points"
            )

        # Canonicalise: store ascending copy + remember original orientation
        self._wn_ascending = np.sort(wn)
        self._wn_original = wn.copy()
        self._reversed = bool(wn[0] > wn[-1])

        # Sanity: strictly monotonic in the ascending copy
        diffs = np.diff(self._wn_ascending)
        if (diffs <= 0).any():
            raise ValueError("wavenumbers must be strictly monotonic")

        self.height = float(height)
        self.prominence = float(prominence)
        self.distance = int(distance)
        self.fit_window_cm = float(fit_window_cm)
        self.fit_quality_threshold = float(fit_quality_threshold)
        self.use_lmfit = bool(use_lmfit) and _HAS_LMFIT

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    @property
    def wavenumbers(self) -> np.ndarray:
        """Return the wavenumber axis in its original orientation."""
        return self._wn_original

    @property
    def n_pixels(self) -> int:
        return int(self._wn_ascending.size)

    def find_peaks_basic(
        self,
        spectrum: np.ndarray,
        *,
        smooth: bool = False,
    ) -> list[dict]:
        """Coarse peak detection via :func:`scipy.signal.find_peaks`.

        Parameters
        ----------
        spectrum : np.ndarray
            1-D intensities of shape (N,), aligned with the wavenumber
            axis given at construction.
        smooth : bool, default False
            If True, apply a small Savitzky-Golay smoother before
            detection (window=5, order=2). Helps on noisy spectra.

        Returns
        -------
        list of dict
            Each entry has keys ``position`` (cm-1, peak centre at the
            pixel grid) and ``intensity`` (raw intensity at that pixel),
            plus ``position_pixel`` for traceability (in the caller's
            original orientation).

        Example
        -------
        >>> wn = np.linspace(800, 1700, 1024)
        >>> ext = PeakExtractor(wn)
        >>> s = np.exp(-0.5 * ((wn - 1003.0) / 5.0) ** 2)
        >>> p = ext.find_peaks_basic(s)
        >>> abs(p[0]["position"] - 1003.0) < 1.0
        True
        """
        s = self._prep_spectrum(spectrum)
        if smooth:
            window = 5
            if s.size >= window:
                s = savgol_filter(s, window_length=window, polyorder=2)

        # Work in ascending order for indexing simplicity
        s_asc = s[::-1] if self._reversed else s

        peak_max = float(np.max(np.abs(s_asc)))
        if peak_max <= 0:
            return []

        s_norm = s_asc / peak_max

        peak_idx, _ = find_peaks(
            s_norm,
            height=self.height,
            prominence=self.prominence,
            distance=self.distance,
        )

        out: list[dict] = []
        for idx_asc in peak_idx:
            pos = float(self._wn_ascending[idx_asc])
            # Map ascending-array idx back to the user's array orientation
            idx_orig = (
                self.n_pixels - 1 - int(idx_asc) if self._reversed else int(idx_asc)
            )
            out.append(
                {
                    "position": pos,
                    "intensity": float(s_asc[idx_asc]),
                    "position_pixel": idx_orig,
                }
            )
        return out

    def fit_voigt(
        self,
        spectrum: np.ndarray,
        peak_pos: float,
        *,
        window: Optional[float] = None,
    ) -> dict:
        """Refine one peak with a Voigt fit.

        Parameters
        ----------
        spectrum : np.ndarray
            1-D intensities (N,) aligned with this extractor's wavenumbers.
        peak_pos : float
            Initial peak centre in cm-1 (typically from
            :meth:`find_peaks_basic`).
        window : float, optional
            Total fit-window width in cm-1. Defaults to
            ``self.fit_window_cm``.

        Returns
        -------
        dict
            Keys: ``position`` (refined centre, cm-1), ``intensity``
            (Voigt peak height in spectrum's units), ``fwhm`` (cm-1),
            ``fit_quality`` (R^2 in [0, 1]; 0.5 on fit fallback).

        Notes
        -----
        If lmfit is unavailable or the fit raises, falls back to a
        local Gaussian moment estimate (peak = local max within window,
        FWHM = full width at half max via linear interpolation).
        """
        s = self._prep_spectrum(spectrum)
        s_asc = s[::-1] if self._reversed else s
        wn = self._wn_ascending

        w = float(window) if window is not None else self.fit_window_cm
        half = w / 2.0

        mask = (wn >= peak_pos - half) & (wn <= peak_pos + half)
        if mask.sum() < 5:
            # Window too narrow for a fit; emit a low-quality fallback
            return self._fallback_estimate(s_asc, wn, peak_pos)

        x = wn[mask]
        y = s_asc[mask]

        if not self.use_lmfit:
            return self._fallback_estimate(s_asc, wn, peak_pos)

        try:
            model = VoigtModel()
            i0 = int(np.argmax(y))
            cen0 = float(x[i0])
            params = model.guess(y, x=x)
            # Re-anchor center to the local max; let amplitude/sigma/gamma vary
            params["center"].set(
                value=cen0,
                min=cen0 - half / 2.0,
                max=cen0 + half / 2.0,
            )
            params["amplitude"].set(min=0.0)
            params["sigma"].set(min=1e-3, max=w)
            params["gamma"].set(
                value=params["sigma"].value, vary=True, min=1e-3, max=w
            )

            result = model.fit(y, params, x=x)

            ss_res = float(np.sum((y - result.best_fit) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            r2 = float(np.clip(r2, 0.0, 1.0))

            center = float(result.params["center"].value)
            fwhm = float(result.params["fwhm"].value)
            # lmfit's VoigtModel exposes 'height' as a derived param =
            # actual peak height (not the integrated amplitude)
            if "height" in result.params:
                height_val = float(result.params["height"].value)
            else:  # pragma: no cover -- older lmfit
                height_val = float(np.max(result.best_fit))

            return {
                "position": center,
                "intensity": height_val,
                "fwhm": fwhm,
                "fit_quality": r2,
            }
        except Exception:
            return self._fallback_estimate(s_asc, wn, peak_pos)

    def extract_full(
        self,
        spectrum: np.ndarray,
        *,
        smooth: bool = False,
        merge_tolerance_cm: float = 4.0,
    ) -> list[Peak]:
        """Run the full extraction pipeline.

        Parameters
        ----------
        spectrum : np.ndarray
            1-D intensities of shape (N,).
        smooth : bool, default False
            Whether to apply Savitzky-Golay smoothing during coarse
            detection only (the fit operates on the raw spectrum).
        merge_tolerance_cm : float, default 4.0
            If two refined peaks land within this distance in cm-1,
            keep the one with higher fit quality. Defeats double-counting
            from overlapping fit windows.

        Returns
        -------
        list[Peak]
            Voigt-refined peaks with ``fit_quality >=
            self.fit_quality_threshold``, sorted by position ascending.

        Example
        -------
        >>> wn = np.linspace(900, 1100, 256)
        >>> ext = PeakExtractor(wn, fit_quality_threshold=0.5)
        >>> s = np.exp(-0.5 * ((wn - 1003.0) / 4.0) ** 2)
        >>> peaks = ext.extract_full(s)
        >>> len(peaks) == 1 and abs(peaks[0].position - 1003.0) < 0.5
        True
        """
        coarse = self.find_peaks_basic(spectrum, smooth=smooth)
        refined: list[Peak] = []
        for c in coarse:
            fit = self.fit_voigt(spectrum, c["position"])
            refined.append(
                Peak(
                    position=fit["position"],
                    intensity=fit["intensity"],
                    fwhm=fit["fwhm"],
                    fit_quality=fit["fit_quality"],
                    position_pixel=c["position_pixel"],
                )
            )

        # Filter by fit quality
        kept = [p for p in refined if p.fit_quality >= self.fit_quality_threshold]

        # Merge near-duplicates (overlapping windows can converge on the same line)
        kept.sort(key=lambda p: p.position)
        deduped: list[Peak] = []
        for p in kept:
            if deduped and abs(p.position - deduped[-1].position) < merge_tolerance_cm:
                if p.fit_quality > deduped[-1].fit_quality:
                    deduped[-1] = p
            else:
                deduped.append(p)

        return deduped

    # -------------------------------------------------------------------------
    # Index <-> wavenumber helpers (public; useful to callers)
    # -------------------------------------------------------------------------

    def pixel_to_cm(self, pixel_idx: int) -> float:
        """Convert a pixel index (in user's wavenumber orientation) to cm-1."""
        return float(self._wn_original[int(pixel_idx)])

    def cm_to_pixel(self, cm: float) -> int:
        """Convert a cm-1 value to the nearest pixel index (in user's orientation)."""
        idx_asc = int(np.argmin(np.abs(self._wn_ascending - float(cm))))
        return self.n_pixels - 1 - idx_asc if self._reversed else idx_asc

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _prep_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        s = np.asarray(spectrum, dtype=np.float64)
        if s.ndim != 1:
            raise ValueError(f"spectrum must be 1-D, got shape {s.shape}")
        if s.size != self.n_pixels:
            raise ValueError(
                f"spectrum length {s.size} does not match wavenumbers length "
                f"{self.n_pixels}"
            )
        return s

    @staticmethod
    def _fallback_estimate(
        s_asc: np.ndarray, wn: np.ndarray, peak_pos: float
    ) -> dict:
        """Cheap Gaussian-like FWHM estimate when Voigt fit is unavailable."""
        idx = int(np.argmin(np.abs(wn - peak_pos)))
        # Walk left/right until half-max
        half = float(s_asc[idx]) / 2.0
        left = idx
        while left > 0 and s_asc[left] > half:
            left -= 1
        right = idx
        while right < s_asc.size - 1 and s_asc[right] > half:
            right += 1
        fwhm = float(wn[right] - wn[left])
        return {
            "position": float(wn[idx]),
            "intensity": float(s_asc[idx]),
            "fwhm": max(fwhm, 0.5),
            "fit_quality": 0.5,  # uncertain; below default 0.8 threshold
        }
