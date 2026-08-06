"""Tests for engine.peak_extractor (T20)."""
from __future__ import annotations

import numpy as np
import pytest

from engine.peak_extractor import PeakExtractor, Peak


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wn_asc() -> np.ndarray:
    """1024-pixel ascending wavenumber axis spanning the AA dataset range."""
    return np.linspace(267.0, 2004.0, 1024)


@pytest.fixture(scope="module")
def wn_desc(wn_asc) -> np.ndarray:
    return wn_asc[::-1].copy()


@pytest.fixture(scope="module")
def histidine_spectrum_asc(wn_asc) -> np.ndarray:
    """Synthetic Histidine: 4 imidazole Gaussian peaks + tiny noise."""
    his_peaks = [1003.0, 1180.0, 1495.0, 1575.0]
    his_intens = [1.0, 0.6, 0.7, 0.8]
    his_sigma = [4.0, 5.0, 6.0, 6.0]
    s = np.zeros_like(wn_asc)
    for p, i, sg in zip(his_peaks, his_intens, his_sigma):
        s += i * np.exp(-0.5 * ((wn_asc - p) / sg) ** 2)
    rng = np.random.default_rng(42)
    s += rng.normal(scale=0.005, size=s.shape)
    return s


@pytest.fixture
def extractor(wn_asc) -> PeakExtractor:
    return PeakExtractor(wn_asc)


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------

class TestConstruction:
    def test_accepts_ascending(self, wn_asc):
        ext = PeakExtractor(wn_asc)
        assert ext.n_pixels == 1024
        assert ext._reversed is False

    def test_accepts_descending(self, wn_desc):
        ext = PeakExtractor(wn_desc)
        assert ext.n_pixels == 1024
        assert ext._reversed is True

    def test_rejects_2d(self):
        with pytest.raises(ValueError, match="1-D"):
            PeakExtractor(np.zeros((10, 5)))

    def test_rejects_short(self):
        with pytest.raises(ValueError, match="too short"):
            PeakExtractor(np.array([100.0, 110.0]))

    def test_rejects_non_monotonic(self):
        wn = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        with pytest.raises(ValueError, match="monotonic"):
            PeakExtractor(wn)

    def test_pixel_to_cm_roundtrip_asc(self, wn_asc):
        ext = PeakExtractor(wn_asc)
        for idx in (0, 100, 500, 1023):
            cm = ext.pixel_to_cm(idx)
            idx2 = ext.cm_to_pixel(cm)
            assert idx2 == idx

    def test_pixel_to_cm_roundtrip_desc(self, wn_desc):
        ext = PeakExtractor(wn_desc)
        for idx in (0, 100, 500, 1023):
            cm = ext.pixel_to_cm(idx)
            idx2 = ext.cm_to_pixel(cm)
            assert idx2 == idx


# -----------------------------------------------------------------------------
# find_peaks_basic
# -----------------------------------------------------------------------------

class TestFindPeaksBasic:
    def test_finds_single_clean_peak(self, extractor, wn_asc):
        s = np.exp(-0.5 * ((wn_asc - 1003.0) / 5.0) ** 2)
        peaks = extractor.find_peaks_basic(s)
        assert len(peaks) == 1
        assert abs(peaks[0]["position"] - 1003.0) < 2.0

    def test_finds_four_histidine_peaks(self, extractor, histidine_spectrum_asc):
        peaks = extractor.find_peaks_basic(histidine_spectrum_asc)
        assert len(peaks) == 4
        positions = sorted(p["position"] for p in peaks)
        targets = [1003.0, 1180.0, 1495.0, 1575.0]
        for found, tgt in zip(positions, targets):
            assert abs(found - tgt) < 2.0

    def test_empty_spectrum_returns_empty(self, extractor, wn_asc):
        s = np.zeros_like(wn_asc)
        peaks = extractor.find_peaks_basic(s)
        assert peaks == []

    def test_low_threshold_rejects_noise(self, wn_asc):
        # Build a single tall real peak plus low-amplitude noise. The
        # find_peaks `height` is RELATIVE to the spectrum max, so noise
        # that is small compared to the main peak should be rejected.
        rng = np.random.default_rng(0)
        noise = rng.normal(scale=0.01, size=wn_asc.shape)
        true_peak = 5.0 * np.exp(-0.5 * ((wn_asc - 1003.0) / 5.0) ** 2)
        s = noise + true_peak
        ext = PeakExtractor(wn_asc, height=0.5, prominence=0.5)
        peaks = ext.find_peaks_basic(s)
        # Only the true peak should survive a 50%-of-max threshold
        assert len(peaks) == 1
        assert abs(peaks[0]["position"] - 1003.0) < 2.0

    def test_descending_axis_returns_same_positions(self, wn_asc, wn_desc):
        # Build spectrum on ascending axis, then flip to descending
        s_asc = np.exp(-0.5 * ((wn_asc - 1003.0) / 5.0) ** 2)
        s_desc = s_asc[::-1].copy()

        peaks_asc = PeakExtractor(wn_asc).find_peaks_basic(s_asc)
        peaks_desc = PeakExtractor(wn_desc).find_peaks_basic(s_desc)

        assert len(peaks_asc) == len(peaks_desc) == 1
        assert abs(peaks_asc[0]["position"] - peaks_desc[0]["position"]) < 1.0

    def test_rejects_wrong_length(self, extractor):
        with pytest.raises(ValueError, match="length"):
            extractor.find_peaks_basic(np.zeros(500))


# -----------------------------------------------------------------------------
# fit_voigt
# -----------------------------------------------------------------------------

class TestFitVoigt:
    def test_refines_clean_peak_subpixel(self, extractor, wn_asc):
        # Target on a grid pixel: the linspace step is ~1.7 cm-1
        target = 1003.5
        s = np.exp(-0.5 * ((wn_asc - target) / 4.0) ** 2)
        fit = extractor.fit_voigt(s, peak_pos=target)
        assert abs(fit["position"] - target) < 0.5
        assert 5 <= fit["fwhm"] <= 20  # reasonable for sigma=4
        assert fit["fit_quality"] > 0.95

    def test_fit_quality_high_for_gaussian(self, extractor, wn_asc):
        s = np.exp(-0.5 * ((wn_asc - 1500.0) / 5.0) ** 2)
        fit = extractor.fit_voigt(s, peak_pos=1500.0)
        assert fit["fit_quality"] > 0.95

    def test_fallback_on_zero_window(self, wn_asc):
        # Force tiny window so the fit can't seat
        ext = PeakExtractor(wn_asc, fit_window_cm=0.5)
        s = np.exp(-0.5 * ((wn_asc - 1003.0) / 5.0) ** 2)
        fit = ext.fit_voigt(s, peak_pos=1003.0)
        # Fallback returns fit_quality = 0.5
        assert fit["fit_quality"] == 0.5

    def test_fwhm_positive(self, extractor, wn_asc):
        s = np.exp(-0.5 * ((wn_asc - 1100.0) / 8.0) ** 2)
        fit = extractor.fit_voigt(s, peak_pos=1100.0)
        assert fit["fwhm"] > 0.0


# -----------------------------------------------------------------------------
# extract_full
# -----------------------------------------------------------------------------

class TestExtractFull:
    def test_four_histidine_peaks_recovered(self, extractor, histidine_spectrum_asc):
        peaks = extractor.extract_full(histidine_spectrum_asc)
        assert len(peaks) == 4

        positions = sorted(p.position for p in peaks)
        targets = [1003.0, 1180.0, 1495.0, 1575.0]
        for p_obs, p_tgt in zip(positions, targets):
            assert abs(p_obs - p_tgt) < 1.5, (
                f"Histidine peak {p_tgt} not recovered (got {p_obs})"
            )

    def test_fwhm_in_physical_range(self, extractor, histidine_spectrum_asc):
        peaks = extractor.extract_full(histidine_spectrum_asc)
        for p in peaks:
            assert 2.0 <= p.fwhm <= 30.0, f"FWHM {p.fwhm} outside expected 2-30 cm-1"

    def test_fit_quality_threshold_drops_garbage(self, wn_asc):
        # Noise-only spectrum produces no high-quality peaks
        rng = np.random.default_rng(123)
        s = rng.normal(scale=0.05, size=wn_asc.shape)
        ext = PeakExtractor(wn_asc, fit_quality_threshold=0.95, height=0.5)
        peaks = ext.extract_full(s)
        assert len(peaks) == 0

    def test_returns_peak_dataclass(self, extractor, histidine_spectrum_asc):
        peaks = extractor.extract_full(histidine_spectrum_asc)
        for p in peaks:
            assert isinstance(p, Peak)
            d = p.to_dict()
            assert set(d) == {"position", "intensity", "fwhm",
                              "fit_quality", "position_pixel"}

    def test_descending_orientation_equivalent(self, wn_asc, wn_desc):
        s_asc = np.zeros_like(wn_asc)
        for p in (1003.0, 1180.0, 1495.0, 1575.0):
            s_asc += np.exp(-0.5 * ((wn_asc - p) / 5.0) ** 2)
        s_desc = s_asc[::-1].copy()

        peaks_asc = PeakExtractor(wn_asc).extract_full(s_asc)
        peaks_desc = PeakExtractor(wn_desc).extract_full(s_desc)

        pos_a = sorted(p.position for p in peaks_asc)
        pos_d = sorted(p.position for p in peaks_desc)
        assert len(pos_a) == len(pos_d)
        for a, d in zip(pos_a, pos_d):
            assert abs(a - d) < 0.5

    def test_merge_duplicates(self, wn_asc):
        # Force coarse over-detection by setting distance=1 (no minimum
        # separation between peaks). The post-filter then merges any
        # near-duplicates within merge_tolerance_cm.
        ext = PeakExtractor(wn_asc, distance=1)
        s = np.exp(-0.5 * ((wn_asc - 1003.0) / 5.0) ** 2)
        peaks = ext.extract_full(s, merge_tolerance_cm=4.0)
        # Even if find_peaks coarse pass over-detects, post-filter merges
        assert len(peaks) == 1
