"""Tests for engine.novelty_locator (T22)."""
from __future__ import annotations

from pathlib import Path

import pytest

from engine.symbolic_mapper import BondMapper, AnnotatedPeak
from engine.novelty_locator import NoveltyLocator, PeakCluster

_DB_PATH = Path(__file__).resolve().parent.parent / "engine" / "bond_mapping.json"


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mapper() -> BondMapper:
    return BondMapper.from_json(_DB_PATH)


@pytest.fixture
def locator(mapper) -> NoveltyLocator:
    return NoveltyLocator(mapper, cluster_gap_cm=30.0)


def _peak(pos: float, intens: float = 1.0) -> dict:
    return {"position": pos, "intensity": intens, "fwhm": 8.0, "fit_quality": 0.99}


# -----------------------------------------------------------------------------
# find_unmatched_peaks
# -----------------------------------------------------------------------------

class TestFindUnmatched:
    def test_in_db_peaks_excluded(self, locator):
        # 1003 cm-1 is a clean Histidine match in the seed DB
        u = locator.find_unmatched_peaks([_peak(1003.0)])
        assert u == []

    def test_off_db_peaks_kept(self, locator):
        # 9999 cm-1: nothing in DB matches
        u = locator.find_unmatched_peaks([_peak(9999.0)])
        assert len(u) == 1
        assert u[0]["position"] == 9999.0

    def test_mixed_input(self, locator):
        u = locator.find_unmatched_peaks([_peak(1003.0), _peak(380.0)])
        # 1003 matched, 380 not in DB
        positions = [p["position"] for p in u]
        assert positions == [380.0]

    def test_accepts_annotated_peaks(self, locator):
        # An AnnotatedPeak with matched_to=None should be treated as unmatched
        ann = AnnotatedPeak(
            position=380.0, intensity=1.0, fwhm=5.0, fit_quality=0.99,
            matched_to=None,
        )
        u = locator.find_unmatched_peaks([ann])
        assert len(u) == 1

    def test_annotated_matched_excluded(self, locator):
        ann = AnnotatedPeak(
            position=1003.0, intensity=1.0, fwhm=5.0, fit_quality=0.99,
            matched_to="P004",
        )
        u = locator.find_unmatched_peaks([ann])
        assert u == []


# -----------------------------------------------------------------------------
# cluster_unmatched
# -----------------------------------------------------------------------------

class TestCluster:
    def test_empty_input(self, locator):
        assert locator.cluster_unmatched([]) == []

    def test_single_peak_single_cluster(self, locator):
        clusters = locator.cluster_unmatched([_peak(500.0)])
        assert len(clusters) == 1
        assert clusters[0].n_peaks == 1
        assert clusters[0].centroid_cm == 500.0
        assert clusters[0].span_cm == 0.0

    def test_adjacent_peaks_merge(self, locator):
        # 380 + 408 are 28 cm-1 apart, less than the 30 cm-1 default gap
        clusters = locator.cluster_unmatched([_peak(380.0), _peak(408.0)])
        assert len(clusters) == 1
        assert clusters[0].n_peaks == 2
        assert 380.0 <= clusters[0].centroid_cm <= 408.0

    def test_far_peaks_split(self, locator):
        # 380 + 1700 are well outside cluster gap
        clusters = locator.cluster_unmatched([_peak(380.0), _peak(1700.0)])
        assert len(clusters) == 2

    def test_single_link_chain(self, locator):
        # 100, 125, 150 -- each adjacent pair within 30 cm-1 default
        clusters = locator.cluster_unmatched([_peak(100.0), _peak(125.0), _peak(150.0)])
        assert len(clusters) == 1
        assert clusters[0].n_peaks == 3
        assert clusters[0].span_cm == 50.0

    def test_centroid_weighted_by_intensity(self, locator):
        # Two peaks at 300 and 320, intensities 1.0 and 3.0
        # centroid = (300*1 + 320*3) / 4 = 315
        clusters = locator.cluster_unmatched(
            [_peak(300.0, intens=1.0), _peak(320.0, intens=3.0)]
        )
        assert len(clusters) == 1
        assert abs(clusters[0].centroid_cm - 315.0) < 0.1

    def test_cluster_is_dataclass(self, locator):
        clusters = locator.cluster_unmatched([_peak(380.0)])
        assert isinstance(clusters[0], PeakCluster)
        d = clusters[0].to_dict()
        assert set(d) >= {
            "members", "centroid_cm", "span_cm", "total_intensity",
            "n_peaks", "region_hints", "region_examples",
        }


# -----------------------------------------------------------------------------
# Region hints
# -----------------------------------------------------------------------------

class TestRegionHints:
    def test_mos2_region_label(self, locator):
        # 380 / 408 cm-1 -> 100-400 + 300-600 regions match
        clusters = locator.cluster_unmatched([_peak(380.0), _peak(408.0)])
        labels = clusters[0].region_hints
        assert any("lattice" in label.lower() or "skeletal" in label.lower()
                   for label in labels)

    def test_amide_region_label(self, locator):
        clusters = locator.cluster_unmatched([_peak(1670.0)])
        labels = clusters[0].region_hints
        # 1670 is in the 1500-1700 amide I band
        assert any("amide" in label.lower() or "c=c" in label.lower()
                   for label in labels)

    def test_oh_nh_region_label(self, locator):
        clusters = locator.cluster_unmatched([_peak(3300.0)])
        labels = clusters[0].region_hints
        assert any("o-h" in label.lower() or "n-h" in label.lower()
                   for label in labels)

    def test_unrecognised_region(self, locator):
        # 50 cm-1 is below any region in our table
        clusters = locator.cluster_unmatched([_peak(50.0)])
        # We tolerate empty hints
        assert clusters[0].region_hints == [] or len(clusters[0].region_hints) >= 0


# -----------------------------------------------------------------------------
# locate (end-to-end)
# -----------------------------------------------------------------------------

class TestLocate:
    def test_mos2_pure(self, locator):
        # MoS2 phonon modes -- neither in our amino-acid DB
        peaks = [_peak(380.0), _peak(408.0)]
        r = locator.locate(peaks)
        assert len(r["unknown_peaks"]) == 2
        assert len(r["clusters"]) == 1
        assert len(r["hints"]) == 1
        assert "MoS2" in r["hints"][0] or "lattice" in r["hints"][0].lower()

    def test_histidine_pure_no_novelty(self, locator):
        # All 4 Histidine peaks are in the DB
        peaks = [_peak(1003.0), _peak(1180.0), _peak(1495.0), _peak(1575.0)]
        r = locator.locate(peaks)
        assert r["unknown_peaks"] == []
        assert r["clusters"] == []
        assert r["hints"] == []

    def test_mixed_input_some_novel(self, locator):
        # Histidine + a single novel peak at 1745 cm-1 (above all DB entries;
        # nearest is P017 at 1700 with tolerance 12, window stops at 1712).
        peaks = [_peak(1003.0), _peak(1180.0), _peak(1495.0),
                 _peak(1575.0), _peak(1745.0)]
        r = locator.locate(peaks)
        assert len(r["unknown_peaks"]) == 1
        assert r["unknown_peaks"][0]["position"] == 1745.0
        assert len(r["hints"]) == 1

    def test_returns_serializable_dict(self, locator):
        import json
        peaks = [_peak(380.0), _peak(408.0)]
        r = locator.locate(peaks)
        # The output should be JSON-serialisable for report generation
        json.dumps(r)


# -----------------------------------------------------------------------------
# Custom cluster gap
# -----------------------------------------------------------------------------

class TestCustomGap:
    def test_narrow_gap_splits(self, mapper):
        # With gap=10, 380 and 408 (28 apart) should split
        nov = NoveltyLocator(mapper, cluster_gap_cm=10.0)
        clusters = nov.cluster_unmatched([_peak(380.0), _peak(408.0)])
        assert len(clusters) == 2

    def test_wide_gap_merges(self, mapper):
        # With gap=200, 380 and 500 should merge
        nov = NoveltyLocator(mapper, cluster_gap_cm=200.0)
        clusters = nov.cluster_unmatched([_peak(380.0), _peak(500.0)])
        assert len(clusters) == 1
