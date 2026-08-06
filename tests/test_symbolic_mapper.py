"""Tests for engine.symbolic_mapper (T21 + base T05 lookup)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from engine.symbolic_mapper import BondMapper, BondEntry, AnnotatedPeak

_DB_PATH = Path(__file__).resolve().parent.parent / "engine" / "bond_mapping.json"


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mapper() -> BondMapper:
    return BondMapper.from_json(_DB_PATH)


@pytest.fixture
def histidine_peaks() -> list[dict]:
    """4 imidazole peaks from extract_full(), idealised."""
    return [
        {"position": 1003.0, "intensity": 1.00, "fwhm": 10.0, "fit_quality": 0.99},
        {"position": 1180.0, "intensity": 0.60, "fwhm": 11.0, "fit_quality": 0.99},
        {"position": 1495.0, "intensity": 0.70, "fwhm": 14.0, "fit_quality": 0.99},
        {"position": 1575.0, "intensity": 0.80, "fwhm": 14.0, "fit_quality": 0.99},
    ]


@pytest.fixture
def glucosamine_peaks() -> list[dict]:
    return [
        {"position": 1080.0, "intensity": 0.9, "fwhm": 10.0, "fit_quality": 0.99},
        {"position": 1100.0, "intensity": 0.7, "fwhm": 10.0, "fit_quality": 0.99},
    ]


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------

class TestConstruction:
    def test_loads_30_entries(self, mapper):
        assert len(mapper) == 30

    def test_canonical_compounds_present(self, mapper):
        canonical = {"Alanine", "Asparagine", "Aspartic Acid",
                     "Glutamic Acid", "Histidine", "Glucosamine"}
        assert set(mapper._known_compounds()) >= canonical

    def test_entries_are_BondEntry(self, mapper):
        for e in mapper.entries:
            assert isinstance(e, BondEntry)
            assert e.id.startswith("P")
            assert e.tolerance_cm_inv > 0
            assert e.wavenumber_cm_inv > 0

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            BondMapper.from_json("/nonexistent/path.json")

    def test_empty_entries_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            BondMapper(entries=[])

    def test_metadata_carried(self, mapper):
        assert "compound_order" in mapper.metadata


# -----------------------------------------------------------------------------
# match_peak (base layer)
# -----------------------------------------------------------------------------

class TestMatchPeak:
    def test_exact_histidine_1003(self, mapper):
        hits = mapper.match_peak(1003.0)
        ids = [h.id for h in hits]
        assert "P004" in ids
        # First hit should be the closest in cm-1
        assert hits[0].wavenumber_cm_inv == 1003

    def test_off_band_returns_empty(self, mapper):
        # 2500 is mid-IR, outside our DB range
        assert mapper.match_peak(2500.0) == []

    def test_tolerance_override(self, mapper):
        # 1003 +- 1.0 should still find P004 even if override is tight
        hits = mapper.match_peak(1003.0, tolerance_override=1.0)
        assert "P004" in [h.id for h in hits]
        # 2200 cm-1 is outside the entire DB range; tight tolerance -> empty
        assert mapper.match_peak(2200.0, tolerance_override=0.5) == []

    def test_sorted_by_distance(self, mapper):
        # 1450 sits between P012 (1450) and P029 (1455)
        hits = mapper.match_peak(1452.0)
        # First hit should be closer to 1452 than the second
        if len(hits) >= 2:
            d0 = abs(hits[0].wavenumber_cm_inv - 1452)
            d1 = abs(hits[1].wavenumber_cm_inv - 1452)
            assert d0 <= d1


# -----------------------------------------------------------------------------
# compute_match_confidence
# -----------------------------------------------------------------------------

class TestMatchConfidence:
    def test_high_confidence_at_center(self, mapper):
        e = next(x for x in mapper.entries if x.id == "P004")  # 1003 +- 8
        assert mapper.compute_match_confidence(1003.0, e) == "high"
        assert mapper.compute_match_confidence(1004.0, e) == "high"  # 1.0 < 4.0

    def test_medium_confidence_near_edge(self, mapper):
        e = next(x for x in mapper.entries if x.id == "P004")  # tol 8
        # |delta|=4.5 is between tol/2 (=4) and tol (=8), so "medium".
        assert mapper.compute_match_confidence(1007.5, e) == "medium"

    def test_low_confidence_outside(self, mapper):
        e = next(x for x in mapper.entries if x.id == "P004")
        # |delta|=9 > tol (=8), so "low".
        assert mapper.compute_match_confidence(1012.0, e) == "low"

    def test_boundary_exact_half_is_medium(self, mapper):
        e = next(x for x in mapper.entries if x.id == "P004")  # tol 8 -> half 4
        # 1003 + 4 = 1007 -> |delta|=4 == tol/2 -> NOT < tol/2, so "medium"
        assert mapper.compute_match_confidence(1007.0, e) == "medium"


# -----------------------------------------------------------------------------
# annotate_peaks
# -----------------------------------------------------------------------------

class TestAnnotatePeaks:
    def test_all_histidine_peaks_match_high(self, mapper, histidine_peaks):
        ann = mapper.annotate_peaks(histidine_peaks)
        assert len(ann) == 4
        for a in ann:
            assert isinstance(a, AnnotatedPeak)
            assert a.matched_to is not None
            assert "Histidine" in a.compounds
            assert a.match_confidence == "high"

    def test_glucosamine_peaks_match(self, mapper, glucosamine_peaks):
        ann = mapper.annotate_peaks(glucosamine_peaks)
        for a in ann:
            assert "Glucosamine" in a.compounds

    def test_unmatched_peak_gets_none(self, mapper):
        peaks = [{"position": 9999.0, "intensity": 0.5, "fwhm": 5.0, "fit_quality": 0.95}]
        ann = mapper.annotate_peaks(peaks)
        assert ann[0].matched_to is None
        assert ann[0].compounds == []
        assert ann[0].match_confidence == "none"
        assert ann[0].delta_cm is None

    def test_prefers_discriminative_when_overlap(self, mapper):
        # 1003 cm-1: P003 (1000, generic Ala/His) AND P004 (1003, Histidine-discriminative)
        # both match. annotate should prefer P004.
        peaks = [{"position": 1003.0, "intensity": 1.0, "fwhm": 8.0, "fit_quality": 0.99}]
        ann = mapper.annotate_peaks(peaks, prefer_discriminative=True)
        assert ann[0].matched_to == "P004"

    def test_disable_discriminative_preference(self, mapper):
        # When prefer_discriminative=False, the nearest entry wins.
        peaks = [{"position": 1003.0, "intensity": 1.0, "fwhm": 8.0, "fit_quality": 0.99}]
        ann = mapper.annotate_peaks(peaks, prefer_discriminative=False)
        # 1003 is closer to P004 (1003) than to P003 (1000), so still P004
        assert ann[0].matched_to == "P004"

    def test_accepts_peak_dataclass(self, mapper):
        from engine.peak_extractor import Peak
        peaks = [Peak(position=1003.0, intensity=1.0, fwhm=8.0,
                      fit_quality=0.99, position_pixel=590)]
        ann = mapper.annotate_peaks(peaks)
        assert ann[0].matched_to == "P004"

    def test_delta_cm_signed(self, mapper):
        peaks = [{"position": 1005.0, "intensity": 1.0, "fwhm": 8.0, "fit_quality": 0.99}]
        ann = mapper.annotate_peaks(peaks)
        assert abs(ann[0].delta_cm - 2.0) < 1e-9  # 1005 - 1003


# -----------------------------------------------------------------------------
# disambiguate_compound
# -----------------------------------------------------------------------------

class TestDisambiguateCompound:
    def test_histidine_unanimous(self, mapper, histidine_peaks):
        ann = mapper.annotate_peaks(histidine_peaks)
        r = mapper.disambiguate_compound(ann)
        assert r["likely_compounds"] == ["Histidine"]
        assert r["votes"]["Histidine"] >= 1.0
        # Other compounds should be in unsupported
        assert "Alanine" in r["unsupported_compounds"]
        assert "Glucosamine" in r["unsupported_compounds"]

    def test_glucosamine_identified(self, mapper, glucosamine_peaks):
        ann = mapper.annotate_peaks(glucosamine_peaks)
        r = mapper.disambiguate_compound(ann)
        assert "Glucosamine" in r["likely_compounds"]

    def test_no_peaks_no_likely(self, mapper):
        r = mapper.disambiguate_compound([])
        assert r["likely_compounds"] == []
        # All known compounds should be unsupported
        canonical = {"Alanine", "Asparagine", "Aspartic Acid",
                     "Glutamic Acid", "Histidine", "Glucosamine"}
        assert canonical <= set(r["unsupported_compounds"])

    def test_medium_confidence_half_weight(self, mapper):
        # Build a single peak at 1009 cm-1: that's |delta|=6 from P004 (tol 8),
        # between tol/2 (=4) and tol (=8), so confidence is medium and vote
        # weight is 0.5.
        peaks = [{"position": 1009.0, "intensity": 1.0, "fwhm": 8.0, "fit_quality": 0.99}]
        ann = mapper.annotate_peaks(peaks)
        assert ann[0].match_confidence == "medium"
        r = mapper.disambiguate_compound(ann)
        # With min_discriminative_hits=1 and only 0.5 vote, Histidine is NOT likely
        assert r["likely_compounds"] == []
        # But with min_discriminative_hits=0.5, it should be
        r2 = mapper.disambiguate_compound(ann, min_discriminative_hits=0.5)
        assert "Histidine" in r2["likely_compounds"]

    def test_mixed_sample_multiple_likely(self, mapper, histidine_peaks, glucosamine_peaks):
        all_peaks = histidine_peaks + glucosamine_peaks
        ann = mapper.annotate_peaks(all_peaks)
        r = mapper.disambiguate_compound(ann)
        assert "Histidine" in r["likely_compounds"]
        assert "Glucosamine" in r["likely_compounds"]


# -----------------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------------

class TestEdgeCases:
    def test_custom_db(self, tmp_path):
        """Mapper can load a hand-rolled DB."""
        data = {
            "metadata": {"compound_order": ["Foo"]},
            "entries": [
                {"id": "X01", "wavenumber_cm_inv": 500.0,
                 "tolerance_cm_inv": 5.0, "bond": "Foo-X stretch",
                 "mode": "stretch", "compounds": ["Foo"],
                 "discriminative_for": ["Foo"], "notes": ""},
            ],
        }
        p = tmp_path / "tiny.json"
        p.write_text(json.dumps(data))
        m = BondMapper.from_json(p)
        assert len(m) == 1
        assert m.match_peak(501.0)[0].id == "X01"