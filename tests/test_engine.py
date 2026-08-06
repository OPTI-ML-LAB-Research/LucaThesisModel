"""Tests for engine.symbolic_mapper.BondMapper.

Run with:
    pytest tests/test_engine.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_JSON = PROJECT_ROOT / "engine" / "bond_mapping.json"


# ───────────────────────────────────────────────────────────────────────────
#  Fixtures
# ───────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mapper():
    if not DB_JSON.is_file():
        pytest.skip(f"DB not found: {DB_JSON}")
    from engine.symbolic_mapper import BondMapper
    return BondMapper.from_json(DB_JSON)


# ───────────────────────────────────────────────────────────────────────────
#  Construction & basic API
# ───────────────────────────────────────────────────────────────────────────

class TestConstruction:
    def test_load_from_json(self, mapper):
        assert len(mapper) >= 30

    def test_compound_canonical_names_loaded(self, mapper):
        expected = {"Alanine", "Asparagine", "Aspartic Acid",
                    "Glutamic Acid", "Histidine", "Glucosamine"}
        assert set(mapper.compound_canonical_names) == expected

    def test_repr_is_informative(self, mapper):
        r = repr(mapper)
        assert "BondMapper" in r
        assert "n_entries" in r

    def test_load_missing_file_raises(self):
        from engine.symbolic_mapper import BondMapper
        with pytest.raises(FileNotFoundError):
            BondMapper.from_json("/nonexistent/path/bond_mapping.json")


# ───────────────────────────────────────────────────────────────────────────
#  Match peak (the core API)
# ───────────────────────────────────────────────────────────────────────────

class TestMatchPeak:
    def test_match_peak_1003_returns_P004(self, mapper):
        """The signature Histidine peak per project spec."""
        hits = mapper.match_peak(1003)
        assert any(h.id == "P004" for h in hits)
        assert any("Histidine" in h.compounds for h in hits)

    def test_match_peak_within_tolerance(self, mapper):
        """Peak at 1004 is still within ±6 of P004 (1003)."""
        hits = mapper.match_peak(1004)
        assert any(h.id == "P004" for h in hits)

    def test_match_peak_outside_tolerance(self, mapper):
        """Peak at 1015 (12 cm⁻¹ off) is outside the ±6 tolerance of P004."""
        hits = mapper.match_peak(1015)
        assert not any(h.id == "P004" for h in hits)

    def test_match_peak_with_override_tolerance(self, mapper):
        """Override loosens / tightens matching."""
        hits_tight = mapper.match_peak(1015, tolerance=5)
        hits_loose = mapper.match_peak(1015, tolerance=20)
        assert len(hits_loose) >= len(hits_tight)

    def test_match_peak_far_returns_empty(self, mapper):
        assert mapper.match_peak(9999) == []

    def test_match_peak_sorted_by_distance(self, mapper):
        hits = mapper.match_peak(1003, tolerance=50)
        if len(hits) > 1:
            distances = [abs(h.wavenumber_cm_inv - 1003) for h in hits]
            assert distances == sorted(distances)

    def test_match_peak_intensity_kwarg_does_not_break(self, mapper):
        hits = mapper.match_peak(1003, intensity=0.85)
        assert any(h.id == "P004" for h in hits)

    def test_match_peaks_vectorised(self, mapper):
        results = mapper.match_peaks([1003, 1080, 9999])
        assert len(results) == 3
        assert any(h.id == "P004" for h in results[0])
        assert any(h.id == "P005" for h in results[1])
        assert results[2] == []


# ───────────────────────────────────────────────────────────────────────────
#  Compound fingerprints
# ───────────────────────────────────────────────────────────────────────────

class TestCompoundFingerprint:
    @pytest.mark.parametrize("required", [1003, 1180, 1495, 1575])
    def test_histidine_required_peaks_present(self, mapper, required):
        """Project spec: Histidine has 4 discriminative imidazole peaks."""
        fp = mapper.get_compound_fingerprint("Histidine")
        wns = [e["wavenumber_cm_inv"] for e in fp["discriminative"]]
        assert required in wns

    @pytest.mark.parametrize("required", [1080, 1100])
    def test_glucosamine_required_peaks_present(self, mapper, required):
        """Project spec: GlcN has 2 pyranose-ring peaks."""
        fp = mapper.get_compound_fingerprint("Glucosamine")
        wns = [e["wavenumber_cm_inv"] for e in fp["discriminative"]]
        assert required in wns

    def test_unknown_compound_returns_empty(self, mapper):
        """Asking for a compound not in the DB returns empty (with a warning)."""
        fp = mapper.get_compound_fingerprint("Tryptophan")
        assert fp["n_total"] == 0
        assert fp["discriminative"] == []
        assert fp["supporting"] == []

    def test_all_six_compounds_have_at_least_one_entry(self, mapper):
        for c in ["Alanine", "Asparagine", "Aspartic Acid",
                  "Glutamic Acid", "Histidine", "Glucosamine"]:
            fp = mapper.get_compound_fingerprint(c)
            assert fp["n_total"] >= 1, f"No entries for {c}"


# ───────────────────────────────────────────────────────────────────────────
#  Validation
# ───────────────────────────────────────────────────────────────────────────

class TestValidate:
    def test_seed_db_validates_clean(self, mapper):
        report = mapper.validate_db()
        assert report["ok"], f"Seed DB has problems: {report['problems']}"
        assert report["n_problems"] == 0

    def test_validate_raises_on_bad_db(self):
        from engine.symbolic_mapper import BondMapper
        bad = BondMapper.from_dict({
            "entries": [
                {"id": "BAD", "wavenumber_cm_inv": -1,
                 "tolerance_cm_inv": 5, "bond": "x", "mode": "y", "compounds": []},
            ],
        })
        with pytest.raises(ValueError):
            bad.validate_db(raise_on_error=True)

    def test_validate_detects_duplicate_ids(self):
        from engine.symbolic_mapper import BondMapper
        dup = BondMapper.from_dict({
            "entries": [
                {"id": "P001", "wavenumber_cm_inv": 100, "tolerance_cm_inv": 5,
                 "bond": "x", "mode": "y", "compounds": []},
                {"id": "P001", "wavenumber_cm_inv": 200, "tolerance_cm_inv": 5,
                 "bond": "z", "mode": "y", "compounds": []},
            ],
        })
        report = dup.validate_db()
        assert not report["ok"]
        assert any("Duplicate" in p for p in report["problems"])

    def test_validate_detects_discriminative_not_in_compounds(self):
        from engine.symbolic_mapper import BondMapper
        bad = BondMapper.from_dict({
            "entries": [
                {"id": "P001", "wavenumber_cm_inv": 100, "tolerance_cm_inv": 5,
                 "bond": "x", "mode": "y",
                 "compounds": ["Histidine"],
                 "discriminative_for": ["Alanine"]},
            ],
        })
        report = bad.validate_db()
        assert not report["ok"]


# ───────────────────────────────────────────────────────────────────────────
#  Lookup by ID
# ───────────────────────────────────────────────────────────────────────────

class TestLookupById:
    def test_lookup_existing(self, mapper):
        e = mapper.lookup_by_id("P004")
        assert e.bond == "Imidazole ring breathing"
        assert e.wavenumber_cm_inv == 1003

    def test_lookup_missing_raises(self, mapper):
        with pytest.raises(KeyError):
            mapper.lookup_by_id("P999")
