"""Symbolic peak -> bond mapping (T05 base + T21 enhancements).

This module provides the deterministic, knowledge-based layer that
translates extracted peaks into chemical interpretation. It does NOT
involve any learning.

Two layers:

1. **Base layer (originally T05 / Groundwork)** -- :class:`BondMapper`
   loads the seed DB (``engine/bond_mapping.json``) and supports
   single-peak lookup via :meth:`BondMapper.match_peak`.

2. **Enhancement layer (T21)**:

   * :meth:`BondMapper.annotate_peaks` -- batch-enrich a peak list from
     T20 with bond / compound / confidence info.
   * :meth:`BondMapper.compute_match_confidence` -- categorical
     {high, medium, low} based on |delta wavenumber| relative to the
     entry's tolerance.
   * :meth:`BondMapper.disambiguate_compound` -- given a list of
     annotated peaks, decide which compound(s) are most likely present
     by counting discriminative-peak matches.

Author: Chat 1 (T05 base) + Chat 4 Phase A (T21 enhancements).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np


# -----------------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------------

@dataclass
class BondEntry:
    """Single bond-mapping database record.

    Mirrors the JSON schema in ``engine/bond_mapping.json``.
    """

    id: str
    wavenumber_cm_inv: float
    tolerance_cm_inv: float
    bond: str
    mode: str
    compounds: list[str] = field(default_factory=list)
    discriminative_for: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnnotatedPeak:
    """Peak from T20 enriched with bond / compound interpretation (T21).

    Attributes
    ----------
    position : float
        Peak centre, cm-1 (from PeakExtractor's Voigt fit).
    intensity : float
        Peak height in spectrum units.
    fwhm : float
        FWHM in cm-1.
    fit_quality : float
        R^2 of the Voigt fit (passthrough from T20).
    matched_to : Optional[str]
        Bond-mapping entry id (e.g. ``P004``), or None if no match.
    bond : Optional[str]
        Human-readable bond/mode description.
    compounds : list[str]
        Compounds this entry is associated with.
    discriminative_for : list[str]
        Compounds this entry is *specifically* discriminative for.
    match_confidence : str
        Categorical: ``"high"`` | ``"medium"`` | ``"low"`` | ``"none"``.
    delta_cm : Optional[float]
        Signed offset (observed - DB) in cm-1, or None if no match.
    """

    position: float
    intensity: float
    fwhm: float
    fit_quality: float
    matched_to: Optional[str] = None
    bond: Optional[str] = None
    compounds: list[str] = field(default_factory=list)
    discriminative_for: list[str] = field(default_factory=list)
    match_confidence: str = "none"
    delta_cm: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# -----------------------------------------------------------------------------
# Mapper
# -----------------------------------------------------------------------------

class BondMapper:
    """Deterministic peak -> bond DB lookup with confidence and compound voting.

    Parameters
    ----------
    entries : list[BondEntry]
        Records to load into the mapper.
    metadata : dict, optional
        Free-form metadata block from the JSON file (sources, etc).

    Example
    -------
    >>> mapper = BondMapper.from_json("engine/bond_mapping.json")
    >>> hits = mapper.match_peak(1003.0)
    >>> hits[0].id
    'P004'
    >>> annotated = mapper.annotate_peaks([
    ...     {"position": 1003.0, "intensity": 0.9, "fwhm": 10.0, "fit_quality": 0.99}
    ... ])
    >>> annotated[0].compounds
    ['Histidine']
    >>> mapper.disambiguate_compound(annotated)["likely_compounds"]
    ['Histidine']
    """

    def __init__(
        self,
        entries: list[BondEntry],
        metadata: Optional[dict] = None,
    ) -> None:
        if not entries:
            raise ValueError("BondMapper requires at least one entry")
        self.entries: list[BondEntry] = list(entries)
        self.metadata: dict = metadata or {}
        # Index for fast nearest-peak lookup
        self._positions = np.array([e.wavenumber_cm_inv for e in self.entries])
        # Sorted view
        order = np.argsort(self._positions)
        self._sorted_idx = order
        self._sorted_positions = self._positions[order]

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "BondMapper":
        """Load the seed DB from disk.

        The JSON file must have an ``entries`` list at the top level; an
        optional ``metadata`` block is also picked up.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"bond_mapping.json not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("entries", data) if isinstance(data, dict) else data
        entries = [
            BondEntry(
                id=str(e["id"]),
                wavenumber_cm_inv=float(e["wavenumber_cm_inv"]),
                tolerance_cm_inv=float(e["tolerance_cm_inv"]),
                bond=str(e["bond"]),
                mode=str(e.get("mode", "")),
                compounds=list(e.get("compounds", [])),
                discriminative_for=list(e.get("discriminative_for", [])),
                notes=str(e.get("notes", "")),
            )
            for e in raw
        ]
        meta = data.get("metadata", {}) if isinstance(data, dict) else {}
        return cls(entries, metadata=meta)

    @classmethod
    def from_dict(cls, data: dict) -> "BondMapper":
        """Build a BondMapper from an in-memory dict (no file I/O).

        Mirrors :meth:`from_json` but accepts the parsed JSON dict
        directly. Useful for tests that need to construct ad-hoc DBs
        (e.g. deliberately malformed ones to exercise ``validate_db``).

        Parameters
        ----------
        data : dict
            Either ``{"entries": [...], "metadata": {...}}`` or just a
            list of entry dicts (in which case metadata is empty).

        Example
        -------
        >>> m = BondMapper.from_dict({"entries": [
        ...     {"id": "X1", "wavenumber_cm_inv": 1003.0,
        ...      "tolerance_cm_inv": 4.0, "bond": "test", "mode": "stretch",
        ...      "compounds": ["Histidine"]},
        ... ]})
        >>> len(m)
        1
        """
        raw = data.get("entries", data) if isinstance(data, dict) else data
        entries = [
            BondEntry(
                id=str(e["id"]),
                wavenumber_cm_inv=float(e["wavenumber_cm_inv"]),
                tolerance_cm_inv=float(e["tolerance_cm_inv"]),
                bond=str(e["bond"]),
                mode=str(e.get("mode", "")),
                compounds=list(e.get("compounds", [])),
                discriminative_for=list(e.get("discriminative_for", [])),
                notes=str(e.get("notes", "")),
            )
            for e in raw
        ]
        meta = data.get("metadata", {}) if isinstance(data, dict) else {}
        return cls(entries, metadata=meta)

    # -------------------------------------------------------------------------
    # Base layer (T05): single-peak lookup
    # -------------------------------------------------------------------------

    def match_peak(
        self,
        wavenumber: float,
        *,
        tolerance_override: Optional[float] = None,
        tolerance: Optional[float] = None,
        intensity: Optional[float] = None,    # noqa: ARG002 -- accepted for back-compat
        **_ignored,
    ) -> list[BondEntry]:
        """Return all DB entries whose tolerance window contains ``wavenumber``.

        Parameters
        ----------
        wavenumber : float
            Observed peak position, cm-1.
        tolerance_override : float, optional
            If given, use this tolerance for ALL entries instead of each
            entry's own ``tolerance_cm_inv``. Useful for permissive
            scanning during exploration.
        tolerance : float, optional
            Alias for ``tolerance_override``. Accepted so legacy callers
            that used ``mapper.match_peak(wn, tolerance=8)`` keep working.
            If both are given, ``tolerance_override`` wins.
        intensity : float, optional
            Ignored. Accepted purely so legacy callers that passed peak
            intensity here (a no-op even in the original API) keep working.

        Returns
        -------
        list[BondEntry]
            All matching entries, sorted by ascending |delta wavenumber|.

        Example
        -------
        >>> mapper = BondMapper.from_json("engine/bond_mapping.json")
        >>> hits = mapper.match_peak(1003.5)
        >>> [h.id for h in hits[:1]]
        ['P004']
        """
        # Resolve the effective override: explicit tolerance_override takes
        # precedence over the legacy ``tolerance`` alias.
        eff_override = (
            tolerance_override if tolerance_override is not None else tolerance
        )
        wn = float(wavenumber)
        hits: list[tuple[float, BondEntry]] = []
        for e in self.entries:
            tol = eff_override if eff_override is not None else e.tolerance_cm_inv
            d = abs(wn - e.wavenumber_cm_inv)
            if d <= tol:
                hits.append((d, e))
        hits.sort(key=lambda x: x[0])
        return [e for _, e in hits]

    # -------------------------------------------------------------------------
    # T21 enhancement layer
    # -------------------------------------------------------------------------

    def compute_match_confidence(
        self,
        peak_position: float,
        entry: BondEntry,
    ) -> str:
        """Categorical confidence of a peak <-> DB-entry match.

        Rules
        -----
        * ``"high"``   if ``|delta| < tolerance / 2``
        * ``"medium"`` if ``tolerance / 2 <= |delta| < tolerance``
        * ``"low"``    if ``|delta| >= tolerance``  (still returned by
          callers if they pass entries from an over-permissive lookup;
          ``annotate_peaks`` itself never produces "low" matches)

        Example
        -------
        >>> mapper = BondMapper.from_json("engine/bond_mapping.json")
        >>> e = next(x for x in mapper.entries if x.id == "P004")  # 1003 +- 4
        >>> mapper.compute_match_confidence(1003.1, e)
        'high'
        >>> mapper.compute_match_confidence(1005.5, e)
        'medium'
        >>> mapper.compute_match_confidence(1008.0, e)
        'low'
        """
        d = abs(float(peak_position) - entry.wavenumber_cm_inv)
        tol = entry.tolerance_cm_inv
        if d < tol / 2.0:
            return "high"
        if d < tol:
            return "medium"
        return "low"

    def annotate_peaks(
        self,
        peaks: Iterable,
        *,
        prefer_discriminative: bool = True,
    ) -> list[AnnotatedPeak]:
        """Enrich a list of peaks from T20 with DB lookups.

        Parameters
        ----------
        peaks : iterable
            Each item must be either a :class:`engine.peak_extractor.Peak`
            dataclass instance or a dict with at least ``position``,
            ``intensity``, ``fwhm``, ``fit_quality``.
        prefer_discriminative : bool, default True
            When more than one DB entry matches the same peak, prefer
            the entry whose ``discriminative_for`` list is non-empty
            (and shortest, i.e. most specific). When False, just pick
            the nearest by wavenumber.

        Returns
        -------
        list[AnnotatedPeak]
            One annotated peak per input peak (same length, same order).
            Unmatched peaks get ``matched_to=None`` and
            ``match_confidence="none"``.

        Example
        -------
        >>> mapper = BondMapper.from_json("engine/bond_mapping.json")
        >>> ann = mapper.annotate_peaks([
        ...     {"position": 1003.0, "intensity": 0.9, "fwhm": 10.0, "fit_quality": 0.99},
        ...     {"position": 9999.0, "intensity": 0.5, "fwhm":  5.0, "fit_quality": 0.95},
        ... ])
        >>> ann[0].matched_to, ann[0].match_confidence
        ('P004', 'high')
        >>> ann[1].matched_to is None
        True
        """
        out: list[AnnotatedPeak] = []
        for p in peaks:
            pos, intens, fwhm, fq = _peak_fields(p)
            hits = self.match_peak(pos)

            chosen: Optional[BondEntry] = None
            if hits:
                if prefer_discriminative:
                    # Prefer entries WITH discriminative_for; ties broken by
                    # |delta| (already the lookup order).
                    discr = [h for h in hits if h.discriminative_for]
                    chosen = discr[0] if discr else hits[0]
                else:
                    chosen = hits[0]

            if chosen is None:
                out.append(
                    AnnotatedPeak(
                        position=pos, intensity=intens,
                        fwhm=fwhm, fit_quality=fq,
                    )
                )
            else:
                conf = self.compute_match_confidence(pos, chosen)
                out.append(
                    AnnotatedPeak(
                        position=pos, intensity=intens,
                        fwhm=fwhm, fit_quality=fq,
                        matched_to=chosen.id,
                        bond=chosen.bond,
                        compounds=list(chosen.compounds),
                        discriminative_for=list(chosen.discriminative_for),
                        match_confidence=conf,
                        delta_cm=pos - chosen.wavenumber_cm_inv,
                    )
                )
        return out

    def disambiguate_compound(
        self,
        annotated_peaks: Iterable[AnnotatedPeak],
        *,
        min_discriminative_hits: int = 1,
        confidence_weights: Optional[dict] = None,
    ) -> dict:
        """Decide which compound(s) are present using peak voting.

        Each annotated peak votes for the compounds in its
        ``discriminative_for`` list, weighted by match confidence:
        ``high`` -> 1.0, ``medium`` -> 0.5, ``low`` / ``none`` -> 0.
        Compounds reaching at least ``min_discriminative_hits``
        equivalent-high votes are declared "likely present".

        Parameters
        ----------
        annotated_peaks : iterable of AnnotatedPeak
        min_discriminative_hits : int, default 1
            Minimum vote total (in units of equivalent-high peaks) for a
            compound to make the "likely_compounds" list.
        confidence_weights : dict, optional
            Override the default {"high": 1.0, "medium": 0.5, "low": 0.0,
            "none": 0.0} mapping.

        Returns
        -------
        dict with keys:
            * ``votes`` : ``{compound: float}``  -- weighted vote totals.
            * ``likely_compounds`` : ``list[str]``  -- compounds at or above
              the threshold, sorted by vote desc.
            * ``unsupported_compounds`` : ``list[str]``  -- compounds
              appearing in the DB's compound list but with zero votes.
              Useful for "we saw no evidence of X" reports.

        Example
        -------
        >>> mapper = BondMapper.from_json("engine/bond_mapping.json")
        >>> ann = mapper.annotate_peaks([
        ...     {"position": 1003.0, "intensity": 1.0, "fwhm": 8.0, "fit_quality": 0.99},
        ...     {"position": 1180.0, "intensity": 0.6, "fwhm": 8.0, "fit_quality": 0.99},
        ...     {"position": 1495.0, "intensity": 0.7, "fwhm": 8.0, "fit_quality": 0.99},
        ... ])
        >>> r = mapper.disambiguate_compound(ann)
        >>> r["likely_compounds"]
        ['Histidine']
        """
        weights = confidence_weights or {
            "high": 1.0,
            "medium": 0.5,
            "low": 0.0,
            "none": 0.0,
        }
        votes: dict[str, float] = {c: 0.0 for c in self._known_compounds()}

        for ap in annotated_peaks:
            w = weights.get(ap.match_confidence, 0.0)
            if w == 0.0 or not ap.discriminative_for:
                continue
            for cmp_name in ap.discriminative_for:
                if cmp_name in votes:
                    votes[cmp_name] += w
                else:
                    votes[cmp_name] = w

        likely = sorted(
            [c for c, v in votes.items() if v >= float(min_discriminative_hits)],
            key=lambda c: -votes[c],
        )
        unsupported = sorted([c for c, v in votes.items() if v == 0.0])

        return {
            "votes": votes,
            "likely_compounds": likely,
            "unsupported_compounds": unsupported,
        }

    # -------------------------------------------------------------------------
    # Vectorised / convenience lookups (T05 legacy + T21 enrichment)
    # -------------------------------------------------------------------------

    def match_peaks(
        self,
        wavenumbers: Iterable[float],
        *,
        tolerance: Optional[float] = None,
    ) -> list[list[BondEntry]]:
        """Vectorised wrapper around :meth:`match_peak`.

        Parameters
        ----------
        wavenumbers : iterable of float
            Observed peak positions, cm-1.
        tolerance : float, optional
            Same semantics as the ``tolerance`` kwarg of
            :meth:`match_peak`: if given, used as a global override for
            every entry; otherwise each entry's own tolerance is used.

        Returns
        -------
        list[list[BondEntry]]
            One match list per input wavenumber, in the same order as
            ``wavenumbers``. Empty list for inputs that match nothing.

        Example
        -------
        >>> mapper = BondMapper.from_json("engine/bond_mapping.json")
        >>> res = mapper.match_peaks([1003.0, 1080.0, 9999.0])
        >>> [r[0].id if r else None for r in res]
        ['P004', 'P005', None]
        """
        return [self.match_peak(float(w), tolerance=tolerance)
                for w in wavenumbers]

    def lookup_by_id(self, entry_id: str) -> BondEntry:
        """Return the :class:`BondEntry` with the given ``id``.

        Raises
        ------
        KeyError
            If no entry matches ``entry_id``.

        Example
        -------
        >>> mapper = BondMapper.from_json("engine/bond_mapping.json")
        >>> mapper.lookup_by_id("P004").bond
        'Imidazole ring breathing'
        """
        for e in self.entries:
            if e.id == entry_id:
                return e
        raise KeyError(f"No bond-mapping entry with id={entry_id!r}")

    # -------------------------------------------------------------------------
    # Compound-level views
    # -------------------------------------------------------------------------

    @property
    def compound_canonical_names(self) -> list[str]:
        """Canonical compound list, in metadata order when available.

        Equivalent to :meth:`_known_compounds` but exposed as a public
        property for callers that just need the list of compounds the
        DB knows about (the report generator uses this).
        """
        return self._known_compounds()

    def get_compound_fingerprint(self, compound_name: str) -> dict:
        """Return the bond-DB fingerprint of a single compound.

        The fingerprint splits DB entries into two buckets:

        * ``discriminative`` -- entries whose ``discriminative_for`` list
          contains ``compound_name``. These are the high-information
          peaks the OOD scorer and report should highlight.
        * ``supporting`` -- entries whose ``compounds`` list contains
          ``compound_name`` but which are NOT discriminative for it
          (i.e. shared with other compounds).

        Unknown compounds return ``{"n_total": 0, "discriminative": [],
        "supporting": []}``.

        Parameters
        ----------
        compound_name : str
            Compound to query. Matched against entry ``compounds`` /
            ``discriminative_for`` lists exactly (case-sensitive).

        Returns
        -------
        dict
            Keys: ``compound``, ``n_total``, ``discriminative`` (list of
            entry dicts), ``supporting`` (list of entry dicts).

        Example
        -------
        >>> mapper = BondMapper.from_json("engine/bond_mapping.json")
        >>> fp = mapper.get_compound_fingerprint("Histidine")
        >>> [e["wavenumber_cm_inv"] for e in fp["discriminative"]][:1]
        [1003.0]
        """
        discriminative: list[dict] = []
        supporting: list[dict] = []
        for e in self.entries:
            if compound_name in e.discriminative_for:
                discriminative.append(e.to_dict())
            elif compound_name in e.compounds:
                supporting.append(e.to_dict())
        # Sort each bucket by wavenumber asc for stable, readable output.
        discriminative.sort(key=lambda d: d["wavenumber_cm_inv"])
        supporting.sort(key=lambda d: d["wavenumber_cm_inv"])
        return {
            "compound": compound_name,
            "n_total": len(discriminative) + len(supporting),
            "discriminative": discriminative,
            "supporting": supporting,
        }

    # -------------------------------------------------------------------------
    # Self-validation
    # -------------------------------------------------------------------------

    def validate_db(self, *, raise_on_error: bool = False) -> dict:
        """Check internal consistency of the loaded DB.

        Rules checked:

        1. ``wavenumber_cm_inv`` is non-negative.
        2. ``tolerance_cm_inv`` is strictly positive.
        3. ``id`` values are unique across entries.
        4. Every name in ``discriminative_for`` also appears in the same
           entry's ``compounds`` list (you can't be discriminative for a
           compound the entry doesn't even mention).
        5. ``compounds`` is non-empty (each entry must explain *something*).

        Parameters
        ----------
        raise_on_error : bool, default False
            If True, raise ``ValueError`` when problems are found; the
            caller doesn't have to inspect the report dict.

        Returns
        -------
        dict
            ``{"ok": bool, "n_problems": int, "problems": list[str]}``.
            ``problems`` is empty when ``ok`` is True.

        Example
        -------
        >>> mapper = BondMapper.from_json("engine/bond_mapping.json")
        >>> mapper.validate_db()["ok"]
        True
        """
        problems: list[str] = []
        seen_ids: dict[str, int] = {}
        for idx, e in enumerate(self.entries):
            tag = f"[entry #{idx} id={e.id!r}]"
            if e.wavenumber_cm_inv < 0:
                problems.append(
                    f"{tag} negative wavenumber_cm_inv: {e.wavenumber_cm_inv}"
                )
            if e.tolerance_cm_inv <= 0:
                problems.append(
                    f"{tag} non-positive tolerance_cm_inv: {e.tolerance_cm_inv}"
                )
            if not e.compounds:
                problems.append(f"{tag} empty compounds list")
            extra_discr = set(e.discriminative_for) - set(e.compounds)
            if extra_discr:
                problems.append(
                    f"{tag} discriminative_for contains compounds NOT in "
                    f"compounds: {sorted(extra_discr)}"
                )
            seen_ids.setdefault(e.id, 0)
            seen_ids[e.id] += 1

        for k, n in seen_ids.items():
            if n > 1:
                problems.append(f"Duplicate id {k!r} appears {n} times")

        report = {
            "ok": not problems,
            "n_problems": len(problems),
            "problems": problems,
        }
        if problems and raise_on_error:
            raise ValueError(
                f"BondMapper.validate_db: {len(problems)} problem(s):\n  - "
                + "\n  - ".join(problems)
            )
        return report

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _known_compounds(self) -> list[str]:
        """All compounds mentioned across DB entries."""
        seen: dict[str, None] = {}
        # Prefer the canonical order from metadata if present
        meta_order = self.metadata.get("compound_order") if self.metadata else None
        if meta_order:
            for c in meta_order:
                seen.setdefault(c, None)
        for e in self.entries:
            for c in e.compounds:
                seen.setdefault(c, None)
        return list(seen.keys())

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:  # pragma: no cover -- repr-only
        return f"BondMapper(n_entries={len(self)}, compounds={self._known_compounds()})"


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _peak_fields(p) -> tuple[float, float, float, float]:
    """Pull (position, intensity, fwhm, fit_quality) from a Peak or dict."""
    if hasattr(p, "position"):
        return (
            float(p.position),
            float(getattr(p, "intensity", 0.0)),
            float(getattr(p, "fwhm", 0.0)),
            float(getattr(p, "fit_quality", 1.0)),
        )
    return (
        float(p["position"]),
        float(p.get("intensity", 0.0)),
        float(p.get("fwhm", 0.0)),
        float(p.get("fit_quality", 1.0)),
    )
