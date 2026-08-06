"""Novelty localisation: unmatched peaks -> clustered hints (T22).

When the symbolic mapper (T21) leaves some peaks without any DB match,
those peaks may be the signature of:

* A compound not yet covered in the seed DB.
* An unknown impurity / contaminant.
* A genuinely novel material (OOD case).

:class:`NoveltyLocator` post-processes such "orphan" peaks by:

1. **filtering** peaks whose nearest DB entry is further than its tolerance,
2. **clustering** orphans by proximity (default 30 cm-1 single-link),
3. **suggesting** plausible chemistry per cluster from a coarse cm-1-region
   lookup table (Socrates 3rd ed style).

The chemistry suggestions are deliberately broad: this module is a
*hint generator*, not a chemistry expert. Pair its output with the
T19 OOD score to flag samples that warrant manual review.

Author: Chat 4 Phase A, Task T22.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable, Optional, Union

import numpy as np

from engine.symbolic_mapper import BondMapper, AnnotatedPeak, _peak_fields


# -----------------------------------------------------------------------------
# Region table for chemistry hints
# -----------------------------------------------------------------------------
#
# Mid-IR / Raman group-frequency assignments, coarse-grained. Each entry:
#   (low_cm, high_cm, label, examples)
# Region overlap is intentional -- some peaks legitimately fit multiple
# categories, so we report all overlapping labels for a cluster.

_REGION_TABLE: list[tuple[float, float, str, str]] = [
    (100,  400,  "lattice / skeletal modes",
     "inorganic phonons (MoS2 E2g ~380, A1g ~408); metal-O stretches; framework breathing"),
    (300,  600,  "metal-X / heavy-atom stretches",
     "M-S, M-O, M-Cl, M-N inorganic stretches; skeletal deformation"),
    (500,  700,  "C-S, C-X bend; ring deformation",
     "thiols, halogenated alkanes, aromatic ring deformation"),
    (700,  900,  "ring / skeletal stretches",
     "alkane C-C, ring breathing of aromatics / heterocycles"),
    (900,  1100, "C-C / C-O stretch",
     "alkane C-C; alcohols / ethers C-O; sugar pyranose ring; P-O"),
    (1100, 1300, "C-C, C-N, C-O stretches",
     "alkyl C-N (amines); ester / ether C-O; aromatic C-H in-plane bend; sulfate S=O"),
    (1300, 1500, "CH bending / COO- symmetric stretch",
     "aliphatic CH2/CH3 deformation; carboxylate sym stretch; amide III in proteins"),
    (1500, 1700, "C=C / C=N / amide I",
     "alkenes; imines; aromatic ring; amide C=O (Amide I)"),
    (1700, 1800, "C=O stretch (ester / ketone / acid)",
     "esters, ketones, carboxylic acid C=O; conjugation lowers towards 1680"),
    (1800, 2300, "triple bonds",
     "alkynes C#C; nitriles C#N; cumulenes"),
    (2400, 2700, "S-H stretch",
     "thiols (weak in Raman)"),
    (2700, 3000, "aliphatic C-H stretch",
     "saturated CH3 / CH2 / CH"),
    (3000, 3200, "aromatic / vinyl C-H stretch",
     "aromatic, alkene C-H"),
    (3200, 3700, "O-H / N-H stretch",
     "hydroxyls (alcohols, water, COOH); amines; broad in H-bonded systems"),
]


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass
class PeakCluster:
    """A group of nearby unmatched peaks.

    Attributes
    ----------
    members : list[dict]
        Each member dict has at least ``position``, ``intensity``,
        ``fwhm``, ``fit_quality``.
    centroid_cm : float
        Intensity-weighted mean of member positions, cm-1.
    span_cm : float
        ``max(position) - min(position)`` across members.
    total_intensity : float
        Sum of member intensities.
    n_peaks : int
        ``len(members)``.
    region_hints : list[str]
        Labels of all region-table rows whose [low, high] overlaps the
        cluster span. Best-effort chemistry suggestions, NOT a diagnosis.
    region_examples : list[str]
        Example-chemistry strings paired one-to-one with ``region_hints``.
    """

    members: list[dict]
    centroid_cm: float
    span_cm: float
    total_intensity: float
    n_peaks: int
    region_hints: list[str] = field(default_factory=list)
    region_examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------

class NoveltyLocator:
    """Cluster unmatched peaks and propose plausible chemistry hints.

    Parameters
    ----------
    mapper : BondMapper
        The bond-mapping DB used to determine which peaks are "matched"
        vs "unmatched". A peak is unmatched iff ``mapper.match_peak`` is
        empty.
    cluster_gap_cm : float, default 30.0
        Single-link clustering distance: two peaks are joined into the
        same cluster iff their position difference is <= this. Larger
        values merge more aggressively (good for broad bands); smaller
        values keep narrow features separate.
    region_table : list, optional
        Override the built-in cm-1 region table. Each row is
        ``(low_cm, high_cm, label, examples)``. Useful for v2 work that
        tailors the hint set to a specific application area.

    Example
    -------
    >>> from engine.symbolic_mapper import BondMapper
    >>> mapper = BondMapper.from_json("engine/bond_mapping.json")
    >>> nov = NoveltyLocator(mapper)
    >>> # A peak at 380 cm-1 (MoS2 region) is not in the bond DB
    >>> peaks = [{"position": 380.0, "intensity": 1.0, "fwhm": 6.0, "fit_quality": 0.99}]
    >>> r = nov.locate(peaks)
    >>> r["unknown_peaks"][0]["position"]
    380.0
    >>> "lattice" in r["hints"][0].lower()
    True
    """

    def __init__(
        self,
        mapper: BondMapper,
        *,
        cluster_gap_cm: float = 30.0,
        region_table: Optional[list[tuple[float, float, str, str]]] = None,
    ) -> None:
        self.mapper = mapper
        self.cluster_gap_cm = float(cluster_gap_cm)
        self._regions = list(region_table) if region_table is not None else list(_REGION_TABLE)

    # -------------------------------------------------------------------------
    # Stage 1 -- filter
    # -------------------------------------------------------------------------

    def find_unmatched_peaks(
        self,
        peaks: Iterable,
    ) -> list[dict]:
        """Return peaks that don't match any DB entry within tolerance.

        Accepts either raw :class:`engine.peak_extractor.Peak` objects /
        dicts (uses :meth:`BondMapper.match_peak` to decide), OR
        :class:`AnnotatedPeak` instances (uses ``matched_to is None``).

        Example
        -------
        >>> from engine.symbolic_mapper import BondMapper
        >>> mapper = BondMapper.from_json("engine/bond_mapping.json")
        >>> nov = NoveltyLocator(mapper)
        >>> peaks = [
        ...     {"position": 1003.0, "intensity": 1.0, "fwhm": 8.0, "fit_quality": 0.99},
        ...     {"position":  380.0, "intensity": 1.0, "fwhm": 6.0, "fit_quality": 0.99},
        ... ]
        >>> u = nov.find_unmatched_peaks(peaks)
        >>> [p["position"] for p in u]
        [380.0]
        """
        out: list[dict] = []
        for p in peaks:
            if isinstance(p, AnnotatedPeak):
                if p.matched_to is None:
                    out.append({
                        "position": p.position,
                        "intensity": p.intensity,
                        "fwhm": p.fwhm,
                        "fit_quality": p.fit_quality,
                    })
                continue
            pos, intens, fwhm, fq = _peak_fields(p)
            hits = self.mapper.match_peak(pos)
            if not hits:
                out.append({
                    "position": pos, "intensity": intens,
                    "fwhm": fwhm, "fit_quality": fq,
                })
        return out

    # -------------------------------------------------------------------------
    # Stage 2 -- cluster
    # -------------------------------------------------------------------------

    def cluster_unmatched(
        self,
        unmatched: list[dict],
    ) -> list[PeakCluster]:
        """Single-link cluster unmatched peaks by position proximity.

        Two peaks are in the same cluster iff they are connected by a
        chain of hops each <= ``self.cluster_gap_cm``. Used for grouping
        broad bands or multi-peak features that come from one chemical
        cause.

        Returns
        -------
        list[PeakCluster]
            Sorted by ascending centroid_cm.
        """
        if not unmatched:
            return []

        items = sorted(unmatched, key=lambda d: d["position"])
        clusters: list[list[dict]] = [[items[0]]]
        for p in items[1:]:
            prev = clusters[-1][-1]
            if p["position"] - prev["position"] <= self.cluster_gap_cm:
                clusters[-1].append(p)
            else:
                clusters.append([p])

        out: list[PeakCluster] = []
        for grp in clusters:
            positions = np.array([m["position"] for m in grp], dtype=np.float64)
            intens = np.array([m["intensity"] for m in grp], dtype=np.float64)
            weight = intens / intens.sum() if intens.sum() > 0 else np.ones_like(intens) / len(intens)
            centroid = float((positions * weight).sum())
            span = float(positions.max() - positions.min())
            total_i = float(intens.sum())
            hints, examples = self._region_hints_for_span(positions.min(), positions.max())
            out.append(
                PeakCluster(
                    members=grp,
                    centroid_cm=centroid,
                    span_cm=span,
                    total_intensity=total_i,
                    n_peaks=len(grp),
                    region_hints=hints,
                    region_examples=examples,
                )
            )
        return out

    # -------------------------------------------------------------------------
    # Stage 3 -- hints
    # -------------------------------------------------------------------------

    def suggest_chemistry(
        self,
        clusters: list[PeakCluster],
    ) -> list[str]:
        """Flatten cluster region_hints into per-cluster human-readable strings.

        Returns one string per cluster suitable for inclusion in a
        report. Format::

            "cluster centroid 1670 cm-1 (3 peaks, span 18 cm-1): C=C / C=N / amide I -- alkenes; imines; ..."
        """
        out: list[str] = []
        for c in clusters:
            label = ", ".join(c.region_hints) if c.region_hints else "no standard region match"
            examples = "; ".join(c.region_examples) if c.region_examples else "n/a"
            out.append(
                f"cluster centroid {c.centroid_cm:.0f} cm-1 "
                f"({c.n_peaks} peak{'s' if c.n_peaks != 1 else ''}, "
                f"span {c.span_cm:.0f} cm-1): {label}  --  {examples}"
            )
        return out

    # -------------------------------------------------------------------------
    # End-to-end convenience
    # -------------------------------------------------------------------------

    def locate(
        self,
        peaks: Iterable,
    ) -> dict:
        """Run unmatched-filter -> cluster -> hint, return a single dict.

        Returns
        -------
        dict with keys:
            * ``unknown_peaks`` -- list of unmatched peak dicts.
            * ``clusters`` -- list of dicts (PeakCluster.to_dict()).
            * ``hints`` -- list of human-readable hint strings, one per cluster.

        Example
        -------
        >>> from engine.symbolic_mapper import BondMapper
        >>> mapper = BondMapper.from_json("engine/bond_mapping.json")
        >>> nov = NoveltyLocator(mapper, cluster_gap_cm=30.0)
        >>> peaks = [
        ...     {"position": 380.0, "intensity": 1.0, "fwhm": 6.0, "fit_quality": 0.99},
        ...     {"position": 408.0, "intensity": 0.8, "fwhm": 5.0, "fit_quality": 0.98},
        ... ]
        >>> r = nov.locate(peaks)
        >>> len(r["clusters"]) == 1 and r["clusters"][0]["n_peaks"] == 2
        True
        """
        unmatched = self.find_unmatched_peaks(peaks)
        clusters = self.cluster_unmatched(unmatched)
        hints = self.suggest_chemistry(clusters)
        return {
            "unknown_peaks": unmatched,
            "clusters": [c.to_dict() for c in clusters],
            "hints": hints,
        }

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _region_hints_for_span(
        self, low_cm: float, high_cm: float
    ) -> tuple[list[str], list[str]]:
        """Return (labels, examples) for region rows overlapping [low, high]."""
        labels: list[str] = []
        examples: list[str] = []
        for r_low, r_high, label, ex in self._regions:
            if (high_cm >= r_low) and (low_cm <= r_high):
                labels.append(label)
                examples.append(ex)
        return labels, examples
