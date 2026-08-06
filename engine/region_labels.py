"""Region labelling for detected peaks.

The bond database (``engine/bond_mapping.json``) and the pure-reference spectra
only cover the model domain (~267..2004 cm-1). Peaks the wide-canvas detector
finds outside that domain cannot be assigned a bond -- but they should still be
*reported* with an honest note rather than silently dropped. This module turns
a peak position into (a) a human-readable spectroscopic band name and (b) an
assignability verdict.
"""
from __future__ import annotations

# Inclusive model / bond-DB trained domain (cm-1). Kept in sync with
# src/data/ingest.FINGERPRINT_RANGE.
MODEL_DOMAIN: tuple[float, float] = (267.0, 2004.0)

# Standard Raman spectroscopic bands (cm-1). Used for the band *name* only;
# assignability is decided separately by MODEL_DOMAIN.
_BANDS: tuple[tuple[float, float, str], ...] = (
    (0.0, 400.0, "low-frequency / lattice region"),
    (400.0, 1800.0, "fingerprint region"),
    (1800.0, 2800.0, "triple-bond / cumulene region"),
    (2800.0, 3700.0, "X-H stretch region (C-H, N-H, O-H)"),
    (3700.0, float("inf"), "high-frequency region"),
)

OUT_OF_DOMAIN_NOTE: str = (
    "Outside fingerprint / model domain (no reference data in this region) "
    "- bond not yet assignable."
)


def band_name(wavenumber: float) -> str:
    """Return the spectroscopic band name for a wavenumber (cm-1)."""
    for lo, hi, name in _BANDS:
        if lo <= wavenumber < hi:
            return name
    return "out of range"


def is_in_model_domain(wavenumber: float) -> bool:
    """True if the peak lies within the model / bond-DB domain (267..2004)."""
    lo, hi = MODEL_DOMAIN
    return lo <= wavenumber <= hi


def label_peak_region(wavenumber: float) -> dict:
    """Describe where a peak falls and whether it is assignable.

    Args:
        wavenumber: Peak position in cm-1.

    Returns:
        Dict with keys:
            ``band``           -- spectroscopic band name,
            ``in_model_domain``-- bool,
            ``assignable``     -- bool (same as ``in_model_domain``; the bond DB
                                  only covers that range),
            ``note``           -- ``""`` if assignable else
                                  :data:`OUT_OF_DOMAIN_NOTE`.
    """
    in_dom = is_in_model_domain(wavenumber)
    return {
        "band": band_name(wavenumber),
        "in_model_domain": in_dom,
        "assignable": in_dom,
        "note": "" if in_dom else OUT_OF_DOMAIN_NOTE,
    }
