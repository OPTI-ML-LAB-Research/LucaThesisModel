"""Shared utilities for stretch handover scripts.

Centralizes the (annoying) job of pulling fields out of predict()'s
output dict in a way that survives:

* changes in field shape (Tensor / ndarray / list / dict-keyed-by-name)
* missing fields (when predict was called with skip_ood=True etc.)
* both flat and batched outputs

Place at `scripts/stretch/_handover_utils.py` and import from each
`run_t*.py`.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

# Canonical order. Keep in sync with src.inference.predict.COMPOUND_ORDER.
COMPOUND_ORDER = [
    "Alanine", "Asparagine", "Aspartic Acid",
    "Glutamic Acid", "Histidine", "Glucosamine",
]


def to_vector(x: Any, order: list[str] = COMPOUND_ORDER) -> list[float]:
    """Normalize a composition output to ``list[float]`` aligned with ``order``.

    Handles every shape predict() has been observed to return:
    - dict[compound_name -> float]    (predict()['composition'], ['composition_std'])
    - np.ndarray (N,)                 (predict()['composition_mean'])
    - torch.Tensor (N,) or (1, N)
    - plain list[float] or list[list[float]]
    - None → zeros

    Parameters
    ----------
    x : any
        Composition-like object to normalize.
    order : list[str]
        Compound names in the canonical order. Used only when ``x`` is a dict.

    Returns
    -------
    list[float]
        Length ``len(order)``. Missing dict keys → 0.0.
    """
    if x is None:
        return [0.0] * len(order)

    # dict[name -> float] : align to canonical order
    if isinstance(x, dict):
        return [float(x.get(c, 0.0)) for c in order]

    # torch.Tensor : detach to numpy
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()

    # np.ndarray or anything with .tolist()
    if hasattr(x, "tolist"):
        x = x.tolist()

    # Unbatch [[a, b, c, ...]] → [a, b, c, ...]
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], (list, tuple)):
        x = list(x[0])

    if not isinstance(x, list):
        x = list(x)

    # Cast and pad/truncate to canonical length.
    out = [float(v) for v in x]
    if len(out) < len(order):
        out += [0.0] * (len(order) - len(out))
    return out[:len(order)]


def get_recon_cosine(result: dict) -> Optional[float]:
    """Read reconstruction cosine similarity. Tries both old + new field names."""
    for key in ("recon_cosine_sim", "recon_cosine", "reconstruction_cosine"):
        v = result.get(key)
        if v is not None:
            return float(v)
    return None


def get_ood(result: dict) -> dict:
    """Read OOD verdict in a uniform shape.

    Returns
    -------
    dict with keys: score, is_ood, threshold, recon_norm, var_norm
    (any may be None).
    """
    # Flat schema (current predict.py)
    if "ood_score" in result:
        comps = result.get("ood_components") or {}
        return {
            "score":      _maybe_float(result.get("ood_score")),
            "is_ood":     result.get("is_ood"),
            "threshold":  _maybe_float(result.get("ood_threshold")),
            "recon_norm": _maybe_float(comps.get("recon_norm")),
            "var_norm":   _maybe_float(comps.get("var_norm")),
        }
    # Legacy nested schema: result["ood"] = {...}
    if isinstance(result.get("ood"), dict):
        ood = result["ood"]
        comps = ood.get("components") or {}
        return {
            "score":      _maybe_float(ood.get("score")),
            "is_ood":     ood.get("is_ood"),
            "threshold":  _maybe_float(ood.get("threshold")),
            "recon_norm": _maybe_float(comps.get("recon_err") or comps.get("recon_norm")),
            "var_norm":   _maybe_float(comps.get("pred_var") or comps.get("var_norm")),
        }
    return {"score": None, "is_ood": None, "threshold": None,
            "recon_norm": None, "var_norm": None}


def get_peaks(result: dict) -> list[dict]:
    """Return a plain list of peak dicts (never a Tensor / ndarray)."""
    p = result.get("peaks", [])
    if p is None:
        return []
    if isinstance(p, list):
        return p
    if hasattr(p, "tolist"):
        return list(p.tolist())
    return list(p)


def _maybe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        if hasattr(v, "item"):
            v = v.item()
        return float(v)
    except (TypeError, ValueError):
        return None
