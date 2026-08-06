"""End-to-end inference pipeline (T_glue).

This module ties together every learned and non-learned component built
in the previous chats:

* **Chat 3 (T18)** -- MC-Dropout uncertainty via
  :func:`src.models.uncertainty.predict_with_uncertainty`.
* **Chat 3 (T19)** -- OOD scoring via :class:`src.inference.ood.OODScorer`.
* **Chat 4 Phase A-B (T20)** -- peak extraction via
  :class:`engine.peak_extractor.PeakExtractor`.
* **Chat 4 Phase A-B (T21)** -- symbolic bond mapping via
  :class:`engine.symbolic_mapper.BondMapper`.
* **Chat 4 Phase A-B (T22)** -- novelty localisation via
  :class:`engine.novelty_locator.NoveltyLocator`.

The single entry point is :func:`predict`, which takes a 1-D Raman
spectrum (assumed already preprocessed -- see Custom Instructions §2)
and returns a single dict ready for the report generator (:mod:`report`)
and visualiser (:mod:`visualize`).

Heavy resources (model, scorer, bond DB) are loaded once on first call
and cached in module-level state -- so repeated calls (e.g. dashboard
batch mode, demo loop) don't pay the load cost every time.

Author: Chat 4 Phase C, Task T_glue.
"""

from __future__ import annotations

import os
import sys
import json
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

# These imports are deferred to runtime via _lazy() so that the module
# can be imported in a sandbox without torch / lmfit. The functions that
# need them check at call time and raise a clear error if missing.

# Project-relative paths (callers may override).
DEFAULT_CHECKPOINT_PATH        = "checkpoints/best.pt"
DEFAULT_WAVENUMBERS_PATH       = "data/processed/wavenumbers.npy"
DEFAULT_BOND_MAPPING_PATH      = "engine/bond_mapping.json"
DEFAULT_OOD_CALIBRATION_PATH   = "results/ood_demo/calibration.json"

# Canonical compound order -- LOCKED, see CHAT4_PHASE_AB_HANDOVER §D.1.
COMPOUND_ORDER = [
    "Alanine", "Asparagine", "Aspartic Acid",
    "Glutamic Acid", "Histidine", "Glucosamine",
]


# -----------------------------------------------------------------------------
# Cached resources (loaded lazily on first call)
# -----------------------------------------------------------------------------

@dataclass
class _Resources:
    """Singleton-style cache for expensive resources."""

    model: Optional[object] = None              # torch.nn.Module
    ood_scorer: Optional[object] = None         # OODScorer | None (None ok)
    extractor: Optional[object] = None          # PeakExtractor
    mapper: Optional[object] = None             # BondMapper
    locator: Optional[object] = None            # NoveltyLocator
    wavenumbers: Optional[np.ndarray] = None    # (P,) float64
    checkpoint_path: Optional[str] = None       # path actually loaded
    n_mc_samples: int = 50


_resources = _Resources()


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def predict(
    spectrum: np.ndarray,
    *,
    model_path: str = DEFAULT_CHECKPOINT_PATH,
    wavenumbers_path: str = DEFAULT_WAVENUMBERS_PATH,
    bond_mapping_path: str = DEFAULT_BOND_MAPPING_PATH,
    ood_calibration_path: Optional[str] = DEFAULT_OOD_CALIBRATION_PATH,
    n_mc_samples: int = 50,
    skip_ood: bool = False,
    verbose: bool = False,
) -> dict:
    """End-to-end inference for one preprocessed Raman spectrum.

    Parameters
    ----------
    spectrum : np.ndarray
        1-D preprocessed spectrum, shape ``(P,)`` matching the model and
        wavenumbers axis (default 1024). NOT raw -- the caller must run
        the same preprocessing pipeline (AsLS baseline + cosmic + SG + SNV)
        that produced the training cache; see Custom Instructions §2.
    model_path : str
        Path to the trained checkpoint, as produced by Chat 3 T15. Must
        contain ``model`` state dict, ``config`` dict, and the
        ``reconstruction.pure_ref`` buffer baked into the state dict.
    wavenumbers_path : str
        Path to ``(P,)`` float wavenumber axis ``.npy`` file.
    bond_mapping_path : str
        Path to the bond DB JSON.
    ood_calibration_path : str or None
        Pre-computed OOD calibration. If None or the file is missing,
        OOD scoring is skipped (and ``ood_score`` is None in the output).
    n_mc_samples : int, default 50
        Number of MC-Dropout forward passes for uncertainty estimation.
        Lower for speed (e.g. dashboard preview); higher for thesis-defense
        figures. Standard MC-Dropout literature uses 30-100.
    skip_ood : bool, default False
        If True, do not compute or load OOD components. Useful for unit
        tests that only need composition + peaks.
    verbose : bool, default False
        Print progress to stderr.

    Returns
    -------
    dict
        Schema (per CHAT4_PHASE_AB_HANDOVER §D.2)::

            {
              # Composition (T18)
              "composition":        {compound_name: float, ...},  # mean simplex
              "composition_std":    {compound_name: float, ...},  # MC std
              "composition_mean":   np.ndarray (6,),
              "composition_std_arr": np.ndarray (6,),
              "predictive_entropy": float,
              "mean_compound_std":  float,

              # Reconstruction
              "reconstructed_spectrum":  np.ndarray (P,),
              "recon_cosine_sim":        float in [-1, 1],

              # OOD (T19)
              "ood_score":          float | None,    # None if skip_ood
              "is_ood":             bool  | None,
              "ood_threshold":      float | None,
              "ood_components":     {"recon_norm": float, "var_norm": float} | None,

              # Peaks (T20 + T21)
              "peaks":              list[dict],  # AnnotatedPeak.to_dict()
              "likely_compounds_symbolic":  list[str],
              "compound_votes":     {compound_name: float, ...},
              "unsupported_compounds":      list[str],

              # Novelty (T22)
              "unknown_peaks":      list[dict],
              "novelty_clusters":   list[dict],
              "novelty_hints":      list[str],

              # Bookkeeping
              "metadata": {
                "model_path":       str,
                "n_mc_samples":     int,
                "spectrum_length":  int,
                "n_peaks_detected": int,
                "n_peaks_matched":  int,
              },
              "input_spectrum":     np.ndarray (P,),   # echoed for plotting
              "wavenumbers":        np.ndarray (P,),
            }

    Raises
    ------
    FileNotFoundError
        If the checkpoint, wavenumbers, or bond DB is missing.
    ValueError
        If the spectrum shape does not match the wavenumbers axis.

    Example
    -------
    >>> import numpy as np
    >>> spectrum = np.load("data/processed/spectra_full.pt")[0]  # row 0
    >>> result = predict(spectrum)
    >>> result["composition"]["Histidine"]
    0.45
    >>> result["likely_compounds_symbolic"]
    ['Histidine']
    >>> result["is_ood"]
    False
    """
    _vprint = (lambda *a, **k: print("[predict]", *a, file=sys.stderr, **k)) \
        if verbose else (lambda *a, **k: None)

    # ---- Validate input ----
    spectrum = np.asarray(spectrum, dtype=np.float64)
    if spectrum.ndim != 1:
        raise ValueError(f"spectrum must be 1-D, got shape {spectrum.shape}")

    # ---- Lazy-load resources ----
    _ensure_loaded(
        model_path=model_path,
        wavenumbers_path=wavenumbers_path,
        bond_mapping_path=bond_mapping_path,
        ood_calibration_path=None if skip_ood else ood_calibration_path,
        n_mc_samples=n_mc_samples,
        verbose=verbose,
    )

    if spectrum.size != _resources.wavenumbers.size:
        raise ValueError(
            f"spectrum length {spectrum.size} does not match wavenumbers "
            f"length {_resources.wavenumbers.size}"
        )

    # ---- T18: MC-Dropout composition + reconstruction ----
    _vprint("MC-Dropout forward...")
    mc = _mc_forward(spectrum, n_samples=n_mc_samples)
    comp_mean = mc["composition_mean"].squeeze(0)        # (6,)
    comp_std  = mc["composition_std"].squeeze(0)         # (6,)
    recon     = mc["reconstruction_mean"].squeeze(0)     # (P,)

    composition       = {c: float(comp_mean[i]) for i, c in enumerate(COMPOUND_ORDER)}
    composition_std   = {c: float(comp_std[i])  for i, c in enumerate(COMPOUND_ORDER)}

    # Cosine similarity between input and reconstruction
    recon_cos = _cosine_similarity(spectrum, recon)

    # ---- T19: OOD scoring ----
    if _resources.ood_scorer is None:
        ood_score = None
        is_ood = None
        ood_threshold = None
        ood_components = None
    else:
        _vprint("OOD scoring...")
        ood_payload = _ood_score_with_components(spectrum)
        ood_score      = float(ood_payload["score"])
        is_ood         = bool(ood_payload["is_ood"])
        ood_threshold  = float(ood_payload["threshold"])
        ood_components = {
            "recon_norm": float(ood_payload["recon_norm"]),
            "var_norm":   float(ood_payload["var_norm"]),
        }

    # ---- T20 + T21: peak extraction & annotation ----
    _vprint("Peak extraction...")
    peaks_raw = _resources.extractor.extract_full(spectrum)
    annotated = _resources.mapper.annotate_peaks(peaks_raw)
    disambig  = _resources.mapper.disambiguate_compound(annotated)

    n_peaks_total   = len(annotated)
    n_peaks_matched = sum(1 for ap in annotated if ap.matched_to is not None)

    # ---- T22: novelty localisation ----
    _vprint("Novelty localisation...")
    novelty = _resources.locator.locate(peaks_raw)

    # ---- Pack the output ----
    result = {
        "composition":       composition,
        "composition_std":   composition_std,
        "composition_mean":  comp_mean,
        "composition_std_arr": comp_std,
        "predictive_entropy":   float(mc["predictive_entropy"].squeeze()),
        "mean_compound_std":    float(mc["mean_compound_std"].squeeze()),

        "reconstructed_spectrum": recon,
        "recon_cosine_sim":       recon_cos,

        "ood_score":      ood_score,
        "is_ood":         is_ood,
        "ood_threshold":  ood_threshold,
        "ood_components": ood_components,

        "peaks":                       [ap.to_dict() for ap in annotated],
        "likely_compounds_symbolic":   disambig["likely_compounds"],
        "compound_votes":              disambig["votes"],
        "unsupported_compounds":       disambig["unsupported_compounds"],

        "unknown_peaks":     novelty["unknown_peaks"],
        "novelty_clusters":  novelty["clusters"],
        "novelty_hints":     novelty["hints"],

        "metadata": {
            "model_path":       str(_resources.checkpoint_path),
            "n_mc_samples":     int(n_mc_samples),
            "spectrum_length":  int(spectrum.size),
            "n_peaks_detected": int(n_peaks_total),
            "n_peaks_matched":  int(n_peaks_matched),
            "compound_order":   list(COMPOUND_ORDER),
        },

        # Echoed for plotting downstream
        "input_spectrum":  spectrum,
        "wavenumbers":     _resources.wavenumbers,
    }
    return result


def predict_batch(
    spectra: np.ndarray,
    **kwargs,
) -> list[dict]:
    """Convenience wrapper for batched inference.

    Loops :func:`predict` per row; resources are loaded once on the
    first call thanks to module-level caching. For a tight inner loop
    you may want to call the underlying components directly; this is
    intended for moderate (<= a few hundred) batch sizes.
    """
    spectra = np.asarray(spectra)
    if spectra.ndim != 2:
        raise ValueError(f"spectra must be 2-D (N, P), got shape {spectra.shape}")
    return [predict(spectra[i], **kwargs) for i in range(spectra.shape[0])]


def reset_cache() -> None:
    """Discard cached resources. Useful for tests and notebooks."""
    global _resources
    _resources = _Resources()


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------

def _ensure_loaded(
    *,
    model_path: str,
    wavenumbers_path: str,
    bond_mapping_path: str,
    ood_calibration_path: Optional[str],
    n_mc_samples: int,
    verbose: bool,
) -> None:
    """Load and cache all resources on first call (lazy init)."""
    global _resources

    _vprint = (lambda *a, **k: print("[predict.load]", *a, file=sys.stderr, **k)) \
        if verbose else (lambda *a, **k: None)

    # Wavenumbers
    if _resources.wavenumbers is None:
        wn_path = Path(wavenumbers_path)
        if not wn_path.exists():
            raise FileNotFoundError(f"Wavenumbers not found at {wn_path}")
        _resources.wavenumbers = np.load(wn_path).astype(np.float64)
        _vprint(f"loaded wavenumbers from {wn_path}")

    # Bond DB + extractor + locator (cheap; co-load)
    if _resources.mapper is None:
        from engine.symbolic_mapper import BondMapper
        from engine.peak_extractor import PeakExtractor
        from engine.novelty_locator import NoveltyLocator
        _resources.mapper = BondMapper.from_json(bond_mapping_path)
        _resources.extractor = PeakExtractor(_resources.wavenumbers)
        _resources.locator = NoveltyLocator(_resources.mapper)
        _vprint(f"loaded engine layer (mapper={len(_resources.mapper)} entries)")

    # Model (heavy)
    if _resources.model is None or _resources.checkpoint_path != model_path:
        import torch
        ck_path = Path(model_path)
        if not ck_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ck_path}")
        _vprint(f"loading checkpoint {ck_path}")
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        _resources.model = _build_model_from_checkpoint(ck)
        _resources.model.eval()
        _resources.checkpoint_path = str(ck_path)
        _vprint(f"model on cpu; epoch={ck.get('epoch', '?')}")

    # OOD scorer (optional)
    if ood_calibration_path is not None and _resources.ood_scorer is None:
        cal_path = Path(ood_calibration_path)
        if cal_path.exists():
            try:
                from src.inference.ood import OODScorer
                _resources.ood_scorer = OODScorer.from_file(
                    _resources.model, str(cal_path)
                )
                _vprint(f"loaded OOD calibration from {cal_path}")
            except Exception as e:
                warnings.warn(
                    f"Failed to load OOD scorer from {cal_path}: {e}. "
                    f"OOD scoring will be skipped."
                )
                _resources.ood_scorer = None
        else:
            warnings.warn(
                f"OOD calibration file {cal_path} not found. "
                f"OOD scoring will be skipped."
            )

    _resources.n_mc_samples = int(n_mc_samples)


def _build_model_from_checkpoint(ck: dict):
    """Build a `RamanPhysicsAI` from a checkpoint dict.

    Follows the recipe in CHAT3_PHASE3_HANDOVER §D.2: rebuild from
    ``config``, then load ``model`` state. The `reconstruction.pure_ref`
    buffer is restored automatically by load_state_dict.
    """
    import torch
    from src.models.full_model import build_full_model_from_config

    cfg = ck["config"]

    # The factory wants a path to the reference spectra. Dump the
    # checkpoint's baked-in buffer to a tempfile and point the factory
    # at it. (Avoids re-reading engine/reference_spectra.npy which may
    # not exist on the user's filesystem -- see T17-B2.)
    refs = ck["model"]["reconstruction.pure_ref"].cpu().numpy()
    tmp_dir = tempfile.mkdtemp(prefix="raman_predict_refs_")
    ref_path = Path(tmp_dir) / "ref.npy"
    np.save(ref_path, refs)

    model = build_full_model_from_config(
        cfg, reference_spectra_path=str(ref_path)
    )
    model.load_state_dict(ck["model"])
    return model


def _mc_forward(spectrum: np.ndarray, *, n_samples: int) -> dict:
    """Run MC-Dropout forward via T18's helper."""
    from src.models.uncertainty import predict_with_uncertainty
    return predict_with_uncertainty(
        _resources.model, spectrum, n_samples=n_samples
    )


def _ood_score_with_components(spectrum: np.ndarray) -> dict:
    """Compute OOD score AND its components for the report.

    The OODScorer's public API only returns the combined score, so we
    re-derive the components by calling the same primitives the scorer
    uses internally. This keeps the report's "why is this OOD" panel
    explainable.
    """
    import torch
    scorer = _resources.ood_scorer
    cal = scorer.calibration

    x = torch.as_tensor(spectrum, dtype=torch.float32).view(1, 1, -1)

    # Deterministic forward for reconstruction error
    scorer.model.eval()
    with torch.no_grad():
        out = scorer.model(x)
    recon = out["reconstruction"]
    if recon.ndim == 3:
        recon = recon.squeeze(1)
    target = x.squeeze(1) if x.ndim == 3 else x
    recon_err = float(((recon - target) ** 2).mean().item())

    # MC forward for variance
    from src.models.uncertainty import predict_with_uncertainty
    mc = predict_with_uncertainty(
        scorer.model, spectrum, n_samples=scorer.calibration.mc_samples
    )
    pred_var = float(mc["mean_compound_std"].squeeze().item())

    # Normalise components
    rn = min(recon_err / cal.recon_p95, 1.0) if cal.recon_p95 > 0 else 0.0
    vn = min(pred_var / cal.var_p95, 1.0) if cal.var_p95 > 0 else 0.0
    score = cal.recon_weight * rn + cal.var_weight * vn

    return {
        "score":     score,
        "is_ood":    score > cal.score_p95,
        "threshold": cal.score_p95,
        "recon_norm": rn,
        "var_norm":   vn,
        "raw_recon_err": recon_err,
        "raw_pred_var":  pred_var,
    }


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Numerically safe cosine similarity for two 1-D arrays."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _vprint(*a, **k):  # pragma: no cover -- top-level stub, real one set inline
    pass
