"""Evaluation metrics for the Raman Physics-Informed AI MVP.

Five metrics, one source of truth. Each function:

* takes numpy arrays (no torch dependency) so it can be used from
  training, evaluation, and report-generation code uniformly;
* validates input shapes and raises informative errors;
* returns either a scalar float or a small dict (for distributional
  metrics);
* documents the formula and intended use in the docstring.

Reference: T09 in Chat-2 spec.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_2d(arr: np.ndarray, name: str) -> np.ndarray:
    """Ensure ``arr`` is float64 2D. Promote (N,) → (N, 1)."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D; got shape {a.shape}")
    return a


def _check_same_shape(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"Shape mismatch: y_true {a.shape} vs y_pred {b.shape}"
        )


# ---------------------------------------------------------------------------
# 1. Quantification MAE
# ---------------------------------------------------------------------------

def quantification_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error between true and predicted composition simplex.

    Formula
    -------
    Given ``y_true`` and ``y_pred`` of shape ``(N, K)`` where each row is
    a composition simplex (entries in [0, 1] summing to ~1)::

        MAE = (1 / (N · K)) · Σᵢⱼ |y_true[i, j] − y_pred[i, j]|

    Notes
    -----
    * Computed across *all* (sample, compound) pairs uniformly, NOT
      averaged-of-averages. This matches scikit-learn's
      ``mean_absolute_error`` with ``multioutput="uniform_average"``.
    * For a 6-compound simplex, MAE = 0.020 means *on average* each
      compound's predicted ratio is off by 2 percentage points.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Shape ``(N, K)``. K = number of components (6 for AA, 8 for AAM).

    Returns
    -------
    float
        Non-negative scalar.
    """
    yt = _as_2d(y_true, "y_true")
    yp = _as_2d(y_pred, "y_pred")
    _check_same_shape(yt, yp)
    return float(np.mean(np.abs(yt - yp)))


# ---------------------------------------------------------------------------
# 2. Identification accuracy
# ---------------------------------------------------------------------------

def identification_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.05,
) -> float:
    """Fraction of samples whose *presence/absence pattern* is fully correct.

    Logic
    -----
    For each sample, compare the binary mask "compound is present"
    (true ratio > ``threshold``) against the same mask on the predictions.
    A sample is correct only if **all K** compounds match.

    This is stricter than per-compound binary accuracy and matches the T09
    spec: *"chấp nhận đúng nếu chất nào có % > threshold trong cả true và
    pred"* — i.e. the predicted presence mask must equal the true
    presence mask.

    Formula
    -------
    Let ``T_ij = (y_true[i, j] > threshold)`` and
    ``P_ij = (y_pred[i, j] > threshold)``::

        sample_correct[i] = ∀j (T_ij == P_ij)
        accuracy = mean(sample_correct)

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Shape ``(N, K)`` simplex compositions.
    threshold : float
        Presence threshold. 0.05 means "considered present if its ratio
        exceeds 5%". Tunable per dataset.

    Returns
    -------
    float
        In [0, 1].
    """
    yt = _as_2d(y_true, "y_true")
    yp = _as_2d(y_pred, "y_pred")
    _check_same_shape(yt, yp)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1]; got {threshold}")

    true_mask = yt > threshold
    pred_mask = yp > threshold
    correct_per_sample = np.all(true_mask == pred_mask, axis=1)
    return float(np.mean(correct_per_sample))


# ---------------------------------------------------------------------------
# 3. OOD AUROC
# ---------------------------------------------------------------------------

def ood_auroc(scores_id: Sequence[float], scores_ood: Sequence[float]) -> float:
    """ROC-AUC for OOD detection given separate ID and OOD score arrays.

    Convention
    ----------
    HIGHER score = MORE OOD (i.e., the score is an "anomaly score", not a
    confidence). If your detector emits "confidence the sample is ID",
    pass it negated.

    Formula
    -------
    Concatenate the scores; build labels (0 = ID, 1 = OOD); compute
    ROC-AUC. AUROC = 1.0 means perfect separation; 0.5 = random; < 0.5
    means the scoring is inverted.

    Falls back to a manual implementation if scikit-learn is missing,
    so the function works in minimal environments.

    Parameters
    ----------
    scores_id : sequence of float
        Anomaly scores for in-distribution samples.
    scores_ood : sequence of float
        Anomaly scores for out-of-distribution samples.

    Returns
    -------
    float
        AUROC in [0, 1].
    """
    s_id = np.asarray(scores_id, dtype=np.float64).ravel()
    s_ood = np.asarray(scores_ood, dtype=np.float64).ravel()
    if s_id.size == 0 or s_ood.size == 0:
        raise ValueError(
            f"Need non-empty score arrays; got |ID|={s_id.size}, "
            f"|OOD|={s_ood.size}."
        )
    labels = np.concatenate([np.zeros_like(s_id), np.ones_like(s_ood)])
    scores = np.concatenate([s_id, s_ood])

    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except ImportError:
        return _manual_auroc(scores, labels)


def _manual_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U based AUROC. Robust ties handling via average ranks."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    # Average ranks for ties (1-based).
    sorted_scores = scores[order]
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    n_pos = float(labels.sum())
    n_neg = float(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[labels == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# 4. Constraint Violation Rate (CVR)
# ---------------------------------------------------------------------------

def reconstruction_cosine_similarity(
    s_input: np.ndarray,
    s_recon: np.ndarray,
    *,
    eps: float = 1e-12,
) -> dict:
    """Per-sample cosine similarity between input and reconstructed spectrum.

    Formula
    -------
    For each sample::

        cos_sim_i = ⟨s_input_i, s_recon_i⟩ / (‖s_input_i‖ · ‖s_recon_i‖ + ε)

    Returns a dict with median, mean, and percentiles to surface the
    distribution shape (a single mean hides bimodality).

    Parameters
    ----------
    s_input, s_recon : np.ndarray
        Shape ``(N, P)`` where P is the spectrum length (1024).
    eps : float
        Numerical stabilizer for zero-norm samples.

    Returns
    -------
    dict
        Keys: ``per_sample`` (np.ndarray (N,)), ``median``, ``mean``,
        ``p05``, ``p25``, ``p75``, ``p95``.
    """
    si = _as_2d(s_input, "s_input")
    sr = _as_2d(s_recon, "s_recon")
    _check_same_shape(si, sr)

    dot = np.sum(si * sr, axis=1)
    n_i = np.linalg.norm(si, axis=1)
    n_r = np.linalg.norm(sr, axis=1)
    cos = dot / (n_i * n_r + eps)
    cos = np.clip(cos, -1.0, 1.0)  # guard against floating drift past ±1

    return {
        "per_sample": cos,
        "median": float(np.median(cos)),
        "mean":   float(np.mean(cos)),
        "p05":    float(np.percentile(cos, 5)),
        "p25":    float(np.percentile(cos, 25)),
        "p75":    float(np.percentile(cos, 75)),
        "p95":    float(np.percentile(cos, 95)),
    }


def constraint_violation_rate(
    s_input: np.ndarray,
    s_recon: np.ndarray,
    threshold: float = 0.85,
) -> float:
    """Fraction of samples whose reconstruction cosine similarity falls below ``threshold``.

    Interpretation
    --------------
    The reconstruction module synthesizes ``s_recon = Σ α_i · scale_i ·
    pure_i`` from the predicted composition. If the resulting spectrum
    has poor cosine similarity to the input, the model has produced a
    composition that does NOT physically explain what it just observed —
    a violation of the Beer-Lambert linearity assumption underlying the
    physics loss.

    Formula
    -------
    ::

        CVR = mean( cos_sim_i < threshold )

    where ``cos_sim_i`` is per-sample cosine similarity (same definition
    as in :func:`reconstruction_cosine_similarity`).

    Lower is better. Target per project spec: ``CVR ≤ 0.05``.

    Parameters
    ----------
    s_input, s_recon : np.ndarray
        Shape ``(N, P)``.
    threshold : float
        Cosine-similarity cutoff. Default 0.85 per project spec.

    Returns
    -------
    float
        In [0, 1].
    """
    if not -1.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [-1, 1]; got {threshold}")
    out = reconstruction_cosine_similarity(s_input, s_recon)
    cos = out["per_sample"]
    return float(np.mean(cos < threshold))


__all__ = [
    "quantification_mae",
    "identification_accuracy",
    "ood_auroc",
    "constraint_violation_rate",
    "reconstruction_cosine_similarity",
]
