"""Publication-quality figures for the evaluation metrics (Day 12).

Turns the scalar eval results (OOD AUROC, reconstruction cosine, quantification
MAE, identification accuracy, constraint violation, baseline comparison) into
figures for the report / dashboard. Each function takes plain arrays and
returns a ``matplotlib.figure.Figure`` (and saves a PNG if ``save_path`` given),
so it works whether the numbers come from ``src/eval/benchmark.py``, a saved
``.npz``, or the live dashboard.

Dependencies: numpy + matplotlib only (ROC/AUROC computed directly, no sklearn).

Author: evaluation visualisation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Success bars from the project spec (Section 6), drawn as reference lines.
COSINE_TARGET, COSINE_FLOOR = 0.95, 0.85
AUROC_TARGET, AUROC_FLOOR = 0.85, 0.75
MAE_TARGET, MAE_FLOOR = 0.020, 0.025

_ID_COLOR, _OOD_COLOR = "#2ca02c", "#d62728"   # green = ID, red = OOD


def _save(fig: Figure, save_path: Optional[Union[str, Path]]) -> None:
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")


# =============================================================================
# OOD: ROC curve + AUROC
# =============================================================================

def roc_auc(is_ood: Sequence[bool], scores: Sequence[float]
            ) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute the ROC curve and AUROC for OOD detection.

    Args:
        is_ood: Binary ground-truth labels (True = out-of-distribution).
        scores: OOD scores; higher should indicate more OOD.

    Returns:
        ``(fpr, tpr, auc)`` where ``fpr``/``tpr`` trace the curve and ``auc``
        is the area under it (0.5 = chance, 1.0 = perfect).
    """
    y = np.asarray(is_ood).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")
    tpr = np.concatenate([[0.0], np.cumsum(y) / n_pos])
    fpr = np.concatenate([[0.0], np.cumsum(~y) / n_neg])
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def plot_roc_curve(is_ood: Sequence[bool], scores: Sequence[float], *,
                   auc_override: Optional[float] = None,
                   save_path=None, title: Optional[str] = None) -> Figure:
    """ROC curve for OOD detection with AUROC and target/floor annotated.

    Args:
        is_ood: Binary OOD labels (True = OOD).
        scores: OOD scores (higher = more OOD).
        auc_override: If given, display this AUROC instead of the
            curve-integrated one. Pass the value from
            :func:`src.eval.metrics.ood_auroc` so the figure and the
            benchmark table always agree (sklearn vs trapezoid can differ in
            the last digit on ties).
        save_path: Optional PNG path.
        title: Optional title override.
    """
    fpr, tpr, auc_curve = roc_auc(is_ood, scores)
    auc = float(auc_override) if auc_override is not None else auc_curve
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2.0, label=f"ROC (AUROC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="0.6", ls="--", lw=1.0, label="chance")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#1f77b4")
    band = ("PASS-target" if auc >= AUROC_TARGET
            else "PASS-floor" if auc >= AUROC_FLOOR else "BELOW floor")
    bcolor = {"PASS-target": "darkgreen", "PASS-floor": "orange",
              "BELOW floor": "crimson"}[band]
    ax.text(0.97, 0.42, f"target {AUROC_TARGET:.2f} / floor {AUROC_FLOOR:.2f}\n[{band}]",
            transform=ax.transAxes, ha="right", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=bcolor))
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title(title or "OOD detection - ROC curve")
    ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); _save(fig, save_path)
    return fig


def plot_score_distribution(is_ood: Sequence[bool], scores: Sequence[float], *,
                            threshold: Optional[float] = None,
                            save_path=None) -> Figure:
    """Overlaid ID vs OOD score histograms with the decision threshold."""
    y = np.asarray(is_ood).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    bins = np.linspace(float(s.min()), float(s.max()), 30)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.hist(s[~y], bins=bins, color=_ID_COLOR, alpha=0.55, label="ID")
    ax.hist(s[y], bins=bins, color=_OOD_COLOR, alpha=0.55, label="OOD")
    if threshold is not None:
        ax.axvline(threshold, color="black", ls="--", lw=1.2,
                   label=f"threshold = {threshold:.3f}")
    ax.set_xlabel("OOD score"); ax.set_ylabel("count")
    ax.set_title("OOD score distribution (ID vs OOD)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); _save(fig, save_path)
    return fig


# =============================================================================
# Physics: reconstruction cosine distribution
# =============================================================================

def plot_cosine_distribution(cosines: Sequence[float], *,
                             cvr_threshold: float = COSINE_FLOOR,
                             save_path=None) -> Figure:
    """Histogram of reconstruction cosine with median, spec lines, and CVR.

    The region below ``cvr_threshold`` (default 0.85, matching
    ``metrics.constraint_violation_rate``) is shaded; the constraint-violation
    rate (fraction of samples below it) is annotated.
    """
    c = np.asarray(cosines, dtype=np.float64)
    median = float(np.median(c))
    cvr = float(np.mean(c < cvr_threshold))
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.hist(c, bins=30, color="#1f77b4", alpha=0.7, edgecolor="white", lw=0.4)
    ax.axvspan(float(min(c.min(), cvr_threshold)) - 1e-6, cvr_threshold,
               color="crimson", alpha=0.10)
    ax.axvline(median, color="navy", lw=2.0, label=f"median = {median:.3f}")
    ax.axvline(COSINE_TARGET, color="darkgreen", ls="--", lw=1.2,
               label=f"target {COSINE_TARGET:.2f}")
    ax.axvline(COSINE_FLOOR, color="orange", ls="--", lw=1.2,
               label=f"floor {COSINE_FLOOR:.2f}")
    ax.text(0.02, 0.95, f"constraint violation rate = {cvr:.3f}\n"
                        f"(cosine < {cvr_threshold:.2f})",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="crimson"))
    ax.set_xlabel("Reconstruction cosine similarity"); ax.set_ylabel("count")
    ax.set_title("Physics validation - reconstruction cosine (test set)")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="upper left",
                                  bbox_to_anchor=(0.0, 0.80))
    fig.tight_layout(); _save(fig, save_path)
    return fig


# =============================================================================
# Quantification: parity plot + per-compound MAE
# =============================================================================

def plot_quant_parity(y_true: np.ndarray, y_pred: np.ndarray,
                      compound_order: Sequence[str], *, save_path=None) -> Figure:
    """Predicted vs true fraction (parity plot), coloured by compound."""
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    mae = float(np.mean(np.abs(yt - yp)))
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    for i, name in enumerate(compound_order):
        ax.scatter(yt[:, i], yp[:, i], s=14, alpha=0.5,
                   color=cmap(i % 10), label=name)
    ax.plot([0, 1], [0, 1], color="0.4", ls="--", lw=1.0)
    band = ("PASS-target" if mae <= MAE_TARGET
            else "PASS-floor" if mae <= MAE_FLOOR else "BELOW floor")
    bcolor = {"PASS-target": "darkgreen", "PASS-floor": "orange",
              "BELOW floor": "crimson"}[band]
    ax.text(0.03, 0.97, f"overall MAE = {mae:.4f}  [{band}]",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=bcolor))
    ax.set_xlabel("True fraction"); ax.set_ylabel("Predicted fraction")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title("Quantification parity (predicted vs true)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout(); _save(fig, save_path)
    return fig


def plot_mae_per_compound(y_true: np.ndarray, y_pred: np.ndarray,
                          compound_order: Sequence[str], *,
                          save_path=None) -> Figure:
    """Bar chart of per-compound MAE with the spec target/floor lines."""
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    per = np.mean(np.abs(yt - yp), axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    colors = ["seagreen" if v <= MAE_TARGET else "orange" if v <= MAE_FLOOR
              else "crimson" for v in per]
    bars = ax.bar(list(compound_order), per, color=colors, alpha=0.85)
    ax.axhline(MAE_TARGET, color="darkgreen", ls="--", lw=1.0,
               label=f"target {MAE_TARGET}")
    ax.axhline(MAE_FLOOR, color="orange", ls="--", lw=1.0,
               label=f"floor {MAE_FLOOR}")
    for b, v in zip(bars, per):
        ax.annotate(f"{v:.3f}", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.set_ylabel("MAE"); ax.set_title("Quantification MAE per compound")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout(); _save(fig, save_path)
    return fig


def plot_identification_presence(y_true: np.ndarray, y_pred: np.ndarray,
                                 compound_order: Sequence[str], *,
                                 threshold: float = 0.05,
                                 save_path=None) -> Figure:
    """Per-compound presence-detection accuracy + overall exact-match.

    Mirrors ``metrics.identification_accuracy``: a compound is "present" when
    its fraction exceeds ``threshold``; a sample is correct only if the whole
    presence mask matches. This shows, per compound, the fraction of samples
    whose present/absent call is right, and annotates the strict exact-match
    accuracy (all 6 correct).
    """
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    tmask, pmask = yt > threshold, yp > threshold
    per_compound_acc = np.mean(tmask == pmask, axis=0)
    exact = float(np.mean(np.all(tmask == pmask, axis=1)))

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    colors = ["seagreen" if v >= 0.90 else "orange" if v >= 0.85 else "crimson"
              for v in per_compound_acc]
    bars = ax.bar(list(compound_order), per_compound_acc, color=colors, alpha=0.85)
    for b, v in zip(bars, per_compound_acc):
        ax.annotate(f"{v:.2f}", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.axhline(0.90, color="darkgreen", ls="--", lw=1.0, label="target 0.90")
    ax.axhline(0.85, color="orange", ls="--", lw=1.0, label="floor 0.85")
    ax.set_ylim(0, 1.08); ax.set_ylabel("Presence-detection accuracy")
    ax.set_title(f"Identification per compound  "
                 f"(strict exact-match accuracy = {exact:.3f}, "
                 f"presence threshold {threshold})")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=9, loc="lower right")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout(); _save(fig, save_path)
    return fig


# =============================================================================
# Canonical metrics table + array dump (numbers match src/eval/metrics.py)
# =============================================================================

def compute_metrics_table(*, y_true=None, y_pred=None,
                          scores_id=None, scores_ood=None,
                          s_input=None, s_recon=None,
                          id_threshold: float = 0.05,
                          cvr_threshold: float = 0.85) -> dict:
    """Compute the spec metrics via ``src.eval.metrics`` (single source of truth).

    Every metric that has inputs is computed with the project's own metric
    functions, so the figure annotations and any saved table match the
    benchmark exactly. Returns a flat ``{metric_name: value}`` dict.
    """
    from src.eval import metrics as M  # canonical definitions

    out: dict[str, float] = {}
    if y_true is not None and y_pred is not None:
        out["MAE"] = M.quantification_mae(y_true, y_pred)
        out["Accuracy"] = M.identification_accuracy(y_true, y_pred,
                                                    threshold=id_threshold)
    if scores_id is not None and scores_ood is not None:
        out["AUROC"] = M.ood_auroc(scores_id, scores_ood)
    if s_input is not None and s_recon is not None:
        out["Cosine median"] = M.reconstruction_cosine_similarity(
            s_input, s_recon)["median"]
        out["CVR"] = M.constraint_violation_rate(s_input, s_recon,
                                                 threshold=cvr_threshold)
    return out


def dump_eval_arrays(npz_path: Union[str, Path], *,
                     compound_order: Sequence[str],
                     y_true=None, y_pred=None,
                     scores_id=None, scores_ood=None, ood_threshold=None,
                     s_input=None, s_recon=None,
                     cosines=None) -> Path:
    """Save evaluation arrays into one ``.npz`` for the figure script / dashboard.

    Call this from ``src/eval/compare.py`` with whatever it already computed.
    Builds the combined OOD label/score vectors from the separate ID/OOD score
    arrays and (if needed) per-sample cosines from input/reconstruction.

    Returns:
        The path written.
    """
    p = Path(npz_path); p.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "compound_order": np.asarray(list(compound_order))}
    if y_true is not None and y_pred is not None:
        arrays["y_true"] = np.asarray(y_true, float)
        arrays["y_pred"] = np.asarray(y_pred, float)
    if scores_id is not None and scores_ood is not None:
        s_id = np.asarray(scores_id, float).ravel()
        s_ood = np.asarray(scores_ood, float).ravel()
        arrays["ood_scores"] = np.concatenate([s_id, s_ood])
        arrays["is_ood"] = np.concatenate([np.zeros(s_id.size, bool),
                                           np.ones(s_ood.size, bool)])
    if ood_threshold is not None:
        arrays["ood_threshold"] = np.asarray(float(ood_threshold))
    if cosines is not None:
        arrays["cosines"] = np.asarray(cosines, float).ravel()
    elif s_input is not None and s_recon is not None:
        from src.eval import metrics as M
        arrays["cosines"] = M.reconstruction_cosine_similarity(
            s_input, s_recon)["per_sample"]
    np.savez(p, **arrays)
    return p


# =============================================================================
# Identification: confusion matrix (optional, for dominant-compound view)
# =============================================================================

def plot_confusion_matrix(true_labels: Sequence[int], pred_labels: Sequence[int],
                          classes: Sequence[str], *, normalize: bool = True,
                          save_path=None) -> Figure:
    """Confusion matrix heatmap for the identification task."""
    t = np.asarray(true_labels, int); p = np.asarray(pred_labels, int)
    k = len(classes)
    cm = np.zeros((k, k), dtype=np.float64)
    for ti, pi in zip(t, p):
        cm[ti, pi] += 1
    acc = float(np.trace(cm) / cm.sum()) if cm.sum() else float("nan")
    shown = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(classes, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    thr = (shown.max() if shown.size else 1.0) / 2.0
    for i in range(k):
        for j in range(k):
            txt = f"{shown[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if shown[i, j] > thr else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Identification confusion matrix (acc = {acc:.3f})")
    fig.tight_layout(); _save(fig, save_path)
    return fig


# =============================================================================
# Baselines: model comparison (small multiples, one panel per metric)
# =============================================================================

def plot_benchmark_comparison(metrics: dict[str, dict[str, float]], *,
                              higher_better: Optional[dict[str, bool]] = None,
                              save_path=None) -> Figure:
    """Grouped bars comparing models across metrics (one panel per metric).

    Separate panels avoid mixing incompatible scales (MAE ~0.02 vs AUROC ~0.85).

    Args:
        metrics: ``{metric_name: {model_name: value}}``.
        higher_better: ``{metric_name: bool}``; best bar is highlighted. MAE and
            "violation" default to lower-is-better, others higher-is-better.
        save_path: optional PNG path.

    Returns:
        A matplotlib Figure.
    """
    metric_names = list(metrics.keys())
    models = list(next(iter(metrics.values())).keys())
    hb = dict(higher_better or {})
    for m in metric_names:
        if m not in hb:
            ml = m.lower()
            hb[m] = not ("mae" in ml or "violation" in ml or "error" in ml)

    cmap = plt.get_cmap("Set2")
    n = len(metric_names)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.4))
    if n == 1:
        axes = [axes]
    for ax, m in zip(axes, metric_names):
        vals = [metrics[m][mod] for mod in models]
        best = (max if hb[m] else min)(range(len(vals)), key=lambda k: vals[k])
        colors = [cmap(i % 8) for i in range(len(models))]
        colors[best] = "#d62728"
        bars = ax.bar(range(len(models)), vals, color=colors, alpha=0.9)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}", xy=(b.get_x() + b.get_width() / 2, v),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
        arrow = "higher better" if hb[m] else "lower better"
        ax.set_title(f"{m}\n({arrow})", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Model comparison (red = best per metric)", fontsize=12)
    fig.tight_layout(); _save(fig, save_path)
    return fig


# =============================================================================
# Convenience: build whatever the available arrays allow
# =============================================================================

def make_all_from_arrays(out_dir: Union[str, Path], *,
                         compound_order: Sequence[str],
                         is_ood=None, ood_scores=None, ood_threshold=None,
                         auc_override=None,
                         cosines=None, y_true=None, y_pred=None,
                         id_threshold: float = 0.05,
                         true_labels=None, pred_labels=None,
                         benchmark: Optional[dict] = None) -> dict[str, Path]:
    """Generate every figure for which the inputs are provided.

    Args:
        out_dir: Directory for the PNGs (created if needed).
        compound_order: Compound names.
        is_ood, ood_scores, ood_threshold: OOD detection arrays.
        auc_override: AUROC from ``metrics.ood_auroc`` to show on the ROC plot.
        cosines: Per-sample reconstruction cosine similarities.
        y_true, y_pred: ``(N, 6)`` true / predicted composition matrices.
        id_threshold: Presence threshold for identification (default 0.05).
        true_labels, pred_labels: Integer class indices (optional confusion
            matrix for a dominant-compound view).
        benchmark: ``{metric: {model: value}}`` for the comparison panel.

    Returns:
        ``{figure_name: path}`` for everything that was generated.
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    made: dict[str, Path] = {}
    if is_ood is not None and ood_scores is not None:
        plot_roc_curve(is_ood, ood_scores, auc_override=auc_override,
                       save_path=out / "roc.png"); plt.close()
        made["roc"] = out / "roc.png"
        plot_score_distribution(is_ood, ood_scores, threshold=ood_threshold,
                                save_path=out / "ood_scores.png"); plt.close()
        made["ood_scores"] = out / "ood_scores.png"
    if cosines is not None:
        plot_cosine_distribution(cosines, save_path=out / "cosine.png"); plt.close()
        made["cosine"] = out / "cosine.png"
    if y_true is not None and y_pred is not None:
        plot_quant_parity(y_true, y_pred, compound_order,
                          save_path=out / "parity.png"); plt.close()
        made["parity"] = out / "parity.png"
        plot_mae_per_compound(y_true, y_pred, compound_order,
                              save_path=out / "mae_per_compound.png"); plt.close()
        made["mae_per_compound"] = out / "mae_per_compound.png"
        plot_identification_presence(y_true, y_pred, compound_order,
                                     threshold=id_threshold,
                                     save_path=out / "identification.png")
        plt.close()
        made["identification"] = out / "identification.png"
    if true_labels is not None and pred_labels is not None:
        plot_confusion_matrix(true_labels, pred_labels, compound_order,
                              save_path=out / "confusion.png"); plt.close()
        made["confusion"] = out / "confusion.png"
    if benchmark is not None:
        plot_benchmark_comparison(benchmark, save_path=out / "benchmark.png")
        plt.close()
        made["benchmark"] = out / "benchmark.png"
    return made
