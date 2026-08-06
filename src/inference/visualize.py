"""Visualisation utilities for inference reports (T27).

Three functions, each producing a publication-quality figure used in
the per-sample report (T26) and the optional Streamlit dashboard (T28):

1. :func:`plot_reconstruction_overlay`
   Input spectrum vs reconstruction, with a difference subplot and the
   cosine-similarity score annotated. Used for the "physics validation"
   section of the report.

2. :func:`plot_peak_annotations`
   Spectrum with arrows + labels at every detected peak. Peaks matched
   to compounds are colour-coded by compound; unmatched peaks are grey
   with "?".

3. :func:`plot_ood_summary`
   Spectrum with shaded regions around novelty clusters, plus an OOD
   score gauge at the top.

All three accept the ``result`` dict from :func:`src.inference.predict.predict`
and return a ``matplotlib.figure.Figure``. They also save to disk if
``save_path`` is given.

Author: Chat 4 Phase C, Task T27.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# Compound-to-colour map -- consistent across all plots so a reviewer
# can scan all three figures for the same sample without re-learning
# the legend.
_COMPOUND_COLORS = {
    "Alanine":       "#1f77b4",   # blue
    "Asparagine":    "#ff7f0e",   # orange
    "Aspartic Acid": "#2ca02c",   # green
    "Glutamic Acid": "#d62728",   # red
    "Histidine":     "#9467bd",   # purple
    "Glucosamine":   "#8c564b",   # brown
}
_UNMATCHED_COLOR = "#7f7f7f"      # gray


# =============================================================================
# Plot 1 -- reconstruction overlay
# =============================================================================

def plot_reconstruction_overlay(
    result: dict,
    *,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    show: bool = False,
) -> Figure:
    """Overlay input spectrum and reconstruction with a difference subplot.

    The top axis shows the two spectra (input solid, reconstruction
    dashed). The bottom axis shows the per-pixel difference. The cosine
    similarity is annotated in the upper-left corner; values close to
    1.0 indicate the model's composition prediction reproduces the input
    well -- physics-loss working as designed.

    Parameters
    ----------
    result : dict
        Output of :func:`src.inference.predict.predict`.
    save_path : str or Path, optional
        If provided, save the figure to disk at this path.
    title : str, optional
        Override the auto-generated title.
    show : bool, default False
        If True, call ``plt.show()`` (Jupyter). Always returns the fig.

    Returns
    -------
    matplotlib.figure.Figure
    """
    wn = np.asarray(result["wavenumbers"])
    s_in = np.asarray(result["input_spectrum"])
    s_re = np.asarray(result["reconstructed_spectrum"])
    cos_sim = float(result["recon_cosine_sim"])

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_top.plot(wn, s_in, color="steelblue", lw=1.0, label="input")
    ax_top.plot(wn, s_re, color="crimson", lw=1.2, ls="--", label="reconstruction")
    ax_top.set_ylabel("Intensity (preprocessed)")
    ax_top.grid(alpha=0.3)
    ax_top.legend(loc="upper right", fontsize=9)

    # Cosine similarity annotation
    band = (
        "PASS-target" if cos_sim >= 0.95
        else "PASS-floor" if cos_sim >= 0.85
        else "FAIL"
    )
    band_color = {"PASS-target": "darkgreen", "PASS-floor": "orange", "FAIL": "crimson"}[band]
    ax_top.text(
        0.02, 0.95, f"cosine sim = {cos_sim:.4f}  [{band}]",
        transform=ax_top.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=band_color, lw=1.2),
    )

    # Difference panel
    diff = s_re - s_in
    ax_bot.plot(wn, diff, color="dimgray", lw=0.8)
    ax_bot.axhline(0, color="black", lw=0.5)
    ax_bot.fill_between(wn, diff, 0, where=(diff >= 0), color="crimson", alpha=0.2)
    ax_bot.fill_between(wn, diff, 0, where=(diff < 0), color="steelblue", alpha=0.2)
    ax_bot.set_xlabel("Raman shift (cm$^{-1}$)")
    ax_bot.set_ylabel("recon − input")
    ax_bot.grid(alpha=0.3)

    if title is None:
        title = "Reconstruction overlay (physics validation)"
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    if show:
        plt.show()
    return fig


# =============================================================================
# Plot 2 -- peak annotations
# =============================================================================

def plot_peak_annotations(
    result: dict,
    *,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    show: bool = False,
    annotate_unmatched: bool = True,
) -> Figure:
    """Spectrum with arrows + labels at every detected peak.

    Peaks are colour-coded by their primary matched compound (taking the
    first entry of ``compounds`` per AnnotatedPeak). Unmatched peaks are
    drawn in gray with a "?" label.

    Parameters
    ----------
    result : dict
        Output of :func:`src.inference.predict.predict`.
    annotate_unmatched : bool, default True
        Whether to draw arrows for unmatched peaks (set False for a
        cleaner figure when there are many spurious detections).

    Returns
    -------
    matplotlib.figure.Figure
    """
    wn = np.asarray(result["wavenumbers"])
    s_in = np.asarray(result["input_spectrum"])
    peaks = result["peaks"]

    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.plot(wn, s_in, color="steelblue", lw=1.0)

    # Track which compounds appear, for the legend
    legend_handles: dict[str, "matplotlib.lines.Line2D"] = {}

    ymax = float(np.max(s_in)) if np.max(s_in) > 0 else 1.0
    ymin = float(np.min(s_in)) if np.min(s_in) < 0 else 0.0
    y_offset = ymax * 0.10
    # Reserve ~30% headroom above the peaks for label annotations
    ax.set_ylim(ymin - 0.02 * (ymax - ymin),
                ymax + 0.30 * (ymax - ymin))

    for p in peaks:
        pos = p["position"]
        i_y = p["intensity"]
        matched = p["matched_to"]
        confidence = p["match_confidence"]
        compounds = p.get("compounds") or []

        if matched is None:
            if not annotate_unmatched:
                continue
            color = _UNMATCHED_COLOR
            label = "?"
            alpha = 0.6
        else:
            primary = compounds[0] if compounds else "unknown"
            color = _COMPOUND_COLORS.get(primary, _UNMATCHED_COLOR)
            label = f"{matched}\n{primary[:8]}"
            alpha = 1.0 if confidence == "high" else 0.7
            # Add to legend
            if primary not in legend_handles:
                legend_handles[primary] = ax.plot(
                    [], [], "o", color=color, label=primary,
                )[0]

        # Vertical line + label above the peak
        ax.axvline(pos, color=color, lw=0.6, ls="--", alpha=alpha * 0.6)
        ax.annotate(
            label, xy=(pos, i_y), xytext=(pos, i_y + y_offset),
            ha="center", fontsize=7, color=color, alpha=alpha,
            arrowprops=dict(arrowstyle="-", color=color, alpha=alpha * 0.7, lw=0.6),
        )

    # Always include an "unmatched" legend entry if any unmatched peaks exist
    if any(p["matched_to"] is None for p in peaks) and annotate_unmatched:
        ax.plot([], [], "o", color=_UNMATCHED_COLOR, label="unmatched", alpha=0.6)

    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity (preprocessed)")
    if title is None:
        n_total = len(peaks)
        n_match = sum(1 for p in peaks if p["matched_to"] is not None)
        title = f"Peak annotations  ({n_match}/{n_total} matched)"
    ax.set_title(title, fontsize=12, pad=12)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    if show:
        plt.show()
    return fig


# =============================================================================
# Plot 3 -- OOD summary
# =============================================================================

def plot_ood_summary(
    result: dict,
    *,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    show: bool = False,
) -> Figure:
    """OOD score gauge + spectrum with shaded novel-peak regions.

    The figure has two parts:

    * **Top gauge** -- a horizontal bar showing the OOD score on a [0, 1]
      scale, with the calibrated threshold marked. Green/orange/red zones
      give the reviewer an at-a-glance sense of how borderline the call is.
    * **Bottom panel** -- the input spectrum with shaded regions around
      each novelty cluster (unknown peaks not in the bond DB), labelled
      with the chemistry hint from the novelty locator.

    If OOD scoring was skipped (``result["ood_score"] is None``), the
    gauge is hidden and only the novelty panel is shown.

    Parameters
    ----------
    result : dict
        Output of :func:`src.inference.predict.predict`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    wn = np.asarray(result["wavenumbers"])
    s_in = np.asarray(result["input_spectrum"])
    score   = result.get("ood_score")
    is_ood  = result.get("is_ood")
    threshold = result.get("ood_threshold")
    clusters = result.get("novelty_clusters", [])
    hints    = result.get("novelty_hints", [])

    has_score = score is not None

    if has_score:
        fig, (ax_g, ax) = plt.subplots(
            2, 1, figsize=(11, 5.5),
            gridspec_kw={"height_ratios": [1, 4]},
        )
        _draw_ood_gauge(ax_g, score, threshold, is_ood)
    else:
        fig, ax = plt.subplots(figsize=(11, 4.5))

    # Spectrum
    ax.plot(wn, s_in, color="steelblue", lw=1.0, label="input spectrum")

    # Shade novelty clusters
    if clusters:
        ymin, ymax = ax.get_ylim()
        for c, hint in zip(clusters, hints):
            members = c["members"]
            pos_lo = min(m["position"] for m in members)
            pos_hi = max(m["position"] for m in members)
            # Pad cluster span by 10 cm-1 each side for visibility
            ax.axvspan(pos_lo - 10, pos_hi + 10, color="crimson", alpha=0.15)
            # Label at the centroid, above the spectrum
            ax.annotate(
                f"novel: {c['n_peaks']} pk",
                xy=(c["centroid_cm"], 0.95 * ymax if ymax > 0 else 1.0),
                xytext=(0, 0), textcoords="offset points",
                ha="center", fontsize=8, color="crimson",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="crimson", alpha=0.8),
            )

    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity (preprocessed)")
    if title is None:
        if has_score:
            verdict = "OUT-OF-DISTRIBUTION" if is_ood else "IN-DISTRIBUTION"
            title = f"OOD assessment  --  {verdict}  ({len(clusters)} novel cluster(s))"
        else:
            title = f"Novelty localisation  ({len(clusters)} novel cluster(s))"
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    if show:
        plt.show()
    return fig


def _draw_ood_gauge(ax, score: float, threshold: float, is_ood: bool) -> None:
    """Horizontal-bar OOD-score gauge."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Background bands: green [0, threshold-0.1], orange [threshold-0.1, threshold], red beyond
    safe_hi = max(0.0, threshold - 0.1)
    ax.axvspan(0,             safe_hi,    color="lightgreen", alpha=0.5)
    ax.axvspan(safe_hi,       threshold,  color="navajowhite", alpha=0.7)
    ax.axvspan(threshold,     1.0,        color="lightcoral", alpha=0.5)

    # Score marker
    score_color = "crimson" if is_ood else "darkgreen"
    ax.axvline(score, color=score_color, lw=3.5)
    ax.text(
        score, 0.5, f"  {score:.3f}",
        va="center", ha="left", fontsize=10, fontweight="bold",
        color=score_color,
    )

    # Threshold marker
    ax.axvline(threshold, color="black", lw=1.0, ls="--")
    ax.text(
        threshold, 1.02, f"thresh {threshold:.3f}",
        ha="center", va="bottom", fontsize=8,
    )

    verdict = "OOD" if is_ood else "ID"
    ax.text(
        0.01, 0.5, f"{verdict}",
        ha="left", va="center", fontsize=11, fontweight="bold",
        color=score_color,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=score_color),
    )

    ax.set_yticks([])
    ax.set_xlabel("OOD score")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="x", labelsize=8)


# =============================================================================
# Convenience: generate all three plots at once
# =============================================================================

def plot_all(
    result: dict,
    *,
    output_dir: Union[str, Path],
    prefix: str = "sample",
    show: bool = False,
) -> dict[str, Path]:
    """Generate all three plots and return a dict of saved paths.

    Convenience wrapper for the report generator (T26) and demo scripts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    p_recon = output_dir / f"{prefix}_reconstruction.png"
    plot_reconstruction_overlay(result, save_path=p_recon, show=show)
    plt.close()
    paths["reconstruction"] = p_recon

    p_peaks = output_dir / f"{prefix}_peaks.png"
    plot_peak_annotations(result, save_path=p_peaks, show=show)
    plt.close()
    paths["peaks"] = p_peaks

    p_ood = output_dir / f"{prefix}_ood.png"
    plot_ood_summary(result, save_path=p_ood, show=show)
    plt.close()
    paths["ood"] = p_ood

    return paths
