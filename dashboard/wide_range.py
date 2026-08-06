"""Wide-range helpers for the dashboard.

Bridges :mod:`src.data.ingest` and :mod:`engine.region_labels` to give the
Streamlit app two new abilities without touching the model path:

  * **Display** only the measured region of a wide (0..4000 cm-1) canvas.
  * **Scan peaks across the full measured range** -- including peaks outside the
    fingerprint / model domain -- and tag each with a band name and an
    assignability note.

The peak scan here is intentionally self-contained (scipy only, no lmfit): it
exists to *surface and locate* out-of-fingerprint peaks honestly. The canonical
in-domain peak numbers + bond assignments still come from the engine /
``predict()`` pipeline on the model grid.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks, peak_widths, savgol_filter

from src.data.ingest import build_canvas, measured_slice, IngestResult
from engine.region_labels import label_peak_region


# --- Light measured-region preprocessing (display / peak scan only) ----------

def _asls_baseline(y: np.ndarray, lam: float = 1.0e5,
                   p: float = 0.01, n_iter: int = 10) -> np.ndarray:
    """Asymmetric Least Squares baseline (Eilers & Boelens).

    Args:
        y: Intensity vector.
        lam: Smoothness penalty (larger = smoother baseline).
        p: Asymmetry (0<p<1; small favours points below the curve).
        n_iter: Reweighting iterations.

    Returns:
        Estimated baseline, same shape as ``y``.
    """
    n = len(y)
    d = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(n, n - 2))
    dtd = lam * (d @ d.transpose())
    w = np.ones(n)
    z = y.copy()
    for _ in range(n_iter):
        wdiag = sparse.spdiags(w, 0, n, n)
        z = spsolve((wdiag + dtd).tocsc(), w * y)
        w = p * (y > z) + (1.0 - p) * (y < z)
    return z


def preprocess_measured(intensity: np.ndarray,
                        sg_window: int = 11,
                        sg_poly: int = 3) -> np.ndarray:
    """Baseline-correct, smooth and min-max scale a measured-region spectrum.

    Runs ONLY on the measured region (never on the zero-filled canvas tails, so
    the baseline/scale statistics are not distorted by padding). Min-max scaling
    to [0, 1] keeps the config's relative peak thresholds meaningful.

    Args:
        intensity: Measured-region intensities (ascending wavenumber order).
        sg_window: Savitzky-Golay window (auto-shrunk for short inputs).
        sg_poly: Savitzky-Golay polynomial order.

    Returns:
        Preprocessed intensities scaled to [0, 1].
    """
    y = np.asarray(intensity, dtype=np.float64)
    if y.size < 5:
        return y
    y = y - _asls_baseline(y)
    win = min(sg_window, y.size if y.size % 2 else y.size - 1)
    if win >= sg_poly + 2 and win % 2 == 1:
        y = savgol_filter(y, win, sg_poly)
    rng = float(y.max() - y.min())
    return (y - y.min()) / rng if rng > 0 else np.zeros_like(y)


# --- Wide-range peak scan ----------------------------------------------------

def scan_peaks_full_range(result: IngestResult,
                          min_height: float = 0.05,
                          min_prominence: float = 0.03) -> list[dict]:
    """Detect peaks across the whole measured range and label their regions.

    Args:
        result: An :class:`~src.data.ingest.IngestResult`.
        min_height: Minimum peak height relative to the max (config
            ``peak_min_height``).
        min_prominence: Minimum prominence relative to the max (config
            ``peak_min_prominence``).

    Returns:
        List of peak dicts sorted by position, each with keys:
        ``position`` (cm-1), ``intensity`` (0..1), ``fwhm`` (cm-1),
        ``band``, ``in_model_domain``, ``assignable``, ``note``.
    """
    wn, raw = measured_slice(result)
    if wn.size < 5:
        return []
    step = float(np.median(np.diff(wn)))
    y = preprocess_measured(raw)

    idx, _ = find_peaks(y, height=min_height, prominence=min_prominence)
    if idx.size == 0:
        return []
    widths, *_ = peak_widths(y, idx, rel_height=0.5)  # FWHM in points

    peaks: list[dict] = []
    for i, w in zip(idx, widths):
        pos = float(wn[i])
        info = label_peak_region(pos)
        peaks.append({
            "position": pos,
            "intensity": float(y[i]),
            "fwhm": float(w * step),
            "band": info["band"],
            "in_model_domain": info["in_model_domain"],
            "assignable": info["assignable"],
            "note": info["note"],
        })
    return sorted(peaks, key=lambda p: p["position"])


# --- Convenience for the app -------------------------------------------------

def ingest_raw_for_display(axis: np.ndarray,
                           intensity: np.ndarray,
                           axis_unit: str = "auto") -> IngestResult:
    """Build a wide-canvas :class:`IngestResult` from a raw (axis, intensity).

    Thin wrapper over :func:`src.data.ingest.build_canvas` so the app has a
    single import surface for the wide-range feature.
    """
    return build_canvas(axis, intensity, axis_unit=axis_unit)


# --- Measured-region figures (for raw uploads) -------------------------------

# --- Reconstruction extraction ----------------------------------------------

_RECON_KEYS = ("reconstructed_spectrum", "reconstruction", "reconstruction_mean",
               "recon_mean", "s_recon", "recon", "reconstructed",
               "reconstruction_spectrum")


def _as_spectrum(v, n: int) -> np.ndarray | None:
    """Coerce ``v`` to a length-``n`` float vector, or None if it can't be."""
    try:
        arr = np.asarray(v, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    return arr if arr.size == n else None


def _get_reconstruction(result: dict, n: int = 1024) -> np.ndarray | None:
    """Best-effort extraction of the model-grid reconstruction.

    Strategy (robust to key-name drift across predict() versions):
      1. Try the known key names in order.
      2. Otherwise scan every ``recon``-family key for a length-``n`` array,
         skipping scalars and std / error / cosine / norm fields.

    Args:
        result: predict() output dict.
        n: Expected model-grid length.

    Returns:
        A length-``n`` reconstruction vector, or None.
    """
    for k in _RECON_KEYS:
        if k in result:
            arr = _as_spectrum(result[k], n)
            if arr is not None:
                return arr
    skip = ("std", "sigma", "err", "cos", "norm", "var")
    for k, v in result.items():
        kl = str(k).lower()
        if "recon" in kl and not any(s in kl for s in skip):
            arr = _as_spectrum(v, n)
            if arr is not None:
                return arr
    return None


def _reconstruct_from_composition(result: dict,
                                  reference_spectra: np.ndarray,
                                  compound_order: list[str]) -> np.ndarray | None:
    """Physics fallback: ``s_recon = sum_i composition_i * reference_i``.

    Faithful to the reconstruction module's formula (modulo the learned
    per-compound scale, which only reweights components and is immaterial for a
    min-max display overlay). Used only when no reconstruction array is in the
    result dict.

    Args:
        result: predict() output dict (must contain ``composition``).
        reference_spectra: ``(n_compounds, n)`` pure references on the model grid.
        compound_order: Compound names in the reference-row order.

    Returns:
        Length-``n`` reconstruction, or None if composition is unavailable.
    """
    comp = result.get("composition")
    if not isinstance(comp, dict):
        return None
    refs = np.asarray(reference_spectra, dtype=np.float64)
    if refs.ndim != 2 or refs.shape[0] != len(compound_order):
        return None
    weights = np.array([float(comp.get(c, 0.0)) for c in compound_order])
    return weights @ refs


def _minmax(y: np.ndarray) -> np.ndarray:
    rng = float(y.max() - y.min())
    return (y - y.min()) / rng if rng > 0 else np.zeros_like(y)


def plot_measured_reconstruction(canvas: IngestResult,
                                 result: dict,
                                 model_wn: np.ndarray,
                                 reference_spectra: np.ndarray | None = None,
                                 compound_order: list[str] | None = None,
                                 save_path=None):
    """Reconstruction-vs-observed overlay, zoomed to the measured region.

    The observed curve is the preprocessed measured-region spectrum. The
    reconstruction (defined only on the 267..2004 model grid) is interpolated
    onto the measured wavenumbers and is 0 outside the model domain -- so a
    spectrum measured entirely outside 267..2004 shows a flat ~0 reconstruction
    against real observed peaks (the physics-can't-reconstruct OOD signal).

    The reconstruction is taken from the result dict if present, else rebuilt
    from ``composition x reference_spectra``. Both curves are min-max scaled for
    display; the quantitative cosine comes from the result.

    Args:
        canvas: Wide-canvas ingest result.
        result: predict() output dict.
        model_wn: Model wavenumber grid (cm-1).
        reference_spectra: Optional ``(6, 1024)`` pure refs for the fallback.
        compound_order: Compound names matching the reference rows.
        save_path: If given, save the figure there.

    Returns:
        A matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    mwn, raw = measured_slice(canvas)
    obs = _minmax(preprocess_measured(raw))

    n = int(np.asarray(model_wn).size)
    recon = _get_reconstruction(result, n)
    recon_label = "reconstruction (model domain only)"
    if recon is None and reference_spectra is not None and compound_order:
        recon = _reconstruct_from_composition(result, reference_spectra,
                                              compound_order)
        recon_label = "reconstruction (composition x refs)"
    cos = result.get("recon_cosine_sim", result.get("recon_cosine"))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(mwn, obs, lw=1.1, color="steelblue", label="observed (measured)")
    if recon is not None:
        mw = np.asarray(model_wn, dtype=np.float64)
        order = np.argsort(mw)
        recon_on_m = np.interp(mwn, mw[order], recon[order], left=0.0, right=0.0)
        ax.plot(mwn, _minmax(recon_on_m), lw=1.1, color="crimson",
                ls="--", label=recon_label)
    else:
        ax.text(0.5, 0.9, "reconstruction unavailable", transform=ax.transAxes,
                ha="center", color="crimson", fontsize=9)

    ax.axvspan(267, 2004, color="seagreen", alpha=0.08,
               label="model domain (267-2004)")
    ax.set_xlim(canvas.measured_lo, canvas.measured_hi)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity (min-max, for display)")
    ttl = "Physics validation - measured region"
    if cos is not None:
        ttl += f"  (cosine = {float(cos):.3f})"
    ax.set_title(ttl)
    ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# --- Bond merge + annotated peak figure --------------------------------------

def merge_bond_info(peaks: list[dict],
                    result_peaks: list[dict] | None,
                    tol_cm: float = 10.0) -> list[dict]:
    """Attach canonical bond info from ``result["peaks"]`` to detected peaks.

    For each in-domain detected peak, finds the nearest annotated peak from
    predict() within ``tol_cm`` and copies its ``matched_to`` / ``bond`` /
    ``compounds``. Out-of-domain peaks are never matched (no reference data).
    This reuses the canonical symbolic-mapper output rather than re-running it,
    so the figure and the report agree.

    Args:
        peaks: Detected peaks from :func:`scan_peaks_full_range`.
        result_peaks: ``result["peaks"]`` (annotated peaks), or None.
        tol_cm: Max position difference to consider a match (cm-1).

    Returns:
        The peaks list with added ``bond`` / ``matched_to`` / ``compounds`` /
        ``has_bond`` keys.
    """
    rp = result_peaks or []
    rp_pos = np.array([float(p.get("position", np.nan)) for p in rp]) \
        if rp else np.empty(0)
    for p in peaks:
        bond = matched_to = None
        compounds: list = []
        if p["assignable"] and rp_pos.size:
            j = int(np.argmin(np.abs(rp_pos - p["position"])))
            if abs(rp_pos[j] - p["position"]) <= tol_cm:
                src = rp[j]
                matched_to = src.get("matched_to")
                bond = src.get("bond")
                compounds = src.get("compounds") or []
        p["bond"] = bond
        p["matched_to"] = matched_to
        p["compounds"] = compounds
        p["has_bond"] = bool(matched_to or bond)
    return peaks


def build_combined_peaks(canvas: IngestResult,
                         result_peaks: list[dict] | None,
                         min_height: float = 0.05,
                         min_prominence: float = 0.03,
                         domain: tuple[float, float] = (267.0, 2004.0)
                         ) -> list[dict]:
    """One reconciled peak set across the whole measured range.

    To keep the peak figure / table consistent with the engine's
    ``result["peaks"]`` (which also drives bond mapping and the novelty / OOD
    figure), this function uses a single source of truth per region:

      * **Inside the model domain (267..2004):** the engine's ``result["peaks"]``
        verbatim -- same detector, thresholds, bonds and novelty status as the
        rest of the pipeline. No independent re-detection here.
      * **Outside the model domain:** :func:`scan_peaks_full_range`, the only
        place the engine never looks. These are reported but not bond-assignable.

    Marker heights use the observed measured curve so all peaks sit on the
    visible trace regardless of which detector found them.

    Args:
        canvas: Wide-canvas ingest result.
        result_peaks: ``result["peaks"]`` from predict() (engine detector).
        min_height: Relative height threshold for the out-of-domain scan.
        min_prominence: Relative prominence threshold for the out-of-domain scan.
        domain: Inclusive model-domain bounds (cm-1).

    Returns:
        Combined peak dicts sorted by position, each with: ``position``,
        ``intensity`` (observed-curve height), ``fwhm``, ``band``,
        ``in_model_domain``, ``assignable``, ``note``, ``bond``, ``matched_to``,
        ``compounds``, ``has_bond``, ``source`` ("engine" | "scan").
    """
    lo, hi = domain
    mwn, raw = measured_slice(canvas)
    obs = preprocess_measured(raw)

    def obs_at(pos: float) -> float:
        return float(np.interp(pos, mwn, obs))

    peaks: list[dict] = []

    # In-domain: defer entirely to the engine's detected peaks.
    for rp in (result_peaks or []):
        pos = float(rp.get("position", np.nan))
        if not (lo <= pos <= hi):
            continue
        if not (canvas.measured_lo <= pos <= canvas.measured_hi):
            continue
        info = label_peak_region(pos)
        bond, matched = rp.get("bond"), rp.get("matched_to")
        peaks.append({
            "position": pos,
            "intensity": obs_at(pos),
            "fwhm": float(rp.get("fwhm", 0.0) or 0.0),
            "band": info["band"],
            "in_model_domain": True,
            "assignable": True,
            "note": "",
            "bond": bond,
            "matched_to": matched,
            "compounds": rp.get("compounds") or [],
            "has_bond": bool(bond or matched),
            "source": "engine",
        })

    # Out-of-domain: the scan's unique contribution.
    for p in scan_peaks_full_range(canvas, min_height=min_height,
                                   min_prominence=min_prominence):
        if lo <= p["position"] <= hi:
            continue
        p.update({
            "intensity": obs_at(p["position"]),
            "bond": None, "matched_to": None, "compounds": [],
            "has_bond": False, "source": "scan",
        })
        peaks.append(p)

    return sorted(peaks, key=lambda x: x["position"])


def plot_measured_peaks(canvas: IngestResult,
                        result_peaks: list[dict] | None = None,
                        min_height: float = 0.05,
                        min_prominence: float = 0.03,
                        save_path=None):
    """Annotated peak figure over the measured region.

    Uses :func:`build_combined_peaks`, so in-domain peaks match the engine's
    ``result["peaks"]`` (and therefore the novelty / OOD figure), while
    out-of-domain peaks come from the wide scan.

    Colour key:
      * green   -- in-domain peak matched to a bond (label shows the bond),
      * grey    -- in-domain peak with no bond match (these are the "novel"
        peaks the OOD figure flags),
      * orange  -- peak outside the fingerprint / model domain.

    Args:
        canvas: Wide-canvas ingest result.
        result_peaks: ``result["peaks"]`` (engine detector + bonds).
        min_height: Relative height threshold (out-of-domain scan).
        min_prominence: Relative prominence threshold (out-of-domain scan).
        save_path: If given, save the figure there.

    Returns:
        A matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    mwn, raw = measured_slice(canvas)
    obs = preprocess_measured(raw)
    peaks = build_combined_peaks(canvas, result_peaks,
                                 min_height=min_height,
                                 min_prominence=min_prominence)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(mwn, obs, lw=1.0, color="steelblue", zorder=1)
    ax.axvspan(267, 2004, color="seagreen", alpha=0.08)

    n_matched = sum(p["has_bond"] for p in peaks)
    n_in = sum(p["assignable"] for p in peaks)
    n_out = len(peaks) - n_in
    for p in peaks:
        if not p["assignable"]:
            color, label = "darkorange", f"{p['position']:.0f}\n(outside fp)"
        elif p["has_bond"]:
            tag = p["bond"] or p["matched_to"]
            color, label = "seagreen", f"{p['position']:.0f}\n{tag}"
        else:
            color, label = "0.45", f"{p['position']:.0f}"
        ax.axvline(p["position"], color=color, alpha=0.55, lw=0.9, zorder=0)
        ax.annotate(label, xy=(p["position"], p["intensity"]),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5, color=color)

    ax.set_xlim(canvas.measured_lo, canvas.measured_hi)
    ax.set_ylim(top=1.25)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity (preprocessed)")
    ax.set_title(f"Peak annotations - measured region "
                 f"({len(peaks)}: {n_matched} bond-matched, "
                 f"{n_in - n_matched} in-domain unmatched, "
                 f"{n_out} outside fingerprint)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# --- Measured-region OOD figure (single source of truth) ---------------------

def _cluster_peaks(peaks: list[dict], gap_cm: float = 30.0) -> list[list[dict]]:
    """Single-link cluster peaks by position (matches NoveltyLocator's rule)."""
    if not peaks:
        return []
    items = sorted(peaks, key=lambda d: d["position"])
    clusters: list[list[dict]] = [[items[0]]]
    for p in items[1:]:
        if p["position"] - clusters[-1][-1]["position"] <= gap_cm:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return clusters


def _draw_ood_gauge(ax, score: float, threshold: float, is_ood: bool) -> None:
    """Horizontal OOD-score gauge (mirrors src/inference/visualize.py)."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    safe_hi = max(0.0, threshold - 0.1)
    ax.axvspan(0, safe_hi, color="lightgreen", alpha=0.5)
    ax.axvspan(safe_hi, threshold, color="navajowhite", alpha=0.7)
    ax.axvspan(threshold, 1.0, color="lightcoral", alpha=0.5)
    sc = "crimson" if is_ood else "darkgreen"
    ax.axvline(score, color=sc, lw=3.5)
    ax.text(score, 0.5, f"  {score:.3f}", va="center", ha="left",
            fontsize=10, fontweight="bold", color=sc)
    ax.axvline(threshold, color="black", lw=1.0, ls="--")
    ax.text(threshold, 1.02, f"thresh {threshold:.3f}", ha="center",
            va="bottom", fontsize=8)
    ax.text(0.01, 0.5, "OOD" if is_ood else "ID", ha="left", va="center",
            fontsize=11, fontweight="bold", color=sc,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=sc))
    ax.set_yticks([]); ax.set_xlabel("OOD score")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.tick_params(axis="x", labelsize=8)


def plot_measured_ood(canvas: IngestResult,
                      result: dict,
                      min_height: float = 0.05,
                      min_prominence: float = 0.03,
                      cluster_gap_cm: float = 30.0,
                      save_path=None):
    """OOD gauge + measured-region spectrum with novelty regions shaded.

    Unlike the engine's ``plot_ood_summary`` (drawn on the model grid from the
    engine's own clusters), this is zoomed to the measured region and shades the
    SAME peak set the peak figure uses (:func:`build_combined_peaks`): in-domain
    peaks with no bond match (crimson) plus out-of-domain peaks (orange). The
    gauge still shows the engine's model-domain OOD score unchanged.

    Args:
        canvas: Wide-canvas ingest result.
        result: predict() output (uses ``ood_score`` / ``ood_threshold`` /
            ``is_ood`` and ``peaks``).
        min_height: Relative height threshold (out-of-domain scan).
        min_prominence: Relative prominence threshold (out-of-domain scan).
        cluster_gap_cm: Single-link clustering distance for novelty regions.
        save_path: If given, save the figure there.

    Returns:
        A matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    mwn, raw = measured_slice(canvas)
    obs = preprocess_measured(raw)
    peaks = build_combined_peaks(canvas, result.get("peaks"),
                                 min_height=min_height,
                                 min_prominence=min_prominence)
    novel = [p for p in peaks if not p["has_bond"]]
    clusters = _cluster_peaks(novel, gap_cm=cluster_gap_cm)

    score = result.get("ood_score")
    threshold = result.get("ood_threshold")
    is_ood = result.get("is_ood")
    has_gauge = score is not None and threshold is not None

    if has_gauge:
        fig, (ax_g, ax) = plt.subplots(
            2, 1, figsize=(11, 5.2), gridspec_kw={"height_ratios": [1, 4]})
        _draw_ood_gauge(ax_g, float(score), float(threshold), bool(is_ood))
    else:
        fig, ax = plt.subplots(figsize=(11, 4.3))

    ax.plot(mwn, obs, color="steelblue", lw=1.0, label="input spectrum")
    ax.axvspan(267, 2004, color="seagreen", alpha=0.06)
    ymax = float(obs.max()) if obs.size and obs.max() > 0 else 1.0
    for grp in clusters:
        lo = min(m["position"] for m in grp)
        hi = max(m["position"] for m in grp)
        out_only = all(not m["assignable"] for m in grp)
        color = "darkorange" if out_only else "crimson"
        ax.axvspan(lo - 10, hi + 10, color=color, alpha=0.15)
        ax.annotate(f"novel: {len(grp)} pk",
                    xy=(0.5 * (lo + hi), 0.95 * ymax),
                    ha="center", fontsize=8, color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=color, alpha=0.85))

    ax.set_xlim(canvas.measured_lo, canvas.measured_hi)
    ax.set_xlabel("Raman shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity (preprocessed)")
    verdict = ("OUT-OF-DISTRIBUTION" if is_ood else "IN-DISTRIBUTION") \
        if is_ood is not None else "novelty"
    ax.set_title(f"OOD assessment - measured region  --  {verdict}  "
                 f"({len(clusters)} novel cluster(s); crimson=in-domain "
                 f"unmatched, orange=outside model)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig