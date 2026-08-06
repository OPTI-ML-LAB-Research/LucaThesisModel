"""Classical preprocessing pipeline for Raman spectra.

The pipeline is **fixed** (not learned) and consists of four independent steps,
applied in this order:

    1. Cosmic ray removal       — eliminate sharp narrow spikes
    2. Baseline correction      — remove fluorescence / drift (Asymmetric Least Squares)
    3. Savitzky-Golay smoothing — reduce shot noise while preserving peak shape
    4. SNV normalization        — Standard Normal Variate, zero-mean / unit-variance

Each step is exposed as a standalone function so it can be ablated, swapped,
or used in isolation (e.g. for Day-2/3 sanity plots). The convenience function
`preprocess_pipeline()` runs all four in sequence and accepts an
`is_preprocessed` flag to skip everything when the spectrum has already been
processed upstream.

The pipeline is **deterministic** — same input always gives the same output.
This is critical for reproducibility and for the train/inference parity rule
(must apply the SAME preprocessing at training and at test time).

Author: Day-2 sprint (T04).
References:
    - Eilers & Boelens (2005), "Baseline correction with asymmetric least squares smoothing"
    - Savitzky & Golay (1964), "Smoothing and Differentiation of Data by Simplified Least Squares"
    - Whitaker & Hayes (2018), "A simple algorithm for despiking Raman spectra"
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from scipy.signal import savgol_filter
from scipy.sparse import diags, eye as speye
from scipy.sparse.linalg import spsolve

log = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
#  Step 1 — Cosmic ray removal
# ───────────────────────────────────────────────────────────────────────────

def remove_cosmic_rays(
    spectrum: np.ndarray,
    *,
    threshold: float = 5.0,
    half_window: int = 2,
) -> np.ndarray:
    """Remove cosmic-ray spikes via 1st-difference detection.

    Method: compute the first differences `d_i = s_{i+1} − s_i`. Any |d| greater
    than `median(|d|) + threshold * std(d)` flags a sharp transition. The
    affected sample is replaced by the average of its non-spike neighbours
    (within ±`half_window`).

    Parameters
    ----------
    spectrum : (P,) float array
        Single Raman spectrum.
    threshold : float, default 5.0
        Standard-deviation multiplier above the median absolute difference.
        Higher → fewer detections.
    half_window : int, default 2
        Window radius used to locate the replacement neighbours.

    Returns
    -------
    (P,) float array — copy with spikes replaced.

    Notes
    -----
    Conservative by design: a 5σ threshold is enough to catch true cosmic rays
    (which are typically 10×+ above the noise floor) while leaving real Raman
    peaks intact. For very clean spectrometers, raise to 7-10.
    """
    if spectrum.ndim != 1:
        raise ValueError(f"Expected 1D spectrum, got shape {spectrum.shape}.")
    s = spectrum.astype(np.float64, copy=True)
    n = s.size
    d = np.diff(s)
    med = float(np.median(np.abs(d)))
    sd = float(np.std(d))
    cutoff = med + threshold * sd
    spike_idx = np.where(np.abs(d) > cutoff)[0]

    for i in spike_idx:
        lo = max(0, i - half_window)
        hi = min(n - 1, i + half_window + 1)
        # Average of *non-spike* neighbours; fall back to nearest valid samples
        s[i] = 0.5 * (s[lo] + s[hi - 1])
    return s.astype(spectrum.dtype, copy=False)


# ───────────────────────────────────────────────────────────────────────────
#  Step 2 — Asymmetric Least Squares baseline correction
# ───────────────────────────────────────────────────────────────────────────

def asls_baseline(
    spectrum: np.ndarray,
    *,
    lam: float = 1.0e5,
    p: float = 0.01,
    max_iter: int = 30,
    tol: float = 1.0e-6,
) -> np.ndarray:
    """Estimate the baseline using Asymmetric Least Squares (Eilers 2005).

    Solves iteratively:
        argmin_z  Σ w_i (s_i − z_i)² + λ Σ (Δ²z)²
    where weights w_i are p when s_i > z_i (peak side) and (1-p) otherwise
    (baseline side). λ controls smoothness; p controls asymmetry.

    Parameters
    ----------
    spectrum : (P,) float array
    lam : float, default 1e5
        Smoothness — larger = stiffer baseline. Typical 1e3 (rough) to 1e7 (smooth).
    p : float, default 0.01
        Asymmetry — peak weight. 0.001 (very asymmetric, all peaks above) to
        0.1 (less aggressive). 0.01 is the standard value.
    max_iter : int, default 30
        Hard cap on iterations.
    tol : float, default 1e-6
        Stop when the L2 change between successive `z` estimates falls below this.

    Returns
    -------
    (P,) float array — estimated baseline, same dtype as input.

    Notes
    -----
    To get the baseline-corrected spectrum, subtract the returned baseline
    from the original spectrum (see `apply_asls_correction`).
    """
    if spectrum.ndim != 1:
        raise ValueError(f"Expected 1D spectrum, got shape {spectrum.shape}.")
    s = spectrum.astype(np.float64, copy=False)
    L = s.size

    # 2nd-order difference matrix D (shape (L-2, L)); use sparse for efficiency
    D = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(L - 2, L), dtype=np.float64)
    DtD = lam * (D.T @ D)

    w = np.ones(L)
    z_prev = None
    for _ in range(max_iter):
        W = diags(w, 0)
        Z = (W + DtD).tocsc()
        z = spsolve(Z, w * s)
        if z_prev is not None and np.linalg.norm(z - z_prev) / max(np.linalg.norm(z_prev), 1e-12) < tol:
            break
        z_prev = z
        w = p * (s > z) + (1 - p) * (s <= z)

    return z.astype(spectrum.dtype, copy=False)


def apply_asls_correction(
    spectrum: np.ndarray,
    *,
    lam: float = 1.0e5,
    p: float = 0.01,
    max_iter: int = 30,
    clip_negative: bool = True,
) -> np.ndarray:
    """Subtract the AsLS baseline from a spectrum.

    Parameters
    ----------
    clip_negative : bool, default True
        If True, set any post-correction negative values to 0. Recommended for
        Raman where intensity is non-negative; disable only for diagnostics.
    """
    baseline = asls_baseline(spectrum, lam=lam, p=p, max_iter=max_iter)
    corrected = spectrum.astype(np.float64, copy=False) - baseline
    if clip_negative:
        corrected = np.maximum(corrected, 0.0)
    return corrected.astype(spectrum.dtype, copy=False)


# ───────────────────────────────────────────────────────────────────────────
#  Step 3 — Savitzky-Golay smoothing
# ───────────────────────────────────────────────────────────────────────────

def savitzky_golay(
    spectrum: np.ndarray,
    *,
    window: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """Savitzky-Golay smoothing (local polynomial least-squares fit).

    Parameters
    ----------
    window : int, default 11
        Window length (must be odd and ≥ polyorder+2).
    polyorder : int, default 3
        Polynomial order used inside each window. Common Raman defaults: 2-4.

    Returns
    -------
    (P,) float array — smoothed spectrum.
    """
    if window % 2 == 0:
        raise ValueError(f"Savitzky-Golay window must be odd, got {window}.")
    if polyorder >= window:
        raise ValueError(f"polyorder ({polyorder}) must be < window ({window}).")
    if spectrum.size < window:
        raise ValueError(f"Spectrum length {spectrum.size} < window {window}.")
    return savgol_filter(spectrum, window_length=window, polyorder=polyorder).astype(
        spectrum.dtype, copy=False
    )


# ───────────────────────────────────────────────────────────────────────────
#  Step 4 — SNV normalization
# ───────────────────────────────────────────────────────────────────────────

def snv_normalize(spectrum: np.ndarray, *, eps: float = 1.0e-8) -> np.ndarray:
    """Standard Normal Variate normalization: subtract mean, divide by std.

    SNV makes spectra of equal *shape* numerically identical regardless of
    overall intensity — essential when sample concentration or laser power
    varies between measurements.

    Parameters
    ----------
    eps : float, default 1e-8
        Numerical floor on the std to avoid division by zero on flat spectra.

    Returns
    -------
    (P,) float array — zero-mean, ~unit-variance spectrum.
    """
    s = spectrum.astype(np.float64, copy=False)
    m = s.mean()
    sd = s.std()
    if sd < eps:
        log.warning("snv_normalize: std < eps; returning mean-centred spectrum only.")
        return (s - m).astype(spectrum.dtype, copy=False)
    return ((s - m) / sd).astype(spectrum.dtype, copy=False)


# ───────────────────────────────────────────────────────────────────────────
#  Composite pipeline
# ───────────────────────────────────────────────────────────────────────────

def preprocess_pipeline(
    spectrum: np.ndarray,
    *,
    is_preprocessed: bool = False,
    cosmic_threshold: float = 5.0,
    asls_lam: float = 1.0e5,
    asls_p: float = 0.01,
    asls_max_iter: int = 30,
    savgol_window: int = 11,
    savgol_polyorder: int = 3,
    snv_eps: float = 1.0e-8,
) -> np.ndarray:
    """Apply the full classical pipeline: cosmic → AsLS → SG → SNV.

    Parameters
    ----------
    spectrum : (P,) float array
        Raw Raman spectrum.
    is_preprocessed : bool, default False
        If True, return the input unchanged (used when the caller has already
        done preprocessing upstream — e.g. cached tensors).
    cosmic_threshold, asls_lam, asls_p, savgol_window, savgol_polyorder, snv_eps
        Pass-through hyperparameters, see individual function docstrings.

    Returns
    -------
    (P,) float array — preprocessed spectrum.
    """
    if is_preprocessed:
        return np.asarray(spectrum)
    s = remove_cosmic_rays(spectrum, threshold=cosmic_threshold)
    s = apply_asls_correction(s, lam=asls_lam, p=asls_p, max_iter=asls_max_iter, clip_negative=True)
    s = savitzky_golay(s, window=savgol_window, polyorder=savgol_polyorder)
    s = snv_normalize(s, eps=snv_eps)
    return s


def preprocess_batch(
    spectra: np.ndarray,
    *,
    is_preprocessed: bool = False,
    **pipeline_kwargs,
) -> np.ndarray:
    """Apply `preprocess_pipeline` to a batch of spectra.

    Parameters
    ----------
    spectra : (N, P) float array
    is_preprocessed : bool, default False
        Pass-through to `preprocess_pipeline`.
    **pipeline_kwargs
        Forwarded to `preprocess_pipeline` for each row.

    Returns
    -------
    (N, P) float array — preprocessed batch with the input dtype preserved.
    """
    if spectra.ndim != 2:
        raise ValueError(f"Expected 2D batch, got shape {spectra.shape}.")
    if is_preprocessed:
        return spectra.copy()
    out = np.empty_like(spectra)
    for i in range(spectra.shape[0]):
        out[i] = preprocess_pipeline(spectra[i], is_preprocessed=False, **pipeline_kwargs)
    return out


# ───────────────────────────────────────────────────────────────────────────
#  Factory: build a callable preprocess function from a config dict
# ───────────────────────────────────────────────────────────────────────────

def make_preprocess_fn(data_config: dict) -> Callable[[np.ndarray], np.ndarray]:
    """Build a preprocess(spectrum) → spectrum closure from `data_config.yaml`.

    Reads `data_config['preprocessing']` and returns a function suitable for
    passing as the `preprocess=` argument of `RamanDataset`.

    If `data_config['preprocessing']['apply_on_the_fly'] == False`, returns None
    (caller should not apply preprocessing — assume tensors are already cached).
    """
    pp = data_config.get("preprocessing", {})
    if not pp.get("apply_on_the_fly", False):
        return None  # type: ignore[return-value]

    kwargs = dict(
        cosmic_threshold=pp.get("cosmic_threshold", 5.0),
        asls_lam=float(pp.get("asls_lam", 1.0e5)),
        asls_p=float(pp.get("asls_p", 0.01)),
        asls_max_iter=int(pp.get("asls_max_iter", 30)),
        savgol_window=int(pp.get("savgol_window", 11)),
        savgol_polyorder=int(pp.get("savgol_polyorder", 3)),
        snv_eps=float(pp.get("snv_eps", 1.0e-8)),
    )
    def _fn(spectrum: np.ndarray) -> np.ndarray:
        return preprocess_pipeline(spectrum, is_preprocessed=False, **kwargs)
    return _fn


# ───────────────────────────────────────────────────────────────────────────
#  Sanity plot helper (for Day-2 visual verification)
# ───────────────────────────────────────────────────────────────────────────

def plot_preprocessing_steps(
    spectra: np.ndarray,
    wavenumbers: np.ndarray,
    output_path: str,
    *,
    titles: list[str] | None = None,
    show_intermediate: bool = True,
) -> None:
    """Plot raw vs. each preprocessing stage for a small set of spectra.

    Parameters
    ----------
    spectra : (N, P) array
        Raw spectra (typically N=5).
    wavenumbers : (P,) array
        Raman shift axis. The plot will flip if the axis is descending so that
        plots read left-to-right in standard cm⁻¹ orientation.
    output_path : str
        Where to save the PNG. Parent directory is created if needed.
    titles : list[str], optional
        Per-spectrum labels (e.g. vial ids). Falls back to "spectrum #i".
    show_intermediate : bool, default True
        If True, plot 5 columns: raw → cosmic-removed → baseline-corrected →
        SG-smoothed → SNV. If False, only plot raw vs. final.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = spectra.shape[0]
    if titles is None:
        titles = [f"spectrum #{i}" for i in range(n)]
    elif len(titles) != n:
        raise ValueError(f"titles length {len(titles)} != n_spectra {n}.")

    # Order x-axis ascending for human readability
    if wavenumbers[0] > wavenumbers[-1]:
        order = np.argsort(wavenumbers)
        wn = wavenumbers[order]
        spectra = spectra[:, order]
    else:
        wn = wavenumbers

    if show_intermediate:
        ncols = 5
        col_titles = [
            "1. Raw",
            "2. Cosmic-removed",
            "3. Baseline-corrected (AsLS)",
            "4. SG-smoothed",
            "5. SNV-normalized (final)",
        ]
    else:
        ncols = 2
        col_titles = ["Raw", "Final (after pipeline)"]

    fig, axes = plt.subplots(n, ncols, figsize=(ncols * 3.4, 2.0 * n + 0.6), sharex=True)
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        s_raw = spectra[i]
        if show_intermediate:
            s1 = remove_cosmic_rays(s_raw, threshold=5.0)
            s2 = apply_asls_correction(s1, lam=1e5, p=0.01)
            s3 = savitzky_golay(s2, window=11, polyorder=3)
            s4 = snv_normalize(s3)
            stages = [s_raw, s1, s2, s3, s4]
        else:
            s_final = preprocess_pipeline(s_raw)
            stages = [s_raw, s_final]

        for j, s in enumerate(stages):
            ax = axes[i, j]
            ax.plot(wn, s, linewidth=0.7, color="steelblue")
            ax.set_xlim(wn.min(), wn.max())
            if i == 0:
                ax.set_title(col_titles[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(titles[i], fontsize=8)
            if i == n - 1:
                ax.set_xlabel("Raman shift (cm⁻¹)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)

    fig.suptitle("Classical preprocessing pipeline — per-stage view", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved sanity plot → {output_path}")
