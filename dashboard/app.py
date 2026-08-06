"""Streamlit dashboard for the Raman Physics-Informed AI MVP.

Thesis-defense demo edition. The live demo runs the AA-only (6-output)
model on three stories — the strongest, cleanest OOD narrative:

    ID     — an Amino-Acid test spectrum    -> reconstructs well, ID
    OOD-1  — an AAM mineral-rich spectrum    -> physics flags it OOD
    OOD-2  — a real Beryl mineral spectrum   -> far-domain OOD (replaces MoS2)

The "after retrain, minerals become ID (MAE 0.0126)" result is shown
separately in the report via the static MAE table + composition_scatter.png
(it does not need to run live, and predict() is locked to the 6-AA order).

Fixes vs the original app.py:
  1. ``_parse_uploaded`` handles RAW spectra (2-column wavenumber/intensity on
     ANY grid) by resampling onto the model's 1024-pt grid + running the SAME
     preprocessing as training (AsLS + cosmic + SG + SNV).
  2. NEW robust text reader ``_read_spectrum_text`` auto-detects the delimiter
     (comma / tab / whitespace), the transposed 2-row layout, and a descending
     wavenumber axis. This fixes the "1600 single-column values" error on the
     comma-delimited Beryl files and makes arbitrary uploads (cells, viruses,
     minerals) run end-to-end.
  3. OOD-2 preset is now Beryl (RRUFF). With 45 beryl spectra available, the
     preset picks a RANDOM file each click (like AA/AAM) -> demonstrates the
     model is stable on the far-OOD case, not tuned to one spectrum.
  4. Preset results are cached so re-clicking is instant during the demo.

Run from the project root:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    raise SystemExit("streamlit is not installed. Install: pip install streamlit")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.inference.predict import predict
from src.inference.report import generate_report
from src.inference.visualize import (
    plot_reconstruction_overlay,
    plot_peak_annotations,
    plot_ood_summary,
)

# OOD re-score controls (đặt cạnh app.py trong dashboard/)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ood_controls import CalStats, rescore, SCALES

# Wide-range ingestion (0..4000 cm-1 canvas) + full-range peak scan.
# build_canvas: src/data/ingest.py ; scan_peaks_full_range: dashboard/wide_range.py
from src.data.ingest import build_canvas
from wide_range import (
    scan_peaks_full_range,
    build_combined_peaks,
    plot_measured_reconstruction,
    plot_measured_peaks,
    plot_measured_ood,
)

COMPOUND_ORDER = [
    "Alanine", "Asparagine", "Aspartic Acid",
    "Glutamic Acid", "Histidine", "Glucosamine",
]

MODEL_PATH = "checkpoints/best.pt"        # AA-only (6 outputs)
WN_PATH = "data/processed/wavenumbers.npy"
REF_PATH = "engine/reference_spectra.npy"  # (6, 1024) pure refs (recon fallback)
BERYL_DIR = "data/raw/ood_demo/beryl"     # folder of Beryl_01 (N).txt files (N = 1..45)


# =============================================================================
# Robust spectrum text reader
# =============================================================================

def _read_spectrum_text(text: str):
    """Parse a Raman spectrum from text in a delimiter/orientation-robust way.

    Handles the heterogeneous formats across demo datasets:
      * 2 columns [wavenumber, intensity]  (comma / tab / whitespace delimited)
      * transposed 2-row layout (row 0 = wavenumbers, row 1 = intensities)
      * descending wavenumber axis (left as-is; downstream resampler sorts it)
      * a bare single-column intensity vector (already preprocessed)
      * a stray header line (skipped automatically)

    Args:
        text: Raw file contents as a decoded string.

    Returns:
        ``(wavenumber, intensity)`` as float arrays, or ``(None, intensity)``
        when the file is a single-column vector (caller resamples by index).

    Raises:
        ValueError: if no numeric data could be parsed.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("empty file")

    # Transposed 2-row layout (e.g. coronavirus files): two long whitespace rows.
    if len(lines) == 2 and len(lines[0].split()) > 4:
        return (np.asarray(lines[0].split(), dtype=np.float64),
                np.asarray(lines[1].split(), dtype=np.float64))

    # Column layout: detect delimiter from a representative middle line.
    sample = lines[len(lines) // 2]
    delim = "," if "," in sample else ("\t" if "\t" in sample else None)

    wn, inten, single = [], [], []
    for ln in lines:
        try:
            vals = [float(x) for x in ln.split(delim)]
        except ValueError:
            continue                      # skip headers / non-numeric lines
        if len(vals) >= 2:
            wn.append(vals[0])
            inten.append(vals[1])
        elif len(vals) == 1:
            single.append(vals[0])

    if wn:
        return np.asarray(wn, dtype=np.float64), np.asarray(inten, dtype=np.float64)
    if single:
        return None, np.asarray(single, dtype=np.float64)
    raise ValueError("no numeric data parsed")


# =============================================================================
# Raw-spectrum preprocessing (mirror of t29a_prepare_aam.py)
# =============================================================================

def _resample_and_preprocess_raw(wn_src: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """Resample a raw (wavenumber, intensity) spectrum onto the model's
    1024-pt grid and apply the training preprocessing pipeline.

    Mirrors t29a_prepare_aam.py: linear interp onto
    data/processed/wavenumbers.npy (ascending), then preprocess_batch
    (cosmic + AsLS + SG + SNV), then nan_to_num.
    """
    from src.data.preprocess import preprocess_batch

    target_wn = np.load(WN_PATH).astype(np.float64)
    flipped = target_wn[0] > target_wn[-1]
    target_asc = target_wn[::-1] if flipped else target_wn

    wn_src = np.asarray(wn_src, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if wn_src[0] > wn_src[-1]:                     # ensure ascending source
        wn_src = wn_src[::-1]
        intensity = intensity[::-1]

    v = np.interp(target_asc, wn_src, intensity, left=0.0, right=0.0)
    if flipped:
        v = v[::-1]

    Xp = preprocess_batch(v[None, :].astype(np.float32)).astype(np.float32)
    Xp = np.nan_to_num(Xp, nan=0.0, posinf=0.0, neginf=0.0)
    return Xp[0]


def _parse_uploaded(upl):
    """Parse an uploaded file into a 1024-pt preprocessed spectrum.

    Accepts:
      * .txt / .csv with 2 columns [wavenumber, intensity] on ANY grid, with
        comma / tab / whitespace delimiters, or a transposed 2-row layout
        -> resampled + preprocessed (path for beryl, cells, viruses, any
        out-of-domain spectrum).
      * .npy / single-column with exactly 1024 values -> used as-is
        (assumed already preprocessed).

    Returns (spectrum, canvas):
      * spectrum -- (1024,) float array for the model, or None on failure.
      * canvas   -- an ``IngestResult`` (wide 0..4000 cm-1 grid) for raw
        uploads, used for display + full-range peak scan; None for
        already-preprocessed 1024-pt inputs.
    """
    try:
        name = upl.name.lower()
        raw = upl.getvalue()

        if name.endswith(".npy"):
            data = np.asarray(np.load(io.BytesIO(raw)), dtype=np.float64)
            if data.ndim == 2 and data.shape[1] >= 2:
                wn_src, inten = data[:, 0], data[:, 1]
                st.info(
                    f"Detected a raw 2-column spectrum ({data.shape[0]} points, "
                    f"{wn_src.min():.0f}-{wn_src.max():.0f} cm-1). "
                    f"Resampling onto the 1024-pt model grid + preprocessing."
                )
                return (_resample_and_preprocess_raw(wn_src, inten),
                        build_canvas(wn_src, inten))
            vec = data.flatten()
            if vec.size == 1024:
                return vec.astype(np.float32), None
            st.error(f"{name}: got {vec.size} values, expected 1024 preprocessed.")
            return None, None

        # .txt / .csv -> robust text parser (comma/tab/space, 2-col or 2-row).
        wn_src, inten = _read_spectrum_text(raw.decode("utf-8", errors="ignore"))

        if wn_src is not None:                     # 2-column raw spectrum
            st.info(
                f"Detected a raw 2-column spectrum ({inten.size} points, "
                f"{wn_src.min():.0f}-{wn_src.max():.0f} cm-1). "
                f"Resampling onto the 1024-pt model grid + preprocessing."
            )
            return (_resample_and_preprocess_raw(wn_src, inten),
                    build_canvas(wn_src, inten))

        vec = inten.flatten()                      # single-column vector
        if vec.size == 1024:
            return vec.astype(np.float32), None
        st.error(
            f"File has {vec.size} single-column values (expected 1024 "
            f"preprocessed, or a 2-column raw spectrum). Provide a "
            f"2-column [wavenumber intensity] file to auto-resample."
        )
        return None, None
    except Exception as e:   # noqa: BLE001
        st.error(f"Could not parse file: {e}")
        return None, None


# =============================================================================
# Cached resources & preset loaders
# =============================================================================

@st.cache_data(show_spinner=False)
def load_aa_cache():
    import torch
    wn = np.load(WN_PATH)
    spectra = torch.load("data/processed/spectra_full.pt", weights_only=True).numpy()
    labels = torch.load("data/processed/labels.pt", weights_only=True).numpy()
    vials = np.load("data/processed/vial_ids.npy", allow_pickle=True)
    split_path = Path("data/splits/split_A_composition_ood.json")
    test_idx = (json.loads(split_path.read_text(encoding="utf-8"))["test"]
                if split_path.exists() else list(range(len(vials))))
    return {"wn": wn, "spectra": spectra, "labels": labels,
            "vials": vials, "test_idx": test_idx}


@st.cache_resource(show_spinner=False)
def load_aam_cache():
    """AAM preprocessed cache (mineral-rich OOD examples)."""
    import torch
    p = Path("data/processed/aam/spectra.pt")
    if not p.exists():
        return None
    spectra = torch.load(p, weights_only=True).numpy()
    if spectra.ndim == 3:
        spectra = spectra[:, 0, :]
    labels7 = torch.load("data/processed/aam/labels_7d.pt", weights_only=True).numpy()
    split = json.loads(Path("data/processed/aam/split.json").read_text(encoding="utf-8"))
    return {"spectra": spectra, "labels7": labels7, "test_idx": split["test"]}


@st.cache_data(show_spinner=False)
def load_beryl_files(dir_path: str):
    """Return the sorted list of beryl .txt spectrum paths (empty if missing)."""
    d = Path(dir_path)
    if not d.exists():
        return []
    return sorted(str(p) for p in d.glob("*.txt"))


@st.cache_data(show_spinner=False)
def load_beryl_preset(path: str):
    """Load + resample + preprocess one Beryl raw spectrum (cached per path).

    The beryl files are comma-delimited 2-column [wavenumber, intensity] with
    the wavenumber column = integer cm-1 (0..1599). The robust reader handles
    the comma delimiter; the resampler maps onto the model grid (same code
    path the MoS2 preset used).
    """
    p = Path(path)
    if not p.exists():
        return None, None
    try:
        wn, inten = _read_spectrum_text(p.read_text(encoding="utf-8", errors="ignore"))
    except ValueError:
        return None, None
    if wn is None:
        return None, None
    return _resample_and_preprocess_raw(wn, inten), build_canvas(wn, inten)


@st.cache_resource
def warm_check():
    needed = [MODEL_PATH, "engine/bond_mapping.json", WN_PATH]
    missing = [p for p in needed if not Path(p).exists()]
    return {"missing": missing}


@st.cache_data(show_spinner=False)
def cached_predict(spectrum_bytes: bytes, shape: tuple, n_mc: int, skip_ood: bool):
    """Cache predict() keyed on spectrum content + settings -> instant re-click."""
    spec = np.frombuffer(spectrum_bytes, dtype=np.float32).reshape(shape)
    return predict(spec, model_path=MODEL_PATH, n_mc_samples=n_mc, skip_ood=skip_ood)


def run_predict(spectrum, n_mc, skip_ood):
    spec = np.ascontiguousarray(spectrum, dtype=np.float32)
    return cached_predict(spec.tobytes(), spec.shape, n_mc, skip_ood)


@st.cache_data(show_spinner=False)
def load_calstats():
    """Đọc recon_p95 / var_p95 / score_p95 cho phần OOD re-score.

    Ưu tiên recommended.json (sinh bởi run_t32_threshold.py), fallback
    calibration.json gốc của repo. Trả về (CalStats | None, f1_threshold | None).
    """
    rec = Path("results/stretch/t32_threshold/recommended.json")
    cal = Path("results/ood_demo/calibration.json")
    if rec.exists():
        d = json.loads(rec.read_text(encoding="utf-8"))
        c = d["calibration"]
        return CalStats(c["recon_p95"], c["var_p95"], c["score_p95"]), d.get("threshold")
    if cal.exists():
        c = json.loads(cal.read_text())
        rp = c.get("recon_p95") or c.get("recon_err_p95")
        vp = c.get("var_p95") or c.get("variance_p95")
        sp = c.get("score_p95") or c.get("threshold")
        if rp is None or vp is None or sp is None:
            return None, None
        return CalStats(float(rp), float(vp), float(sp)), None
    return None, None


@st.cache_data(show_spinner=False)
def load_reference_spectra():
    """Load (6, 1024) pure references for the reconstruction fallback.

    Returns the array on the model grid, or None if unavailable (in which case
    the reconstruction is taken straight from the result dict).
    """
    p = Path(REF_PATH)
    if not p.exists():
        return None
    try:
        refs = np.load(p)
        if refs.ndim == 2 and refs.shape[0] == len(COMPOUND_ORDER):
            return refs.astype(np.float64)
    except Exception:  # noqa: BLE001
        pass
    return None


# =============================================================================
# Page setup
# =============================================================================

st.set_page_config(page_title="Raman Physics-AI", page_icon="🧪", layout="wide")
st.title("Raman Physics-Informed AI - MVP Demo")
st.caption("Composition - peak attribution - physics validation - OOD assessment, in one pass.")

warm = warm_check()
if warm["missing"]:
    st.error("Missing files: " + ", ".join(warm["missing"]) + ". Run from the project root.")
    st.stop()

# ---- Sidebar ----------------------------------------------------------------
st.sidebar.header("Inference settings")
n_mc = st.sidebar.slider("MC Dropout samples", 10, 100, 30, step=10,
                         help="More = tighter uncertainty, slower forward.")
skip_ood = st.sidebar.checkbox("Skip OOD scoring", value=False,
                               help="Use if results/ood_demo/calibration.json is missing.")
st.sidebar.markdown("---")
st.sidebar.subheader("OOD re-score (live)")
ood_scale = st.sidebar.selectbox("Thang chuẩn hóa", list(SCALES.keys()), index=0,
                                 help="clip = gốc repo; no-clip giữ độ phân giải "
                                      "far-OOD; tanh bão hòa mềm.")
recon_w = st.sidebar.slider("recon_weight (physics)", 0.0, 1.0, 0.6, 0.05,
                            help="1.0 = physics-only (bỏ variance). "
                                 "Theo T29B, physics-only cho AUROC cao nhất.")
var_w = round(1.0 - recon_w, 2)
st.sidebar.caption(f"var_weight = {var_w:.2f}  (= 1 - recon_weight)")
use_f1_thr = st.sidebar.checkbox("Dùng threshold F1 (recommended.json)", value=False,
                                 help="Threshold tối ưu F1 từ run_t32_threshold.py.")
manual_thr = st.sidebar.slider("Threshold thủ công", 0.0, 1.5, 0.92, 0.01,
                               disabled=use_f1_thr)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Model: physics-informed 1D-ResNet, AA-only (6 outputs). "
    "Order: Alanine, Asparagine, Aspartic Acid, Glutamic Acid, Histidine, "
    "Glucosamine. The minerals-retrained result is in the report (static)."
)

# ---- Session state ----------------------------------------------------------
for k in ("current_spectrum", "current_label", "current_gt", "result",
          "current_canvas"):
    if k not in st.session_state:
        st.session_state[k] = None

# =============================================================================
# Tabs
# =============================================================================
tab_choose, tab_analysis, tab_report = st.tabs(
    ["1. Choose spectrum", "2. Analysis", "3. Report"]
)

with tab_choose:
    st.subheader("Demo presets")
    st.write("Three stories for the committee: in-distribution -> mild OOD -> hard OOD.")
    aa = load_aa_cache()
    aam = load_aam_cache()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**ID - Amino Acid**")
        st.caption("Random test spectrum - should be ID")
        if st.button("Load random AA spectrum", use_container_width=True):
            ti = np.asarray(aa["test_idx"])
            idx = int(np.random.default_rng().choice(ti))   # random each click
            st.session_state.current_spectrum = aa["spectra"][idx]
            # Name the dominant compound so the story is still readable.
            dom = COMPOUND_ORDER[int(np.argmax(aa["labels"][idx]))]
            st.session_state.current_label = f"AA test ({dom}-dominant) - row {idx}"
            st.session_state.current_gt = {c: float(aa["labels"][idx, i])
                                           for i, c in enumerate(COMPOUND_ORDER)}
            st.session_state.current_canvas = None   # already on model grid
            st.session_state.result = None

    with c2:
        st.markdown("**OOD-1 - AAM (mineral-rich)**")
        st.caption("6 AA refs can't reconstruct minerals -> OOD")
        if aam is None:
            st.warning("AAM cache not found at data/processed/aam/.")
        elif st.button("Load random AAM spectrum", use_container_width=True):
            ti = np.asarray(aam["test_idx"])
            idx = int(np.random.default_rng().choice(ti))   # random each click
            st.session_state.current_spectrum = aam["spectra"][idx]
            mineral_pct = aam["labels7"][idx, 6] * 100
            st.session_state.current_label = (
                f"AAM mineral-rich ({mineral_pct:.0f}% mineral) - row {idx}")
            st.session_state.current_gt = {c: float(aam["labels7"][idx, i])
                                           for i, c in enumerate(COMPOUND_ORDER)}
            st.session_state.current_canvas = None   # already on model grid
            st.session_state.result = None

    with c3:
        st.markdown("**OOD-2 - Beryl (mineral, hard)**")
        st.caption("Inorganic silicate mineral, far out of domain")
        beryl_files = load_beryl_files(BERYL_DIR)
        if not beryl_files:
            st.warning(f"No beryl .txt files found in {BERYL_DIR}/.")
        elif st.button("Load random Beryl spectrum", use_container_width=True):
            path = str(np.random.default_rng().choice(beryl_files))  # random each click
            spec, canvas = load_beryl_preset(path)
            if spec is None:
                st.warning(f"Could not parse {Path(path).name}.")
            else:
                st.session_state.current_spectrum = spec
                st.session_state.current_canvas = canvas
                st.session_state.current_label = (
                    f"Beryl mineral - {Path(path).name} (hard OOD)")
                st.session_state.current_gt = None
                st.session_state.result = None

    st.markdown("---")
    st.subheader("Or upload your own spectrum")
    st.write("`.txt`/`.csv` with 2 columns `wavenumber intensity` (any grid, any "
             "delimiter - auto-resampled), or a 1024-pt preprocessed "
             "`.npy`/single-column file.")
    upl = st.file_uploader("Upload file", type=["npy", "csv", "txt"])
    if upl is not None:
        spectrum, canvas = _parse_uploaded(upl)
        if spectrum is not None and spectrum.size == 1024:
            st.session_state.current_spectrum = spectrum
            st.session_state.current_canvas = canvas
            st.session_state.current_label = f"Uploaded: {upl.name}"
            st.session_state.current_gt = None
            st.session_state.result = None
            st.success(f"Loaded {upl.name} -> 1024-pt preprocessed spectrum.")

    # Preview + run
    if st.session_state.current_spectrum is not None:
        st.markdown("### Preview")
        canvas = st.session_state.get("current_canvas")
        if canvas is not None:
            # Raw upload/preset: show ONLY the measured region of the wide
            # 0..4000 cm-1 canvas (zero-filled tails are hidden). The shaded
            # band marks the 267..2004 model / bond-DB domain.
            m = canvas.measured_mask
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(canvas.canvas_wn[m], canvas.canvas_intensity[m],
                    lw=1.0, color="steelblue")
            ax.axvspan(267, 2004, color="seagreen", alpha=0.08,
                       label="model domain (267-2004)")
            ax.set_xlabel("Raman shift (cm$^{-1}$)")
            ax.set_ylabel("Intensity (raw, measured region)")
            ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8)
            ax.set_title(st.session_state.current_label)
            st.pyplot(fig); plt.close(fig)
            st.caption(
                f"Axis detected as **{canvas.unit_detected}** | measured "
                f"{canvas.measured_lo:.0f}-{canvas.measured_hi:.0f} cm-1. "
                "The model + composition use only the 267-2004 cm-1 region; "
                "peaks outside it are reported in Tab 2 but not assigned a bond.")
        else:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(aa["wn"], st.session_state.current_spectrum,
                    lw=1.0, color="steelblue")
            ax.set_xlabel("Raman shift (cm$^{-1}$)")
            ax.set_ylabel("Intensity (preprocessed)")
            ax.grid(alpha=0.3); ax.set_title(st.session_state.current_label)
            st.pyplot(fig); plt.close(fig)

        if st.button("Run analysis", type="primary"):
            with st.spinner(f"Running predict() with {n_mc} MC samples..."):
                st.session_state.result = run_predict(
                    st.session_state.current_spectrum, n_mc, skip_ood)
            st.success("Analysis complete - see Tab 2.")


# =============================================================================
# Tab 2 - Analysis
# =============================================================================
with tab_analysis:
    if st.session_state.result is None:
        st.info("Pick a preset (or upload) and click Run analysis in Tab 1.")
    else:
        result = st.session_state.result

        st.subheader("Composition")
        col_pie, col_table = st.columns([1, 1])
        with col_pie:
            fig, ax = plt.subplots(figsize=(5, 5))
            values = [result["composition"][c] for c in COMPOUND_ORDER]
            ax.pie(values, labels=COMPOUND_ORDER, autopct="%1.1f%%", startangle=90,
                   textprops={"fontsize": 9})
            ax.set_title("Predicted composition")
            st.pyplot(fig); plt.close(fig)
        with col_table:
            st.markdown("**Composition with MC-Dropout uncertainty:**")
            rows = []
            for c in COMPOUND_ORDER:
                gt = (st.session_state.current_gt or {}).get(c)
                rows.append({
                    "Compound": c,
                    "Mean": f"{result['composition'][c]:.3f}",
                    "Std": f"+/-{result['composition_std'][c]:.3f}",
                    "Ground truth": f"{gt:.3f}" if gt is not None else "-",
                    "Symbolic vote": f"{result['compound_votes'].get(c, 0):.1f}",
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

        likely = result["likely_compounds_symbolic"]
        if likely:
            st.success(f"**Symbolic head likely_compounds:** {', '.join(likely)}")
        else:
            st.warning("Symbolic head found no compounds above the vote threshold "
                       "(peaks present but not discriminative - typical for OOD).")

        st.subheader("Detected peaks (model domain 267-2004)"
                     if st.session_state.get("current_canvas") is not None
                     else "Detected peaks")
        peak_rows = []
        for p in sorted(result["peaks"], key=lambda x: x["position"]):
            peak_rows.append({
                "Position (cm-1)": f"{p['position']:.1f}",
                "Intensity": f"{p['intensity']:.3f}",
                "FWHM": f"{p['fwhm']:.1f}",
                "Matched to": p["matched_to"] or "-",
                "Bond": p.get("bond") or "-",
                "Compound": ", ".join(p.get("compounds") or []) or "-",
                "Confidence": p.get("match_confidence") or "-",
            })
        st.dataframe(peak_rows, use_container_width=True, hide_index=True)

        # ---- Full measured-range peak scan (incl. out-of-fingerprint) -------
        canvas = st.session_state.get("current_canvas")
        if canvas is not None:
            st.subheader("Full measured-range peak scan")
            st.caption(
                "In-domain peaks (267-2004) come from the engine's detector "
                "(same source as the bond mapping and the OOD/novelty figure); "
                "peaks outside 267-2004 come from the wide scan and are located "
                "but not bond-assigned.")
            wide_peaks = build_combined_peaks(canvas, result.get("peaks"))
            if not wide_peaks:
                st.info("No peaks passed the height/prominence thresholds.")
            else:
                wide_rows = [{
                    "Position (cm-1)": f"{p['position']:.1f}",
                    "Intensity": f"{p['intensity']:.3f}",
                    "FWHM": f"{p['fwhm']:.1f}",
                    "Band": p["band"],
                    "Bond": p.get("bond") or p.get("matched_to") or "-",
                    "Compound": ", ".join(p.get("compounds") or []) or "-",
                    "Status": ("in model domain" if p["assignable"]
                               else "outside fingerprint"),
                    "Note": p["note"] or "-",
                } for p in wide_peaks]
                st.dataframe(wide_rows, use_container_width=True, hide_index=True)
                n_out = sum(not p["assignable"] for p in wide_peaks)
                n_bond = sum(p.get("has_bond") for p in wide_peaks)
                st.caption(f"{n_bond} peak(s) matched to a known bond; "
                           f"{n_out} peak(s) outside the fingerprint / model "
                           "domain (not yet assignable).")

        st.subheader("Physics validation")
        canvas = st.session_state.get("current_canvas")
        if canvas is not None:
            # Raw upload/preset: draw on the MEASURED region, not the model grid.
            # Reconstruction is ~0 outside 267-2004 (the model can't reconstruct
            # there) -> honest physics-mismatch signal for far-OOD spectra.
            model_wn = np.load(WN_PATH)
            fig_recon = plot_measured_reconstruction(
                canvas, result, model_wn,
                reference_spectra=load_reference_spectra(),
                compound_order=COMPOUND_ORDER, save_path=None)
            st.pyplot(fig_recon); plt.close(fig_recon)

            st.subheader("Peak annotations")
            fig_peaks = plot_measured_peaks(
                canvas, result_peaks=result.get("peaks"), save_path=None)
            st.pyplot(fig_peaks); plt.close(fig_peaks)
        else:
            fig_recon = plot_reconstruction_overlay(result, save_path=None)
            st.pyplot(fig_recon); plt.close(fig_recon)

            st.subheader("Peak annotations")
            fig_peaks = plot_peak_annotations(result, save_path=None)
            st.pyplot(fig_peaks); plt.close(fig_peaks)

        st.subheader("OOD re-score (tùy chỉnh trọng số / threshold)")
        calstats, f1_thr = load_calstats()
        if calstats is None:
            st.warning("Chưa có calibration. Chạy "
                       "`python scripts/stretch/run_t32_threshold.py --n-id 973` "
                       "hoặc `scripts/run_phase3_stretch.py` trước.")
        else:
            cos = result.get("recon_cosine_sim", result.get("recon_cosine"))
            recon_raw = result.get("recon_err_raw")
            if recon_raw is None and cos is not None:
                recon_raw = 1.0 - float(cos)          # = 1 - cosine
            var_raw = (result.get("var_raw") or result.get("mean_compound_std")
                       or result.get("predictive_variance"))
            if var_raw is None:
                std = result.get("composition_std")
                if isinstance(std, dict) and std:
                    var_raw = float(np.mean(list(std.values())))

            thr = (f1_thr if (use_f1_thr and f1_thr is not None) else manual_thr)
            if recon_raw is None:
                st.error("result thiếu recon_cosine -> không rescore được.")
            else:
                if var_raw is None and var_w > 0:
                    st.info("result không có variance thô -> chỉ physics-only đúng "
                            "(đặt recon_weight = 1.0). Tạm coi var = 0.")
                    var_raw = 0.0
                score, is_ood = rescore(recon_raw, var_raw or 0.0, calstats,
                                        recon_w=recon_w, var_w=var_w,
                                        scale=ood_scale, threshold=thr)
                score = float(np.asarray(score).item())
                is_ood = bool(np.asarray(is_ood).item())
                # lưu lại để (tùy chọn) đồng bộ vào hình OOD assessment bên dưới
                st.session_state._rescore = {"score": score, "is_ood": is_ood,
                                             "threshold": float(thr)}
                m1, m2, m3 = st.columns(3)
                m1.metric("OOD score", f"{score:.3f}")
                m2.metric("Threshold", f"{thr:.3f}")
                m3.metric("Verdict", "OOD" if is_ood else "ID",
                          delta=("vượt ngưỡng" if is_ood else "dưới ngưỡng"),
                          delta_color=("inverse" if is_ood else "normal"))
                st.caption(f"score = {recon_w:.2f}·recon_norm + {var_w:.2f}·var_norm "
                           f"| thang: {ood_scale} | "
                           f"threshold: {'F1-optimal' if (use_f1_thr and f1_thr) else 'thủ công'}")

        st.subheader("OOD assessment")
        # Mặc định hình này vẽ verdict GỐC của predict() (threshold cũ ~0.92).
        # Bật ô dưới để đồng bộ hình với verdict re-score ở trên.
        sync = st.checkbox(
            "Đồng bộ hình với verdict re-score ở trên",
            value=True,
            help="Bật: hình dùng score/threshold mới (khớp panel trên). "
                 "Tắt: hình giữ verdict gốc của predict() để so sánh before/after.")
        result_for_plot = result
        rs = st.session_state.get("_rescore")
        if sync and rs is not None:
            # ghi đè 3 field mà plot_ood_summary đọc (decision D6: ood_score,
            # is_ood, ood_threshold). Dùng bản copy để không làm bẩn result gốc.
            result_for_plot = dict(result)
            result_for_plot["ood_score"] = rs["score"]
            result_for_plot["is_ood"] = rs["is_ood"]
            result_for_plot["ood_threshold"] = rs["threshold"]
            st.caption(f"Hình đang hiển thị verdict RE-SCORE "
                       f"(score {rs['score']:.3f} vs threshold {rs['threshold']:.3f}).")
        else:
            st.caption("Hình đang hiển thị verdict GỐC của predict() (threshold cũ).")
        if st.session_state.get("current_canvas") is not None:
            fig_ood = plot_measured_ood(st.session_state.current_canvas,
                                        result_for_plot, save_path=None)
        else:
            fig_ood = plot_ood_summary(result_for_plot, save_path=None)
        st.pyplot(fig_ood); plt.close(fig_ood)

        # ---- Export figures for the report ----
        st.markdown("---")
        if st.button("Export figures (PNG for report)"):
            safe = "".join(ch if ch.isalnum() else "_"
                           for ch in str(st.session_state.current_label))[:50]
            out_dir = Path("results/reports/demo_app") / safe
            out_dir.mkdir(parents=True, exist_ok=True)
            canvas = st.session_state.get("current_canvas")
            if canvas is not None:
                model_wn = np.load(WN_PATH)
                plot_measured_reconstruction(
                    canvas, result, model_wn,
                    reference_spectra=load_reference_spectra(),
                    compound_order=COMPOUND_ORDER,
                    save_path=out_dir / "reconstruction.png")
                plot_measured_peaks(canvas, result_peaks=result.get("peaks"),
                                    save_path=out_dir / "peaks.png")
                plot_measured_ood(canvas, result, save_path=out_dir / "ood.png")
            else:
                plot_reconstruction_overlay(result, save_path=out_dir / "reconstruction.png")
                plot_peak_annotations(result, save_path=out_dir / "peaks.png")
                plot_ood_summary(result, save_path=out_dir / "ood.png")
            plt.close("all")
            st.success(f"Saved 3 PNGs -> {out_dir}")


# =============================================================================
# Tab 3 - Report
# =============================================================================
with tab_report:
    if st.session_state.result is None:
        st.info("Run analysis in Tab 1 first.")
    else:
        rep = generate_report(
            st.session_state.result,
            sample_id=str(st.session_state.current_label),
            ground_truth=st.session_state.current_gt,
        )
        st.markdown(rep["markdown"])
        st.download_button("Download Markdown", rep["markdown"],
                           file_name="raman_report.md", mime="text/markdown")
        st.download_button(
            "Download JSON",
            json.dumps(rep["json"], indent=2,
                       default=lambda o: float(o) if hasattr(o, "item") else str(o)),
            file_name="raman_report.json", mime="application/json")