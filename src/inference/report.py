"""Per-sample report generator (T26).

Takes a result dict from :func:`src.inference.predict.predict` and
generates three forms of the same content:

* **Markdown** -- pretty per-sample report, suitable for the thesis
  defense / Streamlit dashboard / commit-in-repo evidence.
* **JSON** -- structured dump for downstream tooling (e.g. external
  benchmarking pipelines).
* **plain text** -- one-paragraph summary for terminal output.

Two design rules that are NOT negotiable (CHAT4_PHASE_AB_HANDOVER §D.3,
upgraded to mandatory after P4AB-9 was observed on real mixture data):

1. **Composition always shown with uncertainty.** The mean *and* the std
   from MC Dropout are presented together. A point estimate alone would
   misleadingly suggest higher confidence than the model actually has,
   especially on Glucosamine + Asparagine where Chat 2 T17 measured very
   wide MC variance.

2. **Cross-check between learned and symbolic heads is shown explicitly.**
   The "agreement / learned-says-yes-symbolic-no / symbolic-says-yes-
   learned-no / agreement-absent" tag is computed per compound and
   shown in a single column of the composition table.

Author: Chat 4 Phase C, Task T26.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import json
import numpy as np


COMPOUND_ORDER = [
    "Alanine", "Asparagine", "Aspartic Acid",
    "Glutamic Acid", "Histidine", "Glucosamine",
]


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def generate_report(
    result: dict,
    *,
    sample_id: str = "unknown",
    ground_truth: Optional[dict] = None,
    benchmark_context: Optional[dict] = None,
    model_version: str = "v0.1.0-mvp",
    image_paths: Optional[dict] = None,
) -> dict:
    """Generate Markdown, JSON, and plain-text views of a prediction.

    Parameters
    ----------
    result : dict
        Output of :func:`src.inference.predict.predict`.
    sample_id : str
        Identifier for the sample (e.g. ``"a17-row42"`` or ``"MoS2-demo"``).
    ground_truth : dict, optional
        ``{compound_name: float}`` if available. Adds an "Error" column to
        the composition table.
    benchmark_context : dict, optional
        Numbers from ``results/benchmark_table.json`` (Chat 2 Phase B):
        ``{"pca_svm_mae": 0.0479, "resnet_only_mae": 0.0462,
        "ours_mae": 0.0550, "test_n": 540}``. If provided, adds the
        comparison sub-section to the Markdown report.
    model_version : str
        Tag string for the "Model Version" header.
    image_paths : dict, optional
        ``{"reconstruction": Path, "peaks": Path, "ood": Path}`` -- if
        provided, image links are embedded in the Markdown.

    Returns
    -------
    dict with keys:
        * ``markdown``   -- str
        * ``json``       -- dict (JSON-serialisable)
        * ``plain_text`` -- str (single paragraph, terminal-friendly)

    Example
    -------
    >>> from src.inference.predict import predict
    >>> import numpy as np
    >>> result = predict(np.load("data/processed/spectra_full.pt")[0])
    >>> report = generate_report(result, sample_id="row0",
    ...                          ground_truth={"Histidine": 1.0})
    >>> print(report["plain_text"])
    Sample row0: composition...
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Cross-check between learned and symbolic heads
    cross_check = _compute_cross_check(result)

    md = _render_markdown(
        result,
        sample_id=sample_id,
        timestamp=timestamp,
        ground_truth=ground_truth,
        cross_check=cross_check,
        benchmark_context=benchmark_context,
        model_version=model_version,
        image_paths=image_paths,
    )
    js = _render_json(
        result,
        sample_id=sample_id,
        timestamp=timestamp,
        ground_truth=ground_truth,
        cross_check=cross_check,
    )
    pt = _render_plain_text(result, sample_id=sample_id, cross_check=cross_check)

    return {"markdown": md, "json": js, "plain_text": pt}


def save_report(
    report: dict,
    output_dir: Union[str, Path],
    *,
    base_name: str,
) -> dict[str, Path]:
    """Write the three views to disk under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    md_path = output_dir / f"{base_name}.md"
    md_path.write_text(report["markdown"], encoding="utf-8")
    paths["markdown"] = md_path

    js_path = output_dir / f"{base_name}.json"
    js_path.write_text(json.dumps(report["json"], indent=2, default=_json_default),
                       encoding="utf-8")
    paths["json"] = js_path

    return paths


# -----------------------------------------------------------------------------
# Cross-check (P4AB-9 mandatory pattern)
# -----------------------------------------------------------------------------

def _compute_cross_check(result: dict) -> dict[str, str]:
    """Per-compound agreement tag between learned and symbolic heads.

    Returns ``{compound: tag}`` where ``tag`` is one of:

    * ``"agreement: present"``   -- both heads say > 5% / likely
    * ``"agreement: absent"``    -- both heads say absent
    * ``"learned-only"``         -- learned says > 5%, symbolic does not
    * ``"symbolic-only"``        -- symbolic says likely, learned says < 5%

    Per CHAT4 §D.3 / P4AB-9 the row-780 mixture observation showed both
    error modes simultaneously, so the report must distinguish them.
    """
    composition = result["composition"]
    likely = set(result.get("likely_compounds_symbolic", []))
    out: dict[str, str] = {}
    for c in COMPOUND_ORDER:
        mean = composition.get(c, 0.0)
        in_learned  = mean > 0.05
        in_symbolic = c in likely
        if in_learned and in_symbolic:
            out[c] = "agreement: present"
        elif (not in_learned) and (not in_symbolic):
            out[c] = "agreement: absent"
        elif in_learned and not in_symbolic:
            out[c] = "learned-only"
        else:
            out[c] = "symbolic-only"
    return out


# -----------------------------------------------------------------------------
# Markdown rendering
# -----------------------------------------------------------------------------

def _render_markdown(
    result: dict,
    *,
    sample_id: str,
    timestamp: str,
    ground_truth: Optional[dict],
    cross_check: dict[str, str],
    benchmark_context: Optional[dict],
    model_version: str,
    image_paths: Optional[dict],
) -> str:
    composition     = result["composition"]
    composition_std = result["composition_std"]
    likely  = result.get("likely_compounds_symbolic", [])
    votes   = result.get("compound_votes", {})
    peaks   = result.get("peaks", [])
    md = []

    # --- Header ---
    md.append(f"# Raman Spectrum Analysis Report\n")
    md.append(f"- **Sample ID:** `{sample_id}`")
    md.append(f"- **Date:** {timestamp}")
    md.append(f"- **Model Version:** {model_version}")
    md.append(f"- **n MC samples:** {result['metadata']['n_mc_samples']}")
    md.append("")

    # --- §1 Composition with cross-check ---
    md.append("## 1. Composition Analysis\n")
    md.append("Composition predicted by the learned head (mean ± MC std), "
              "with the symbolic head's cross-check tag (per "
              "CHAT4_PHASE_AB_HANDOVER §D.3).")
    md.append("")

    header = "| Compound | Predicted (mean) | Uncertainty (±std) |"
    sep = "|---|---|---|"
    if ground_truth is not None:
        header += " Ground truth | Error |"
        sep += "---|---|"
    header += " Symbolic vote | Cross-check |"
    sep += "---|---|"
    md.append(header)
    md.append(sep)

    for c in COMPOUND_ORDER:
        mean = composition.get(c, 0.0)
        std  = composition_std.get(c, 0.0)
        vote = votes.get(c, 0.0)
        cc   = cross_check[c]
        row = f"| {c} | {mean:.3f} ({mean*100:.1f}%) | ±{std:.3f} |"
        if ground_truth is not None:
            gt = ground_truth.get(c, None)
            if gt is not None:
                err = mean - gt
                row += f" {gt:.3f} | {err:+.3f} |"
            else:
                row += " — | — |"
        # Flag confidence in the vote column
        if vote >= 1.0:
            vote_str = f"**{vote:.1f}**"
        elif vote > 0:
            vote_str = f"{vote:.1f}"
        else:
            vote_str = "—"
        row += f" {vote_str} | {_format_crosscheck(cc)} |"
        md.append(row)
    md.append("")

    # Summary of likely_compounds
    if likely:
        md.append(f"**Symbolic head ⇒ likely present:** {', '.join(likely)}.")
    else:
        md.append("**Symbolic head ⇒ no compounds reached the vote threshold.**  "
                  "(See peak table for individual matches.)")
    md.append("")

    # Uncertainty summary
    md.append(f"**Uncertainty summary:** "
              f"predictive entropy = {result['predictive_entropy']:.4f}, "
              f"mean compound std = {result['mean_compound_std']:.4f}.")
    md.append("")

    # --- §2 Peak Analysis ---
    md.append("## 2. Peak Analysis\n")
    n_total   = len(peaks)
    n_matched = sum(1 for p in peaks if p["matched_to"] is not None)
    n_unmatched = n_total - n_matched
    md.append(f"Detected **{n_total} peaks**. **{n_matched}** matched to known "
              f"compounds, **{n_unmatched}** unmatched.")
    md.append("")

    if peaks:
        md.append("| Position (cm⁻¹) | Intensity | FWHM | Bond / Mode | "
                  "Compound | DB-id | Conf. | Δν |")
        md.append("|---|---|---|---|---|---|---|---|")
        for p in sorted(peaks, key=lambda x: x["position"]):
            bond = p.get("bond") or "—"
            compounds = ", ".join(p.get("compounds") or []) or "—"
            db_id = p.get("matched_to") or "—"
            conf = p.get("match_confidence") or "—"
            delta = p.get("delta_cm")
            delta_str = f"{delta:+.1f}" if delta is not None else "—"
            md.append(
                f"| {p['position']:.1f} | {p['intensity']:.3f} | "
                f"{p['fwhm']:.1f} | {bond} | {compounds} | {db_id} | "
                f"{conf} | {delta_str} |"
            )
        md.append("")

    # --- §3 Physics Validation ---
    md.append("## 3. Physics Validation\n")
    recon = result["recon_cosine_sim"]
    band = (
        "✓ PASS-target (≥ 0.95)" if recon >= 0.95
        else "✓ PASS-floor (≥ 0.85)" if recon >= 0.85
        else "✗ Floor missed"
    )
    md.append(f"- **Reconstruction cosine similarity:** {recon:.4f}  ({band})")
    if recon >= 0.85:
        md.append("- The predicted composition reproduces the input spectrum "
                  "well via the linear Beer-Lambert combination of pure "
                  "references. Constraint satisfied.")
    else:
        md.append("- The predicted composition cannot reproduce the input "
                  "spectrum (cosine sim below floor 0.85). **Constraint "
                  "violated** -- this prediction should be treated with "
                  "extreme caution.")
    md.append("")

    # --- §4 OOD Assessment ---
    md.append("## 4. OOD Assessment\n")
    if result["ood_score"] is None:
        md.append("- OOD scoring was skipped (no calibration loaded).")
    else:
        score    = result["ood_score"]
        thresh   = result["ood_threshold"]
        is_ood   = result["is_ood"]
        comps    = result["ood_components"] or {}
        verdict  = "⚠️ **OUT-OF-DISTRIBUTION**" if is_ood else "✓ IN-DISTRIBUTION"
        md.append(f"- **OOD Score:** {score:.4f}  (threshold {thresh:.4f})")
        md.append(f"- **Verdict:** {verdict}")
        md.append(f"- **Components:** recon_norm = {comps.get('recon_norm', 0):.3f}, "
                  f"var_norm = {comps.get('var_norm', 0):.3f}")
    # Novelty clusters always shown
    n_novel = len(result.get("novelty_clusters", []))
    md.append(f"- **Novel peaks:** {len(result.get('unknown_peaks', []))} "
              f"unmatched, grouped into {n_novel} cluster(s).")
    hints = result.get("novelty_hints", [])
    if hints:
        md.append("")
        md.append("**Cluster details:**")
        for h in hints:
            md.append(f"- {h}")
    md.append("")

    # --- §5 Visualisations ---
    if image_paths:
        md.append("## 5. Visualisations\n")
        if image_paths.get("reconstruction"):
            md.append(f"![Reconstruction overlay]({image_paths['reconstruction']})")
            md.append("")
        if image_paths.get("peaks"):
            md.append(f"![Peak annotations]({image_paths['peaks']})")
            md.append("")
        if image_paths.get("ood"):
            md.append(f"![OOD summary]({image_paths['ood']})")
            md.append("")

    # --- §6 Benchmark context ---
    if benchmark_context:
        md.append("## 6. Benchmark Context\n")
        md.append(f"On the same test split (Scheme-A composition-OOD, "
                  f"{benchmark_context.get('test_n', '?')} rows):")
        md.append("")
        md.append("| Model | Quant MAE | Notes |")
        md.append("|---|---|---|")
        rendered_any_row = False
        if benchmark_context.get("pca_svm_mae") is not None:
            md.append(f"| PCA + SVM | {benchmark_context['pca_svm_mae']:.4f} | "
                      f"classical baseline |")
            rendered_any_row = True
        if benchmark_context.get("resnet_only_mae") is not None:
            md.append(f"| ResNet-only (no physics) | "
                      f"{benchmark_context['resnet_only_mae']:.4f} | "
                      f"black-box DL baseline |")
            rendered_any_row = True
        if benchmark_context.get("ours_mae") is not None:
            md.append(f"| **Ours (physics-informed)** | "
                      f"{benchmark_context['ours_mae']:.4f} | "
                      f"+ recon validation, + OOD, + interpretability |")
            rendered_any_row = True
        if not rendered_any_row:
            md.append("| _(no benchmark numbers available yet)_ | -- | "
                      "run T23/T24/T25 to populate `results/benchmark_table.json` |")
        if ground_truth is not None:
            # This sample's MAE
            this_mae = np.mean([
                abs(composition.get(c, 0.0) - ground_truth.get(c, 0.0))
                for c in COMPOUND_ORDER if c in ground_truth
            ])
            md.append("")
            md.append(f"**This sample's MAE: {this_mae:.4f}**.")
        md.append("")

    return "\n".join(md)


def _format_crosscheck(tag: str) -> str:
    """Compact emoji-decorated cross-check tag for the Markdown table."""
    return {
        "agreement: present":  "✓ present",
        "agreement: absent":   "✓ absent",
        "learned-only":        "⚠️ learned-only",
        "symbolic-only":       "⚠️ symbolic-only",
    }.get(tag, tag)


# -----------------------------------------------------------------------------
# JSON rendering
# -----------------------------------------------------------------------------

def _render_json(
    result: dict,
    *,
    sample_id: str,
    timestamp: str,
    ground_truth: Optional[dict],
    cross_check: dict[str, str],
) -> dict:
    """Build a JSON-serialisable dict from the prediction result."""
    out = {
        "sample_id": sample_id,
        "timestamp": timestamp,
        "composition": {
            c: {
                "mean": float(result["composition"].get(c, 0.0)),
                "std":  float(result["composition_std"].get(c, 0.0)),
                "ground_truth": (None if ground_truth is None
                                 else float(ground_truth.get(c, 0.0))),
                "symbolic_vote": float(result["compound_votes"].get(c, 0.0)),
                "cross_check": cross_check[c],
            }
            for c in COMPOUND_ORDER
        },
        "physics": {
            "reconstruction_cosine_sim": float(result["recon_cosine_sim"]),
            "predictive_entropy": float(result["predictive_entropy"]),
            "mean_compound_std":  float(result["mean_compound_std"]),
        },
        "ood": {
            "score":     result["ood_score"],
            "is_ood":    result["is_ood"],
            "threshold": result["ood_threshold"],
            "components": result["ood_components"],
        },
        "peaks": result["peaks"],
        "symbolic_head": {
            "likely_compounds": result["likely_compounds_symbolic"],
            "votes": result["compound_votes"],
            "unsupported": result["unsupported_compounds"],
        },
        "novelty": {
            "unknown_peaks": result["unknown_peaks"],
            "n_clusters":    len(result["novelty_clusters"]),
            "hints":         result["novelty_hints"],
        },
        "metadata": result["metadata"],
    }
    return out


# -----------------------------------------------------------------------------
# Plain-text rendering
# -----------------------------------------------------------------------------

def _render_plain_text(
    result: dict,
    *,
    sample_id: str,
    cross_check: dict[str, str],
) -> str:
    """Single-paragraph summary suitable for terminal output."""
    composition = result["composition"]
    top = max(COMPOUND_ORDER, key=lambda c: composition.get(c, 0.0))
    top_pct = composition[top] * 100
    top_std = result["composition_std"][top] * 100

    n_peaks = len(result.get("peaks", []))
    n_matched = sum(1 for p in result.get("peaks", []) if p["matched_to"] is not None)

    likely = result.get("likely_compounds_symbolic", [])
    likely_str = ", ".join(likely) if likely else "none"

    recon = result["recon_cosine_sim"]

    parts = [
        f"Sample {sample_id}: composition dominated by {top} "
        f"({top_pct:.1f}% ± {top_std:.1f}%).",
        f"Detected {n_peaks} peaks ({n_matched} matched).",
        f"Symbolic head likely_compounds = [{likely_str}].",
        f"Reconstruction cosine = {recon:.3f}.",
    ]
    if result["ood_score"] is not None:
        ood_label = "OOD" if result["is_ood"] else "ID"
        parts.append(f"OOD = {result['ood_score']:.3f} ({ood_label}).")

    # Flag disagreements
    disagreements = [c for c, t in cross_check.items()
                     if t in ("learned-only", "symbolic-only")]
    if disagreements:
        parts.append(f"Cross-check disagreements on: {', '.join(disagreements)}.")

    return " ".join(parts)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _json_default(o):
    """JSON serialiser fallback for numpy types."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")