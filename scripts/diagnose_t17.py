"""Diagnostic for T17 FAIL: explain WHY quant_mae is stuck around 0.05.

Reads `results/midcheckpoint_predictions.npz` and the checkpoint, then
runs 5 diagnostics that pinpoint which failure mode the model is in:

  D1. Prediction distribution — variance, range, mode collapse?
  D2. Marginal-baseline comparison — is model just predicting mean(y_train)?
  D3. Per-compound MAE — which compounds is the model worst at?
  D4. Reconstruction `scale` parameter — has it absorbed the gradient?
  D5. Prediction signal — does correlation(y_true, y_pred) exist per-compound?

Outputs `results/midcheckpoint_diagnosis.md` and prints a verdict.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


CANONICAL_ORDER = [
    "Alanine", "Asparagine", "Aspartic Acid",
    "Glutamic Acid", "Histidine", "Glucosamine",
]


def main() -> int:
    preds_path = _PROJECT_ROOT / "results" / "midcheckpoint_predictions.npz"
    ckpt_path = _PROJECT_ROOT / "checkpoints" / "best.pt"
    labels_path = _PROJECT_ROOT / "data" / "processed" / "labels.pt"
    split_path = _PROJECT_ROOT / "data" / "splits" / "split_A_composition_ood.json"

    for p in (preds_path, ckpt_path, labels_path, split_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    print(f"[load] predictions.npz")
    d = np.load(preds_path)
    y_true = d["y_true"].astype(np.float64)       # (N_test, 6)
    y_pred = d["y_pred"].astype(np.float64)
    s_input = d["s_input"].astype(np.float64)     # (N_test, 1024)
    s_recon = d["s_recon"].astype(np.float64)

    print(f"[load] checkpoint")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model"]

    print(f"[load] full training labels for marginal-baseline")
    labels_full = torch.load(labels_path, weights_only=False)
    if isinstance(labels_full, torch.Tensor):
        labels_full = labels_full.numpy()
    labels_full = np.asarray(labels_full, dtype=np.float64)

    import json
    with open(split_path) as f:
        split = json.load(f)
    y_train = labels_full[split["train"]]         # (N_train, 6)

    N, K = y_true.shape
    print(f"[info] N_test={N}, K={K}, N_train={len(y_train)}")
    print()

    out_lines = ["# T17 Failure Diagnosis", "",
                 f"N_test = {N}, N_train = {len(y_train)}, K = {K}", ""]

    # ── D1: Prediction distribution ──
    print("[D1] Prediction distribution")
    pred_mean = y_pred.mean(axis=0)
    pred_std = y_pred.std(axis=0)
    pred_min = y_pred.min(axis=0)
    pred_max = y_pred.max(axis=0)
    print(f"      Per-compound y_pred mean: {[f'{x:.3f}' for x in pred_mean]}")
    print(f"      Per-compound y_pred std : {[f'{x:.3f}' for x in pred_std]}")
    print(f"      Per-compound y_pred range: "
          f"[{', '.join(f'{lo:.2f}->{hi:.2f}' for lo, hi in zip(pred_min, pred_max))}]")
    overall_pred_std = float(pred_std.mean())
    out_lines += ["## D1 — Prediction distribution",
                  "",
                  f"| Compound | mean | std | min | max |",
                  f"|---|---|---|---|---|"]
    for i, c in enumerate(CANONICAL_ORDER):
        out_lines.append(
            f"| {c} | {pred_mean[i]:.3f} | {pred_std[i]:.3f} | {pred_min[i]:.3f} | {pred_max[i]:.3f} |"
        )
    out_lines.append("")
    if overall_pred_std < 0.05:
        d1_verdict = "🚨 MODE COLLAPSE — predictions barely vary across samples"
    elif overall_pred_std < 0.10:
        d1_verdict = "⚠ LOW VARIANCE — model is hedging toward a single point"
    else:
        d1_verdict = "✅ Reasonable variance in predictions"
    print(f"      {d1_verdict}\n")
    out_lines += [f"**Verdict:** {d1_verdict}", ""]

    # ── D2: Marginal-baseline comparison ──
    print("[D2] Marginal-baseline comparison")
    y_train_mean = y_train.mean(axis=0)
    print(f"      mean(y_train) = {[f'{x:.3f}' for x in y_train_mean]}")
    # MAE of a model that predicts mean(y_train) for every test sample
    baseline_pred = np.tile(y_train_mean, (N, 1))
    baseline_mae = float(np.mean(np.abs(y_true - baseline_pred)))
    model_mae = float(np.mean(np.abs(y_true - y_pred)))
    distance_to_marginal = float(np.mean(np.linalg.norm(y_pred - y_train_mean, axis=1)))
    print(f"      Baseline (predict mean(y_train)) MAE = {baseline_mae:.4f}")
    print(f"      Model MAE                           = {model_mae:.4f}")
    print(f"      Avg L2 distance of y_pred to mean(y_train) = {distance_to_marginal:.4f}")
    out_lines += ["## D2 — Marginal baseline comparison",
                  "",
                  f"- mean(y_train) per compound: {[f'{x:.3f}' for x in y_train_mean]}",
                  f"- **Baseline MAE** (predict mean): **{baseline_mae:.4f}**",
                  f"- **Model MAE**                 : **{model_mae:.4f}**",
                  f"- Avg L2 distance of model preds to baseline: {distance_to_marginal:.4f}",
                  ""]
    if model_mae > baseline_mae * 0.95:
        d2_verdict = (
            "🚨 MODEL ≈ BASELINE — predicting mean(y_train) would do "
            f"~{baseline_mae:.4f} MAE; model does {model_mae:.4f}. "
            "Model has learned the marginal but not the per-sample signal."
        )
    elif distance_to_marginal < 0.10:
        d2_verdict = (
            f"⚠ Model predictions cluster near mean(y_train) "
            f"(avg L2 distance {distance_to_marginal:.3f}); weak per-sample differentiation."
        )
    else:
        d2_verdict = "✅ Model beats marginal baseline; some sample-level signal exists"
    print(f"      {d2_verdict}\n")
    out_lines += [f"**Verdict:** {d2_verdict}", ""]

    # ── D3: Per-compound MAE ──
    print("[D3] Per-compound MAE")
    per_compound_mae = np.mean(np.abs(y_true - y_pred), axis=0)
    out_lines += ["## D3 — Per-compound MAE",
                  "",
                  f"| Compound | MAE | y_true mean | y_pred mean |",
                  f"|---|---|---|---|"]
    for i, c in enumerate(CANONICAL_ORDER):
        print(f"      {c:>15}: MAE={per_compound_mae[i]:.4f}  "
              f"true_mean={y_true[:, i].mean():.3f}  pred_mean={pred_mean[i]:.3f}")
        out_lines.append(
            f"| {c} | {per_compound_mae[i]:.4f} | {y_true[:, i].mean():.3f} | {pred_mean[i]:.3f} |"
        )
    out_lines.append("")
    worst = CANONICAL_ORDER[int(np.argmax(per_compound_mae))]
    best = CANONICAL_ORDER[int(np.argmin(per_compound_mae))]
    print(f"      Worst compound: {worst}, Best: {best}\n")
    out_lines += [f"Worst: **{worst}**, best: **{best}**", ""]

    # ── D4: Reconstruction scale parameter ──
    print("[D4] Reconstruction `scale` parameter")
    scale_key = None
    for k in sd.keys():
        if "reconstruction" in k and "scale" in k:
            scale_key = k
            break
    if scale_key is None:
        scale_msg = "⚠ No 'reconstruction.scale' parameter found in checkpoint"
        d4_verdict = scale_msg
        out_lines += ["## D4 — Reconstruction `scale` parameter", "", scale_msg, ""]
    else:
        scale = sd[scale_key].detach().cpu().numpy()
        print(f"      Found '{scale_key}' shape={scale.shape}")
        print(f"      Values: {[f'{x:.3f}' for x in scale.ravel()]}")
        print(f"      Mean: {scale.mean():.3f}  Std: {scale.std():.3f}  "
              f"Min: {scale.min():.3f}  Max: {scale.max():.3f}")
        out_lines += ["## D4 — Reconstruction `scale` parameter",
                      "",
                      f"- key: `{scale_key}`",
                      f"- shape: {scale.shape}",
                      f"- per-compound: {[f'{x:.3f}' for x in scale.ravel()]}",
                      f"- summary: mean={scale.mean():.3f}, std={scale.std():.3f}, "
                      f"min={scale.min():.3f}, max={scale.max():.3f}",
                      ""]
        max_abs = float(np.max(np.abs(scale)))
        ratio = max_abs / 1.0   # init was 1.0
        if max_abs > 5.0:
            d4_verdict = (
                f"🚨 SCALE INFLATED — max |scale| = {max_abs:.2f} (started at 1.0). "
                "Reconstruction module is compensating for wrong compositions "
                "by amplifying the references. This 'cheats' physics loss."
            )
        elif max_abs > 2.0:
            d4_verdict = (
                f"⚠ Scale somewhat inflated (max |scale| = {max_abs:.2f}, started 1.0). "
                "Some compensation happening but not catastrophic."
            )
        else:
            d4_verdict = (
                f"✅ Scale stable (max |scale| = {max_abs:.2f}, near init 1.0). "
                "Recon module is NOT cheating."
            )
        print(f"      {d4_verdict}\n")
        out_lines += [f"**Verdict:** {d4_verdict}", ""]

    # ── D5: Per-compound correlation (any signal at all?) ──
    print("[D5] Per-compound correlation y_true ↔ y_pred")
    out_lines += ["## D5 — Per-compound Pearson correlation (y_true vs y_pred)",
                  "",
                  "Pearson r tells whether the model has learned anything per-compound. "
                  "r ≈ 0 → no signal (model is decoupled from input).",
                  "",
                  "| Compound | r | interpretation |",
                  "|---|---|---|"]
    correlations = []
    for i, c in enumerate(CANONICAL_ORDER):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]
        if col_true.std() < 1e-6 or col_pred.std() < 1e-6:
            r = float("nan")
            label = "constant column (no variance)"
        else:
            r = float(np.corrcoef(col_true, col_pred)[0, 1])
            if abs(r) < 0.1:
                label = "no signal"
            elif abs(r) < 0.3:
                label = "weak signal"
            elif abs(r) < 0.6:
                label = "moderate"
            else:
                label = "strong signal"
        correlations.append(r)
        print(f"      {c:>15}: r = {r:+.3f}  ({label})")
        out_lines.append(f"| {c} | {r:+.3f} | {label} |")
    out_lines.append("")
    valid_r = [r for r in correlations if not np.isnan(r)]
    avg_r = float(np.mean(np.abs(valid_r))) if valid_r else float("nan")
    if avg_r < 0.1:
        d5_verdict = (
            f"🚨 NO SIGNAL — avg |r| = {avg_r:.3f}. Model output is decoupled from "
            "input spectra. Likely cause: backbone or head not updating, OR "
            "all the gradient is going into reconstruction.scale."
        )
    elif avg_r < 0.3:
        d5_verdict = f"⚠ WEAK SIGNAL — avg |r| = {avg_r:.3f}. Model has learned something but not enough."
    else:
        d5_verdict = f"✅ SIGNAL EXISTS — avg |r| = {avg_r:.3f}. Model differentiates samples."
    print(f"      {d5_verdict}\n")
    out_lines += [f"**Verdict:** {d5_verdict}", ""]

    # ── Synthesis ──
    print("=" * 60)
    print("SYNTHESIS")
    print("=" * 60)
    out_lines += ["## Synthesis", ""]

    flags = {
        "mode_collapse": overall_pred_std < 0.05,
        "marginal_baseline": model_mae > baseline_mae * 0.95,
        "scale_inflated": (scale_key is not None and float(np.max(np.abs(sd[scale_key].cpu().numpy()))) > 5.0),
        "no_signal": avg_r < 0.1 if valid_r else False,
    }
    for name, fired in flags.items():
        status = "🚨 YES" if fired else "OK"
        print(f"  {name:>20}: {status}")
        out_lines.append(f"- **{name}**: {status}")
    out_lines.append("")

    # Diagnosis tree
    if flags["scale_inflated"] and flags["no_signal"]:
        msg = (
            "**PRIMARY CAUSE: Recon scale absorbed all the gradient.** "
            "Reconstruction module's `scale` parameter inflated to compensate "
            "for wrong compositions, satisfying physics_loss while quant_loss "
            "is ignored. Backbone+head learned nothing useful."
        )
    elif flags["no_signal"]:
        msg = (
            "**PRIMARY CAUSE: Backbone/head not learning.** No correlation "
            "between input spectra and predictions. Check: learning rate too low, "
            "gradient vanishing, output simplex saturation, augmentation too aggressive."
        )
    elif flags["marginal_baseline"]:
        msg = (
            "**PRIMARY CAUSE: Model collapsed to marginal.** Predicting "
            "mean(y_train) achieves ~same MAE as your model. Loss landscape "
            "favors low-variance solutions. Need to penalize uniform predictions "
            "or strengthen per-sample signal."
        )
    elif flags["mode_collapse"]:
        msg = (
            "**PRIMARY CAUSE: Mode collapse without marginal alignment.** "
            "Model produces nearly identical outputs but they're not the marginal. "
            "Check augmentation noise level and output saturation."
        )
    else:
        msg = (
            "**No single dominant failure mode** — diagnostic flags mixed. "
            "Review per-compound MAE (D3) and worst-compound spectrum to identify "
            "specific weakness."
        )
    print(f"\n{msg}")
    out_lines += [msg, ""]

    # Write report
    out_path = _PROJECT_ROOT / "results" / "midcheckpoint_diagnosis.md"
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n[saved] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())