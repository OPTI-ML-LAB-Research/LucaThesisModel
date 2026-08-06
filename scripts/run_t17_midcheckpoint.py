"""T17 — Mid-Checkpoint evaluation on TEST set (v1.1).

What's different from v1.0:
* Reads compound names + spectrum_length from checkpoint config (no hard-coded)
* Pre-flight check: shape consistency BEFORE forward
* Pre-flight check: warns if config['paths']['data_raw_csv'] is legacy data.csv
* Reconstruction module uses pure_ref FROM CHECKPOINT (not a fresh .npy file),
  because the checkpoint stored it as a buffer at training time.
  This avoids "config says one ref file, checkpoint has different weights"
  silent corruption.

Outputs:
* results/midcheckpoint_report.md
* results/midcheckpoint_report.json
* results/midcheckpoint_predictions.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.splits import load_split  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    constraint_violation_rate,
    identification_accuracy,
    quantification_mae,
    reconstruction_cosine_similarity,
)


TARGETS = {
    "quant_mae":                 {"target": 0.020, "floor": 0.025, "lower_is_better": True},
    "ident_accuracy":            {"target": 0.90,  "floor": 0.85,  "lower_is_better": False},
    "recon_cosine_median":       {"target": 0.95,  "floor": 0.85,  "lower_is_better": False},
    "constraint_violation_rate": {"target": 0.05,  "floor": 0.10,  "lower_is_better": True},
}


def _build_model_from_checkpoint(ck: dict, ref_override: np.ndarray | None = None) -> torch.nn.Module:
    """Build RamanPhysicsAI from checkpoint config, then restore weights.

    Implementation notes
    --------------------
    ``build_full_model_from_config`` (see ``src/models/full_model.py``)
    accepts ``reference_spectra_path: Optional[Union[str, Path]]`` — a
    file path, not an array. We therefore:

    1. Extract pure_ref from the checkpoint buffer (or use ``ref_override``)
       to guarantee shape compatibility.
    2. Dump it to a temporary ``.npy`` file.
    3. Pass that path to the factory.
    4. Call ``load_state_dict``, which overwrites the buffer with the
       exact same values the checkpoint stored — so the tmp file is just
       scaffolding to get the shape right.

    Using the checkpoint's own ``reconstruction.pure_ref`` avoids the
    classic silent corruption where the script reads from
    ``engine/reference_spectra.npy`` (possibly rebuilt since training)
    while the model weights expect a different reference.
    """
    import tempfile

    from src.models.full_model import build_full_model_from_config

    cfg = ck.get("config")
    if cfg is None:
        raise ValueError("Checkpoint has no 'config' key.")

    if ref_override is not None:
        refs = ref_override.astype(np.float32)
    else:
        sd = ck["model"]
        if "reconstruction.pure_ref" not in sd:
            raise KeyError(
                "Checkpoint state_dict has no 'reconstruction.pure_ref'. "
                "Pass --references explicitly."
            )
        pr = sd["reconstruction.pure_ref"]
        refs = (
            pr.detach().cpu().numpy().astype(np.float32)
            if isinstance(pr, torch.Tensor)
            else np.asarray(pr, np.float32)
        )

    # Factory wants a path to an .npy file. Dump and forward.
    tmp_dir = tempfile.mkdtemp(prefix="t17_refs_")
    tmp_path = Path(tmp_dir) / "ref_from_ckpt.npy"
    np.save(tmp_path, refs)
    model = build_full_model_from_config(cfg, reference_spectra_path=str(tmp_path))

    # Restore exact weights (incl. pure_ref buffer, overwriting tmp values).
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def _evaluate(model, spectra, labels, indices, *, batch_size=128, device="cpu") -> dict:
    model = model.to(device)
    ds = TensorDataset(spectra[indices], labels[indices])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    y_pred_list, s_recon_list, s_input_list, y_true_list = [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            x_in = x.unsqueeze(1) if x.ndim == 2 else x
            out = model(x_in)
            y_pred_list.append(out["composition"].cpu().numpy())
            s_recon_list.append(out["reconstruction"].cpu().numpy())
            s_input_list.append(x.cpu().numpy())
            y_true_list.append(y.cpu().numpy())
    return {
        "y_true":  np.concatenate(y_true_list, axis=0),
        "y_pred":  np.concatenate(y_pred_list, axis=0),
        "s_input": np.concatenate(s_input_list, axis=0),
        "s_recon": np.concatenate(s_recon_list, axis=0),
    }


def _verdict_for_metric(name: str, value: float) -> tuple[str, str]:
    spec = TARGETS[name]
    t, f, lb = spec["target"], spec["floor"], spec["lower_is_better"]
    if lb:
        if value <= t: return "PASS-target", f"{value:.4f} ≤ target {t}"
        if value <= f: return "PASS-floor", f"target {t} < {value:.4f} ≤ floor {f}"
        return "FAIL", f"{value:.4f} > floor {f}"
    if value >= t: return "PASS-target", f"{value:.4f} ≥ target {t}"
    if value >= f: return "PASS-floor", f"floor {f} ≤ {value:.4f} < target {t}"
    return "FAIL", f"{value:.4f} < floor {f}"


def _overall_verdict(per_metric: dict[str, str]) -> str:
    if any(v == "FAIL" for v in per_metric.values()):
        return "FAIL" if per_metric.get("quant_mae") == "FAIL" else "BORDERLINE"
    if all(v == "PASS-target" for v in per_metric.values()):
        return "PASS-target"
    return "PASS-floor"


def _render_report(*, metrics, per_metric_band, per_metric_comment, overall,
                   checkpoint_info, n_test, pred_npz_path, preflight_warnings):
    L = ["# T17 — Mid-Checkpoint Report", "",
         f"*Generated: {datetime.now().isoformat(timespec='seconds')}*", ""]

    if preflight_warnings:
        L.append("## ⚠ Pre-flight warnings"); L.append("")
        for w in preflight_warnings:
            L.append(f"- {w}")
        L.append("")

    L += ["## Inputs", "",
          f"- **Checkpoint:** epoch {checkpoint_info.get('epoch')}, "
          f"best_val_mae {checkpoint_info.get('val_mae')}",
          f"- **Split:** `{checkpoint_info.get('split_path')}` "
          f"(scheme `{checkpoint_info.get('split_scheme')}`, seed `{checkpoint_info.get('split_seed')}`)",
          f"- **Test rows evaluated:** {n_test}",
          f"- **Device:** {checkpoint_info.get('device', 'cpu')}", "",
          "## Metrics on TEST set", "",
          "| Metric | Value | Target | Floor | Band | Note |",
          "|---|---|---|---|---|---|"]

    rows = [
        ("Quantification MAE",                          "quant_mae"),
        ("Identification Accuracy",                     "ident_accuracy"),
        ("Reconstruction cosine (median)",              "recon_cosine_median"),
        ("Constraint Violation Rate (cos < 0.85)",      "constraint_violation_rate"),
    ]
    for label, key in rows:
        spec = TARGETS[key]
        L.append(f"| {label} | **{metrics[key]:.4f}** | {spec['target']} "
                 f"| {spec['floor']} | **{per_metric_band[key]}** "
                 f"| {per_metric_comment[key]} |")

    L += ["", "Note: `ood_auroc` is skipped — needs OOD samples not yet defined. "
          "Evaluated in Phase 3 after T19.", ""]

    rs = metrics["recon_full_stats"]
    L += ["Reconstruction distribution (additional diagnostics):", "",
          f"- Mean: **{rs['mean']:.4f}**, Median: **{rs['median']:.4f}**",
          f"- Percentiles: p05={rs['p05']:.4f}, p25={rs['p25']:.4f}, "
          f"p75={rs['p75']:.4f}, p95={rs['p95']:.4f}", ""]
    if rs["p25"] < 0.5 < rs["p75"]:
        L += ["> ⚠ Wide spread between p25 and p75 — distribution may be bimodal.", ""]

    L += ["## Overall verdict", ""]
    if overall == "PASS-target":
        L += ["### **PASS — TARGET MET** ✅", "",
              "All 4 metrics meet target. Phase 3 unlocked. Proceed per CHAT3 §F.1."]
    elif overall == "PASS-floor":
        L += ["### **PASS — FLOOR MET** ✅", "",
              "All 4 metrics meet at least the floor. Phase 3 unlocked. "
              "Document caveats in REPORT.md."]
    elif overall == "BORDERLINE":
        L += ["### **BORDERLINE** ⚠", "",
              "Non-headline metrics fail floor but quant_mae OK. Phase 3 proceeds; "
              "use T18 MC Dropout as honest-uncertainty diagnostic per CHAT3 §F.2."]
    else:
        L += ["### **FAIL** ❌", "",
              "Headline quant_mae missed the floor. **Phase 3 paused.** "
              "Apply fallback retrain per CHAT3 §F.3:",
              "", "```yaml",
              "# configs/train_config.yaml overrides",
              "loss:",
              "  beta_phys: 0.0           # was 0.5",
              "  alpha_quant: 1.0",
              "training:",
              "  early_stopping_patience: 15  # was 8",
              "data:",
              "  augmentation:",
              "    gaussian_noise_sigma: 0.002  # was 0.005",
              "```", "",
              "After retrain produces new `best.pt`, re-run this script."]

    L += ["", "## Files written", "",
          f"- This report",
          f"- `{pred_npz_path}` — `y_true`, `y_pred`, `s_input`, `s_recon`", ""]
    return "\n".join(L)


def _preflight(args, ck) -> list[str]:
    warnings: list[str] = []
    cfg = ck.get("config", {})

    raw_csv = cfg.get("paths", {}).get("data_raw_csv", "")
    if raw_csv.endswith("data.csv") and "AA_Data" not in raw_csv:
        warnings.append(
            f"Config has `data_raw_csv = '{raw_csv}'` (legacy). If "
            "`spectra_full.pt` was rebuilt from `AA_Data.csv` since training, "
            "row order may differ and T17 will silently use wrong indices. "
            "Verify by checking vial_ids.npy naming (a01-a48 vs aa01-aa48)."
        )

    for path_attr in ("spectra", "labels", "split"):
        path = getattr(args, path_attr)
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    spectra = torch.load(args.spectra, weights_only=False)
    labels  = torch.load(args.labels,  weights_only=False)
    sp_len = cfg.get("data", {}).get("spectrum_length", 1024)
    n_classes = cfg.get("compounds", {}).get("num_classes", 6)

    if spectra.shape[1] != sp_len:
        warnings.append(
            f"spectra_full.pt has {spectra.shape[1]} channels but config "
            f"expects {sp_len}. Forward may crash."
        )
    if labels.shape[1] != n_classes:
        warnings.append(
            f"labels.pt has {labels.shape[1]} cols but config expects {n_classes}."
        )
    if spectra.shape[0] != labels.shape[0]:
        raise ValueError(
            f"spectra has {spectra.shape[0]} rows but labels has {labels.shape[0]}."
        )

    split = load_split(args.split)
    if max(split.test) >= spectra.shape[0]:
        raise ValueError(
            f"split.test index {max(split.test)} exceeds spectra rows "
            f"{spectra.shape[0]}. Split JSON is for a different dataset."
        )
    return warnings


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="T17 — Mid-Checkpoint TEST-set evaluation (v1.1).")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    p.add_argument("--spectra",    type=Path, default=Path("data/processed/spectra_full.pt"))
    p.add_argument("--labels",     type=Path, default=Path("data/processed/labels.pt"))
    p.add_argument("--split",      type=Path, default=Path("data/splits/split_A_composition_ood.json"))
    p.add_argument("--references", type=Path, default=None,
                   help="Optional: override references (default = use checkpoint's)")
    p.add_argument("--out-report", type=Path, default=Path("results/midcheckpoint_report.md"))
    p.add_argument("--out-preds",  type=Path, default=Path("results/midcheckpoint_predictions.npz"))
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--cvr-threshold",    type=float, default=0.85)
    p.add_argument("--id-acc-threshold", type=float, default=0.05)
    args = p.parse_args(argv)

    print(f"[1/6] Loading checkpoint {args.checkpoint} ...")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    epoch = int(ck.get("epoch", -1))
    vm = ck.get("val_metrics", {}) or {}
    val_mae_raw = vm.get("val_mae")
    val_mae_str = f"{val_mae_raw:.4f}" if isinstance(val_mae_raw, (int, float)) else str(val_mae_raw)
    print(f"      epoch={epoch}, val_mae={val_mae_str}")

    print(f"[2/6] Pre-flight checks ...")
    warnings = _preflight(args, ck)
    for w in warnings:
        print(f"      ⚠ {w}")

    print(f"[3/6] Building model ...")
    ref_override = np.load(args.references).astype(np.float32) if args.references else None
    if ref_override is not None:
        print(f"      using --references override")
    else:
        print(f"      using pure_ref baked into checkpoint")
    model = _build_model_from_checkpoint(ck, ref_override=ref_override)

    print(f"[4/6] Loading test split ...")
    spectra = torch.load(args.spectra, weights_only=False)
    labels  = torch.load(args.labels,  weights_only=False)
    if not isinstance(spectra, torch.Tensor):
        spectra = torch.as_tensor(spectra)
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)
    split = load_split(args.split)
    test_idx = split.test
    print(f"      |test|={len(test_idx)}, spectra={tuple(spectra.shape)}")

    print(f"[5/6] Forward (batch={args.batch_size}) ...")
    out = _evaluate(model, spectra, labels, test_idx,
                    batch_size=args.batch_size, device=args.device)
    y_true, y_pred = out["y_true"], out["y_pred"]
    s_input, s_recon = out["s_input"], out["s_recon"]
    print(f"      collected y_pred {y_pred.shape}, s_recon {s_recon.shape}")

    print(f"[6/6] Metrics ...")
    quant_mae = quantification_mae(y_true, y_pred)
    ident_acc = identification_accuracy(y_true, y_pred, threshold=args.id_acc_threshold)
    recon = reconstruction_cosine_similarity(s_input, s_recon)
    cvr = constraint_violation_rate(s_input, s_recon, threshold=args.cvr_threshold)
    metrics = {
        "quant_mae": float(quant_mae),
        "ident_accuracy": float(ident_acc),
        "recon_cosine_median": float(recon["median"]),
        "constraint_violation_rate": float(cvr),
        "recon_full_stats": {k: float(v) for k, v in recon.items() if k != "per_sample"},
    }
    bands, comments = {}, {}
    for key in ["quant_mae", "ident_accuracy", "recon_cosine_median", "constraint_violation_rate"]:
        b, c = _verdict_for_metric(key, metrics[key])
        bands[key] = b; comments[key] = c
    overall = _overall_verdict(bands)

    args.out_preds.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_preds,
             y_true=y_true.astype(np.float32),
             y_pred=y_pred.astype(np.float32),
             s_input=s_input.astype(np.float32),
             s_recon=s_recon.astype(np.float32))
    print(f"      saved preds → {args.out_preds}")

    metrics_json = args.out_report.with_suffix(".json")
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump({
            "overall": overall, "metrics": metrics, "bands": bands,
            "n_test": int(y_true.shape[0]),
            "checkpoint": {"path": str(args.checkpoint), "epoch": epoch,
                           "val_mae": vm.get("val_mae")},
            "thresholds": {"id_acc_threshold": args.id_acc_threshold,
                           "cvr_threshold": args.cvr_threshold},
            "preflight_warnings": warnings,
        }, f, indent=2)
    print(f"      saved JSON → {metrics_json}")

    report = _render_report(
        metrics=metrics, per_metric_band=bands, per_metric_comment=comments,
        overall=overall,
        checkpoint_info={"epoch": epoch, "val_mae": vm.get("val_mae"),
                         "split_path": str(args.split),
                         "split_scheme": split.scheme, "split_seed": split.seed,
                         "device": args.device},
        n_test=int(y_true.shape[0]), pred_npz_path=args.out_preds,
        preflight_warnings=warnings,
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(report, encoding="utf-8")
    print(f"      saved report → {args.out_report}")

    print()
    print(f"========================= T17 RESULT =========================")
    print(f"  quant_mae        = {metrics['quant_mae']:.4f}   [{bands['quant_mae']}]")
    print(f"  ident_accuracy   = {metrics['ident_accuracy']:.4f}   [{bands['ident_accuracy']}]")
    print(f"  recon_cos_median = {metrics['recon_cosine_median']:.4f}   [{bands['recon_cosine_median']}]")
    print(f"  CVR              = {metrics['constraint_violation_rate']:.4f}   [{bands['constraint_violation_rate']}]")
    print(f"  ---")
    print(f"  OVERALL: {overall}")
    print(f"==============================================================")
    return 0 if overall != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())