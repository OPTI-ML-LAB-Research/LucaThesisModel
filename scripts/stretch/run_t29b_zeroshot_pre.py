"""Stretch T29B — Zero-shot test of AA-only model on AAM test set.

This is the BEFORE side of the paired comparison. The AA-trained model
(checkpoints/best.pt with 6 outputs) sees the AAM test set for the first
time. Expected behaviour:
- Composition prediction "force-fitted" into 6 AA outputs (cannot
  predict minerals).
- For mineral-rich samples: high OOD score, low recon cosine.
- For low-mineral samples: should look like normal AA mixtures.

Output:
    results/stretch/t29b_zeroshot_pre/
        results.json       (per-sample metrics)
        summary.json       (aggregate)
        histograms.png
    results/stretch/t29b_handover.md

Usage:
    python scripts/stretch/run_t29b_zeroshot_pre.py
    python scripts/stretch/run_t29b_zeroshot_pre.py --mc 15 --max-samples 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Make the project root importable (parents[2] = repo root) AND this
# script's own dir (for _handover_utils). Insert BEFORE importing the
# helper so the import below resolves regardless of CWD.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR.parents[1]))

from _handover_utils import to_vector, get_recon_cosine, get_ood  # noqa: E402


def _fmt(v, nd: int = 4) -> str:
    """Format a number, or return 'N/A' for None / non-numeric.

    Note: bool is a subclass of int, but we only ever pass float-valued
    metrics (means, medians, AUROC) here, so this is safe.
    """
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else "N/A"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aam-dir", default="data/processed/aam")
    p.add_argument("--mc", type=int, default=30, help="MC samples (lower=faster)")
    p.add_argument("--out-dir", default="results/stretch/t29b_zeroshot_pre")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap test samples (default: use all)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aam_dir = Path(args.aam_dir)
    if not (aam_dir / "spectra.pt").exists():
        print(f"  ERROR: {aam_dir / 'spectra.pt'} not found. Run T29A first.")
        sys.exit(1)

    # ---- Load AAM test split ----
    print("\n[T29B] Loading AAM test set ...")
    spectra = torch.load(aam_dir / "spectra.pt", weights_only=True).numpy()
    labels7 = torch.load(aam_dir / "labels_7d.pt", weights_only=True).numpy()
    has_minerals = np.load(aam_dir / "has_minerals.npy")
    split = json.loads((aam_dir / "split.json").read_text())
    test_idx = split["test"]
    if args.max_samples:
        test_idx = test_idx[:args.max_samples]
    print(f"  using {len(test_idx)} test samples")

    n_low = int((~has_minerals[test_idx]).sum())
    n_high = int(has_minerals[test_idx].sum())
    print(f"  low-mineral: {n_low}, mineral-rich: {n_high}")
    if n_low == 0:
        print("  [note] this test split has 0 low-mineral samples; "
              "low-mineral aggregates and the mineral-vs-low AUROC will be N/A.")

    # ---- Load model + scorer via predict() lazy cache ----
    print("\n[T29B] Loading AA-only model + OOD scorer ...")
    from src.inference.predict import predict, reset_cache
    reset_cache()
    # Prime the cache (lazy-loads model + OOD scorer on first call).
    _ = predict(spectra[test_idx[0]], skip_ood=False, n_mc_samples=10, verbose=False)
    from src.inference.predict import _resources
    scorer = getattr(_resources, "ood_scorer", None)
    if scorer is None:
        print("  [warn] OOD scorer not loaded (calibration file missing?). "
              "OOD scores will be None; recon cosine still computed.")

    # ---- Score all test samples ----
    print(f"\n[T29B] Running predict() on {len(test_idx)} samples (mc={args.mc}) ...")
    results = []
    for i, idx in enumerate(test_idx):
        res = predict(spectra[idx], n_mc_samples=args.mc, skip_ood=False, verbose=False)
        # Use shared helpers (handle every field-name / shape variant).
        comp_val = res.get("composition_mean")
        if comp_val is None:
            comp_val = res.get("composition")
        comp = to_vector(comp_val)

        recon_cos = get_recon_cosine(res)        # reads "recon_cosine_sim" correctly
        ood = get_ood(res)                        # reads flat "ood_score"/"is_ood"
        results.append({
            "test_idx": int(idx),
            "has_minerals": bool(has_minerals[idx]),
            "true_minerals_frac": float(labels7[idx, 6]),
            "composition_pred_6aa": [float(v) for v in comp],
            "recon_cosine": recon_cos,
            "ood_score": ood["score"],
            "ood_flag": ood["is_ood"],
        })
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(test_idx)}")

    # ---- Aggregate metrics ----
    def get(name, mask=None):
        if mask is None:
            vals = [r[name] for r in results if r[name] is not None]
        else:
            vals = [r[name] for r, m in zip(results, mask)
                    if r[name] is not None and m]
        return np.array(vals, dtype=np.float64)

    has_mins = np.array([r["has_minerals"] for r in results])
    recon_low = get("recon_cosine", ~has_mins)
    recon_high = get("recon_cosine", has_mins)
    ood_low = get("ood_score", ~has_mins)
    ood_high = get("ood_score", has_mins)

    def _stats(arr):
        return {
            "mean": float(arr.mean()) if len(arr) else None,
            "median": float(np.median(arr)) if len(arr) else None,
        }

    summary = {
        "task": "T29B AAM zero-shot test (AA-only model)",
        "n_samples": len(results),
        "n_low_mineral": int((~has_mins).sum()),
        "n_mineral_rich": int(has_mins.sum()),
        "mc_samples": args.mc,
        "recon_cosine": {
            "low_mineral": _stats(recon_low),
            "mineral_rich": _stats(recon_high),
        },
        "ood_score": {
            "low_mineral": _stats(ood_low),
            "mineral_rich": _stats(ood_high),
        },
    }

    # AUROC for mineral_rich vs low_mineral discrimination (needs both classes).
    if len(ood_low) and len(ood_high):
        from src.eval.metrics import ood_auroc
        auroc = ood_auroc(id_scores=ood_low, ood_scores=ood_high)
        summary["mineral_ood_auroc"] = float(auroc)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[T29B] saved {out_dir}/")
    print(f"  summary: {json.dumps(summary, indent=2)}")

    # ---- Plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        if len(recon_low) and len(recon_high):
            axes[0].hist(recon_low, bins=30, alpha=0.6, color="#2a9d8f",
                         label=f"Low mineral (n={len(recon_low)})")
            axes[0].hist(recon_high, bins=30, alpha=0.6, color="#e76f51",
                         label=f"Mineral-rich (n={len(recon_high)})")
        elif len(recon_high):
            axes[0].hist(recon_high, bins=30, alpha=0.7, color="#e76f51",
                         label=f"Mineral-rich (n={len(recon_high)})")
        axes[0].set_xlabel("Reconstruction cosine")
        axes[0].set_ylabel("count")
        axes[0].set_title("AAM zero-shot: Recon cosine")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        if len(ood_low) and len(ood_high):
            axes[1].hist(ood_low, bins=30, alpha=0.6, color="#2a9d8f",
                         label=f"Low mineral (n={len(ood_low)})")
            axes[1].hist(ood_high, bins=30, alpha=0.6, color="#e76f51",
                         label=f"Mineral-rich (n={len(ood_high)})")
        elif len(ood_high):
            axes[1].hist(ood_high, bins=30, alpha=0.7, color="#e76f51",
                         label=f"Mineral-rich (n={len(ood_high)})")
        axes[1].set_xlabel("OOD score")
        axes[1].set_ylabel("count")
        t = "AAM zero-shot: OOD score"
        if "mineral_ood_auroc" in summary:
            t += f" (AUROC={summary['mineral_ood_auroc']:.3f})"
        axes[1].set_title(t)
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / "histograms.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  [warn] plot failed: {e}")

    # ---- Handover ----
    auroc = summary.get("mineral_ood_auroc")
    rc = summary["recon_cosine"]
    od = summary["ood_score"]
    handover = Path("results/stretch/t29b_handover.md")
    handover.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stretch T29B — AAM Zero-shot Test (BEFORE Retrain) — Handover",
        "",
        "**Setup:** model AA-only (6 outputs), test trên AAM test set.",
        "Model **chưa từng thấy minerals**. Câu hỏi: physics + OOD có nhận "
        "biết được mineral-rich samples là OOD?",
        "",
        "## Kết quả",
        "",
        f"- Test n = {summary['n_samples']} "
        f"({summary['n_low_mineral']} low-mineral + "
        f"{summary['n_mineral_rich']} mineral-rich)",
        f"- MC samples = {summary['mc_samples']}",
        "",
        "### Reconstruction cosine",
        "",
        "| Class | Mean | Median |",
        "|---|---|---|",
        f"| Low mineral | {_fmt(rc['low_mineral']['mean'])} | "
        f"{_fmt(rc['low_mineral']['median'])} |",
        f"| Mineral-rich | {_fmt(rc['mineral_rich']['mean'])} | "
        f"{_fmt(rc['mineral_rich']['median'])} |",
        "",
        "### OOD score",
        "",
        "| Class | Mean | Median |",
        "|---|---|---|",
        f"| Low mineral | {_fmt(od['low_mineral']['mean'])} | "
        f"{_fmt(od['low_mineral']['median'])} |",
        f"| Mineral-rich | {_fmt(od['mineral_rich']['mean'])} | "
        f"{_fmt(od['mineral_rich']['median'])} |",
        "",
    ]
    if auroc is not None:
        verdict = ("PASS-target" if auroc >= 0.85 else
                   "PASS-floor" if auroc >= 0.75 else "FAIL")
        lines += [
            f"### **AUROC (mineral_rich vs low_mineral) = {_fmt(auroc)}** "
            f"— {verdict}",
            "",
        ]
    else:
        lines += [
            "### AUROC (mineral_rich vs low_mineral) = N/A",
            "",
            "> Không tính được AUROC vì test split này thiếu một trong hai "
            "lớp (cần cả low-mineral lẫn mineral-rich). Xem ghi chú bên dưới.",
            "",
        ]

    lines += [
        "## Diễn giải",
        "",
        "- Mineral-rich samples chứa quartz (~464 cm⁻¹) và calcite (~1086 cm⁻¹).",
        "- 6 AA pure references KHÔNG thể tái dựng các peaks này.",
        "- Reconstruction error mineral-rich KỲ VỌNG cao hơn low-mineral.",
        "- OOD score (combine recon + variance) KỲ VỌNG cao hơn cho mineral-rich.",
        "",
    ]
    if auroc is not None:
        lines.append(f"AUROC = {_fmt(auroc)} cho thấy mức độ phân biệt thực tế.")
    else:
        lines.append(
            "Vì split chỉ có một lớp, dùng giá trị tuyệt đối của recon "
            "cosine / OOD score (bảng trên) làm bằng chứng thay cho AUROC."
        )
    lines += [
        "",
        "## Files được tạo",
        "",
        f"- `{(out_dir / 'results.json').as_posix()}` -- per-sample records",
        f"- `{(out_dir / 'summary.json').as_posix()}`",
        f"- `{(out_dir / 'histograms.png').as_posix()}`",
        "",
        "## Cách dùng trong báo cáo",
        "",
        "Đây là 'BEFORE' phần của paired comparison. Sau khi T29E xong, "
        "viết Chương 3 subsection:",
        "",
        "'Bảng X — AAM evaluation: AA-only model vs AAM-retrained model "
        "trên cùng test set'",
        "",
        "Highlight:",
    ]
    if auroc is not None:
        lines.append(f"- AA-only thấy mineral-rich là OOD với AUROC = {_fmt(auroc)}")
    else:
        lines.append("- AA-only: dùng recon cosine / OOD score tuyệt đối "
                     "(AUROC N/A do thiếu lớp low-mineral)")
    lines += [
        "- Sau retrain (T29E), composition prediction cho minerals KỲ VỌNG "
        "chính xác",
        "- Đây là evidence cho thesis: physics + OOD work, retrain solves "
        "shortcomings.",
    ]
    handover.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[handover] {handover}")
    print("\n[T29B done]")


if __name__ == "__main__":
    main()