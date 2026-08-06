"""Stretch T30A — Real MoS2 demo (replace synthetic spike).

Loads the real MoS2 spectrum, resamples to AA grid, runs through
predict(), and produces a thesis-quality demo report that replaces
the synthetic-spike demo 3 from Phase C.

Output:
    results/reports/demo_3_real_mos2/         (4 files: .md, .json, 3 PNGs)
    results/stretch/t30a_mos2_handover.md     (paste to report-writing chat)

Usage:
    python scripts/stretch/run_t30a_mos2.py
    python scripts/stretch/run_t30a_mos2.py --mc 30   # fewer MC samples

Estimated runtime: ~2-3 min on CPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest import result

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.inference.predict import predict
from src.inference.report import generate_report, save_report
from src.inference.visualize import plot_all
from _handover_utils import to_vector, get_recon_cosine, get_ood, get_peaks, COMPOUND_ORDER

COMPOUND_ORDER = [
    "Alanine", "Asparagine", "Aspartic Acid",
    "Glutamic Acid", "Histidine", "Glucosamine",
]


def load_benchmark_context() -> dict:
    """Return benchmark numbers if available."""
    ctx = {"test_n": 540}
    p = Path("results/benchmark_table.json")
    if not p.exists():
        return ctx
    try:
        data = json.loads(p.read_text())
        rows = data if isinstance(data, list) else [data]
        for row in rows:
            name = str(row.get("name", "")).lower()
            mae = row.get("quant_mae")
            if mae is None:
                continue
            if "pca" in name:
                ctx["pca_svm_mae"] = mae
            elif "resnet" in name and "physics" not in name:
                ctx["resnet_only_mae"] = mae
            elif "ours" in name or "physics" in name:
                ctx["ours_mae"] = mae
    except Exception:
        pass
    return ctx


def write_handover(result: dict, n_mc: int, out_root: Path) -> Path:
    """Generate handover Markdown summary."""
    lines = [
        "# Stretch T30A — Real MoS₂ Demo — Handover",
        "",
        "**Context:** thay synthetic 380 cm⁻¹ Gaussian spike trong demo 3 (Phase C)",
        "bằng phổ MoS₂ thực được tự tay đo. Đây là OOD case 'sạch' nhất vì MoS₂",
        "là chất rắn vô cơ hoàn toàn ngoài domain amino acid.",
        "",
        "## Kết quả",
        "",
        f"- Input: `data/raw/ood_demo/MoS2-160o-12h-ph5.txt`",
        f"- MC samples: {n_mc}",
        f"- Output folder: `results/reports/demo_3_real_mos2/`",
        "",
    ]

    if result:
        comp_raw = result.get("composition_mean", [])
        std_raw = result.get("composition_std", [])

        def _to_vector(x):
            """Normalize composition output to a list of 6 floats aligned with COMPOUND_ORDER.

            Handles: dict[name→float], torch.Tensor, np.ndarray, list, list-of-lists.
            """
            if x is None:
                return [0.0] * len(COMPOUND_ORDER)
            # dict keyed by compound name → align to COMPOUND_ORDER
            if isinstance(x, dict):
                return [float(x.get(c, 0.0)) for c in COMPOUND_ORDER]
            # Tensor → ndarray → list
            if hasattr(x, "detach"):
                x = x.detach().cpu().numpy()
            if hasattr(x, "tolist"):
                x = x.tolist()
            # Unbatch: [[a,b,c,...]] → [a,b,c,...]
            if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
                x = x[0]
            if not isinstance(x, list):
                x = list(x)
            return [float(v) for v in x]

        comp = _to_vector(comp_raw)
        std = _to_vector(std_raw)

        lines.append("### Composition do model dự đoán")
        lines.append("")
        lines.append("Model bị buộc gán MoS₂ vào 6-simplex AA (không có ground truth).")
        lines.append("Composition KHÔNG có ý nghĩa hoá học — chỉ là 'chữ ký AA gần nhất'.")
        lines.append("")
        lines.append("| Compound | Predicted | ±Std |")
        lines.append("|---|---|---|")
        for i, c in enumerate(COMPOUND_ORDER):
            v = float(comp[i]) if i < len(comp) else 0.0
            s = float(std[i]) if i < len(std) else 0.0
            lines.append(f"| {c} | {v:.3f} | ±{s:.3f} |")
        lines.append("")

        # Reconstruction
        recon_cos = result.get("recon_cosine", None)
        if recon_cos is None and "reconstruction" in result:
            # try compute later
            pass
        if recon_cos is not None:
            lines.append(f"### Reconstruction cosine: **{float(recon_cos):.4f}**")
            lines.append("")
            lines.append("So với demo 1 (ID Histidine): 0.965")
            lines.append("So với demo 3 synthetic spike (Phase C): 0.943")
            v = float(recon_cos)
            if v < 0.9:
                lines.append(f"→ Cosine = {v:.3f} THẤP nhiều — physics constraint phát hiện input không thể tái dựng bằng 6 AA refs. **Đúng behavior cho OOD.**")
            else:
                lines.append(f"→ Cosine = {v:.3f} cao hơn dự kiến — model 'gắng' tái dựng dùng AA refs gần nhất.")
            lines.append("")

        # OOD
        ood = result.get("ood", None)
        if ood and isinstance(ood, dict):
            score = ood.get("score", None)
            is_ood = ood.get("is_ood", None)
            lines.append(f"### OOD verdict")
            lines.append("")
            lines.append(f"- **Score**: {score}")
            lines.append(f"- **Flag**: {'OOD ✓ (đúng)' if is_ood else 'ID ✗ (false negative)'}")
            comp_dict = ood.get("components", {})
            if comp_dict:
                lines.append(f"- Reconstruction error component: {comp_dict.get('recon_err', 'N/A')}")
                lines.append(f"- Predictive variance component: {comp_dict.get('pred_var', 'N/A')}")
            lines.append("")
            lines.append("**So sánh với demo 3 synthetic spike (Phase C):** OOD = 0.916 (flagged ID, false negative)")
            lines.append("")
            if is_ood:
                lines.append("Kết luận: thay synthetic bằng real MoS₂ → OOD detector hoạt động đúng.")
            else:
                lines.append("Kết luận: ngay cả real MoS₂ cũng không bị flag → OOD threshold quá cao.")
                lines.append("Hypothesis: threshold p95 calibrate trên val set Split A vẫn chưa đủ discriminative.")

        # Peaks
        peaks = result.get("peaks", []) or []
        if len(peaks) > 0:
            lines.append("")
            lines.append(f"### Peak detection: {len(peaks)} peaks detected")
            lines.append("")
            n_matched = sum(1 for p in peaks if p.get("matched_to") or p.get("compounds"))
            lines.append(f"- {n_matched} matched với DB")
            lines.append(f"- {len(peaks) - n_matched} unmatched (likely lattice modes)")
            lines.append("")
            lines.append("MoS₂ chính peaks (literature):")
            lines.append("- E2g mode ~ 380 cm⁻¹")
            lines.append("- A1g mode ~ 408 cm⁻¹")

    # Cách dùng trong báo cáo
    lines += [
        "",
        "---",
        "",
        "## Cách dùng kết quả này trong báo cáo",
        "",
        "**Trong Chương 3.6 — Demo case studies**: thay 'Demo 3 — Hard OOD Synthetic',",
        "đổi tên thành 'Demo 3 — Real OOD MoS₂', cập nhật:",
        "",
        "- Input: phổ MoS₂ tự đo (thay synthetic spike)",
        "- Composition prediction: như bảng trên",
        "- Reconstruction cosine: số mới (kỳ vọng thấp hơn 0.943)",
        "- OOD verdict: nếu bây giờ đúng → success story, nếu vẫn fail → consistent limitation",
        "",
        "**Trong Chương 3.7.2 — Hạn chế**: cập nhật mục 3 (OOD detection):",
        "- TRƯỚC: 'OOD detector chỉ test trên synthetic spike, demo 3 false negative'",
        "- SAU: 'OOD detector đã test trên real MoS₂. Verdict: {kết quả thực tế}'",
        "",
        "**Hình mới cần chèn vào báo cáo**:",
        "- `results/reports/demo_3_real_mos2/demo_3_real_mos2_reconstruction.png`",
        "- `results/reports/demo_3_real_mos2/demo_3_real_mos2_peaks.png`",
        "- `results/reports/demo_3_real_mos2/demo_3_real_mos2_ood.png`",
    ]

    out_root.mkdir(parents=True, exist_ok=True)
    handover_path = out_root / "t30a_mos2_handover.md"
    handover_path.write_text("\n".join(lines), encoding="utf-8")
    return handover_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mc", type=int, default=50)
    p.add_argument("--mos2-path", default="data/raw/ood_demo/MoS2-160o-12h-ph5.txt")
    p.add_argument("--output-dir", default="results/reports")
    args = p.parse_args()

    # Load MoS2
    from src.data.mos2_loader import load_mos2_spectrum
    print("[T30A] Loading real MoS₂ spectrum ...")
    mos2_path = Path(args.mos2_path)
    if not mos2_path.exists():
        print(f"  ERROR: {mos2_path} not found"); sys.exit(1)
    spec = load_mos2_spectrum(mos2_path, apply_preprocessing=True)
    print(f"  resampled: {spec.shape}, range [{spec.min():.3f}, {spec.max():.3f}]")

    # Run predict
    print(f"[T30A] Running predict() with {args.mc} MC samples ...")
    result = predict(spec, n_mc_samples=args.mc, skip_ood=False, verbose=False)

    # Save outputs
    sample_id = "demo_3_real_mos2"
    demo_dir = Path(args.output_dir) / sample_id
    demo_dir.mkdir(parents=True, exist_ok=True)

    print("[T30A] Generating plots ...")
    fig_paths = plot_all(result, output_dir=demo_dir, prefix=sample_id, show=False)
    image_paths = {k: v.name for k, v in fig_paths.items()}

    print("[T30A] Generating report ...")
    report = generate_report(
        result,
        sample_id=sample_id,
        ground_truth=None,
        benchmark_context=load_benchmark_context(),
        image_paths={
            "reconstruction": image_paths.get("reconstruction", ""),
            "peaks": image_paths.get("peaks", ""),
            "ood": image_paths.get("ood", ""),
        },
    )
    paths = save_report(report, demo_dir, base_name=sample_id)
    print(f"  saved: {paths['markdown']}")
    print(f"  plain_text: {report['plain_text']}")

    # Write handover
    handover = write_handover(result, args.mc, Path("results/stretch"))
    print(f"\n[T30A handover] {handover}")
    print("\n[T30A done]")


if __name__ == "__main__":
    main()
