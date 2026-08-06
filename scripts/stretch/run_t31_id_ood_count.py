"""Stretch T31 — Per-sample ID/OOD count, PAIRED across two models.

Chạy CÙNG hai lớp mẫu qua HAI model rồi so sánh:
    - Lớp ID  = AA-pure (mineral = 0)              -> nhãn thật ID
    - Lớp OOD = AAM mineral-rich (mineral > 5%)     -> nhãn thật OOD

    Model A = AA-only 6-output      ("before retrain")
    Model B = AAM-retrained 7-output ("after retrain")

Với mỗi model: gán nhãn ID/OOD per-sample (score > threshold => OOD, đúng logic
``OODScorer.is_ood``), đếm confusion matrix (TP/TN/FP/FN), tính accuracy /
precision / recall / specificity / F1, rồi đặt hai model cạnh nhau.

DIỄN GIẢI KỲ VỌNG (quan trọng cho thesis):
    - Model A (AA-only): mineral-rich là OOD thật. Model KHÔNG tái dựng được
      (thiếu mineral references) -> OOD score cao -> phát hiện đúng.
      Kỳ vọng: accuracy cao, TP cao, FN thấp.  (đây là kết quả AUROC 0.99 ở T29B)
    - Model B (AAM-retrained): đã học mineral -> tái dựng mineral-rich TỐT ->
      OOD score THẤP -> model coi mineral-rich là ID.
      Kỳ vọng: TP thấp, FN cao (xét theo nhãn "OOD" cũ).
      Đây KHÔNG phải lỗi: nó chứng minh modular design — cấp thêm data thì
      mineral chuyển từ OOD sang ID. Bảng so sánh làm nổi bật tương phản này.

Quy ước "đúng / sai" (nhãn thật ID=0, OOD=1; gán OOD nếu score > threshold):
    ID  thật, gán ID  -> ĐÚNG (TN)      ID  thật, gán OOD -> SAI (FP, báo nhầm)
    OOD thật, gán OOD -> ĐÚNG (TP)      OOD thật, gán ID  -> SAI (FN, bỏ sót)

Usage:
    # Mặc định: cả hai model với đường dẫn chuẩn
    python scripts/stretch/run_t31_id_ood_count.py

    # Cân bằng hai lớp + chỉ định checkpoint
    python scripts/stretch/run_t31_id_ood_count.py --n-id 973 \
        --ckpt-a checkpoints/best.pt \
        --ckpt-b checkpoints/aam_retrained_v2/best.pt

    # Chỉ chạy MỘT model (bỏ qua model kia bằng "none")
    python scripts/stretch/run_t31_id_ood_count.py --ckpt-b none

Output (results/stretch/t31_id_ood_count/):
    summary.json        kết quả cả hai model + so sánh
    per_sample.csv      mỗi (phổ × model) một dòng: score, nhãn thật, nhãn gán, đúng/sai
    comparison.md       bảng so sánh BEFORE vs AFTER (Markdown, để dán vào báo cáo)
    distribution.png    histogram score + confusion matrix cho từng model
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


# ───────────────────────────── helpers ──────────────────────────────────────

def _first_existing(paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None


def _load_any(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    t = torch.load(path, weights_only=True)
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy().astype(np.float32)
    return np.asarray(t, dtype=np.float32)


def _to_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3 and x.shape[1] == 1:
        return x[:, 0, :]
    return x


def _load_checkpoint_state(path: str):
    """Load state_dict; bóc tách checkpoint bọc {'model': ...} nếu cần.

    (Fix cho lỗi state_dict mismatch đã gặp: checkpoint best.pt thực ra là dict
    {'model', 'optimizer', 'scheduler', 'epoch', ...} chứ không phải raw state.)
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    return ckpt


# ──────────────────────── chấm điểm một model ───────────────────────────────

def evaluate_model(name, ckpt, ref_path, n_compounds, Xid, Xood,
                   *, mc, recon_w, var_w, batch_size, manual_threshold):
    """Calibrate + score + đếm confusion matrix cho MỘT model.

    Returns dict gồm: threshold, score_id, score_ood, pred_id, pred_ood,
    confusion (TP/TN/FP/FN), metrics.
    """
    from src.models.full_model import RamanPhysicsAI
    from src.inference.ood import OODScorer

    n_id, n_ood = len(Xid), len(Xood)
    print(f"\n[T31] === Model '{name}' (ckpt={ckpt}, n_compounds={n_compounds}) ===")

    model = RamanPhysicsAI(reference_spectra=ref_path,
                           n_compounds=n_compounds,
                           spectrum_length=Xood.shape[1])
    model.load_state_dict(_load_checkpoint_state(ckpt))
    model.eval()

    # calibrate trên lớp ID -> threshold
    id_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xid).float().unsqueeze(1)),
        batch_size=batch_size, shuffle=False)
    scorer = OODScorer(model, recon_weight=recon_w,
                       var_weight=var_w, mc_samples=mc)
    cal = scorer.calibrate(id_loader)
    threshold = manual_threshold if manual_threshold is not None else cal.score_p95
    print(f"[T31]   threshold = {threshold:.4f} "
          f"({'manual' if manual_threshold is not None else 'calibrated score_p95'})")

    def score_block(X, tag=""):
        # Chấm theo mini-batch để tránh OOM: nạp cả N phổ × 50 MC cùng lúc
        # sẽ ngốn RAM khổng lồ và bị OS kill lặng lẽ (không traceback).
        out = []
        n = len(X)
        for i in range(0, n, batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).float().unsqueeze(1)
            comp = scorer.score_batch(xb, return_components=True)
            out.append(comp["score"].cpu().numpy())
            print(f"\r[T31]   scoring {tag}: {min(i + batch_size, n)}/{n}",
                  end="", flush=True)
        print()
        return np.concatenate(out)

    print(f"[T31]   scoring ID ({n_id}) + OOD ({n_ood}) ...")
    score_id = score_block(Xid, "ID")
    score_ood = score_block(Xood, "OOD")

    pred_id = (score_id > threshold).astype(int)    # 1 = gán OOD (sai cho lớp ID)
    pred_ood = (score_ood > threshold).astype(int)  # 1 = gán OOD (đúng cho lớp OOD)

    TN = int((pred_id == 0).sum())
    FP = int((pred_id == 1).sum())
    TP = int((pred_ood == 1).sum())
    FN = int((pred_ood == 0).sum())
    total = TN + FP + TP + FN
    n_correct, n_wrong = TP + TN, FP + FN

    eps = 1e-12
    metrics = {
        "accuracy": n_correct / total,
        "precision": TP / (TP + FP + eps),
        "recall_tpr": TP / (TP + FN + eps),
        "specificity": TN / (TN + FP + eps),
        "fpr": FP / (FP + TN + eps),
        "f1": 2 * TP / (2 * TP + FP + FN + eps),
    }

    print(f"[T31]   ĐÚNG {n_correct}/{total} ({100*metrics['accuracy']:.2f}%)  "
          f"| TP={TP} TN={TN} FP={FP} FN={FN}")
    print(f"[T31]   score mean: ID={score_id.mean():.4f}  OOD={score_ood.mean():.4f}")

    return {
        "name": name, "ckpt": str(ckpt), "n_compounds": n_compounds,
        "threshold": float(threshold),
        "threshold_source": ("manual" if manual_threshold is not None
                             else "calibrated_score_p95"),
        "score_id": score_id, "score_ood": score_ood,
        "pred_id": pred_id, "pred_ood": pred_ood,
        "confusion": {"TP": TP, "TN": TN, "FP": FP, "FN": FN},
        "n_total": total, "n_correct": n_correct, "n_wrong": n_wrong,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "score_stats": {
            "id": {"mean": float(score_id.mean()),
                   "median": float(np.median(score_id))},
            "ood": {"mean": float(score_ood.mean()),
                    "median": float(np.median(score_ood))},
        },
    }


# ───────────────────────────────── main ─────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aam-dir", default="data/processed/aam")
    p.add_argument("--aa-spectra", default=None)
    p.add_argument("--aa-labels", default=None)
    # Model A — AA-only (before retrain)
    p.add_argument("--ckpt-a", default="checkpoints/best.pt",
                   help="Checkpoint model A (AA-only 6-output). 'none' để bỏ qua.")
    p.add_argument("--ref-a", default="engine/reference_spectra.npy")
    p.add_argument("--ncomp-a", type=int, default=6)
    # Model B — AAM-retrained (after retrain)
    p.add_argument("--ckpt-b", default="checkpoints/aam_retrained_v2/best.pt",
                   help="Checkpoint model B (AAM-retrained 7-output). 'none' để bỏ qua.")
    p.add_argument("--ref-b", default="engine/reference_spectra_aam.npy")
    p.add_argument("--ncomp-b", type=int, default=7)
    # chung
    p.add_argument("--mineral-threshold", type=float, default=0.05)
    p.add_argument("--n-id", type=int, default=None)
    p.add_argument("--mc", type=int, default=50)
    p.add_argument("--recon-weight", type=float, default=0.6)
    p.add_argument("--var-weight", type=float, default=0.4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=None,
                   help="Ép threshold thủ công cho CẢ hai model. None = score_p95.")
    p.add_argument("--out-dir", default="results/stretch/t31_id_ood_count")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Lớp OOD: mineral-rich từ AAM test split ----
    aam_dir = Path(args.aam_dir)
    aam_spec = _to_2d(_load_any(aam_dir / "spectra.pt"))
    aam_lab7 = _load_any(aam_dir / "labels_7d.pt")
    split = json.loads((aam_dir / "split.json").read_text())
    Xood = aam_spec[split["test"]]
    minerals = aam_lab7[split["test"]][:, 6]
    Xood = Xood[minerals > args.mineral_threshold]
    n_ood = len(Xood)
    print(f"[T31] OOD class (mineral-rich, AAM test): {n_ood}")

    # ---- 2. Lớp ID: AA-pure (mineral = 0) ----
    aa_spec_path = _first_existing([
        args.aa_spectra, "data/processed/spectra.pt",
        "data/processed/spectra_full.pt", "data/processed/X.pt",
        "data/processed/X.npy",
    ])
    aa_lab_path = _first_existing([
        args.aa_labels, "data/processed/labels.pt",
        "data/processed/labels_6d.pt", "data/processed/Y.pt",
        "data/processed/Y.npy",
    ])
    if aa_spec_path is None:
        print("\n[FATAL] Không tìm thấy cache phổ AA. Chạy lại với --aa-spectra.")
        sys.exit(2)
    print(f"[T31] AA spectra cache: {aa_spec_path}")

    Xaa = _to_2d(_load_any(aa_spec_path))
    if aa_lab_path is not None:
        Yaa = _load_any(aa_lab_path)
        Xid = Xaa[Yaa[:, 6] <= args.mineral_threshold] if Yaa.shape[1] >= 7 else Xaa
    else:
        Xid = Xaa

    if Xid.shape[1] != Xood.shape[1]:
        print(f"\n[FATAL] Grid mismatch: AA P={Xid.shape[1]} vs AAM P={Xood.shape[1]}.")
        sys.exit(3)

    rng = np.random.default_rng(args.seed)
    if args.n_id is not None and args.n_id < len(Xid):
        Xid = Xid[rng.choice(len(Xid), size=args.n_id, replace=False)]
    n_id = len(Xid)
    print(f"[T31] ID class (low-mineral, AA-pure): {n_id}")
    if n_id == 0 or n_ood == 0:
        print("\n[FATAL] Một lớp rỗng; không thể đếm.")
        sys.exit(4)

    # ---- 3. Chấm điểm từng model ----
    results = []
    if args.ckpt_a.lower() != "none":
        results.append(evaluate_model(
            "AA-only (before retrain)", args.ckpt_a, args.ref_a, args.ncomp_a,
            Xid, Xood, mc=args.mc, recon_w=args.recon_weight,
            var_w=args.var_weight, batch_size=args.batch_size,
            manual_threshold=args.threshold))
    if args.ckpt_b.lower() != "none":
        results.append(evaluate_model(
            "AAM-retrained (after retrain)", args.ckpt_b, args.ref_b, args.ncomp_b,
            Xid, Xood, mc=args.mc, recon_w=args.recon_weight,
            var_w=args.var_weight, batch_size=args.batch_size,
            manual_threshold=args.threshold))
    if not results:
        print("\n[FATAL] Không có model nào được chấm (cả hai --ckpt = none).")
        sys.exit(5)

    # ---- 4. summary.json ----
    def _strip_arrays(r):
        return {k: v for k, v in r.items()
                if k not in ("score_id", "score_ood", "pred_id", "pred_ood")}
    summary = {
        "task": "T31 paired per-sample ID/OOD count (AA=ID, AAM=OOD)",
        "n_id_true": n_id, "n_ood_true": n_ood, "n_total": n_id + n_ood,
        "mineral_threshold": args.mineral_threshold,
        "weights": {"recon": args.recon_weight, "var": args.var_weight},
        "mc_samples": args.mc,
        "models": [_strip_arrays(r) for r in results],
        "sources": {"aa_spectra": str(aa_spec_path), "aam_dir": str(aam_dir)},
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    # ---- 5. per_sample.csv (mỗi phổ × model một dòng) ----
    with (out_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "idx", "source", "true_label",
                                          "ood_score", "pred_label", "correct"])
        w.writeheader()
        for r in results:
            for i, (s, pr) in enumerate(zip(r["score_id"], r["pred_id"])):
                w.writerow({"model": r["name"], "idx": i, "source": "AA_pure",
                            "true_label": "ID", "ood_score": round(float(s), 6),
                            "pred_label": "OOD" if pr == 1 else "ID",
                            "correct": int(pr == 0)})
            for i, (s, pr) in enumerate(zip(r["score_ood"], r["pred_ood"])):
                w.writerow({"model": r["name"], "idx": i,
                            "source": "AAM_mineral_rich", "true_label": "OOD",
                            "ood_score": round(float(s), 6),
                            "pred_label": "OOD" if pr == 1 else "ID",
                            "correct": int(pr == 1)})

    # ---- 6. comparison.md ----
    lines = [
        "# T31 — So sánh nhận định ID/OOD: AA-only vs AAM-retrained",
        "",
        f"Cùng test set: ID = {n_id} phổ AA-pure (mineral≈0), "
        f"OOD = {n_ood} phổ AAM mineral-rich (mineral>{args.mineral_threshold}).",
        "",
        "## Bảng tổng hợp",
        "",
        "| Chỉ số | " + " | ".join(r["name"] for r in results) + " |",
        "|---|" + "---|" * len(results),
    ]

    def row(label, fn):
        return "| " + label + " | " + " | ".join(fn(r) for r in results) + " |"

    lines += [
        row("Số mẫu thử", lambda r: str(r["n_total"])),
        row("Nhận định ĐÚNG", lambda r: f"{r['n_correct']} ({100*r['metrics']['accuracy']:.1f}%)"),
        row("Nhận định SAI", lambda r: str(r["n_wrong"])),
        row("TP (OOD→OOD, đúng)", lambda r: str(r["confusion"]["TP"])),
        row("TN (ID→ID, đúng)", lambda r: str(r["confusion"]["TN"])),
        row("FP (ID→OOD, báo nhầm)", lambda r: str(r["confusion"]["FP"])),
        row("FN (OOD→ID, bỏ sót)", lambda r: str(r["confusion"]["FN"])),
        row("Accuracy", lambda r: f"{r['metrics']['accuracy']:.4f}"),
        row("Recall / TPR", lambda r: f"{r['metrics']['recall_tpr']:.4f}"),
        row("Specificity", lambda r: f"{r['metrics']['specificity']:.4f}"),
        row("F1", lambda r: f"{r['metrics']['f1']:.4f}"),
        row("OOD score TB (lớp ID)", lambda r: f"{r['score_stats']['id']['mean']:.4f}"),
        row("OOD score TB (lớp OOD)", lambda r: f"{r['score_stats']['ood']['mean']:.4f}"),
        "",
        "## Diễn giải",
        "",
        "**AA-only (before retrain):** mineral-rich là OOD thật. Model thiếu "
        "mineral references nên KHÔNG tái dựng được → OOD score cao → phát hiện "
        "đúng. Accuracy cao xác nhận model 'biết khi không biết'.",
        "",
        "**AAM-retrained (after retrain):** model đã học mineral fingerprint nên "
        "tái dựng tốt cả phổ mineral-rich → OOD score thấp → coi mineral-rich là "
        "ID. Do đó TP giảm / FN tăng *khi xét theo nhãn OOD cũ*. Đây KHÔNG phải "
        "lỗi: nó chứng minh khi cấp thêm dữ liệu huấn luyện, mẫu trước đây là OOD "
        "chuyển thành ID — đúng tinh thần modular design.",
        "",
        "**Kết luận cho thesis:** OOD detector hoạt động đúng (model A bắt được "
        "mẫu lạ); và framework mở rộng được (model B hấp thụ được phân phối mới). "
        "Hai kết quả bổ trợ nhau, củng cố cho thiết kế physics + OOD.",
    ]
    (out_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- 7. distribution.png (mỗi model 2 ô: histogram + confusion matrix) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        nm = len(results)
        fig, axes = plt.subplots(nm, 2, figsize=(13, 4.5 * nm),
                                 gridspec_kw={"width_ratios": [2, 1]})
        if nm == 1:
            axes = axes[None, :]
        for k, r in enumerate(results):
            ax0, ax1 = axes[k, 0], axes[k, 1]
            si, so = r["score_id"], r["score_ood"]
            thr = r["threshold"]
            cm = r["confusion"]
            bins = np.linspace(min(si.min(), so.min()),
                               max(si.max(), so.max()), 35)
            ax0.hist(si, bins=bins, alpha=0.6, color="#2a9d8f",
                     label=f"ID / low-mineral (n={n_id})")
            ax0.hist(so, bins=bins, alpha=0.6, color="#e76f51",
                     label=f"OOD / mineral-rich (n={n_ood})")
            ax0.axvline(thr, color="k", ls="--", lw=1.5,
                        label=f"threshold = {thr:.3f}")
            ax0.set_xlabel("OOD score"); ax0.set_ylabel("Số mẫu")
            ax0.set_title(f"{r['name']}  —  accuracy = {r['metrics']['accuracy']:.3f}")
            ax0.legend(fontsize=8); ax0.grid(alpha=0.3)

            ax1.axis("off")
            mat = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
            ax1.imshow(mat, cmap="Blues", alpha=0.5)
            ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
            ax1.set_xticklabels(["Gán ID", "Gán OOD"])
            ax1.set_yticklabels(["Thật ID", "Thật OOD"])
            ax1.xaxis.set_ticks_position("top")
            ax1.xaxis.set_label_position("top")
            txt = [[f"TN\n{cm['TN']}", f"FP\n{cm['FP']}"],
                   [f"FN\n{cm['FN']}", f"TP\n{cm['TP']}"]]
            for rr in range(2):
                for cc in range(2):
                    ax1.text(cc, rr, txt[rr][cc], ha="center", va="center",
                             fontsize=12, fontweight="bold")
            ax1.set_title("Confusion matrix", pad=28)
            for sp in ax1.spines.values():
                sp.set_visible(False)
        fig.tight_layout()
        fig.savefig(out_dir / "distribution.png", dpi=130)
        plt.close(fig)
        print("\n[T31] saved distribution.png")
    except Exception as e:
        print(f"\n[T31] (plot skipped: {e})")

    # ---- 8. In bảng so sánh ra console ----
    print("\n" + "=" * 64)
    print("  SO SÁNH NHẬN ĐỊNH ID / OOD")
    print("=" * 64)
    for r in results:
        m, c = r["metrics"], r["confusion"]
        print(f"  [{r['name']}]")
        print(f"    ĐÚNG {r['n_correct']}/{r['n_total']} "
              f"({100*m['accuracy']:.2f}%)  SAI {r['n_wrong']}")
        print(f"    TP={c['TP']} TN={c['TN']} FP={c['FP']} FN={c['FN']}  "
              f"| Recall={m['recall_tpr']:.3f} Spec={m['specificity']:.3f} "
              f"F1={m['f1']:.3f}")
    print("=" * 64)
    print(f"\n[T31 done] -> {out_dir}")
    print("  - summary.json")
    print("  - per_sample.csv")
    print("  - comparison.md   <- bảng BEFORE vs AFTER, dán thẳng vào báo cáo")
    print("  - distribution.png")


if __name__ == "__main__":
    main()