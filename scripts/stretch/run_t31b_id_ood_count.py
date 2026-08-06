"""Stretch T31b — Confusion matrix với THANG + TRỌNG SỐ + THRESHOLD tùy chỉnh.

Khác T31 cũ: T31 cũ gọi ``OODScorer.score_batch`` trực tiếp -> dùng thang
**clip nội tại** của OODScorer, không đổi được sang tanh / no-clip dù truyền
tham số. T31b lấy thành phần THÔ (recon_err_raw + var_raw) rồi rescore qua
``ood_controls.rescore`` (cùng logic với dashboard) -> confusion matrix khớp
100% với verdict bạn thấy trên dashboard.

Mặc định = cấu hình bạn đã chốt qua demo: **tanh, recon_w=0.7, threshold=0.8**.

Usage:
    # Mặc định (tanh, recon_w=0.7, threshold=0.8) — đúng cấu hình bạn chốt
    python scripts/stretch/run_t31b_id_ood_count.py --n-id 973

    # So sánh nhiều cấu hình một lần (chấm thành phần thô một lần, rescore N lần)
    python scripts/stretch/run_t31b_id_ood_count.py --n-id 973 --multi

    # Đổi cấu hình
    python scripts/stretch/run_t31b_id_ood_count.py --n-id 973 \
        --scale "no-clip (giữ độ phân giải far-OOD)" \
        --recon-w 1.0 --threshold 1.5

Output (results/stretch/t31b_id_ood_count/):
    summary.json         confusion matrix + metrics cho mỗi cấu hình
    per_sample.csv       mỗi (phổ × cấu hình) một dòng
    distribution.png     histogram score + confusion matrix
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


def _first_existing(paths):
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None


def _load_any(path):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    t = torch.load(path, weights_only=True)
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy().astype(np.float32)
    return np.asarray(t, dtype=np.float32)


def _to_2d(x):
    return x[:, 0, :] if (x.ndim == 3 and x.shape[1] == 1) else x


def _load_ckpt_state(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return ck["model"] if isinstance(ck, dict) and "model" in ck else ck


def confusion(score_id, score_ood, threshold):
    """Tính TP/TN/FP/FN + metrics từ score đã rescore."""
    pred_id = (score_id > threshold).astype(int)
    pred_ood = (score_ood > threshold).astype(int)
    TN = int((pred_id == 0).sum()); FP = int((pred_id == 1).sum())
    TP = int((pred_ood == 1).sum()); FN = int((pred_ood == 0).sum())
    total = TN + FP + TP + FN
    eps = 1e-12
    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "n_total": total, "n_correct": TP + TN, "n_wrong": FP + FN,
        "metrics": {
            "accuracy": (TP + TN) / total,
            "precision": TP / (TP + FP + eps),
            "recall_tpr": TP / (TP + FN + eps),
            "specificity": TN / (TN + FP + eps),
            "fpr": FP / (FP + TN + eps),
            "f1": 2 * TP / (2 * TP + FP + FN + eps),
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aam-dir", default="data/processed/aam")
    p.add_argument("--aa-spectra", default=None)
    p.add_argument("--aa-labels", default=None)
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--ref-path", default="engine/reference_spectra.npy")
    p.add_argument("--n-compounds", type=int, default=6)
    p.add_argument("--mineral-threshold", type=float, default=0.05)
    p.add_argument("--n-id", type=int, default=None)
    p.add_argument("--mc", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    # cấu hình rescore — mặc định = bạn đã chốt
    p.add_argument("--scale", default="tanh (bão hòa mềm)",
                   help="Thang chuẩn hóa (xem ood_controls.SCALES).")
    p.add_argument("--recon-w", type=float, default=0.7)
    p.add_argument("--threshold", type=float, default=0.8,
                   help="Threshold thủ công (mặc định = 0.8, cấu hình demo).")
    p.add_argument("--multi", action="store_true",
                   help="So sánh nhiều cấu hình: cấu hình bạn chốt + F1-optimal từ T32.")
    p.add_argument("--recommended", default="results/stretch/t32_threshold/recommended.json",
                   help="Đọc cấu hình F1-optimal từ T32 nếu --multi.")
    p.add_argument("--out-dir", default="results/stretch/t31b_id_ood_count")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dashboard"))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    from ood_controls import CalStats, rescore, SCALES

    if args.scale not in SCALES:
        print(f"[FATAL] --scale phải là một trong: {list(SCALES)}")
        sys.exit(1)

    # ---- nạp ID + OOD ----
    aam_dir = Path(args.aam_dir)
    aam_spec = _to_2d(_load_any(aam_dir / "spectra.pt"))
    aam_lab7 = _load_any(aam_dir / "labels_7d.pt")
    test_idx = json.loads((aam_dir / "split.json").read_text())["test"]
    Xood = aam_spec[test_idx]
    Xood = Xood[aam_lab7[test_idx][:, 6] > args.mineral_threshold]
    n_ood = len(Xood)
    print(f"[T31b] OOD (AAM mineral-rich): {n_ood}")

    aa_spec_path = _first_existing([args.aa_spectra, "data/processed/spectra.pt",
                                    "data/processed/spectra_full.pt"])
    aa_lab_path = _first_existing([args.aa_labels, "data/processed/labels.pt",
                                   "data/processed/labels_6d.pt"])
    if aa_spec_path is None:
        print("[FATAL] không tìm thấy cache AA"); sys.exit(2)
    Xaa = _to_2d(_load_any(aa_spec_path))
    if aa_lab_path is not None:
        Yaa = _load_any(aa_lab_path)
        Xid = Xaa[Yaa[:, 6] <= args.mineral_threshold] if Yaa.shape[1] >= 7 else Xaa
    else:
        Xid = Xaa
    rng = np.random.default_rng(args.seed)
    if args.n_id and args.n_id < len(Xid):
        Xid = Xid[rng.choice(len(Xid), size=args.n_id, replace=False)]
    n_id = len(Xid)
    print(f"[T31b] ID (AA low-mineral): {n_id}")
    if n_id == 0 or n_ood == 0:
        print("[FATAL] một lớp rỗng"); sys.exit(3)

    # ---- model + scorer (chỉ để lấy thành phần thô + calibration) ----
    from src.models.full_model import RamanPhysicsAI
    from src.inference.ood import OODScorer

    model = RamanPhysicsAI(reference_spectra=args.ref_path,
                           n_compounds=args.n_compounds,
                           spectrum_length=Xid.shape[1])
    model.load_state_dict(_load_ckpt_state(args.ckpt)); model.eval()

    id_loader = DataLoader(TensorDataset(torch.from_numpy(Xid).float().unsqueeze(1)),
                           batch_size=args.batch_size)
    # recon_weight/var_weight ở đây CHỈ phục vụ calibrate (lấy recon_p95/var_p95);
    # score thực sự sẽ tính lại qua rescore() phía dưới với cấu hình của user.
    scorer = OODScorer(model, recon_weight=0.6, var_weight=0.4, mc_samples=args.mc)
    cal = scorer.calibrate(id_loader)
    calstats = CalStats(recon_p95=float(cal.recon_p95),
                        var_p95=float(cal.var_p95),
                        score_p95=float(cal.score_p95))
    print(f"[T31b] cal: recon_p95={calstats.recon_p95:.4f} "
          f"var_p95={calstats.var_p95:.4e} score_p95={calstats.score_p95:.4f}")

    # ---- chấm thành phần THÔ một lần cho cả hai lớp ----
    def raw_block(X, tag):
        rec, var = [], []
        for i in range(0, len(X), args.batch_size):
            xb = torch.from_numpy(X[i:i + args.batch_size]).float().unsqueeze(1)
            c = scorer.score_batch(xb, return_components=True)
            rec.append(c["recon_err_raw"].cpu().numpy())
            vk = ("var_raw" if "var_raw" in c
                  else ("pred_var" if "pred_var" in c else None))
            var.append(c[vk].cpu().numpy() if vk else np.zeros(len(xb)))
            print(f"\r[T31b] {tag}: {min(i+args.batch_size, len(X))}/{len(X)}",
                  end="", flush=True)
        print()
        return np.concatenate(rec), np.concatenate(var)

    rec_id, var_id = raw_block(Xid, "ID raw")
    rec_ood, var_ood = raw_block(Xood, "OOD raw")

    # ---- xây danh sách cấu hình ----
    configs = [{
        "name": "chốt demo (user)",
        "scale": args.scale,
        "recon_w": args.recon_w,
        "var_w": round(1.0 - args.recon_w, 4),
        "threshold": args.threshold,
    }]

    if args.multi:
        # thêm F1-optimal từ T32 nếu có
        rec_path = Path(args.recommended)
        if rec_path.exists():
            r = json.loads(rec_path.read_text(encoding="utf-8"))
            configs.append({
                "name": "F1-optimal (từ T32)",
                "scale": r["scale"],
                "recon_w": r["recon_w"], "var_w": r["var_w"],
                "threshold": r["threshold"],
            })
        # và cấu hình cũ (clip 0.6/0.4, score_p95 cũ) để đối chiếu
        configs.append({
            "name": "cũ (clip 0.6/0.4, score_p95)",
            "scale": "clip (gốc repo)",
            "recon_w": 0.6, "var_w": 0.4,
            "threshold": calstats.score_p95,
        })

    # ---- rescore + confusion cho từng cấu hình ----
    results = []
    print("\n" + "=" * 70)
    for cfg in configs:
        s_id, _ = rescore(rec_id, var_id, calstats,
                          recon_w=cfg["recon_w"], var_w=cfg["var_w"],
                          scale=cfg["scale"], threshold=cfg["threshold"])
        s_ood, _ = rescore(rec_ood, var_ood, calstats,
                           recon_w=cfg["recon_w"], var_w=cfg["var_w"],
                           scale=cfg["scale"], threshold=cfg["threshold"])
        cm = confusion(s_id, s_ood, cfg["threshold"])
        m = cm["metrics"]
        print(f"[{cfg['name']}]")
        print(f"  scale={cfg['scale']}  recon_w={cfg['recon_w']} "
              f"var_w={cfg['var_w']}  threshold={cfg['threshold']:.4f}")
        print(f"  ĐÚNG {cm['n_correct']}/{cm['n_total']} "
              f"({100*m['accuracy']:.2f}%)  | TP={cm['TP']} TN={cm['TN']} "
              f"FP={cm['FP']} FN={cm['FN']}")
        print(f"  Recall={m['recall_tpr']:.3f}  Spec={m['specificity']:.3f}  "
              f"F1={m['f1']:.3f}")
        print("-" * 70)
        results.append({"config": cfg, "score_id": s_id, "score_ood": s_ood,
                        "confusion": cm})

    # ---- ghi summary.json ----
    summary = {
        "task": "T31b confusion với thang/trọng số/threshold tùy chỉnh",
        "n_id_true": n_id, "n_ood_true": n_ood,
        "calibration": calstats.__dict__,
        "configs": [
            {"config": r["config"],
             "confusion": {k: v for k, v in r["confusion"].items()},
             "score_stats": {
                 "id_mean": float(r["score_id"].mean()),
                 "id_median": float(np.median(r["score_id"])),
                 "ood_mean": float(r["score_ood"].mean()),
                 "ood_median": float(np.median(r["score_ood"])),
             }}
            for r in results
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8")

    # ---- per_sample.csv ----
    with (out_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["config", "idx", "source",
                                          "true_label", "ood_score",
                                          "pred_label", "correct"])
        w.writeheader()
        for r in results:
            cfg, thr = r["config"], r["config"]["threshold"]
            cname = cfg["name"]
            for i, s in enumerate(r["score_id"]):
                pr = "OOD" if s > thr else "ID"
                w.writerow({"config": cname, "idx": i, "source": "AA_pure",
                            "true_label": "ID", "ood_score": round(float(s), 6),
                            "pred_label": pr, "correct": int(pr == "ID")})
            for i, s in enumerate(r["score_ood"]):
                pr = "OOD" if s > thr else "ID"
                w.writerow({"config": cname, "idx": i,
                            "source": "AAM_mineral_rich", "true_label": "OOD",
                            "ood_score": round(float(s), 6),
                            "pred_label": pr, "correct": int(pr == "OOD")})

    # ---- biểu đồ ----
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        nrow = len(results)
        fig, axes = plt.subplots(nrow, 2, figsize=(13, 4.5 * nrow),
                                 gridspec_kw={"width_ratios": [2, 1]})
        if nrow == 1:
            axes = axes[None, :]
        for k, r in enumerate(results):
            ax0, ax1 = axes[k, 0], axes[k, 1]
            si, so = r["score_id"], r["score_ood"]
            cfg = r["config"]; thr = cfg["threshold"]; cm = r["confusion"]
            bins = np.linspace(min(si.min(), so.min()),
                               max(si.max(), so.max()), 35)
            ax0.hist(si, bins=bins, alpha=0.6, color="#2a9d8f",
                     label=f"ID (n={n_id})")
            ax0.hist(so, bins=bins, alpha=0.6, color="#e76f51",
                     label=f"OOD (n={n_ood})")
            ax0.axvline(thr, color="k", ls="--", lw=1.5,
                        label=f"thr={thr:.3f}")
            ax0.set_xlabel("OOD score"); ax0.set_ylabel("Số mẫu")
            ax0.set_title(f"{cfg['name']} — {cfg['scale']}\n"
                          f"recon_w={cfg['recon_w']}, accuracy="
                          f"{cm['metrics']['accuracy']:.3f}")
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
        print("[T31b] saved distribution.png")
    except Exception as e:
        print(f"[T31b] (plot skipped: {e})")

    print("=" * 70)
    print(f"[T31b done] -> {out_dir}")
    print("  summary.json, per_sample.csv, distribution.png")


if __name__ == "__main__":
    main()
