"""Stretch T32 — Hiệu chỉnh ngưỡng OOD đúng cách (giải quyết H2).

Vấn đề (H2 trong báo cáo): ngưỡng OOD hiện đặt ở quantile-95 trên Split A val,
vốn đã có composition-shift nhẹ -> ngưỡng quá cao -> MoS2 (0.853) và các hard-OOD
bị gán ID sai (false negative).

Giải pháp: calibrate ngưỡng trên một tập CÓ NHÃN ID/OOD rõ ràng (AA=ID 973 +
AAM=OOD 973), tối ưu theo F1 thay vì đặt mù ở quantile-95. Đồng thời quét:
    - Các tỉ lệ trọng số (recon_w, var_w): 0.6/0.4, 0.8/0.2, 1.0/0.0 (physics-only)
    - Các thang chuẩn hóa: clip / no-clip / tanh

Với mỗi cấu hình: tìm threshold tối ưu F1, báo AUROC + confusion matrix, và —
nếu có file MoS2/SkMel28 — chấm luôn chúng để xem cấu hình nào bắt được hard-OOD.

Xuất: results/stretch/t32_threshold/
    sweep.json          mọi cấu hình + threshold F1-optimal + metrics
    recommended.json    cấu hình tốt nhất, dùng để nạp vào dashboard
    sweep_table.md      bảng so sánh (dán vào báo cáo)

Usage:
    python scripts/stretch/run_t32_threshold.py --n-id 973
    python scripts/stretch/run_t32_threshold.py --n-id 973 \
        --hard-ood data/raw/ood_demo/MoS2-160o-12h-ph5.txt skmel28.txt

Phụ thuộc: dùng OODScorer của repo để lấy thành phần THÔ (recon_err_raw + var),
sau đó rescore bằng dashboard/ood_controls.py (cùng logic, không lệch).
"""
from __future__ import annotations

import argparse
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


def _load_any(path: Path) -> np.ndarray:
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


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    alls = np.concatenate([pos, neg])
    _, inv, counts = np.unique(alls, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts); start = csum - counts + 1
    ranks = ((start + csum) / 2.0)[inv]
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


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
    p.add_argument("--hard-ood", nargs="*", default=[],
                   help="File .txt 2 cột [wn intensity] (MoS2, SkMel28...) để chấm thử.")
    p.add_argument("--wn-path", default="data/processed/wavenumbers.npy")
    p.add_argument("--out-dir", default="results/stretch/t32_threshold")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # cùng thư mục để import ood_controls (đặt cạnh app.py trong dashboard/)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dashboard"))
    from ood_controls import CalStats, rescore, best_f1_threshold, SCALES

    # ---- nạp ID (AA) + OOD (AAM mineral-rich) ----
    aam_dir = Path(args.aam_dir)
    aam_spec = _to_2d(_load_any(aam_dir / "spectra.pt"))
    aam_lab7 = _load_any(aam_dir / "labels_7d.pt")
    test_idx = json.loads((aam_dir / "split.json").read_text())["test"]
    Xood = aam_spec[test_idx]
    Xood = Xood[aam_lab7[test_idx][:, 6] > args.mineral_threshold]

    aa_spec_path = _first_existing([args.aa_spectra, "data/processed/spectra.pt",
                                    "data/processed/spectra_full.pt"])
    aa_lab_path = _first_existing([args.aa_labels, "data/processed/labels.pt",
                                   "data/processed/labels_6d.pt"])
    Xaa = _to_2d(_load_any(aa_spec_path))
    if aa_lab_path is not None:
        Yaa = _load_any(aa_lab_path)
        Xid = Xaa[Yaa[:, 6] <= args.mineral_threshold] if Yaa.shape[1] >= 7 else Xaa
    else:
        Xid = Xaa
    rng = np.random.default_rng(args.seed)
    if args.n_id and args.n_id < len(Xid):
        Xid = Xid[rng.choice(len(Xid), size=args.n_id, replace=False)]
    print(f"[T32] ID(AA)={len(Xid)}  OOD(AAM)={len(Xood)}")

    # ---- model + scorer (lấy thành phần THÔ) ----
    from src.models.full_model import RamanPhysicsAI
    from src.inference.ood import OODScorer

    model = RamanPhysicsAI(reference_spectra=args.ref_path,
                           n_compounds=args.n_compounds,
                           spectrum_length=Xid.shape[1])
    model.load_state_dict(_load_ckpt_state(args.ckpt)); model.eval()

    id_loader = DataLoader(TensorDataset(torch.from_numpy(Xid).float().unsqueeze(1)),
                           batch_size=args.batch_size)
    scorer = OODScorer(model, recon_weight=0.6, var_weight=0.4, mc_samples=args.mc)
    cal = scorer.calibrate(id_loader)
    calstats = CalStats(recon_p95=float(cal.recon_p95),
                        var_p95=float(cal.var_p95),
                        score_p95=float(cal.score_p95))
    print(f"[T32] cal: recon_p95={calstats.recon_p95:.4f} "
          f"var_p95={calstats.var_p95:.4e} score_p95={calstats.score_p95:.4f}")

    # ---- lấy thành phần thô cho cả hai lớp (chấm theo mini-batch, tránh OOM) ----
    def raw_components(X, tag):
        rec, var = [], []
        for i in range(0, len(X), args.batch_size):
            xb = torch.from_numpy(X[i:i + args.batch_size]).float().unsqueeze(1)
            c = scorer.score_batch(xb, return_components=True)
            rec.append(c["recon_err_raw"].cpu().numpy())
            # var thô: ưu tiên key 'var_raw', fallback 'pred_var'
            vk = "var_raw" if "var_raw" in c else ("pred_var" if "pred_var" in c else None)
            var.append(c[vk].cpu().numpy() if vk else np.zeros(len(xb)))
            print(f"\r[T32]   {tag}: {min(i+args.batch_size, len(X))}/{len(X)}",
                  end="", flush=True)
        print()
        return np.concatenate(rec), np.concatenate(var)

    rec_id, var_id = raw_components(Xid, "ID")
    rec_ood, var_ood = raw_components(Xood, "OOD")

    # ---- chấm hard-OOD nếu có ----
    def preprocess_raw(path):
        from src.data.preprocess import preprocess_batch
        data = np.genfromtxt(path)
        if data.ndim != 2 or data.shape[1] < 2:
            return None
        wn_src, inten = data[:, 0].astype(np.float64), data[:, 1].astype(np.float64)
        tgt = np.load(args.wn_path).astype(np.float64)
        flip = tgt[0] > tgt[-1]; tgt_asc = tgt[::-1] if flip else tgt
        if wn_src[0] > wn_src[-1]:
            wn_src, inten = wn_src[::-1], inten[::-1]
        v = np.interp(tgt_asc, wn_src, inten, left=0.0, right=0.0)
        if flip:
            v = v[::-1]
        Xp = preprocess_batch(v[None, :].astype(np.float32)).astype(np.float32)
        return np.nan_to_num(Xp, nan=0.0, posinf=0.0, neginf=0.0)

    hard = {}
    for hp in args.hard_ood:
        Xp = preprocess_raw(hp)
        if Xp is None:
            print(f"[T32] (bỏ qua {hp}: không phải 2 cột)")
            continue
        r, v = raw_components(Xp, Path(hp).stem)
        hard[Path(hp).stem] = (float(r[0]), float(v[0]))

    # ---- quét cấu hình ----
    weight_grid = [(0.6, 0.4), (0.7, 0.3), (0.8, 0.2), (1.0, 0.0)]
    sweep = []
    for scale in SCALES:
        for rw, vw in weight_grid:
            s_id, _ = rescore(rec_id, var_id, calstats, recon_w=rw, var_w=vw, scale=scale)
            s_ood, _ = rescore(rec_ood, var_ood, calstats, recon_w=rw, var_w=vw, scale=scale)
            au = auroc(s_ood, s_id)
            f1info = best_f1_threshold(s_id, s_ood)
            # chấm hard-OOD với threshold F1-optimal
            hard_verdicts = {}
            for nm, (r, v) in hard.items():
                sc, is_o = rescore(r, v, calstats, recon_w=rw, var_w=vw,
                                   scale=scale, threshold=f1info["best_threshold"])
                hard_verdicts[nm] = {"score": float(np.asarray(sc).item()),
                                     "verdict": "OOD" if bool(np.asarray(is_o).item()) else "ID"}
            sweep.append({
                "scale": scale, "recon_w": rw, "var_w": vw,
                "auroc": au,
                "f1_threshold": f1info["best_threshold"],
                "f1": f1info["f1"], "precision": f1info["precision"],
                "recall": f1info["recall"], "fpr": f1info["fpr"],
                "confusion": {k: f1info[k] for k in ("TP", "FP", "TN", "FN")},
                "hard_ood": hard_verdicts,
            })

    sweep.sort(key=lambda d: (-d["auroc"], -d["f1"]))
    best = sweep[0]

    (out_dir / "sweep.json").write_text(
        json.dumps({"calibration": calstats.__dict__, "sweep": sweep},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")
    (out_dir / "recommended.json").write_text(json.dumps({
        "calibration": calstats.__dict__,
        "recon_w": best["recon_w"], "var_w": best["var_w"],
        "scale": best["scale"], "threshold": best["f1_threshold"],
        "auroc": best["auroc"], "f1": best["f1"],
    }, indent=2, ensure_ascii=False),
        encoding="utf-8")

    # ---- bảng markdown ----
    L = ["# T32 — Quét trọng số × thang × threshold-F1 (AA=ID vs AAM=OOD)", "",
         f"ID(AA)={len(Xid)}, OOD(AAM)={len(Xood)}. Threshold tối ưu F1 trên chính tập này.",
         "", "| Thang | recon_w | var_w | AUROC | F1 | Threshold | Recall | FPR |",
         "|---|---|---|---|---|---|---|---|"]
    for d in sweep:
        L.append(f"| {d['scale']} | {d['recon_w']} | {d['var_w']} | "
                 f"{d['auroc']:.4f} | {d['f1']:.4f} | {d['f1_threshold']:.4f} | "
                 f"{d['recall']:.3f} | {d['fpr']:.3f} |")
    if hard:
        L += ["", "## Hard-OOD verdict (với threshold F1-optimal mỗi cấu hình)", ""]
        names = list(hard.keys())
        L.append("| Thang | recon_w/var_w | " + " | ".join(names) + " |")
        L.append("|---|---|" + "---|" * len(names))
        for d in sweep:
            cells = " | ".join(f"{d['hard_ood'][n]['verdict']} ({d['hard_ood'][n]['score']:.3f})"
                               for n in names)
            L.append(f"| {d['scale']} | {d['recon_w']}/{d['var_w']} | {cells} |")
    L += ["", "## Khuyến nghị",
          f"- Cấu hình tốt nhất (AUROC rồi F1): **{best['scale']}, "
          f"recon_w={best['recon_w']}, var_w={best['var_w']}, "
          f"threshold={best['f1_threshold']:.4f}** (AUROC {best['auroc']:.4f}).",
          "- Nạp `recommended.json` vào dashboard để dùng trực tiếp."]
    (out_dir / "sweep_table.md").write_text("\n".join(L), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"  BEST: {best['scale']}  recon_w={best['recon_w']} var_w={best['var_w']}")
    print(f"        AUROC={best['auroc']:.4f}  F1={best['f1']:.4f}  "
          f"threshold={best['f1_threshold']:.4f}")
    for nm, hv in best["hard_ood"].items():
        print(f"        {nm}: {hv['verdict']} (score {hv['score']:.3f})")
    print("=" * 60)
    print(f"[T32 done] -> {out_dir}  (sweep.json, recommended.json, sweep_table.md)")


if __name__ == "__main__":
    main()
