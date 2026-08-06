"""Stretch T29B-fix — TWO-CLASS OOD test with REAL AUROC.

Matches the actual repo API (verified against src/inference/ood.py and
src/models/uncertainty.py):
    - src.models.uncertainty.predict_with_uncertainty
    - src.inference.ood.OODScorer  (+ compute_reconstruction_error)

PROBLEM
-------
The AAM dataset (Zarei 2023) has NO low-mineral samples — every spectrum is
48–91% mineral. The original T29B/T29E "test set" had ONE class only, so AUROC
was N/A and the "before-retrain = OOD" claim had no control group.

FIX
---
Borrow the LOW-MINERAL (ID) class from the ORIGINAL AA dataset (mineral = 0),
which lives in data/processed/ on the SAME wavenumber grid + SAME preprocessing
+ SAME instrument as AAM (user-confirmed). The resulting AUROC therefore reflects
MINERAL PRESENCE, not an instrument/domain artifact.

    Class 0 (ID,  low-mineral)  := AA-pure spectra        [data/processed/spectra.pt]
    Class 1 (OOD, mineral-rich) := AAM mineral-rich test  [data/processed/aam/...]

OOD scoring follows the repo's design exactly: an OODScorer is CALIBRATED on the
ID (low-mineral) class — the in-distribution reference for the AA-only model —
then used to score BOTH classes. AUROC is computed two ways:
    (a) on the raw reconstruction error  (pure physics signal, calibration-free)
    (b) on the calibrated combined OOD score

Run the AA-only (6-output) model = the "BEFORE retrain" model.

Usage:
    python scripts/stretch/run_t29b_fix_twoclass.py
    python scripts/stretch/run_t29b_fix_twoclass.py --n-id 973   # balance classes

Output (results/stretch/t29b_twoclass/):
    summary.json   two-class metrics incl. real AUROC
    raw.npz        per-sample arrays
    histograms.png overlaid ID vs OOD distributions
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


def _to_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3 and x.shape[1] == 1:
        return x[:, 0, :]
    return x


def auroc(pos_scores, neg_scores):
    """AUROC = P(score(pos) > score(neg)) via Mann–Whitney U. No sklearn."""
    pos = np.asarray(pos_scores, dtype=np.float64)
    neg = np.asarray(neg_scores, dtype=np.float64)
    alls = np.concatenate([pos, neg])
    order = alls.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(alls) + 1)
    # average ties
    # (simple tie handling: group equal values)
    _, inv, counts = np.unique(alls, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts + 1
    avg_rank_per_group = (start + csum) / 2.0
    ranks = avg_rank_per_group[inv]
    r_pos = ranks[:len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aam-dir", default="data/processed/aam")
    p.add_argument("--aa-spectra", default=None,
                   help="AA-pure spectra cache (auto-discovered if omitted)")
    p.add_argument("--aa-labels", default=None,
                   help="AA labels cache (optional; used to filter mineral=0)")
    p.add_argument("--ckpt", default="checkpoints/best.pt",
                   help="AA-only (6-output) checkpoint = 'before retrain' model")
    p.add_argument("--ref-path", default="engine/reference_spectra.npy",
                   help="6-component AA references")
    p.add_argument("--mineral-threshold", type=float, default=0.05)
    p.add_argument("--n-id", type=int, default=None,
                   help="Cap on low-mineral (ID) samples; None = all AA-pure")
    p.add_argument("--mc", type=int, default=50)
    p.add_argument("--recon-weight", type=float, default=0.6)
    p.add_argument("--var-weight", type=float, default=0.4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/stretch/t29b_twoclass")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. OOD class: mineral-rich from AAM test split ----
    aam_dir = Path(args.aam_dir)
    aam_spec = _to_2d(_load_any(aam_dir / "spectra.pt"))
    aam_lab7 = _load_any(aam_dir / "labels_7d.pt")
    split = json.loads((aam_dir / "split.json").read_text())
    test_idx = split["test"]
    Xood = aam_spec[test_idx]
    minerals = aam_lab7[test_idx][:, 6]
    Xood = Xood[minerals > args.mineral_threshold]
    n_ood = len(Xood)
    print(f"[fix] OOD class (mineral-rich, AAM test): {n_ood}")

    # ---- 2. ID class: AA-pure (mineral = 0), borrowed ----
    aa_spec_path = _first_existing([
        args.aa_spectra,
        "data/processed/spectra.pt",
        "data/processed/spectra_full.pt",
        "data/processed/X.pt", "data/processed/X.npy",
    ])
    aa_lab_path = _first_existing([
        args.aa_labels,
        "data/processed/labels.pt", "data/processed/labels_6d.pt",
        "data/processed/Y.pt", "data/processed/Y.npy",
    ])
    if aa_spec_path is None:
        print("\n[FATAL] Could not find original AA spectra cache under data/processed/.")
        print("        Re-run with --aa-spectra <path> --aa-labels <path>.")
        sys.exit(2)
    print(f"[fix] AA spectra cache: {aa_spec_path}")
    print(f"[fix] AA labels  cache: {aa_lab_path}")

    Xaa = _to_2d(_load_any(aa_spec_path))
    if aa_lab_path is not None:
        Yaa = _load_any(aa_lab_path)
        if Yaa.shape[1] >= 7:  # has a mineral column -> keep mineral≈0
            Xid = Xaa[Yaa[:, 6] <= args.mineral_threshold]
        else:                  # 6-d AA labels => all are mineral-free by definition
            Xid = Xaa
    else:
        Xid = Xaa

    if Xid.shape[1] != Xood.shape[1]:
        print(f"\n[FATAL] Grid mismatch: AA P={Xid.shape[1]} vs AAM P={Xood.shape[1]}.")
        sys.exit(3)

    rng = np.random.default_rng(args.seed)
    if args.n_id is not None and args.n_id < len(Xid):
        Xid = Xid[rng.choice(len(Xid), size=args.n_id, replace=False)]
    n_id = len(Xid)
    print(f"[fix] ID class (low-mineral, AA-pure): {n_id}")
    if n_id == 0 or n_ood == 0:
        print("\n[FATAL] One class empty; AUROC uncomputable.")
        sys.exit(4)

    # ---- 3. Build AA-only model, load 'before retrain' checkpoint ----
    from src.models.full_model import RamanPhysicsAI
    from src.inference.ood import OODScorer, compute_reconstruction_error

    model = RamanPhysicsAI(reference_spectra=args.ref_path,
                           n_compounds=6, spectrum_length=Xood.shape[1])
    ckpt = torch.load(args.ckpt, weights_only=False)
    # Checkpoints come in two flavours in this repo:
    #   (a) bare state_dict            (run_t29d: torch.save(model.state_dict()))
    #   (b) wrapped dict with "model"  (training loop: {"model":..., "optimizer":...})
    if isinstance(ckpt, dict) and "model" in ckpt and any(
        k.startswith(("backbone", "quantification", "reconstruction"))
        for k in ckpt.get("model", {})
    ):
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt  # already a bare state_dict
    model.load_state_dict(state)
    model.eval()
    print(f"[fix] loaded AA-only checkpoint {args.ckpt} "
          f"({'wrapped' if state is not ckpt else 'bare'} format)")

    # ---- 4. Calibrate OOD scorer on the ID (low-mineral) class ----
    # Per OOD design, calibration set = in-distribution reference.
    id_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xid).float().unsqueeze(1)),
        batch_size=args.batch_size, shuffle=False)
    scorer = OODScorer(model, recon_weight=args.recon_weight,
                       var_weight=args.var_weight, mc_samples=args.mc)
    cal = scorer.calibrate(id_loader)
    print(f"[fix] calibrated on ID: recon_p95={cal.recon_p95:.4f} "
          f"var_p95={cal.var_p95:.4e} score_p95(thresh)={cal.score_p95:.4f}")

    # ---- 5. Score BOTH classes ----
    def score_block(X):
        comp = scorer.score_batch(torch.from_numpy(X).float().unsqueeze(1),
                                  return_components=True)
        return (comp["score"].cpu().numpy(),
                comp["recon_err_raw"].cpu().numpy())   # raw = 1 - cosine

    print(f"[fix] scoring ID ({n_id}) ...")
    score_id, recon_err_id = score_block(Xid)
    print(f"[fix] scoring OOD ({n_ood}) ...")
    score_ood, recon_err_ood = score_block(Xood)

    cos_id = 1.0 - recon_err_id
    cos_ood = 1.0 - recon_err_ood

    # ---- 6. AUROC ----
    auroc_score = auroc(score_ood, score_id)          # higher OOD score = good
    auroc_recon = auroc(recon_err_ood, recon_err_id)  # higher recon err = good

    # how many OOD flagged with the ID-calibrated threshold?
    tpr = float((score_ood > cal.score_p95).mean())
    fpr = float((score_id > cal.score_p95).mean())

    summary = {
        "task": "T29B-fix two-class OOD (AA-only model, before retrain)",
        "n_id_low_mineral": int(n_id),
        "n_ood_mineral_rich": int(n_ood),
        "mc_samples": args.mc,
        "weights": {"recon": args.recon_weight, "var": args.var_weight},
        "calibration": {"recon_p95": cal.recon_p95, "var_p95": cal.var_p95,
                        "score_p95_threshold": cal.score_p95},
        "AUROC_ood_score": auroc_score,
        "AUROC_recon_error": auroc_recon,
        "tpr_at_threshold": tpr,
        "fpr_at_threshold": fpr,
        "ood_score": {
            "low_mineral":  {"mean": float(score_id.mean()),  "median": float(np.median(score_id))},
            "mineral_rich": {"mean": float(score_ood.mean()), "median": float(np.median(score_ood))},
        },
        "recon_cosine": {
            "low_mineral":  {"mean": float(cos_id.mean()),  "median": float(np.median(cos_id))},
            "mineral_rich": {"mean": float(cos_ood.mean()), "median": float(np.median(cos_ood))},
        },
        "sources": {"aa_spectra": str(aa_spec_path), "ckpt": args.ckpt},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(out_dir / "raw.npz",
             score_id=score_id, score_ood=score_ood,
             cos_id=cos_id, cos_ood=cos_ood)

    # ---- 7. Histograms ----
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        ax[0].hist(cos_id, bins=30, alpha=0.6, label=f"Low-mineral / ID (n={n_id})")
        ax[0].hist(cos_ood, bins=30, alpha=0.6, label=f"Mineral-rich / OOD (n={n_ood})")
        ax[0].set_xlabel("Reconstruction cosine"); ax[0].set_ylabel("count")
        ax[0].set_title(f"Recon cosine  (AUROC={auroc_recon:.3f})"); ax[0].legend()
        ax[1].hist(score_id, bins=30, alpha=0.6, label=f"Low-mineral / ID (n={n_id})")
        ax[1].hist(score_ood, bins=30, alpha=0.6, label=f"Mineral-rich / OOD (n={n_ood})")
        ax[1].axvline(cal.score_p95, color="k", ls="--", label="threshold")
        ax[1].set_xlabel("OOD score"); ax[1].set_ylabel("count")
        ax[1].set_title(f"OOD score  (AUROC={auroc_score:.3f})"); ax[1].legend()
        fig.tight_layout(); fig.savefig(out_dir / "histograms.png", dpi=120)
        print("[fix] saved histograms.png")
    except Exception as e:
        print(f"[fix] (plot skipped: {e})")

    print("\n" + "=" * 60)
    print(f"  REAL AUROC (OOD score)   = {auroc_score:.4f}")
    print(f"  REAL AUROC (recon error) = {auroc_recon:.4f}")
    print(f"  TPR @ threshold = {tpr:.3f}   FPR @ threshold = {fpr:.3f}")
    print(f"  recon cos:  ID mean={cos_id.mean():.4f}  OOD mean={cos_ood.mean():.4f}")
    print("=" * 60)
    print(f"\n[fix done] -> {out_dir}")


if __name__ == "__main__":
    main()