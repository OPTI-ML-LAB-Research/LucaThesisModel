"""Stretch T32 — Bacteria-ID cross-instrument zero-shot OOD test.

The Bacteria-ID dataset (Ho 2019, 30 bacteria species) was collected
on a different instrument from AA (Wasatch WP-785). Cross-instrument
OOD test: can AA model detect bacteria as foreign?

Note: bacteria_ID is large (~60K reference, ~3K test). We use the test
split by default (~3K spectra) to keep runtime manageable.

Output:
    results/stretch/t32_bacteria_auroc.json
    results/stretch/t32_bacteria_score_distribution.png
    results/stretch/t32_bacteria_handover.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from _handover_utils import to_vector, get_recon_cosine, get_ood, get_peaks, COMPOUND_ORDER

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-bacteria", type=int, default=1000)
    p.add_argument("--bacteria-split", default="test",
                   choices=["test", "reference", "2018clinical"])
    p.add_argument("--n-id-max", type=int, default=540)
    p.add_argument("--bacteria-dir", default="data/raw/bacteria_ID")
    p.add_argument("--out-dir", default="results/stretch")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[T32] Loading {args.n_bacteria} bacteria-ID spectra ...")
    from src.data.bacteria_id_loader import load_bacteria_id_subset
    try:
        bact_spectra = load_bacteria_id_subset(
            n=args.n_bacteria, split=args.bacteria_split, seed=42,
            apply_preprocessing=True, bacteria_dir=args.bacteria_dir,
        )
    except FileNotFoundError as e:
        print(f"  ERROR: {e}"); sys.exit(1)
    print(f"  loaded: {bact_spectra.shape}")

    # OOD scorer
    print("[T32] loading OOD scorer ...")
    from src.inference.predict import predict, reset_cache
    reset_cache()
    _ = predict(bact_spectra[0], skip_ood=False, n_mc_samples=10, verbose=False)
    from src.inference import predict as predict_module
    scorer = predict_module._resources.get("ood_scorer")
    if scorer is None:
        print("  ERROR: OOD scorer not in cache"); sys.exit(1)

    # Score
    print(f"\n[T32] scoring {len(bact_spectra)} bacteria spectra ...")
    bact_scores = []
    for i, spec in enumerate(bact_spectra):
        s = scorer.score(spec)
        if isinstance(s, dict):
            bact_scores.append(s.get("score", s.get("combined", 0.0)))
        else:
            bact_scores.append(float(s))
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(bact_spectra)}")
    bact_scores = np.array(bact_scores)
    print(f"  bacteria: mean={bact_scores.mean():.3f}, std={bact_scores.std():.3f}")

    print(f"\n[T32] scoring {args.n_id_max} AA test (ID) samples ...")
    spectra_full = torch.load("data/processed/spectra_full.pt", weights_only=True).numpy()
    split = json.loads(Path("data/splits/split_A_composition_ood.json").read_text())
    test_idx = split["test"][:args.n_id_max]
    id_scores = []
    for i, idx in enumerate(test_idx):
        s = scorer.score(spectra_full[idx])
        if isinstance(s, dict):
            id_scores.append(s.get("score", s.get("combined", 0.0)))
        else:
            id_scores.append(float(s))
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(test_idx)}")
    id_scores = np.array(id_scores)

    from src.eval.metrics import ood_auroc
    auroc = ood_auroc(id_scores=id_scores, ood_scores=bact_scores)
    print(f"\n[T32 RESULT] cross-instrument OOD AUROC = {auroc:.4f}")

    summary = {
        "task": "T32 Bacteria-ID cross-instrument OOD",
        "n_id": len(id_scores), "n_ood": len(bact_scores),
        "bacteria_split": args.bacteria_split,
        "id_score_stats": {"mean": float(id_scores.mean()),
                             "std": float(id_scores.std()),
                             "min": float(id_scores.min()),
                             "max": float(id_scores.max())},
        "ood_score_stats": {"mean": float(bact_scores.mean()),
                              "std": float(bact_scores.std()),
                              "min": float(bact_scores.min()),
                              "max": float(bact_scores.max())},
        "auroc": float(auroc),
    }
    (out_dir / "t32_bacteria_auroc.json").write_text(json.dumps(summary, indent=2))
    np.savez(out_dir / "t32_bacteria_raw_scores.npz",
              id_scores=id_scores, ood_scores=bact_scores)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(id_scores, bins=30, alpha=0.6, color="#2a9d8f",
                label=f"AA test ID (n={len(id_scores)})")
        ax.hist(bact_scores, bins=30, alpha=0.6, color="#f1c40f",
                label=f"Bacteria-ID OOD (n={len(bact_scores)})")
        ax.set_xlabel("OOD score"); ax.set_ylabel("count")
        ax.set_title(f"Bacteria-ID cross-instrument OOD  (AUROC = {auroc:.3f})")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "t32_bacteria_score_distribution.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  [warn] plot failed: {e}")

    verdict = ("PASS-target" if auroc >= 0.85 else
                "PASS-floor" if auroc >= 0.75 else "FAIL")
    lines = [
        "# Stretch T32 — Bacteria-ID Cross-instrument OOD — Handover",
        "",
        "**Setup:** AA-trained model (Wasatch WP-785) test trên Bacteria-ID dataset",
        "(Ho 2019, 30 species, instrument khác).",
        "",
        f"## AUROC = **{auroc:.4f}** — {verdict}",
        "",
        f"- n_ID (AA test, Wasatch): {len(id_scores)}",
        f"- n_OOD (Bacteria-ID, different instrument): {len(bact_scores)}",
        f"- Bacteria split used: `{args.bacteria_split}`",
        "",
        "### Phân bố OOD score",
        "",
        "| Class | Mean | Std |",
        "|---|---|---|",
        f"| AA test (ID) | {id_scores.mean():.3f} | {id_scores.std():.3f} |",
        f"| Bacteria-ID (OOD) | {bact_scores.mean():.3f} | {bact_scores.std():.3f} |",
        "",
        "## Cách dùng trong báo cáo",
        "",
        "Thêm vào bảng tổng hợp OOD evaluation (tổng 3-4 nguồn):",
        "",
        f"- `{(out_dir / 't32_bacteria_auroc.json').as_posix()}`",
        f"- `{(out_dir / 't32_bacteria_score_distribution.png').as_posix()}`",
    ]
    (out_dir / "t32_bacteria_handover.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[handover] {out_dir / 't32_bacteria_handover.md'}")
    print("\n[T32 done]")


if __name__ == "__main__":
    main()
