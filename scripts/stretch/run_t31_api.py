"""Stretch T31 — API pharmaceutical cross-domain zero-shot OOD test.

The API dataset (33 pharmaceutical chemicals, ~3511 spectra) is fully
out-of-domain for the AA-trained model. Test that the OOD scorer
correctly flags pharmaceutical samples vs AA test samples.

Output:
    results/stretch/t31_api_auroc.json
    results/stretch/t31_api_score_distribution.png
    results/stretch/t31_api_handover.md
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
    p.add_argument("--n-api", type=int, default=500)
    p.add_argument("--n-id-max", type=int, default=540)
    p.add_argument("--api-csv", default="data/raw/API/API_data.csv")
    p.add_argument("--out-dir", default="results/stretch")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[T31] Loading API dataset ({args.n_api} samples) ...")
    from src.data.api_loader import load_api_subset
    try:
        api_spectra, api_classes = load_api_subset(
            n=args.n_api, seed=42, apply_preprocessing=True, api_csv=args.api_csv,
        )
    except FileNotFoundError as e:
        print(f"  ERROR: {e}"); sys.exit(1)
    print(f"  loaded: {api_spectra.shape}")

    # Load OOD scorer
    print("[T31] loading OOD scorer ...")
    from src.inference.predict import predict, reset_cache
    reset_cache()
    _ = predict(api_spectra[0], skip_ood=False, n_mc_samples=10, verbose=False)
    from src.inference import predict as predict_module
    scorer = predict_module._resources.get("ood_scorer")
    if scorer is None:
        print("  ERROR: OOD scorer not in cache"); sys.exit(1)

    # Score API
    print(f"\n[T31] scoring {len(api_spectra)} API samples ...")
    api_scores = []
    for i, spec in enumerate(api_spectra):
        s = scorer.score(spec)
        if isinstance(s, dict):
            api_scores.append(s.get("score", s.get("combined", 0.0)))
        else:
            api_scores.append(float(s))
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(api_spectra)}")
    api_scores = np.array(api_scores)
    print(f"  API: mean={api_scores.mean():.3f}, std={api_scores.std():.3f}")

    # Score AA test (ID)
    print(f"\n[T31] scoring {args.n_id_max} AA test (ID) samples ...")
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
    print(f"  ID: mean={id_scores.mean():.3f}, std={id_scores.std():.3f}")

    # AUROC
    from src.eval.metrics import ood_auroc
    auroc = ood_auroc(id_scores=id_scores, ood_scores=api_scores)
    print(f"\n[T31 RESULT] cross-domain OOD AUROC (AA vs API) = {auroc:.4f}")

    summary = {
        "task": "T31 API pharmaceutical cross-domain OOD",
        "n_id": len(id_scores), "n_ood": len(api_scores),
        "id_score_stats": {"mean": float(id_scores.mean()),
                             "std": float(id_scores.std()),
                             "min": float(id_scores.min()),
                             "max": float(id_scores.max())},
        "ood_score_stats": {"mean": float(api_scores.mean()),
                              "std": float(api_scores.std()),
                              "min": float(api_scores.min()),
                              "max": float(api_scores.max())},
        "auroc": float(auroc),
    }
    (out_dir / "t31_api_auroc.json").write_text(json.dumps(summary, indent=2))
    np.savez(out_dir / "t31_api_raw_scores.npz",
              id_scores=id_scores, ood_scores=api_scores)

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(id_scores, bins=30, alpha=0.6, color="#2a9d8f",
                label=f"AA test ID (n={len(id_scores)})")
        ax.hist(api_scores, bins=30, alpha=0.6, color="#9b5de5",
                label=f"API pharmaceutical OOD (n={len(api_scores)})")
        ax.set_xlabel("OOD score"); ax.set_ylabel("count")
        ax.set_title(f"API cross-domain OOD discrimination  (AUROC = {auroc:.3f})")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "t31_api_score_distribution.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"  [warn] plot failed: {e}")

    # Handover
    verdict = ("PASS-target" if auroc >= 0.85 else
                "PASS-floor" if auroc >= 0.75 else "FAIL")
    lines = [
        "# Stretch T31 — API Cross-domain OOD — Handover",
        "",
        "**Setup:** AA-trained model test trên dataset API (33 pharmaceutical compounds).",
        "API hoàn toàn ngoài domain amino acid → KỲ VỌNG high OOD score.",
        "",
        f"## AUROC = **{auroc:.4f}** — {verdict}",
        "",
        f"- n_ID (AA test): {len(id_scores)}",
        f"- n_OOD (API pharmaceutical): {len(api_scores)}",
        "",
        "### Phân bố OOD score",
        "",
        "| Class | Mean | Std | Min | Max |",
        "|---|---|---|---|---|",
        f"| AA test (ID) | {id_scores.mean():.3f} | {id_scores.std():.3f} | {id_scores.min():.3f} | {id_scores.max():.3f} |",
        f"| API (OOD) | {api_scores.mean():.3f} | {api_scores.std():.3f} | {api_scores.min():.3f} | {api_scores.max():.3f} |",
        "",
        "## Cách dùng trong báo cáo",
        "",
        "Thêm vào Chương 3 'OOD evaluation' subsection, bảng tổng hợp 3 nguồn OOD:",
        "",
        "| OOD source | Type | AUROC |",
        "|---|---|---|",
        "| Synthetic spike (Phase C) | in-range artifact | (existing) |",
        "| AAM mineral-rich (T29B) | same-domain shift | (from T29B) |",
        f"| API pharmaceutical (T31) | cross-domain | **{auroc:.4f}** |",
        "",
        f"- `{(out_dir / 't31_api_auroc.json').as_posix()}`",
        f"- `{(out_dir / 't31_api_score_distribution.png').as_posix()}`",
    ]
    (out_dir / "t31_api_handover.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[handover] {out_dir / 't31_api_handover.md'}")
    print("\n[T31 done]")


if __name__ == "__main__":
    main()
