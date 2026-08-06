"""T18+T19 stretch test: discriminative power on ID vs OOD samples.

Loads:
    * checkpoints/best.pt   -- T17 Round-2 model (val_mae=0.0523)
    * data/processed/spectra_full.pt, labels.pt
    * data/splits/split_A_composition_ood.json

Runs:
    1. Calibrates OODScorer on VAL split (in-distribution).
    2. Scores N_TEST samples from TEST split (in-distribution, untouched
       during training) and reports the score distribution.
    3. For each ID sample, synthesises an OOD analogue (4 modes:
       spike / noise / mask / scale) and scores those.
    4. Prints a side-by-side table and saves:
        - results/ood_demo/calibration.json
        - results/ood_demo/score_table.csv
        - results/ood_demo/score_distribution.png

Usage:
    python scripts/run_phase3_stretch.py
    python scripts/run_phase3_stretch.py --n-samples 10 --mc 50
    python scripts/run_phase3_stretch.py --device cpu

This script is a self-contained smoke test for T18 + T19, separate
from the eventual T20 ``predict()``-based pipeline. Use it after
dropping the new uncertainty.py / ood.py files in to confirm both
modules work against the real checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from src.inference.ood import OODScorer, make_synthetic_ood
from src.models.full_model import build_full_model_from_config


# Compound order is LOCKED (CHAT2_PHASE_A_HANDOVER decision C1).
COMPOUND_ORDER = [
    "Alanine", "Asparagine", "Aspartic Acid",
    "Glutamic Acid", "Histidine", "Glucosamine",
]

OOD_MODES = ["spike", "noise", "mask", "scale"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--spectra", default="data/processed/spectra_full.pt")
    p.add_argument("--labels", default="data/processed/labels.pt")
    p.add_argument("--split", default="data/splits/split_A_composition_ood.json")
    p.add_argument("--out-dir", default="results/ood_demo")
    p.add_argument("--n-samples", type=int, default=10,
                   help="Number of ID samples to use for the demo (also "
                        "number of OOD samples per mode).")
    p.add_argument("--mc", type=int, default=50,
                   help="MC-Dropout samples per spectrum.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def resolve_device(prefer: str) -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device):
    """Build + load model using the contract from CHAT2_TASK17_HANDOVER §5.

    Refs are dispatched to the factory via a tempfile, then overwritten
    by the actual state_dict (which contains the baked-in pure_ref buffer).
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]

    refs = ck["model"]["reconstruction.pure_ref"].cpu().numpy()
    tmpdir = Path(tempfile.mkdtemp())
    ref_tmp = tmpdir / "ref.npy"
    np.save(ref_tmp, refs)

    model = build_full_model_from_config(cfg, reference_spectra_path=str(ref_tmp))
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()

    print(f"  Loaded checkpoint from {ckpt_path}")
    print(f"    epoch        : {ck.get('epoch')}")
    print(f"    val_metrics  : {ck.get('val_metrics')}")
    print(f"    pure_ref     : shape {refs.shape}")
    return model


def build_loaders(
    spectra_path: Path, labels_path: Path, split_path: Path, batch_size: int = 64,
) -> Tuple[DataLoader, Subset]:
    """Build val DataLoader (for calibration) + test Subset (for demo)."""
    spectra = torch.load(spectra_path, weights_only=True).float()
    labels = torch.load(labels_path, weights_only=True).float()
    with open(split_path) as f:
        split = json.load(f)

    base = TensorDataset(spectra, labels)
    val_ds = Subset(base, split["val"])
    test_ds = Subset(base, split["test"])

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0)
    return val_loader, test_ds


def summarise_scores(name: str, scores: np.ndarray) -> Dict[str, float]:
    return {
        "label": name,
        "n": len(scores),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "p25": float(np.percentile(scores, 25)),
        "median": float(np.median(scores)),
        "p75": float(np.percentile(scores, 75)),
        "max": float(scores.max()),
    }


def fmt_row(d: Dict[str, float]) -> str:
    return (f"  {d['label']:<24} n={d['n']:>3} "
            f"mean={d['mean']:.3f} std={d['std']:.3f} "
            f"[{d['min']:.3f} .. {d['max']:.3f}] "
            f"p50={d['median']:.3f}")


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Device: {device}\n")

    print("== Loading model ==")
    model = load_model_from_checkpoint(Path(args.checkpoint), device)

    print("\n== Building loaders ==")
    val_loader, test_ds = build_loaders(
        Path(args.spectra), Path(args.labels), Path(args.split),
    )
    print(f"  val   : {len(val_loader.dataset)} samples")
    print(f"  test  : {len(test_ds)} samples")

    print("\n== Calibrating OOD scorer on VAL set ==")
    scorer = OODScorer(model, recon_weight=0.6, var_weight=0.4,
                       mc_samples=args.mc, threshold_percentile=95.0)
    cal = scorer.calibrate(val_loader)
    print(f"  recon_p95 : {cal.recon_p95:.4f}")
    print(f"  var_p95   : {cal.var_p95:.4e}")
    print(f"  score_p95 : {cal.score_p95:.4f}  <- OOD threshold")
    print(f"  n_cal     : {cal.n_calibration_samples}")
    scorer.save(out_dir / "calibration.json")

    # ----- Pick N ID samples from TEST set -----
    rng = np.random.default_rng(args.seed)
    id_indices = rng.choice(len(test_ds), size=args.n_samples, replace=False)
    id_spectra_list = [test_ds[int(i)][0] for i in id_indices]
    # Each element is (P,) or (1, P) -- stack into (N, P)
    id_spectra = torch.stack([
        s.squeeze(0) if s.ndim == 2 else s for s in id_spectra_list
    ]).to(device)

    print(f"\n== Scoring {args.n_samples} ID samples from TEST set ==")
    id_scores = scorer.score_batch(id_spectra).cpu().numpy()

    print(f"\n== Building synthetic OOD analogues ({args.n_samples} per mode) ==")
    results: Dict[str, np.ndarray] = {"ID (test)": id_scores}
    for mode in OOD_MODES:
        ood_list = [
            make_synthetic_ood(id_spectra[i].cpu(), mode=mode, seed=args.seed + i)
            for i in range(args.n_samples)
        ]
        ood_batch = torch.stack(ood_list).to(device)
        results[f"OOD ({mode})"] = scorer.score_batch(ood_batch).cpu().numpy()

    # ----- Print table -----
    print("\n" + "=" * 72)
    print(f"OOD DISCRIMINATION REPORT  -- threshold = {cal.score_p95:.3f}")
    print("=" * 72)
    summaries: List[Dict[str, float]] = []
    for label, scores in results.items():
        s = summarise_scores(label, scores)
        s["above_threshold"] = int((scores > cal.score_p95).sum())
        summaries.append(s)
        print(fmt_row(s)
              + f"  flagged={s['above_threshold']}/{s['n']}")

    print("\n  Discriminative check (each OOD mode mean > ID mean?):")
    id_mean = id_scores.mean()
    for mode in OOD_MODES:
        ood_mean = results[f"OOD ({mode})"].mean()
        verdict = "OK" if ood_mean > id_mean else "WEAK"
        print(f"    {mode:>10}: ID={id_mean:.3f}  OOD={ood_mean:.3f}  [{verdict}]")

    # ----- Save CSV + raw scores -----
    csv_path = out_dir / "score_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "label", "n", "mean", "std", "min", "p25", "median", "p75", "max",
            "above_threshold",
        ])
        w.writeheader()
        for s in summaries:
            w.writerow(s)
    print(f"\nWrote {csv_path}")

    np.savez(out_dir / "raw_scores.npz", **{
        k.replace(" ", "_").replace("(", "").replace(")", ""): v
        for k, v in results.items()
    })
    print(f"Wrote {out_dir / 'raw_scores.npz'}")

    # ----- Optional plot -----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        labels = list(results.keys())
        data = [results[lab] for lab in labels]
        positions = np.arange(len(labels))
        bp = ax.boxplot(data, positions=positions, widths=0.6,
                        showfliers=True, patch_artist=True)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(
                "#88bcff" if i == 0 else "#ffb088"
            )
        ax.axhline(cal.score_p95, color="red", ls="--", lw=1,
                   label=f"threshold = {cal.score_p95:.3f}")
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("OOD score")
        ax.set_title("T18+T19 stretch test: ID vs synthetic OOD")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        plot_path = out_dir / "score_distribution.png"
        fig.savefig(plot_path, dpi=140)
        print(f"Wrote {plot_path}")
    except ImportError:
        print("matplotlib not available -- skipping plot")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
