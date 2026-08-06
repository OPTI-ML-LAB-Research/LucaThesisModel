"""Plot loss / MAE curves from results/training_log.csv.

Standalone utility: works on a partial CSV (e.g. after early stop or crash).

Usage:
    python scripts/plot_training_curves.py
    python scripts/plot_training_curves.py --csv results/training_log.csv \
                                           --out results/figures/training_curves.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict


def read_csv(path: Path) -> Dict[str, List[float]]:
    """Read a training log CSV; return columns as lists of floats.

    Skips rows where the requested columns aren't numeric (e.g. a stray
    header row from a resumed run).
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    cols: Dict[str, List[float]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                cols.setdefault(k, []).append(fv)
    return cols


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="results/training_log.csv")
    p.add_argument("--out", type=str,
                   default="results/figures/training_curves.png")
    p.add_argument("--show", action="store_true",
                   help="Open the figure interactively (in addition to saving).")
    args = p.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = read_csv(Path(args.csv))
    if not data.get("epoch"):
        print(f"No epoch column found in {args.csv}; nothing to plot.")
        return 1

    epochs = data["epoch"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    # (0,0) Total loss
    ax = axes[0, 0]
    if "train_loss_total" in data:
        ax.plot(epochs, data["train_loss_total"], label="train", lw=1.5)
    if "val_loss_total" in data:
        ax.plot(epochs, data["val_loss_total"], label="val", lw=1.5)
    ax.set_ylabel("Total loss")
    ax.set_title("Combined loss (alpha*quant + beta*physics + gamma*L2)")
    ax.legend(); ax.grid(alpha=0.3)

    # (0,1) MAE
    ax = axes[0, 1]
    if "train_mae" in data:
        ax.plot(epochs, data["train_mae"], label="train MAE", lw=1.5)
    if "val_mae" in data:
        ax.plot(epochs, data["val_mae"], label="val MAE", lw=1.5)
    if "best_val_mae" in data:
        ax.plot(epochs, data["best_val_mae"], label="best val MAE",
                lw=1.0, ls="--", color="gray")
    # Reference bars from Custom Instructions Section 6.
    ax.axhline(0.020, color="green", ls=":", lw=1, label="target 0.020")
    ax.axhline(0.025, color="orange", ls=":", lw=1, label="floor 0.025")
    ax.set_ylabel("MAE")
    ax.set_title("Quantification MAE")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (1,0) Quant vs physics
    ax = axes[1, 0]
    if "train_loss_quant" in data:
        ax.plot(epochs, data["train_loss_quant"], label="train quant", lw=1.5)
    if "train_loss_physics" in data:
        ax.plot(epochs, data["train_loss_physics"],
                label="train physics", lw=1.5)
    if "val_loss_physics" in data:
        ax.plot(epochs, data["val_loss_physics"],
                label="val physics", lw=1.5, ls="--")
    ax.set_xlabel("epoch"); ax.set_ylabel("component loss")
    ax.set_title("Loss components (train)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (1,1) LR + epoch time
    ax = axes[1, 1]
    if "lr" in data:
        ax.plot(epochs, data["lr"], color="purple", label="lr")
        ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("learning rate")
    if "epoch_seconds" in data:
        ax2 = ax.twinx()
        ax2.plot(epochs, data["epoch_seconds"], color="gray",
                 alpha=0.5, label="epoch sec")
        ax2.set_ylabel("seconds / epoch", color="gray")
    ax.set_title("LR schedule + epoch time")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("Raman Physics-AI training", y=0.995, fontsize=12)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_path}")
    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
