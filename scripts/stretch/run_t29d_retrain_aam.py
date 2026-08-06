"""Stretch T29D — Retrain RamanPhysicsAI with 7 outputs on AAM.

Trains a new model with n_compounds=7 (6 AA + 1 minerals combined)
on the AAM train split. Reuses the existing training loop and most
hyperparameters from the AA config; only n_compounds and reference
spectra change.

Output:
    checkpoints/aam_retrained/
        best.pt              (best val MAE checkpoint)
        last.pt              (final epoch checkpoint)
        training_log.csv
        config.json

Usage:
    python scripts/stretch/run_t29d_retrain_aam.py
    python scripts/stretch/run_t29d_retrain_aam.py --epochs 30  # shorter
    python scripts/stretch/run_t29d_retrain_aam.py --smoke      # 2-epoch test

Estimated runtime: 60-120 min on CPU for 50 epochs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Make the project root importable (parents[2] = repo root) AND this
# script's own dir (for _handover_utils), BEFORE importing the helper.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR.parents[1]))

from _handover_utils import COMPOUND_ORDER  # noqa: E402  (kept for parity)


def _to_3d(spec: torch.Tensor) -> torch.Tensor:
    """Ensure spectrum is (B, 1, P), matching train.py's _to_device_3d."""
    if spec.ndim == 2:
        spec = spec.unsqueeze(1)
    return spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aam-dir", default="data/processed/aam")
    p.add_argument("--ref-path", default="engine/reference_spectra_aam.npy")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--alpha-quant", type=float, default=1.0)
    p.add_argument("--beta-phys", type=float, default=0.5)
    p.add_argument("--gamma-reg", type=float, default=0.01)
    p.add_argument("--lambda-cosine", type=float, default=0.3)
    p.add_argument("--smoke", action="store_true",
                   help="2-epoch + 200-sample smoke test")
    p.add_argument("--out-dir", default="checkpoints/aam_retrained")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    aam_dir = Path(args.aam_dir)
    spectra = torch.load(aam_dir / "spectra.pt", weights_only=True)
    labels7 = torch.load(aam_dir / "labels_7d.pt", weights_only=True)
    split = json.loads((aam_dir / "split.json").read_text())
    train_idx = split["train"]
    val_idx = split["val"]

    if args.smoke:
        print("  [SMOKE] capping to 200 train / 50 val, 2 epochs")
        train_idx = train_idx[:200]
        val_idx = val_idx[:50]
        args.epochs = 2

    # Keep spectra as (B, 1, P) to match what the model + physics_loss expect.
    X_tr = _to_3d(spectra[train_idx])
    Y_tr = labels7[train_idx]
    X_va = _to_3d(spectra[val_idx])
    Y_va = labels7[val_idx]
    print(f"  train: X {tuple(X_tr.shape)}, Y {tuple(Y_tr.shape)}")
    print(f"  val:   X {tuple(X_va.shape)}, Y {tuple(Y_va.shape)}")

    train_loader = DataLoader(TensorDataset(X_tr, Y_tr),
                              batch_size=args.batch_size, shuffle=True,
                              drop_last=False)
    val_loader = DataLoader(TensorDataset(X_va, Y_va),
                            batch_size=args.batch_size, shuffle=False)

    # ---- Build model with n_compounds=7 ----
    from src.models.full_model import RamanPhysicsAI
    print("\n[T29D] Building RamanPhysicsAI with n_compounds=7 ...")
    model = RamanPhysicsAI(
        reference_spectra=args.ref_path,
        n_compounds=7,
        spectrum_length=1024,
    )
    n_params = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
    print(f"  params: {n_params:,}")

    # ---- Loss (FUNCTION, called exactly like src/training/train.py) ----
    from src.training.losses import combined_loss
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # ---- Training loop ----
    print(f"\n[T29D] Training for {args.epochs} epochs ...")
    history = []
    best_val_mae = float("inf")
    patience_counter = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        # ----- train -----
        model.train()
        train_total, train_mae, n_train = 0.0, 0.0, 0
        for X, Y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            out = model(X)
            components = combined_loss(
                Y, out["composition"], X, out["reconstruction"],
                model_parameters=model.parameters(),
                alpha=args.alpha_quant,
                beta=args.beta_phys,
                gamma=args.gamma_reg,
                lambda_cosine=args.lambda_cosine,
                return_components=True,
            )
            loss = components["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            with torch.no_grad():
                mae = (out["composition"] - Y).abs().mean().item()
            train_total += loss.item() * len(X)
            train_mae += mae * len(X)
            n_train += len(X)
        train_total /= max(n_train, 1)
        train_mae /= max(n_train, 1)

        # ----- val -----
        model.eval()
        val_total, val_mae, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for X, Y in val_loader:
                out = model(X)
                components = combined_loss(
                    Y, out["composition"], X, out["reconstruction"],
                    model_parameters=None,   # no L2 in val (matches train.py)
                    alpha=args.alpha_quant,
                    beta=args.beta_phys,
                    gamma=0.0,
                    lambda_cosine=args.lambda_cosine,
                    return_components=True,
                )
                loss = components["total"]
                mae = (out["composition"] - Y).abs().mean().item()
                val_total += loss.item() * len(X)
                val_mae += mae * len(X)
                n_val += len(X)
        val_total /= max(n_val, 1)
        val_mae /= max(n_val, 1)

        scheduler.step()
        elapsed = time.time() - t0
        print(f"  Epoch {epoch + 1:3d}/{args.epochs}: "
              f"train_loss={train_total:.4f} train_mae={train_mae:.4f}  "
              f"val_loss={val_total:.4f} val_mae={val_mae:.4f}  "
              f"({elapsed:.0f}s elapsed)")
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_total, "train_mae": train_mae,
            "val_loss": val_total, "val_mae": val_mae,
            "lr": scheduler.get_last_lr()[0],
        })

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), out_dir / "best.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  early stopping (patience={args.patience})")
                break

    # ---- Save final outputs ----
    torch.save(model.state_dict(), out_dir / "last.pt")
    import csv
    with open(out_dir / "training_log.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    config = {
        "task": "T29D AAM retrain",
        "n_compounds": 7,
        "epochs_run": len(history),
        "best_val_mae": float(best_val_mae),
        "hyperparams": {
            "lr": args.lr, "weight_decay": args.weight_decay,
            "batch_size": args.batch_size, "patience": args.patience,
            "alpha_quant": args.alpha_quant, "beta_phys": args.beta_phys,
            "gamma_reg": args.gamma_reg, "lambda_cosine": args.lambda_cosine,
        },
        "n_train": int(len(X_tr)), "n_val": int(len(X_va)),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"\n[T29D done] best val MAE = {best_val_mae:.4f}")
    print(f"  saved → {out_dir}")
    print("Next: run T29E to test on AAM test set with retrained checkpoint")


if __name__ == "__main__":
    main()