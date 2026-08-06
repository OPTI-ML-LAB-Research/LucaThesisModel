"""T24 — Baseline 2: 1D-ResNet WITHOUT physics loss / reconstruction.

This is the architectural twin of ``RamanPhysicsAI`` (Chat 3) minus:
  * reconstruction module
  * physics loss term
  * per-compound loss weights

The training objective is pure MAE on the composition simplex. Everything
else (backbone, quant head, optimiser, scheduler, augmentation,
early-stopping, batch size, learning rate) matches T17 exactly so the
T25 comparison is apples-to-apples.

Why duplicate the backbone code instead of importing from
``src.models.full_model``? Because this file must run **without** the
reconstruction / physics-loss machinery, and because deleting that
machinery would break the running RamanPhysicsAI. Duplicating ~150
lines is the lesser evil; the duplication is one-off (Phase B
deliverable, doesn't grow).

Output
------
``checkpoints/baselines/resnet_only_best.pt`` saved in the same dict
schema as ``checkpoints/best.pt`` so downstream evaluation reuses code:
  ``{"model": state_dict, "epoch": int, "val_metrics": {...},
     "config": cfg_dict}``

Plus ``checkpoints/baselines/resnet_only_log.csv`` for the learning curves.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.splits import load_split  # noqa: E402


COMPOUND_ORDER = ["Alanine", "Asparagine", "Aspartic Acid",
                  "Glutamic Acid", "Histidine", "Glucosamine"]


# ---------------------------------------------------------------------------
# Architecture — mirrors Chat 3 backbone + head exactly
# ---------------------------------------------------------------------------

class BasicBlock1D(nn.Module):
    """Two-conv residual block. Stride > 1 in conv1 downsamples spatially."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        # Shortcut: identity when shapes match; 1×1 conv otherwise.
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNet1DBackbone(nn.Module):
    """1D-ResNet with 4 stages [32, 64, 128, 256], 2 blocks per stage.

    Input shape: (B, 1, P=1024). Output shape: (B, 256) after
    AdaptiveAvgPool1d(1) → flatten.
    """
    def __init__(self, channels: Sequence[int] = (32, 64, 128, 256),
                 blocks_per_stage: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )
        stages = []
        prev_ch = channels[0]
        for i, ch in enumerate(channels):
            for j in range(blocks_per_stage):
                stride = 2 if (i > 0 and j == 0) else 1
                stages.append(BasicBlock1D(prev_ch, ch, stride=stride))
                prev_ch = ch
        # `stages` is flat; group into 4 stage-lists to match Chat-3 key naming
        # backbone.stages.<i>.<j>.<...>
        self.stages = nn.ModuleList([
            nn.Sequential(*stages[i * blocks_per_stage:(i + 1) * blocks_per_stage])
            for i in range(len(channels))
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        for stage in self.stages:
            out = stage(out)
        out = self.pool(out).squeeze(-1)
        return out  # (B, channels[-1])


class QuantificationHead(nn.Module):
    """Linear(256→128) → ReLU → Dropout(p) → Linear(128→6) → Softmax."""
    def __init__(self, feature_dim: int = 256, n_compounds: int = 6,
                 dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, n_compounds)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(feat), inplace=True)
        h = self.dropout(h)
        logits = self.fc2(h)
        return F.softmax(logits, dim=-1)


class ResNetOnly(nn.Module):
    """Backbone + Head only. No reconstruction, no physics.

    Forward returns ``{"composition": (B, 6)}`` so downstream eval code
    that expects a dict still works.
    """
    def __init__(self, feature_dim: int = 256, dropout_rate: float = 0.2,
                 n_compounds: int = 6):
        super().__init__()
        self.backbone = ResNet1DBackbone()
        self.head = QuantificationHead(
            feature_dim=feature_dim, n_compounds=n_compounds,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> dict:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        feat = self.backbone(x)
        comp = self.head(feat)
        return {"composition": comp}


# ---------------------------------------------------------------------------
# Augmentation — minimal, identical to T17 train_config.yaml
# ---------------------------------------------------------------------------

class _AugmentedDataset(Dataset):
    """Per-sample independent augmentation: shift, scale, noise."""
    def __init__(self, base: Dataset, *,
                 shift_max_px: int = 6, intensity_range=(0.9, 1.1),
                 noise_sigma: float = 0.002,
                 shift_p: float = 0.5, scale_p: float = 0.5,
                 noise_p: float = 0.5,
                 enabled: bool = True, seed: int | None = None):
        self.base = base
        self.enabled = enabled
        self.shift_max_px = shift_max_px
        self.intensity_range = intensity_range
        self.noise_sigma = noise_sigma
        self.shift_p = shift_p
        self.scale_p = scale_p
        self.noise_p = noise_p
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if not self.enabled:
            return x, y
        # x is a torch tensor (P,) typically
        x = x.clone()
        # Shift (circular roll over spectrum length)
        if self.rng.random() < self.shift_p and self.shift_max_px > 0:
            shift = int(self.rng.integers(-self.shift_max_px, self.shift_max_px + 1))
            x = torch.roll(x, shift, dims=-1)
        # Scale
        if self.rng.random() < self.scale_p:
            lo, hi = self.intensity_range
            s = float(self.rng.uniform(lo, hi))
            x = x * s
        # Noise
        if self.rng.random() < self.noise_p and self.noise_sigma > 0:
            x = x + torch.from_numpy(
                self.rng.normal(0.0, self.noise_sigma, size=x.shape).astype(np.float32)
            )
        return x, y


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def _mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return (y_pred - y_true).abs().mean()


def train_resnet_only(
    spectra: np.ndarray,
    labels: np.ndarray,
    split,
    *,
    cfg: dict,
    out_dir: Path,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Train + select best on val MAE. Saves checkpoint + log CSV.

    Returns the same dict that gets saved into the checkpoint (so callers
    can introspect without re-loading).
    """
    seed = int(cfg.get("reproducibility", {}).get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    spec_t = torch.from_numpy(spectra.astype(np.float32))
    lab_t = torch.from_numpy(labels.astype(np.float32))

    full_ds = TensorDataset(spec_t, lab_t)
    train_ds = Subset(full_ds, split.train)
    val_ds = Subset(full_ds, split.val)

    aug_cfg = cfg.get("data", {}).get("augmentation", {})
    train_ds_aug = _AugmentedDataset(
        train_ds, seed=seed,
        shift_max_px=aug_cfg.get("shift_max_px", 6),
        intensity_range=tuple(aug_cfg.get("intensity_range", (0.9, 1.1))),
        noise_sigma=aug_cfg.get("noise_sigma", 0.002),
        shift_p=aug_cfg.get("shift_p", 0.5),
        scale_p=aug_cfg.get("scale_p", 0.5),
        noise_p=aug_cfg.get("noise_p", 0.5),
        enabled=aug_cfg.get("enabled", True),
    )

    bs = int(cfg.get("data", {}).get("batch_size", 64))
    num_workers = int(cfg.get("data", {}).get("num_workers", 0))
    train_loader = DataLoader(train_ds_aug, batch_size=bs, shuffle=True,
                              num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=max(bs, 128), shuffle=False,
                            num_workers=num_workers)

    model_cfg = cfg.get("model", {})
    model = ResNetOnly(
        feature_dim=int(model_cfg.get("feature_dim", 256)),
        dropout_rate=float(model_cfg.get("dropout_rate", 0.2)),
        n_compounds=6,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"  ResNetOnly: {n_params:,} trainable params")

    tcfg = cfg.get("training", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg.get("learning_rate", 1e-3)),
        weight_decay=float(tcfg.get("weight_decay", 1e-5)),
        betas=tuple(tcfg.get("betas", (0.9, 0.999))),
    )
    epochs = int(tcfg.get("epochs", 50))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs,
        eta_min=float(tcfg.get("scheduler_eta_min", 1e-6)),
    )
    grad_clip = float(tcfg.get("grad_clip_norm", 1.0))
    patience = int(tcfg.get("early_stopping_patience", 15))
    min_delta = float(tcfg.get("early_stopping_min_delta", 1e-5))

    best_val_mae = math.inf
    best_state = None
    best_epoch = -1
    no_improve = 0
    log_rows: list[dict] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "resnet_only_log.csv"
    ck_path = out_dir / "resnet_only_best.pt"

    t0 = time.perf_counter()
    for epoch in range(epochs):
        epoch_t0 = time.perf_counter()
        model.train()
        train_losses = []
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = _mae(out["composition"], y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(loss.item())
        train_mae = float(np.mean(train_losses))

        # ---- Val ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device); y = y.to(device)
                out = model(x)
                val_losses.append(_mae(out["composition"], y).item())
        val_mae = float(np.mean(val_losses))

        scheduler.step()
        epoch_sec = time.perf_counter() - epoch_t0
        improved = val_mae < (best_val_mae - min_delta)
        if improved:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        log_rows.append({
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_mae": train_mae,
            "val_mae": val_mae,
            "best_val_mae": best_val_mae,
            "epoch_seconds": epoch_sec,
        })
        if verbose:
            tag = " *" if improved else ""
            print(f"  ep {epoch:>2d}  train_mae={train_mae:.4f}  "
                  f"val_mae={val_mae:.4f}  best={best_val_mae:.4f}  "
                  f"({epoch_sec:.1f}s){tag}")
        if no_improve >= patience:
            if verbose:
                print(f"  early stop @ epoch {epoch} "
                      f"(patience {patience} reached)")
            break

    total_seconds = time.perf_counter() - t0
    if verbose:
        print(f"  total: {total_seconds/60:.1f} min, best epoch {best_epoch} "
              f"@ val_mae {best_val_mae:.4f}")

    # Save log
    if log_rows:
        with log_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader()
            w.writerows(log_rows)

    # Save checkpoint (mirrors RamanPhysicsAI schema so downstream is uniform)
    payload = {
        "model": best_state,
        "epoch": best_epoch,
        "val_metrics": {"val_mae": float(best_val_mae)},
        "config": cfg,
        "meta": {
            "arch": "ResNetOnly",
            "n_params": int(n_params),
            "n_train": len(split.train),
            "n_val": len(split.val),
            "total_seconds": float(total_seconds),
        },
    }
    torch.save(payload, ck_path)
    if verbose:
        print(f"  saved checkpoint -> {ck_path}")
        print(f"  saved log        -> {log_path}")
    return payload


def predict_resnet_only(model: ResNetOnly, X: np.ndarray,
                       device: str = "cpu", batch_size: int = 128) -> np.ndarray:
    """Forward all rows of X through ``model``. Returns (N, 6) compositions."""
    model.eval().to(device)
    out_chunks = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = torch.from_numpy(X[i:i+batch_size].astype(np.float32)).to(device)
            yb = model(xb)["composition"].cpu().numpy()
            out_chunks.append(yb)
    return np.concatenate(out_chunks, axis=0)


def load_resnet_only(ck_path: str | Path, device: str = "cpu") -> tuple[ResNetOnly, dict]:
    """Reverse of training: load checkpoint and rebuild model."""
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    mc = cfg.get("model", {})
    model = ResNetOnly(
        feature_dim=int(mc.get("feature_dim", 256)),
        dropout_rate=float(mc.get("dropout_rate", 0.2)),
    )
    model.load_state_dict(ck["model"])
    return model.to(device), ck


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spectra", type=Path,
                        default=Path("data/processed/spectra_full.pt"))
    parser.add_argument("--labels", type=Path,
                        default=Path("data/processed/labels.pt"))
    parser.add_argument("--split", type=Path,
                        default=Path("data/splits/split_A_composition_ood.json"))
    parser.add_argument("--config", type=Path,
                        default=Path("configs/train_config.yaml"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("checkpoints/baselines"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--epochs-override", type=int, default=None,
                        help="Override training.epochs (e.g. 5 for smoke).")
    args = parser.parse_args(argv)

    print(f"[1/3] Loading config + data ...")
    import yaml
    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.epochs_override is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs_override
        print(f"  epochs override: {args.epochs_override}")

    spectra = torch.load(args.spectra, weights_only=False)
    labels = torch.load(args.labels, weights_only=False)
    if not isinstance(spectra, np.ndarray):
        spectra = spectra.numpy() if hasattr(spectra, "numpy") else np.asarray(spectra)
    if not isinstance(labels, np.ndarray):
        labels = labels.numpy() if hasattr(labels, "numpy") else np.asarray(labels)
    print(f"  spectra={spectra.shape}, labels={labels.shape}, device={args.device}")

    split = load_split(args.split)
    print(f"  split: train={len(split.train)} val={len(split.val)} test={len(split.test)}")

    print(f"[2/3] Training ResNet-only (no physics) ...")
    payload = train_resnet_only(spectra, labels, split, cfg=cfg,
                                out_dir=args.out_dir, device=args.device,
                                verbose=True)

    print(f"[3/3] Done. Best val MAE = {payload['val_metrics']['val_mae']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPOUND_ORDER", "BasicBlock1D", "ResNet1DBackbone",
    "QuantificationHead", "ResNetOnly",
    "train_resnet_only", "predict_resnet_only", "load_resnet_only",
]
