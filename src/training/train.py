"""End-to-end training entry point for RamanPhysicsAI (T15).

Usage:
    python -m src.training.train --config configs/train_config.yaml
    python -m src.training.train --config configs/train_config.yaml --smoke
    python -m src.training.train --config configs/train_config.yaml --no-wandb

The script:

    1. Loads + merges configs/default.yaml + configs/train_config.yaml.
    2. Loads cached preprocessed spectra + labels + split JSON
       (built by scripts/prepare_data.py and Phase A T07).
    3. Builds train / val DataLoaders (training set wrapped in
       AugmentedDataset).
    4. Constructs RamanPhysicsAI via build_full_model_from_config.
    5. Optimizer = AdamW; scheduler = CosineAnnealingLR.
    6. Per epoch: train + val + log + checkpoint.
       Saves best by val MAE; also a "last" checkpoint each epoch.
    7. Early stopping after ``patience`` epochs of no improvement.
    8. Writes results/training_log.csv (one row per epoch).

The script does NOT run a final test-set evaluation -- that's reserved
for Chat 2 T17 (mid-checkpoint), which loads ``checkpoints/best.pt`` and
runs all 5 metrics from src/eval/metrics.py.

Designed to fail loud and fail early: every config key, file path, and
shape is validated up front before training begins.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset, TensorDataset

# Local imports.
from src.data.augmentation import AugmentedDataset, RamanAugmentation
from src.models.full_model import RamanPhysicsAI, build_full_model_from_config
from src.training.losses import combined_loss, compute_per_compound_weights


# Optional: tqdm + wandb. Both are soft deps -- script must run without them.
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import wandb  # type: ignore
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

LOG = logging.getLogger("train")


def configure_logging(verbose: bool = False) -> None:
    """Root logger -> stderr with timestamps."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def set_seeds(seed: int, deterministic_cudnn: bool = False) -> None:
    """Seed all RNGs that the training pipeline touches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(prefer: str) -> torch.device:
    """Pick a torch.device from a 'auto' | 'cuda' | 'cpu' string."""
    prefer = prefer.lower()
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device.prefer='cuda' but CUDA not available.")
        return torch.device("cuda")
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unknown device.prefer={prefer!r}")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``.

    Lists are replaced wholesale (NOT element-wise merged).
    """
    result = dict(base)
    for k, v in override.items():
        if (k in result and isinstance(result[k], dict)
                and isinstance(v, dict)):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(train_cfg_path: Path) -> Dict[str, Any]:
    """Load default + training config and merge.

    Convention: ``configs/default.yaml`` lives next to ``train_cfg_path``.
    """
    train_cfg_path = Path(train_cfg_path)
    if not train_cfg_path.exists():
        raise FileNotFoundError(f"Training config not found: {train_cfg_path}")

    default_path = train_cfg_path.parent / "default.yaml"
    base: Dict[str, Any] = {}
    if default_path.exists():
        with open(default_path, "r", encoding='utf-8') as f:
            base = yaml.safe_load(f) or {}
        LOG.info(f"Loaded base config: {default_path}")
    else:
        LOG.warning(f"No default.yaml at {default_path}; using train_config alone.")

    with open(train_cfg_path, "r", encoding='utf-8') as f:
        train = yaml.safe_load(f) or {}
    LOG.info(f"Loaded train config: {train_cfg_path}")

    return deep_merge(base, train)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _validate_path(label: str, path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Required input '{label}' not found: {p.resolve()}"
        )
    return p


def load_cached_data(cfg: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, list]]:
    """Load preprocessed spectra, labels, and the train/val/test split.

    Returns:
        spectra: (N, P) float32
        labels:  (N, n_compounds) float32
        split:   dict with keys ``train``, ``val``, ``test`` -> list[int]
    """
    paths = cfg["paths"]
    spectra_p = _validate_path("spectra_cache", paths["spectra_cache"])
    labels_p = _validate_path("labels_cache", paths["labels_cache"])
    split_p = _validate_path("split_train_val_test", paths["split_train_val_test"])

    spectra = torch.load(spectra_p, weights_only=True)
    labels = torch.load(labels_p, weights_only=True)
    if not isinstance(spectra, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("Cache files must contain torch.Tensor objects.")
    spectra = spectra.float()
    labels = labels.float()

    if spectra.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Sample count mismatch: spectra={spectra.shape[0]}, "
            f"labels={labels.shape[0]}"
        )

    expected_P = cfg["data"]["spectrum_length"]
    if spectra.shape[1] != expected_P:
        raise ValueError(
            f"Spectrum length mismatch: cache has P={spectra.shape[1]}, "
            f"config expects {expected_P}"
        )

    with open(split_p, "r") as f:
        split = json.load(f)
    for key in ("train", "val", "test"):
        if key not in split:
            raise KeyError(f"Split JSON missing required key '{key}'")

    LOG.info(f"Loaded {spectra.shape[0]} spectra, {labels.shape[1]} compounds")
    LOG.info(
        f"Split sizes: train={len(split['train'])}, "
        f"val={len(split['val'])}, test={len(split['test'])}"
    )
    return spectra, labels, split


def build_dataloaders(
    spectra: torch.Tensor,
    labels: torch.Tensor,
    split: Dict[str, list],
    cfg: Dict[str, Any],
    smoke: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Build train (augmented) and val (deterministic) DataLoaders."""
    train_idx = split["train"]
    val_idx = split["val"]

    if smoke:
        # Subsample 200 train + 50 val for a quick pipeline check.
        rng = np.random.default_rng(0)
        train_idx = rng.choice(train_idx, size=min(200, len(train_idx)),
                               replace=False).tolist()
        val_idx = rng.choice(val_idx, size=min(50, len(val_idx)),
                             replace=False).tolist()
        LOG.warning(f"SMOKE MODE: subsampled to train={len(train_idx)}, "
                    f"val={len(val_idx)}")

    base = TensorDataset(spectra, labels)
    train_base = Subset(base, train_idx)
    val_base = Subset(base, val_idx)

    aug_cfg = cfg["data"].get("augmentation", {}) or {}
    if aug_cfg.get("enabled", True):
        aug = RamanAugmentation(
            shift_max_px=aug_cfg.get("shift_max_px", 6),
            intensity_range=tuple(aug_cfg.get("intensity_range", [0.9, 1.1])),
            noise_sigma=aug_cfg.get("noise_sigma", 0.005),
            shift_p=aug_cfg.get("shift_p", 0.5),
            scale_p=aug_cfg.get("scale_p", 0.5),
            noise_p=aug_cfg.get("noise_p", 0.5),
            seed=cfg["reproducibility"]["seed"],
        )
        train_ds: torch.utils.data.Dataset = AugmentedDataset(
            train_base, aug, keep_channel_dim=False,
        )
        LOG.info(
            f"Augmentation ENABLED: shift+/-{aug.shift_max_px}px, "
            f"scale={aug.intensity_range}, noise_sigma={aug.noise_sigma}"
        )
    else:
        train_ds = train_base
        LOG.info("Augmentation DISABLED")

    bs = cfg["data"]["batch_size"]
    nw = cfg["data"].get("num_workers", 0)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, drop_last=False, pin_memory=False,
    )
    val_loader = DataLoader(
        val_base, batch_size=bs, shuffle=False,
        num_workers=nw, drop_last=False, pin_memory=False,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model / optimizer / scheduler factories
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    tcfg = cfg["training"]
    name = tcfg.get("optimizer", "adamw").lower()
    lr = float(tcfg["learning_rate"])
    wd = float(tcfg.get("weight_decay", 0.0))
    betas = tuple(tcfg.get("betas", [0.9, 0.999]))

    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd,
                                betas=betas)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                                 betas=betas)
    raise ValueError(f"Unsupported optimizer: {name!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Dict[str, Any], epochs: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    name = cfg["training"].get("scheduler", "cosine").lower()
    if name in ("none", "off"):
        return None
    if name == "cosine":
        eta_min = float(cfg["training"].get("scheduler_eta_min", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=eta_min,
        )
    raise ValueError(f"Unsupported scheduler: {name!r}")


def build_per_compound_weights(
    cfg: Dict[str, Any],
    n_compounds: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Resolve per-compound weights from config.

    Config field: ``loss.per_compound_weights`` (under loss section).
    Three accepted forms:

    * **Missing or null** → returns None → unweighted MAE (legacy).
    * **List of K floats** (e.g. ``[1.07, 1.20, 1.10, 1.18, 0.47, 0.98]``)
      → used as-is.
    * **String ``"auto:<csv path>"``** → reads per-compound MAE from CSV,
      column ``mae``, in canonical compound order. Currently NOT wired
      up — pass an explicit list instead.

    Args:
        cfg: Merged config dict.
        n_compounds: Expected weight vector length.
        device: Target device for the returned tensor.

    Returns:
        Tensor of shape (n_compounds,) on ``device``, or None.
    """
    raw = (cfg.get("loss", {}) or {}).get("per_compound_weights", None)
    if raw is None:
        LOG.info("Per-compound weights: NONE (using unweighted MAE)")
        return None

    if isinstance(raw, (list, tuple)):
        if len(raw) != n_compounds:
            raise ValueError(
                f"loss.per_compound_weights has {len(raw)} entries but "
                f"n_compounds = {n_compounds}."
            )
        weights = torch.as_tensor(raw, dtype=torch.float32, device=device)
        # Sanity check: non-negative
        if (weights < 0).any():
            raise ValueError(
                f"loss.per_compound_weights contains negatives: {raw}"
            )
        mean_w = float(weights.mean())
        LOG.info(
            f"Per-compound weights ENABLED: {[round(v, 3) for v in raw]} "
            f"(mean={mean_w:.3f})"
        )
        if abs(mean_w - 1.0) > 0.2:
            LOG.warning(
                f"mean(weights) = {mean_w:.3f} deviates from 1.0; total "
                "loss magnitude will differ from unweighted MAE — consider "
                "rescaling so other hyperparameters keep their meaning."
            )
        return weights

    raise TypeError(
        f"loss.per_compound_weights must be a list of floats or null; "
        f"got {type(raw).__name__}: {raw!r}"
    )


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

@dataclass
class EpochStats:
    """Aggregates over an epoch."""
    n_samples: int = 0
    sum_total: float = 0.0
    sum_quant: float = 0.0
    sum_physics: float = 0.0
    sum_l2: float = 0.0
    sum_mae: float = 0.0     # equal to sum_quant; kept for clarity in logs
    sum_raw_mae: float = 0.0 # unweighted MAE, for fair comparison w/ T17 metrics

    def add(self, batch_size: int, total: float, quant: float,
            physics: float, l2: float, raw_mae: Optional[float] = None) -> None:
        self.n_samples += batch_size
        self.sum_total += total * batch_size
        self.sum_quant += quant * batch_size
        self.sum_physics += physics * batch_size
        self.sum_l2 += l2 * batch_size
        self.sum_mae += quant * batch_size
        # If raw_mae not passed (legacy path / no weights), fall back to quant.
        self.sum_raw_mae += (raw_mae if raw_mae is not None else quant) * batch_size

    def means(self) -> Dict[str, float]:
        n = max(self.n_samples, 1)
        return {
            "loss_total": self.sum_total / n,
            "loss_quant": self.sum_quant / n,
            "loss_physics": self.sum_physics / n,
            "loss_l2": self.sum_l2 / n,
            "mae": self.sum_mae / n,         # weighted (or raw, if no weights)
            "raw_mae": self.sum_raw_mae / n, # always raw, comparable to T17
        }


def _to_device_3d(spec: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Ensure spectrum is (B, 1, P) on the right device."""
    if spec.ndim == 2:
        spec = spec.unsqueeze(1)
    return spec.to(device, non_blocking=True)


def train_one_epoch(
    model: RamanPhysicsAI,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    device: torch.device,
    epoch: int,
    per_compound_weights: Optional[torch.Tensor] = None,
) -> EpochStats:
    model.train()
    stats = EpochStats()
    loss_cfg = cfg["loss"]

    iterator = loader
    if HAS_TQDM and cfg["logging"].get("use_tqdm", True):
        iterator = tqdm(loader, desc=f"epoch {epoch:03d} train",
                        leave=False, dynamic_ncols=True)

    grad_clip = float(cfg["training"].get("grad_clip_norm", 0.0) or 0.0)

    for batch in iterator:
        spectra, labels = batch[0], batch[1]
        spectra = _to_device_3d(spectra, device)
        labels = labels.to(device, non_blocking=True)

        out = model(spectra)
        components = combined_loss(
            labels, out["composition"], spectra, out["reconstruction"],
            model_parameters=model.parameters(),
            alpha=loss_cfg["alpha_quant"],
            beta=loss_cfg["beta_phys"],
            gamma=loss_cfg["gamma_reg"],
            lambda_cosine=loss_cfg["lambda_cosine"],
            per_compound_weights=per_compound_weights,
            return_components=True,
        )
        loss = components["total"]

        # Raw (unweighted) MAE for logging — comparable to T17 quant_mae.
        with torch.no_grad():
            raw_mae = (out["composition"] - labels).abs().mean().item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = spectra.shape[0]
        stats.add(
            bs,
            total=loss.item(),
            quant=components["quant"].item(),
            physics=components["physics"].item(),
            l2=components["l2"].item(),
            raw_mae=raw_mae,
        )
    return stats


@torch.no_grad()
def validate(
    model: RamanPhysicsAI,
    loader: DataLoader,
    cfg: Dict[str, Any],
    device: torch.device,
    per_compound_weights: Optional[torch.Tensor] = None,
) -> EpochStats:
    model.eval()
    stats = EpochStats()
    loss_cfg = cfg["loss"]
    for batch in loader:
        spectra, labels = batch[0], batch[1]
        spectra = _to_device_3d(spectra, device)
        labels = labels.to(device, non_blocking=True)

        out = model(spectra)
        components = combined_loss(
            labels, out["composition"], spectra, out["reconstruction"],
            model_parameters=None,            # no L2 in val (we already report it)
            alpha=loss_cfg["alpha_quant"],
            beta=loss_cfg["beta_phys"],
            gamma=0.0,                         # ditto
            lambda_cosine=loss_cfg["lambda_cosine"],
            per_compound_weights=per_compound_weights,
            return_components=True,
        )
        raw_mae = (out["composition"] - labels).abs().mean().item()
        bs = spectra.shape[0]
        stats.add(
            bs,
            total=components["total"].item(),
            quant=components["quant"].item(),
            physics=components["physics"].item(),
            l2=0.0,
            raw_mae=raw_mae,
        )
    return stats


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    val_metrics: Dict[str, float],
    cfg: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "val_metrics": val_metrics,
        "config": cfg,
    }
    torch.save(payload, path)


def maybe_resume(
    cfg: Dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    resume: bool,
) -> Tuple[int, float]:
    """If resume=True and a 'last' checkpoint exists, load and return
    (start_epoch, best_val_mae). Otherwise (0, +inf)."""
    if not resume:
        return 0, float("inf")
    last_path = Path(cfg["paths"]["checkpoint_last"])
    if not last_path.exists():
        LOG.info("No 'last' checkpoint to resume from; starting fresh.")
        return 0, float("inf")
    LOG.info(f"Resuming from {last_path}")
    ck = torch.load(last_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    if scheduler is not None and ck.get("scheduler") is not None:
        scheduler.load_state_dict(ck["scheduler"])
    start_epoch = int(ck.get("epoch", 0)) + 1
    best = float(ck.get("val_metrics", {}).get("best_val_mae", float("inf")))
    return start_epoch, best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RamanPhysicsAI.")
    p.add_argument("--config", type=str, default="configs/train_config.yaml",
                   help="Path to training config YAML.")
    p.add_argument("--smoke", action="store_true",
                   help="Run 2 epochs on a subset of data to sanity-check.")
    p.add_argument("--no-wandb", action="store_true",
                   help="Force-disable wandb logging.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from checkpoints/last.pt if present.")
    p.add_argument("--device", type=str, default=None,
                   help="Override device.prefer (auto/cuda/cpu).")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def init_wandb(cfg: Dict[str, Any], no_wandb: bool, smoke: bool) -> bool:
    """Initialise wandb if requested + available. Returns True if active."""
    if no_wandb or smoke:
        return False
    if not cfg["logging"].get("use_wandb", False):
        return False
    if not HAS_WANDB:
        LOG.warning("wandb requested but not installed; continuing without it.")
        return False
    wandb.init(
        project=cfg["logging"].get("wandb_project", "raman-physics-ai"),
        name=cfg["logging"].get("wandb_run_name"),
        config=cfg,
    )
    return True


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    cfg = load_config(Path(args.config))
    if args.device:
        cfg.setdefault("device", {})["prefer"] = args.device

    set_seeds(
        cfg["reproducibility"]["seed"],
        deterministic_cudnn=cfg["reproducibility"].get("deterministic_cudnn", False),
    )
    device = resolve_device(cfg["device"]["prefer"])
    LOG.info(f"Device: {device}")

    # ---- Data ----
    spectra, labels, split = load_cached_data(cfg)
    train_loader, val_loader = build_dataloaders(
        spectra, labels, split, cfg, smoke=args.smoke,
    )

    # ---- Model ----
    model = build_full_model_from_config(
        cfg, reference_spectra_path=cfg["paths"]["reference_spectra"],
    ).to(device)
    counts = model.count_parameters()
    LOG.info(
        f"Model params: backbone={counts['backbone']:,} "
        f"head={counts['head']:,} recon={counts['reconstruction']:,} "
        f"total={counts['total']:,}"
    )

    # ---- Per-compound weights (Hướng 1 fix for T17 FAIL) ----
    per_compound_weights = build_per_compound_weights(
        cfg, n_compounds=model.n_compounds, device=device,
    )

    # ---- Optimizer + scheduler ----
    epochs = 2 if args.smoke else int(cfg["training"]["epochs"])
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, epochs)

    # ---- Resume ----
    start_epoch, best_val_mae = maybe_resume(
        cfg, model, optimizer, scheduler, resume=args.resume,
    )

    # ---- Logging dest ----
    log_csv_path = Path(cfg["paths"]["training_log_csv"])
    log_csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_exists = log_csv_path.exists() and args.resume
    csv_file = open(log_csv_path, "a" if csv_exists else "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "epoch", "lr", "train_loss_total", "train_loss_quant",
            "train_loss_physics", "train_mae",
            "val_loss_total", "val_loss_quant", "val_loss_physics",
            "val_mae", "best_val_mae", "epoch_seconds",
        ],
    )
    if not csv_exists:
        csv_writer.writeheader()

    use_wandb = init_wandb(cfg, no_wandb=args.no_wandb, smoke=args.smoke)

    # ---- Training loop ----
    patience = int(cfg["training"]["early_stopping_patience"])
    min_delta = float(cfg["training"].get("early_stopping_min_delta", 0.0))
    epochs_no_improve = 0
    LOG.info(f"Starting training: {epochs} epochs, "
             f"early-stop patience={patience}")

    try:
        for epoch in range(start_epoch, epochs):
            t0 = time.time()
            train_stats = train_one_epoch(
                model, train_loader, optimizer, cfg, device, epoch,
                per_compound_weights=per_compound_weights,
            )
            val_stats = validate(
                model, val_loader, cfg, device,
                per_compound_weights=per_compound_weights,
            )
            if scheduler is not None:
                scheduler.step()

            train_m = train_stats.means()
            val_m = val_stats.means()
            elapsed = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]

            # Early stopping & best-checkpoint use RAW (unweighted) val MAE,
            # so that "best" model selection is comparable to T17 quant_mae
            # regardless of whether weighted loss is enabled.
            improved = val_m["raw_mae"] < best_val_mae - min_delta
            if improved:
                best_val_mae = val_m["raw_mae"]
                epochs_no_improve = 0
                save_checkpoint(
                    Path(cfg["paths"]["checkpoint_best"]),
                    model, optimizer, scheduler, epoch,
                    val_metrics={"val_mae": val_m["raw_mae"],
                                 "val_weighted_mae": val_m["mae"],
                                 "best_val_mae": best_val_mae,
                                 "val_loss_total": val_m["loss_total"]},
                    cfg=cfg,
                )
            else:
                epochs_no_improve += 1

            # Always update last.pt for resume.
            save_checkpoint(
                Path(cfg["paths"]["checkpoint_last"]),
                model, optimizer, scheduler, epoch,
                val_metrics={"val_mae": val_m["raw_mae"],
                             "val_weighted_mae": val_m["mae"],
                             "best_val_mae": best_val_mae,
                             "val_loss_total": val_m["loss_total"]},
                cfg=cfg,
            )

            LOG.info(
                f"epoch {epoch:03d} | lr={current_lr:.2e} | "
                f"train raw_mae={train_m['raw_mae']:.4f} "
                f"loss={train_m['loss_total']:.4f} | "
                f"val raw_mae={val_m['raw_mae']:.4f} "
                f"loss={val_m['loss_total']:.4f} | "
                f"best={best_val_mae:.4f} {'*' if improved else ''} | "
                f"{elapsed:.1f}s"
            )

            row = {
                "epoch": epoch, "lr": current_lr,
                "train_loss_total": train_m["loss_total"],
                "train_loss_quant": train_m["loss_quant"],
                "train_loss_physics": train_m["loss_physics"],
                "train_mae": train_m["raw_mae"],
                "val_loss_total": val_m["loss_total"],
                "val_loss_quant": val_m["loss_quant"],
                "val_loss_physics": val_m["loss_physics"],
                "val_mae": val_m["raw_mae"],
                "best_val_mae": best_val_mae,
                "epoch_seconds": round(elapsed, 2),
            }
            csv_writer.writerow(row)
            csv_file.flush()
            if use_wandb:
                wandb.log(row, step=epoch)

            if epochs_no_improve >= patience:
                LOG.info(
                    f"Early stop: no improvement for {patience} epochs "
                    f"(best val_mae={best_val_mae:.4f})"
                )
                break
    finally:
        csv_file.close()
        if use_wandb:
            wandb.finish()

    LOG.info(f"Done. Best val MAE = {best_val_mae:.4f}")
    LOG.info(f"Best checkpoint:    {cfg['paths']['checkpoint_best']}")
    LOG.info(f"Last checkpoint:    {cfg['paths']['checkpoint_last']}")
    LOG.info(f"Training log CSV:   {cfg['paths']['training_log_csv']}")
    LOG.info("Next step: run T17 mid-checkpoint evaluation in Chat 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())