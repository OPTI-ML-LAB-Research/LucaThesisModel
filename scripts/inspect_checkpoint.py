"""Inspect a PyTorch checkpoint without needing torch installed.

Reports:
* Top-level keys (model / optimizer / scheduler / epoch / val_metrics / config)
* Full config dict (JSON)
* State-dict tensor shapes (grouped by submodule)
* Saved pure_ref and scale values from reconstruction module
* Discrepancy warnings (legacy data.csv path, scale drift, etc.)

Usage::

    python scripts/inspect_checkpoint.py
    python scripts/inspect_checkpoint.py --checkpoint path/to/best.pt --dump-arrays
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._pt_inspect import load_checkpoint, TensorRecord


def _group_keys(sd: dict) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for k in sd.keys():
        # Group by first two dot-segments (backbone.stem, backbone.stages.0, etc.)
        parts = k.split(".")
        prefix = ".".join(parts[:3]) if k.startswith("backbone.stages") else ".".join(parts[:2])
        groups.setdefault(prefix, []).append(k)
    return groups


def _fmt_int(n: int) -> str:
    return f"{n:>12,}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Inspect a Raman-AI .pt checkpoint without torch.")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    p.add_argument("--dump-arrays", action="store_true",
                   help="Save pure_ref and scale to .npy files next to the checkpoint.")
    args = p.parse_args(argv)

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    obj, z, _ = load_checkpoint(args.checkpoint)

    print("=" * 70)
    print(f" CHECKPOINT INSPECTION  :  {args.checkpoint}")
    print("=" * 70)
    print()

    # ---- Header ----
    print("HEADER")
    print("-" * 70)
    print(f"  epoch         : {obj.get('epoch')}")
    vm = obj.get("val_metrics", {})
    print(f"  val_metrics   : val_mae={vm.get('val_mae'):.6f}, "
          f"best_val_mae={vm.get('best_val_mae'):.6f}, "
          f"val_loss_total={vm.get('val_loss_total'):.6f}")
    print(f"  top-level keys: {list(obj.keys())}")
    print()

    # ---- Config ----
    cfg = obj.get("config", {})
    print("CONFIG  (full)")
    print("-" * 70)
    print(json.dumps(cfg, indent=2, default=str))
    print()

    # ---- Discrepancy / warning checks ----
    warnings: list[str] = []

    raw_csv = cfg.get("paths", {}).get("data_raw_csv", "")
    if raw_csv.endswith("data.csv") and "AA_Data" not in raw_csv:
        warnings.append(
            f"⚠ Config points to legacy '{raw_csv}'. PROJECT_REVISION v2 §1.1 "
            f"says the primary dataset is now 'AA_Data.csv'. The checkpoint was "
            "trained on the OLD data, so test-set indices in the new "
            "split_A_composition_ood.json may NOT match the spectra the model "
            "expects. Verify spectra_full.pt was rebuilt with the same source "
            "before running T17."
        )

    beta_phys = cfg.get("loss", {}).get("beta_phys", None)
    if beta_phys is not None and beta_phys >= 0.5:
        warnings.append(
            f"ℹ beta_phys = {beta_phys} (project default). Combined with the "
            "observed train physics/quant ratio of ~5×, this is the suspected "
            "cause of the underfitting plateau. Documented in CHAT3 handover P1-4/P2-7."
        )

    # ---- State dict summary ----
    sd = obj.get("model", {})
    if not isinstance(sd, dict):
        print(f"WARN: model is {type(sd)}, not a dict")
        z.close()
        return 0

    print("STATE-DICT GROUPS")
    print("-" * 70)
    grand_total = 0
    for prefix, keys in _group_keys(sd).items():
        sub_total = 0
        for k in keys:
            v = sd[k]
            if isinstance(v, TensorRecord):
                sub_total += int(np.prod(v.size))
        grand_total += sub_total
        print(f"  {prefix:<35s} {len(keys):>4d} tensors {_fmt_int(sub_total)} params")
    print(f"  {'TOTAL':<35s} {len(sd):>4d} tensors {_fmt_int(grand_total)} params")
    print()

    # ---- Reconstruction module specifics ----
    pure_ref_rec = sd.get("reconstruction.pure_ref")
    scale_rec = sd.get("reconstruction.scale")
    if pure_ref_rec is not None and scale_rec is not None:
        pure_ref = pure_ref_rec.materialize()
        scale = scale_rec.materialize()
        print("RECONSTRUCTION MODULE")
        print("-" * 70)
        print(f"  pure_ref shape : {pure_ref.shape}, dtype {pure_ref.dtype}")
        print(f"  pure_ref per-compound mean / std:")
        compound_names = cfg.get("compounds", {}).get("full_names", [f"#{i}" for i in range(6)])
        for i, name in enumerate(compound_names):
            print(f"    {i}: {name:<14s} mean={pure_ref[i].mean():+.4f} "
                  f"std={pure_ref[i].std():.4f} "
                  f"range=[{pure_ref[i].min():+.3f}, {pure_ref[i].max():+.3f}]")
        snv_ok = (np.allclose(pure_ref.mean(axis=1), 0, atol=0.05)
                  and np.allclose(pure_ref.std(axis=1), 1, atol=0.05))
        print(f"  approximately SNV-normalized? {'YES' if snv_ok else 'NO'}")
        print()
        print(f"  scale (learned) : {scale}")
        drift = scale - 1.0
        print(f"  drift from init=1.0 : {drift}")
        if np.max(np.abs(drift)) > 0.3:
            for i, name in enumerate(compound_names):
                if abs(drift[i]) > 0.3:
                    warnings.append(
                        f"ℹ scale[{i}] = {scale[i]:.3f} for {name} ({drift[i]:+.3f} from "
                        f"init). Large drift = pure_ref intensity differs strongly from "
                        f"what the model needed → possibly a reference-quality issue, or "
                        f"compensation working correctly."
                    )

        if args.dump_arrays:
            out_dir = args.checkpoint.parent
            np.save(out_dir / "_inspected_pure_ref.npy", pure_ref)
            np.save(out_dir / "_inspected_scale.npy", scale)
            print(f"  dumped pure_ref → {out_dir/'_inspected_pure_ref.npy'}")
            print(f"  dumped scale    → {out_dir/'_inspected_scale.npy'}")
        print()

    # ---- Quantification head sanity ----
    fc2_w_rec = sd.get("quantification_head.fc2.weight")
    fc2_b_rec = sd.get("quantification_head.fc2.bias")
    if fc2_w_rec is not None and fc2_b_rec is not None:
        fc2_w = fc2_w_rec.materialize()
        fc2_b = fc2_b_rec.materialize()
        print("QUANTIFICATION HEAD (final layer)")
        print("-" * 70)
        print(f"  fc2.weight shape: {fc2_w.shape}, std={fc2_w.std():.4f}, "
              f"norm={np.linalg.norm(fc2_w):.3f}")
        print(f"  fc2.bias (pre-softmax logit bias per compound):")
        for i, name in enumerate(compound_names):
            print(f"    {i}: {name:<14s} bias={fc2_b[i]:+.5f}")
        bias_range = float(fc2_b.max() - fc2_b.min())
        if bias_range < 0.1:
            print(f"  Bias spread {bias_range:.4f} — fairly balanced (good).")
        else:
            print(f"  Bias spread {bias_range:.4f} — moderately tilted toward "
                  f"{compound_names[int(fc2_b.argmax())]}.")
        print()

    # ---- Warnings ----
    if warnings:
        print("FLAGS / WARNINGS")
        print("-" * 70)
        for w in warnings:
            print(f"  {w}")
            print()

    z.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
