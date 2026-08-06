"""Stretch T29A — AAM data preparation.

One-time setup for AAM-based stretch tasks:
1. Load AAM_Data.csv (~12,956 rows, wavelength columns).
2. Detect mineral vs AA ratios.
3. Resample wavelength → wavenumber on AA grid.
4. Apply preprocessing pipeline.
5. Build 7-component labels (6 AA + 1 minerals = quartz+calcite combined).
6. Save train/val/test splits (~85/7.5/7.5 by row, since no vial structure in AAM).
7. Save processed cache for downstream T29B / T29D / T29E.

Output (under data/processed/aam/):
    spectra.pt          (N, 1024) float32, preprocessed
    labels_7d.pt        (N, 7)    float32 [Ala, Asn, Asp, Glu, His, Glc, Minerals]
    labels_full.pt      (N, 8)    float32 [...AA, Quartz, Calcite] (for diagnostics)
    has_minerals.npy    (N,) bool
    split.json          {train: [...], val: [...], test: [...]}

Engine reference:
    engine/reference_spectra_aam.npy   (7, 1024) -- AA refs + minerals ref

Usage:
    python scripts/stretch/t29a_prepare_aam.py
    python scripts/stretch/t29a_prepare_aam.py --n-limit 5000  # subsample
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _wavelength_to_wavenumber(wl_nm: np.ndarray,
                               laser_nm: float = 784.815734863281) -> np.ndarray:
    return 1e7 * (1.0 / laser_nm - 1.0 / wl_nm)


def detect_columns(df: pd.DataFrame) -> tuple[list, list, list]:
    """Detect spec / meta / ratio cols by header values."""
    spec, meta, ratio = [], [], []
    for c in df.columns:
        s = str(c)
        try:
            val = float(s)
            # Wavelength range for AA-family data
            if 700 <= val <= 1000:
                spec.append(c)
                continue
        except ValueError:
            pass
        low = s.lower().strip()
        if any(k in low for k in ["alanine", "asparagine", "aspartic",
                                    "glutamic", "histidine", "glucosamine",
                                    "quartz", "calcite"]):
            ratio.append(c)
        else:
            meta.append(c)
    return spec, meta, ratio


# Canonical order: 6 AA + Quartz + Calcite (last 2 will be combined)
CANON_8 = ["Alanine", "Asparagine", "Aspartic Acid", "Glutamic Acid",
           "Histidine", "Glucosamine", "Quartz", "Calcite"]


def reorder_ratios(df_ratio: pd.DataFrame) -> np.ndarray:
    """Match user column names to CANON_8 order. Returns (N, 8) float array."""
    aliases = {
        "Alanine":       ["alanine", "ala", "dl-alanine", "l-alanine"],
        "Asparagine":    ["asparagine", "asn", "l-asparagine"],
        "Aspartic Acid": ["aspartic acid", "aspartic-acid", "asp", "l-aspartic"],
        "Glutamic Acid": ["glutamic acid", "glutamic-acid", "glu", "l-glutamic"],
        "Histidine":     ["histidine", "his", "l-histidine"],
        "Glucosamine":   ["glucosamine", "glc", "d-glucosamine"],
        "Quartz":        ["quartz", "qtz", "sio2"],
        "Calcite":       ["calcite", "cal", "caco3"],
    }
    n = len(df_ratio)
    out = np.zeros((n, 8), dtype=np.float32)
    df_lower = {str(c).lower().strip(): c for c in df_ratio.columns}
    for i, canon in enumerate(CANON_8):
        found = False
        for alias in aliases[canon]:
            for header_low, original_c in df_lower.items():
                if alias in header_low:
                    out[:, i] = df_ratio[original_c].values
                    found = True
                    break
            if found:
                break
        if not found:
            print(f"  [warn] no column for {canon} -- leaving as 0")
    # Check sum to 1
    s = out.sum(axis=1)
    bad = np.abs(s - 1.0) > 1e-2
    if bad.any():
        print(f"  [warn] {bad.sum()}/{n} rows have ratio sum ≠ 1 "
              f"(min={s.min():.4f}, max={s.max():.4f}) -- renormalising")
        out = out / np.clip(s[:, None], 1e-6, None)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aam-csv", default="data/raw/AAM_Data.csv")
    p.add_argument("--n-limit", type=int, default=None,
                   help="Subsample to N rows (None = use all 12,956)")
    p.add_argument("--mineral-threshold", type=float, default=0.05)
    p.add_argument("--train-frac", type=float, default=0.85)
    p.add_argument("--val-frac",   type=float, default=0.075)
    p.add_argument("--test-frac",  type=float, default=0.075)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="data/processed/aam")
    p.add_argument("--mineral-ref-path", default="data/raw/50_50_quartz_calcite.csv",
                   help="Path to 50/50 mineral mix CSV")
    p.add_argument("--engine-ref-out", default="engine/reference_spectra_aam.npy")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: load AAM CSV ----
    print(f"\n[T29A] Loading {args.aam_csv} (this may take 10-30 s) ...")
    df = pd.read_csv(args.aam_csv)
    print(f"  shape: {df.shape}")

    spec_cols, meta_cols, ratio_cols = detect_columns(df)
    print(f"  spec={len(spec_cols)}, meta={len(meta_cols)}, ratio={len(ratio_cols)}")
    if len(spec_cols) != 1024:
        print(f"  ERROR: expected 1024 spec cols, got {len(spec_cols)}"); sys.exit(1)
    if len(ratio_cols) < 6:
        print(f"  ERROR: expected ≥6 ratio cols, got {len(ratio_cols)}"); sys.exit(1)

    # ---- Step 2: optional subsample ----
    if args.n_limit and args.n_limit < len(df):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(df), size=args.n_limit, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
        print(f"  subsampled to {len(df)} rows")

    # ---- Step 3: extract arrays ----
    X_raw = df[spec_cols].values.astype(np.float32)
    Y_full = reorder_ratios(df[ratio_cols])
    wl_nm = np.array([float(c) for c in spec_cols], dtype=np.float64)
    wn_aam = _wavelength_to_wavenumber(wl_nm)
    if wn_aam[0] > wn_aam[-1]:
        wn_aam_asc = wn_aam[::-1]
        X_raw_asc = X_raw[:, ::-1]
    else:
        wn_aam_asc = wn_aam
        X_raw_asc = X_raw

    # ---- Step 4: resample to AA grid ----
    target_wn = np.load("data/processed/wavenumbers.npy")
    target = np.asarray(target_wn, dtype=np.float64)
    if target[0] > target[-1]:
        target_asc = target[::-1]
        flipped = True
    else:
        target_asc = target
        flipped = False

    print(f"  resampling {len(X_raw)} spectra to AA grid ...")
    Xr = np.empty((len(X_raw), 1024), dtype=np.float32)
    for i in range(len(X_raw)):
        v = np.interp(target_asc, wn_aam_asc, X_raw_asc[i], left=0.0, right=0.0)
        if flipped:
            v = v[::-1]
        Xr[i] = v
        if (i + 1) % 2000 == 0:
            print(f"    {i+1}/{len(X_raw)}")

    # ---- Step 5: preprocess ----
    print("  applying preprocessing (cosmic + AsLS + SG + SNV) ...")
    from src.data.preprocess import preprocess_batch
    Xp = preprocess_batch(Xr).astype(np.float32)
    Xp = np.nan_to_num(Xp, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- Step 6: build 7-component labels ----
    # 6 AA + minerals (quartz + calcite combined)
    Y7 = np.zeros((len(Y_full), 7), dtype=np.float32)
    Y7[:, :6] = Y_full[:, :6]
    Y7[:, 6]  = Y_full[:, 6] + Y_full[:, 7]  # minerals total
    minerals_total = Y_full[:, 6] + Y_full[:, 7]
    has_minerals = minerals_total > args.mineral_threshold
    print(f"  has_minerals > {args.mineral_threshold}: "
          f"{has_minerals.sum()} / {len(has_minerals)}")

    # ---- Step 7: splits ----
    n = len(Xp)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_tr = int(n * args.train_frac)
    n_va = int(n * args.val_frac)
    train_idx = perm[:n_tr].tolist()
    val_idx   = perm[n_tr:n_tr + n_va].tolist()
    test_idx  = perm[n_tr + n_va:].tolist()
    print(f"  split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    print(f"  train has_minerals: {has_minerals[train_idx].sum()}/{len(train_idx)}")
    print(f"  test  has_minerals: {has_minerals[test_idx].sum()}/{len(test_idx)}")

    # ---- Step 8: save cache ----
    torch.save(torch.from_numpy(Xp), out_dir / "spectra.pt")
    torch.save(torch.from_numpy(Y7), out_dir / "labels_7d.pt")
    torch.save(torch.from_numpy(Y_full), out_dir / "labels_full.pt")
    np.save(out_dir / "has_minerals.npy", has_minerals)
    split = {"train": train_idx, "val": val_idx, "test": test_idx,
             "scheme": "random row-level (AAM)",
             "seed": args.seed,
             "fractions": [args.train_frac, args.val_frac, args.test_frac]}
    (out_dir / "split.json").write_text(json.dumps(split, indent=2))
    print(f"  saved cache → {out_dir}")

    # ---- Step 9: build 7-row reference_spectra (AA + minerals) ----
    print("\n[T29A] Building 7-component reference (6 AA + 1 minerals) ...")
    aa_refs = np.load("engine/reference_spectra.npy")  # (6, 1024) preprocessed
    if aa_refs.shape != (6, 1024):
        print(f"  WARN: AA refs shape {aa_refs.shape}"); 

    # Mineral ref from 50/50 mix CSV
    mineral_csv = Path(args.mineral_ref_path)
    if not mineral_csv.exists():
        print(f"  WARN: {mineral_csv} not found -- using mean of mineral-rich AAM rows")
        mineral_rich_idx = np.where(has_minerals)[0]
        mineral_ref = Xp[mineral_rich_idx].mean(axis=0).astype(np.float32)
    else:
        # Parse ENLIGHTEN-style mineral CSV (header rows + ~1024 pixels x ~10 spectra)
        # Best-effort: read CSV, find numeric block, take mean
        print(f"  parsing {mineral_csv} ...")
        try:
            mdf = pd.read_csv(mineral_csv, header=None, skip_blank_lines=True,
                               on_bad_lines='skip')
            # Find numeric columns (skip metadata rows at top)
            numeric_block = mdf.apply(pd.to_numeric, errors='coerce').dropna(how='all')
            arr = numeric_block.values.astype(np.float64)
            # Try to find a (1024, k) or (k, 1024) block
            if arr.shape[0] == 1024:
                spectra_min = arr.T  # (k, 1024)
            elif arr.shape[1] == 1024:
                spectra_min = arr
            else:
                print(f"    [warn] expected 1024-dim, got {arr.shape}; falling back to AAM mineral mean")
                mineral_rich_idx = np.where(has_minerals)[0]
                spectra_min = Xp[mineral_rich_idx]
            mean_min = np.nanmean(spectra_min, axis=0).astype(np.float32)
            # Need to resample if from different wavenumber grid -- best effort assume same length
            if len(mean_min) != 1024:
                print(f"    [warn] mineral ref len {len(mean_min)} ≠ 1024; using AAM fallback")
                mineral_rich_idx = np.where(has_minerals)[0]
                mean_min = Xp[mineral_rich_idx].mean(axis=0).astype(np.float32)
            # Apply preprocessing
            from src.data.preprocess import preprocess_pipeline
            mineral_ref = preprocess_pipeline(mean_min).astype(np.float32)
        except Exception as e:
            print(f"    [warn] failed to parse: {e}. Falling back to AAM mineral mean.")
            mineral_rich_idx = np.where(has_minerals)[0]
            mineral_ref = Xp[mineral_rich_idx].mean(axis=0).astype(np.float32)

    refs7 = np.concatenate([aa_refs, mineral_ref[None, :]], axis=0).astype(np.float32)
    refs7 = np.nan_to_num(refs7, nan=0.0, posinf=0.0, neginf=0.0)
    Path(args.engine_ref_out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.engine_ref_out, refs7)
    print(f"  saved 7-component ref → {args.engine_ref_out}, shape={refs7.shape}")

    print("\n[T29A done]")
    print(f"Next: run T29B (zero-shot test) then T29D (retrain) then T29E (post-train test)")


if __name__ == "__main__":
    main()
