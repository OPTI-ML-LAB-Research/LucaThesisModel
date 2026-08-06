"""One-time data preparation script.

Steps:
    1. Load `data/raw/{AA_Data.csv | data.csv}` via
       `src.data.dataloader.load_raw_csv`.
    2. Build the requested split scheme(s) — calls the Phase-A T07 API
       (`split_A_vial_level` and/or `split_A_sample_level`).
    3. Persist each split to `data/splits/split_{name}.json` using
       contract names that the rest of the pipeline (notably Chat 3
       training) expects.
    4. Apply the classical preprocessing pipeline to all spectra
       (cosmic → AsLS → SG → SNV).
    5. Cache preprocessed tensors to `data/processed/spectra_full.pt`
       (single file with the full preprocessed array; downstream code
       slices by split indices).

Usage
-----
    # Defaults: build BOTH scheme A and A', then preprocess.
    python scripts/prepare_data.py

    # Single scheme:
    python scripts/prepare_data.py --scheme A
    python scripts/prepare_data.py --scheme A_prime

    # Skip preprocessing cache (only build splits):
    python scripts/prepare_data.py --no-preprocess

Output files
------------
    data/splits/split_A_composition_ood.json    — Scheme A (vial-level OOD)
    data/splits/split_A_prime_sample_level.json — Scheme A' (random)
    data/processed/spectra_full.pt              — preprocessed (N, P) torch.float32
    data/processed/labels.pt                    — (N, C) torch.float32
    data/processed/wavenumbers.npy              — (P,) float64 (Raman shift cm-1)
    data/processed/vial_ids.npy                 — (N,) string
    data/processed/preprocess_meta.json         — pipeline hyperparameters used

Notes
-----
* Scheme B (component-level holdout) was removed in Phase A. Every
  amino acid appears in 49/54 vials, so train pool would collapse to
  ~50 rows — see GROUNDWORK_SUMMARY §3 issue #1 and CHAT2_PHASE_A
  decision C6.
* Scheme C (mix_method holdout) is deferred to Phase B/C.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Make `src` importable when running this script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.data.dataloader import load_raw_csv  # noqa: E402
from src.data.splits import (  # noqa: E402
    split_A_vial_level,
    split_A_sample_level,
    save_split,
)


log = logging.getLogger("prepare_data")


# Output filenames are part of the Chat-3 contract — DO NOT rename
# without coordinating with `src/training/train.py`.
SCHEME_OUTPUT_NAME = {
    "A":       "split_A_composition_ood.json",
    "A_prime": "split_A_prime_sample_level.json",
}


def _save_torch_tensor(arr: np.ndarray, path: Path) -> None:
    """Save a numpy array as a torch tensor (lazy import to avoid hard dep)."""
    import torch
    t = torch.from_numpy(arr.astype(np.float32, copy=False))
    torch.save(t, path)


def _resolve_csv_path(defaults: dict) -> Path:
    """Resolve the raw-CSV path, preferring AA_Data.csv if present.

    Per PROJECT_REVISION_v2 §1.1, `AA_Data.csv` supersedes `data.csv`.
    Some configs may still reference the old path; fall back gracefully.
    """
    configured = PROJECT_ROOT / defaults["paths"]["data_raw_csv"]

    # Prefer configured path if it exists.
    if configured.exists():
        return configured

    # Otherwise look for AA_Data.csv next to it (likely case after data
    # reorganisation but configs/ wasn't updated).
    candidate = configured.parent / "AA_Data.csv"
    if candidate.exists():
        log.warning(
            "  configured path %s not found; using %s instead",
            configured.relative_to(PROJECT_ROOT),
            candidate.relative_to(PROJECT_ROOT),
        )
        return candidate

    # Fall back to data.csv if user kept legacy name.
    candidate2 = configured.parent / "data.csv"
    if candidate2.exists():
        log.warning(
            "  configured path %s not found; falling back to legacy %s",
            configured.relative_to(PROJECT_ROOT),
            candidate2.relative_to(PROJECT_ROOT),
        )
        return candidate2

    raise FileNotFoundError(
        f"Could not find raw CSV. Tried:\n"
        f"  {configured}\n"
        f"  {configured.parent / 'AA_Data.csv'}\n"
        f"  {configured.parent / 'data.csv'}\n"
        f"Update configs/default.yaml -> paths.data_raw_csv."
    )


def _build_splits(table, scheme: str, seed: int) -> dict:
    """Build the requested split(s). Returns dict {scheme_key: SplitIndices}."""
    out: dict = {}
    labels = np.asarray(table.labels, dtype=np.float32)
    vial_ids = list(table.vial_ids)

    if scheme in ("A", "all"):
        log.info("  building Scheme A (vial-level, composition-OOD)...")
        sp = split_A_vial_level(
            vial_ids=vial_ids,
            labels=labels,
            seed=seed,
            include_pure_in_train=True,
        )
        out["A"] = sp
        log.info(
            "    train=%d  val=%d  test=%d  (n_total=%d)",
            len(sp.train), len(sp.val), len(sp.test), sp.n_total,
        )

    if scheme in ("A_prime", "all"):
        log.info("  building Scheme A' (sample-level, random 60/20/20)...")
        sp = split_A_sample_level(
            n_rows=len(vial_ids),
            labels=labels,
            seed=seed,
        )
        out["A_prime"] = sp
        log.info(
            "    train=%d  val=%d  test=%d  (n_total=%d)",
            len(sp.train), len(sp.val), len(sp.test), sp.n_total,
        )

    if not out:
        raise ValueError(
            f"Unknown scheme '{scheme}'. Use one of: A, A_prime, all."
        )
    return out


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Prepare splits and preprocessed cache.")
    parser.add_argument("--default-config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "configs/data_config.yaml"))
    parser.add_argument(
        "--scheme",
        default="all",
        choices=["A", "A_prime", "all"],
        help="Which split scheme(s) to build. Default: all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for splits. Falls back to data_config.yaml -> split.seed (default 42).",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip the preprocessing cache step (only build splits).",
    )
    args = parser.parse_args()

    with open(args.default_config, encoding='utf-8') as f:
        defaults = yaml.safe_load(f)
    with open(args.data_config, encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f) or {}

    # Seed precedence: CLI > data_config.yaml > 42
    seed = (
        args.seed
        if args.seed is not None
        else int((data_cfg.get("split") or {}).get("seed", 42))
    )

    # -- Step 1: load raw CSV --
    log.info("Step 1: load CSV")
    csv_path = _resolve_csv_path(defaults)
    log.info(f"  source: {csv_path.relative_to(PROJECT_ROOT)}")

    table = load_raw_csv(
        csv_path=csv_path,
        compound_full_names=defaults["compounds"]["full_names"],
        laser_wl_nm=defaults["wavenumber"]["laser_wavelength_nm"],
        expected_num_points=defaults["wavenumber"]["expected_num_points"],
    )
    log.info(
        f"  loaded: {table.num_samples} spectra x {table.num_points} points; "
        f"{table.num_compounds} compounds"
    )

    # -- Step 2: build splits --
    log.info(f"Step 2: build split(s) - scheme='{args.scheme}', seed={seed}")
    splits = _build_splits(table, scheme=args.scheme, seed=seed)

    # -- Step 3: save splits --
    log.info("Step 3: save splits to data/splits/")
    splits_dir = PROJECT_ROOT / defaults["paths"]["data_splits_dir"]
    splits_dir.mkdir(parents=True, exist_ok=True)
    for key, sp in splits.items():
        out_name = SCHEME_OUTPUT_NAME[key]
        out_path = splits_dir / out_name
        save_split(sp, out_path)
        log.info(f"  saved {out_path.relative_to(PROJECT_ROOT)}")

    # -- Step 4: classical preprocessing (full table) --
    if args.no_preprocess:
        log.info("Step 4 SKIPPED - flag --no-preprocess.")
        return 0

    log.info(
        f"Step 4: apply classical preprocessing to all {table.num_samples} "
        "spectra (cosmic -> AsLS -> SG -> SNV)"
    )
    from src.data.preprocess import preprocess_batch

    pp_cfg = data_cfg.get("preprocessing", {}) or {}
    pp_kwargs = dict(
        cosmic_threshold=float(pp_cfg.get("cosmic_threshold", 5.0)),
        asls_lam=float(pp_cfg.get("asls_lam", 1.0e5)),
        asls_p=float(pp_cfg.get("asls_p", 0.01)),
        asls_max_iter=int(pp_cfg.get("asls_max_iter", 30)),
        savgol_window=int(pp_cfg.get("savgol_window", 11)),
        savgol_polyorder=int(pp_cfg.get("savgol_polyorder", 3)),
        snv_eps=float(pp_cfg.get("snv_eps", 1.0e-8)),
    )
    log.info(f"  pipeline kwargs: {pp_kwargs}")

    t0 = time.time()
    spectra_pp = preprocess_batch(table.spectra, is_preprocessed=False, **pp_kwargs)
    elapsed = time.time() - t0
    log.info(
        f"  done in {elapsed:.1f}s  "
        f"(~{elapsed / table.num_samples * 1000:.1f} ms/spectrum)"
    )

    # -- Step 5: cache to data/processed/ --
    log.info("Step 5: cache preprocessed tensors to data/processed/")
    proc_dir = PROJECT_ROOT / defaults["paths"]["data_processed_dir"]
    proc_dir.mkdir(parents=True, exist_ok=True)

    spec_path = proc_dir / "spectra_full.pt"
    label_path = proc_dir / "labels.pt"
    wn_path = proc_dir / "wavenumbers.npy"
    meta_path = proc_dir / "preprocess_meta.json"
    vial_path = proc_dir / "vial_ids.npy"

    try:
        _save_torch_tensor(spectra_pp, spec_path)
        _save_torch_tensor(np.asarray(table.labels), label_path)
    except ImportError:
        log.warning("  torch not installed; saving as .npy instead.")
        spec_path = spec_path.with_suffix(".npy")
        label_path = label_path.with_suffix(".npy")
        np.save(spec_path, spectra_pp.astype(np.float32, copy=False))
        np.save(
            label_path,
            np.asarray(table.labels).astype(np.float32, copy=False),
        )

    np.save(wn_path, np.asarray(table.wavenumbers).astype(np.float64, copy=False))
    np.save(vial_path, np.asarray(table.vial_ids))

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "pipeline_kwargs": pp_kwargs,
            "num_spectra": int(table.num_samples),
            "num_points": int(table.num_points),
            "num_compounds": int(table.num_compounds),
            "preprocessing_seconds": round(elapsed, 2),
            "compound_names": list(table.compound_names),
            "source_csv": str(csv_path.relative_to(PROJECT_ROOT)),
            "splits_built": list(splits.keys()),
            "seed": seed,
        }, f, indent=2)

    for p in [spec_path, label_path, wn_path, vial_path, meta_path]:
        if p.exists():
            log.info(
                f"  saved {p.relative_to(PROJECT_ROOT)} "
                f"({p.stat().st_size / 1024:.0f} KB)"
            )

    log.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())