"""PyTorch Dataset and DataLoader for the Raman amino-acid dataset.

Module purpose
--------------
Load `data/raw/data.csv` (4378 × 1031), parse spectral / label / vial-id columns,
convert wavelength → wavenumber, and serve PyTorch tensors of shape:

    spectrum : (1, num_points)         e.g. (1, 1024)
    label    : (num_compounds,)        simplex over 6 amino-acid fractions
    vial_id  : str                     for OOD / split bookkeeping

This module does NOT apply the classical preprocessing pipeline
(AsLS + cosmic + SG + SNV) — that lives in `src.data.preprocess` and is
applied either offline (recommended) by `scripts/prepare_data.py` or
on-the-fly when `apply_on_the_fly=True`.

Augmentation (random shift / intensity / noise) is delegated to
`src.data.augmentation` and only enabled for the *training* split.

Design notes
------------
* CSV column layout (verified on real & mock data):
    - first  N spectral columns: wavelength (nm) headers as floats, e.g. "801.62"
    - column 'vial #'            : sample identifier (str)
    - last 6 columns             : compound mass-fractions, sum to 1.0
* Extra non-numeric metadata columns (e.g. 'file_name', 'Repitation',
  'mix_method' in AA_Data.csv — see PROJECT_REVISION_v2 §1.1) are silently
  dropped: any column whose header does not parse as a float and is not
  the vial / compound column is ignored.
* Pure samples (`L-histidine`, `D-glucosamine`, ...) are detected via
  `is_pure_vial()`. They are kept in the split iff
  `data_config.split.include_pure_in_split == True`.
* Wavelength → wavenumber conversion (cm⁻¹):
    nu = (1 / laser_wl_nm − 1 / wavelength_nm) × 1e7
* The dataset stores spectra as **float32** numpy arrays in RAM
  (~4378 × 1024 × 4 B ≈ 17 MB — fits easily).

Author: Day-1 sprint (T03)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
#  Constants & helpers
# ───────────────────────────────────────────────────────────────────────────

VIAL_COL = "vial #"


def _normalize_compound_name(name: str) -> str:
    """Strip everything except letters and lowercase. 'Aspartic Acid' → 'asparticacid'."""
    return re.sub(r"[^a-z]", "", str(name).lower())


def is_pure_vial(vial_id: str, compound_full_names: Sequence[str]) -> bool:
    """Return True if `vial_id` matches any pure-compound naming pattern.

    Real data uses formats like 'L-histidine', 'D-glucosamine', 'L-alanine'.
    We strip stereoisomer prefixes ('L-', 'D-', 'DL-') and non-letter chars,
    then compare to the canonical compound list.

    Examples
    --------
    >>> is_pure_vial('L-histidine', ['Histidine', 'Alanine'])
    True
    >>> is_pure_vial('a01', ['Histidine', 'Alanine'])
    False
    """
    s = str(vial_id)
    # Drop a leading L-/D-/DL- (case-insensitive) if present
    s = re.sub(r"^(d|l|dl)\s*[-]\s*", "", s, flags=re.IGNORECASE)
    cleaned = _normalize_compound_name(s)
    targets = {_normalize_compound_name(n) for n in compound_full_names}
    return cleaned in targets


def wavelength_nm_to_wavenumber_cm_inv(
    wavelength_nm: np.ndarray,
    laser_wl_nm: float,
) -> np.ndarray:
    """Convert sample wavelength(s) (nm) to Raman shift (cm⁻¹)."""
    return (1.0 / laser_wl_nm - 1.0 / wavelength_nm) * 1.0e7


# ───────────────────────────────────────────────────────────────────────────
#  Container for a parsed dataset (raw, pre-split)
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class RawSpectraTable:
    """Holds the entire parsed CSV in numpy form, with metadata.

    Attributes
    ----------
    spectra        : (N, P) float32        — raw intensities
    labels         : (N, C) float32        — simplex of compound fractions
    vial_ids       : (N,)   object str     — sample identifiers
    wavelengths_nm : (P,)   float64        — original column headers
    wavenumbers    : (P,)   float64        — converted Raman shift (cm⁻¹)
    compound_names : list[str]             — canonical order, len = C
    """
    spectra: np.ndarray
    labels: np.ndarray
    vial_ids: np.ndarray
    wavelengths_nm: np.ndarray
    wavenumbers: np.ndarray
    compound_names: list[str] = field(default_factory=list)

    @property
    def num_samples(self) -> int:
        return self.spectra.shape[0]

    @property
    def num_points(self) -> int:
        return self.spectra.shape[1]

    @property
    def num_compounds(self) -> int:
        return self.labels.shape[1]

    @property
    def unique_vials(self) -> np.ndarray:
        return np.unique(self.vial_ids)

    def pure_mask(self) -> np.ndarray:
        """Boolean mask: True for rows whose vial is a pure-compound sample."""
        return np.array(
            [is_pure_vial(v, self.compound_names) for v in self.vial_ids],
            dtype=bool,
        )


# ───────────────────────────────────────────────────────────────────────────
#  CSV loader
# ───────────────────────────────────────────────────────────────────────────

def load_raw_csv(
    csv_path: str | Path,
    compound_full_names: Sequence[str],
    laser_wl_nm: float,
    expected_num_points: int | None = None,
) -> RawSpectraTable:
    """Parse the Raman CSV into a `RawSpectraTable`.

    Parameters
    ----------
    csv_path
        Path to data/raw/data.csv.
    compound_full_names
        Canonical compound order (e.g. ['Alanine', 'Asparagine', ...]).
        Must exactly match the last 6 column headers in the CSV.
    laser_wl_nm
        Excitation wavelength used to convert column headers to cm⁻¹.
    expected_num_points
        If given, log a warning when the detected spectral length differs.

    Returns
    -------
    RawSpectraTable
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    log.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # ── 1. Verify mandatory non-spectral columns ──
    if VIAL_COL not in df.columns:
        raise ValueError(f"Missing '{VIAL_COL}' column in CSV.")
    missing = [c for c in compound_full_names if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing compound columns in CSV: {missing}. "
            f"Expected last columns to be {list(compound_full_names)}."
        )

    # ── 2. Identify spectral columns: everything that's NOT vial / compound ──
    # AND has a numeric header (wavelength in nm). The numeric-header filter
    # excludes extra metadata columns like 'file_name', 'Repitation',
    # 'mix_method' that AA_Data.csv adds vs. legacy data.csv (see
    # PROJECT_REVISION_v2 §1.1).
    non_spec = {VIAL_COL, *compound_full_names}
    candidate_cols = [c for c in df.columns if c not in non_spec]

    def _is_numeric_header(h) -> bool:
        try:
            float(h)
            return True
        except (ValueError, TypeError):
            return False

    spec_cols = [c for c in candidate_cols if _is_numeric_header(c)]
    skipped_meta_cols = [c for c in candidate_cols if not _is_numeric_header(c)]
    if skipped_meta_cols:
        log.info(
            f"  ignoring {len(skipped_meta_cols)} non-numeric metadata "
            f"column(s): {skipped_meta_cols}"
        )

    if not spec_cols:
        raise ValueError(
            "No numeric spectral columns detected after filtering "
            f"out {{{VIAL_COL}}}, the compound columns, and any metadata "
            f"columns. Candidates were: {candidate_cols[:10]}..."
        )

    # All remaining headers are guaranteed numeric here.
    wavelengths_nm = np.array([float(c) for c in spec_cols], dtype=np.float64)

    if expected_num_points is not None and len(spec_cols) != expected_num_points:
        log.warning(
            f"Detected {len(spec_cols)} spectral columns but config expected "
            f"{expected_num_points}. Continuing with the detected value."
        )

    # ── 3. Slice into arrays ──
    spectra = df[spec_cols].to_numpy(dtype=np.float32)
    labels = df[list(compound_full_names)].to_numpy(dtype=np.float32)
    vial_ids = df[VIAL_COL].to_numpy()  # dtype object (strings)

    # ── 4. Sanity checks ──
    if np.isnan(spectra).any():
        n_nan = int(np.isnan(spectra).sum())
        raise ValueError(f"NaN values in spectra ({n_nan} cells).")
    if np.isnan(labels).any():
        n_nan = int(np.isnan(labels).sum())
        raise ValueError(f"NaN values in labels ({n_nan} cells).")
    label_sums = labels.sum(axis=1)
    if not np.allclose(label_sums, 1.0, atol=1e-3):
        bad = int((np.abs(label_sums - 1.0) > 1e-3).sum())
        log.warning(f"{bad} rows have label sum != 1.0 (tol 1e-3).")

    # ── 5. Convert wavelength → wavenumber ──
    wavenumbers = wavelength_nm_to_wavenumber_cm_inv(wavelengths_nm, laser_wl_nm)

    log.info(
        f"  shape={spectra.shape}, "
        f"wavelength range {wavelengths_nm.min():.2f}–{wavelengths_nm.max():.2f} nm, "
        f"wavenumber range {wavenumbers.min():.1f}–{wavenumbers.max():.1f} cm⁻¹"
    )
    log.info(f"  unique vials = {len(np.unique(vial_ids))}")

    return RawSpectraTable(
        spectra=spectra,
        labels=labels,
        vial_ids=vial_ids,
        wavelengths_nm=wavelengths_nm,
        wavenumbers=wavenumbers,
        compound_names=list(compound_full_names),
    )


# ───────────────────────────────────────────────────────────────────────────
#  PyTorch Dataset
# ───────────────────────────────────────────────────────────────────────────

class RamanDataset(Dataset):
    """A PyTorch Dataset over a fixed subset of spectra.

    Yields per-item dicts:
        {
          'spectrum': torch.float32 of shape (1, P),
          'label'   : torch.float32 of shape (C,),
          'vial_id' : str,
          'index'   : int  (row index within the *full* table; useful for debug)
        }

    Parameters
    ----------
    table
        The full parsed table.
    indices
        Row indices (into `table.spectra`) that this dataset will iterate over.
    transform
        Optional callable applied to each spectrum AFTER conversion to torch.
        Signature: (spectrum_tensor: (1, P)) -> (1, P).
        Used for on-the-fly augmentation during training.
    preprocess
        Optional callable applied to each spectrum BEFORE conversion to torch.
        Signature: (spectrum_np: (P,)) -> (P,).
        If None, the raw spectrum is passed through unchanged.
    """

    def __init__(
        self,
        table: RawSpectraTable,
        indices: Sequence[int] | np.ndarray,
        transform=None,
        preprocess=None,
    ):
        self._table = table
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
        self.preprocess = preprocess

        # Validation
        if self.indices.size == 0:
            raise ValueError("RamanDataset received an empty index list.")
        bad = (self.indices < 0) | (self.indices >= table.num_samples)
        if bad.any():
            raise IndexError(f"{int(bad.sum())} indices out of bounds for table.")

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, i: int) -> dict:
        if i < 0:
            i += len(self)
        if i < 0 or i >= len(self):
            raise IndexError(i)
        row_idx = int(self.indices[i])

        spec_np = self._table.spectra[row_idx]              # (P,) float32
        if self.preprocess is not None:
            spec_np = self.preprocess(spec_np).astype(np.float32, copy=False)

        spec = torch.from_numpy(spec_np).unsqueeze(0)        # (1, P)
        if self.transform is not None:
            spec = self.transform(spec)

        label = torch.from_numpy(self._table.labels[row_idx])  # (C,)
        vial_id = str(self._table.vial_ids[row_idx])

        return {
            "spectrum": spec,
            "label": label,
            "vial_id": vial_id,
            "index": row_idx,
        }

    @property
    def num_points(self) -> int:
        return self._table.num_points

    @property
    def num_compounds(self) -> int:
        return self._table.num_compounds

    @property
    def wavenumbers(self) -> np.ndarray:
        return self._table.wavenumbers

    @property
    def compound_names(self) -> list[str]:
        return self._table.compound_names


# ───────────────────────────────────────────────────────────────────────────
#  Collate function (handles the str field gracefully)
# ───────────────────────────────────────────────────────────────────────────

def raman_collate(batch: list[dict]) -> dict:
    """Collate a list of dataset items into a batched dict.

    Stacks tensors but keeps `vial_id` as a list of strings (since PyTorch's
    default collate would attempt to tensorize them and fail).
    """
    return {
        "spectrum": torch.stack([b["spectrum"] for b in batch], dim=0),  # (B, 1, P)
        "label":    torch.stack([b["label"]    for b in batch], dim=0),  # (B, C)
        "vial_id":  [b["vial_id"] for b in batch],                       # list[str], len B
        "index":    torch.tensor([b["index"] for b in batch], dtype=torch.long),
    }


# ───────────────────────────────────────────────────────────────────────────
#  DataModule — convenience wrapper for build_dataloaders()
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class SplitIndices:
    """Indices defining one (train, val, test) partition of the full table."""
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    scheme: str = ""

    def summary(self) -> str:
        return (
            f"scheme={self.scheme!r}: "
            f"train={len(self.train)}, val={len(self.val)}, test={len(self.test)}"
        )


def build_dataloaders(
    table: RawSpectraTable,
    split: SplitIndices,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle_train: bool = True,
    drop_last_train: bool = False,
    train_transform=None,
    eval_transform=None,
    preprocess=None,
) -> dict[str, DataLoader]:
    """Construct PyTorch DataLoaders for the three splits.

    Returns
    -------
    dict with keys 'train', 'val', 'test', each mapping to a DataLoader.

    Notes
    -----
    * Augmentation (`train_transform`) is applied ONLY to the training loader.
    * `eval_transform` is applied to both val and test (typically None or a
      deterministic normalisation).
    * `preprocess` is applied to ALL splits (it represents the deterministic
      classical pipeline, which must be identical at train and eval time).
    """
    sets = {
        "train": RamanDataset(table, split.train, transform=train_transform, preprocess=preprocess),
        "val":   RamanDataset(table, split.val,   transform=eval_transform,  preprocess=preprocess),
        "test":  RamanDataset(table, split.test,  transform=eval_transform,  preprocess=preprocess),
    }
    loaders = {
        "train": DataLoader(
            sets["train"],
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last_train,
            collate_fn=raman_collate,
        ),
        "val": DataLoader(
            sets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=raman_collate,
        ),
        "test": DataLoader(
            sets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=raman_collate,
        ),
    }
    return loaders


# ───────────────────────────────────────────────────────────────────────────
#  Smoke test (run as: python -m src.data.dataloader)
# ───────────────────────────────────────────────────────────────────────────

def _smoke_test() -> None:
    """End-to-end check: load CSV, build datasets, iterate one batch."""
    import yaml

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s — %(message)s",
                        datefmt="%H:%M:%S")

    with open("configs/default.yaml", encoding="utf-8") as f:
        defaults = yaml.safe_load(f)
    with open("configs/data_config.yaml", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    table = load_raw_csv(
        csv_path=defaults["paths"]["data_raw_csv"],
        compound_full_names=defaults["compounds"]["full_names"],
        laser_wl_nm=defaults["wavenumber"]["laser_wavelength_nm"],
        expected_num_points=defaults["wavenumber"]["expected_num_points"],
    )

    log.info(f"  pure-vial rows detected: {int(table.pure_mask().sum())}")

    # Use a trivial random split for the smoke test (the real splits live in splits.py)
    n = table.num_samples
    cut1, cut2 = int(0.8 * n), int(0.9 * n)
    perm = np.random.RandomState(0).permutation(n)
    split = SplitIndices(
        train=perm[:cut1], val=perm[cut1:cut2], test=perm[cut2:],
        scheme="smoke_random_802010",
    )
    log.info(split.summary())

    loaders = build_dataloaders(
        table, split,
        batch_size=data_cfg["dataloader"]["batch_size"],
        num_workers=0,         # smoke-test always uses 0 workers
        pin_memory=False,
    )

    for name, loader in loaders.items():
        batch = next(iter(loader))
        log.info(
            f"[{name}] batch shapes: spectrum={tuple(batch['spectrum'].shape)} "
            f"label={tuple(batch['label'].shape)} vial_ids={len(batch['vial_id'])}"
        )
        assert batch["spectrum"].dim() == 3 and batch["spectrum"].shape[1] == 1, \
            "spectrum must have shape (B, 1, P)"
        assert batch["label"].shape[1] == table.num_compounds, \
            f"label dim mismatch: {batch['label'].shape}"
        assert torch.isnan(batch["spectrum"]).sum() == 0, "NaN in batch spectra"
        assert torch.isnan(batch["label"]).sum() == 0, "NaN in batch labels"

    log.info("✓ Smoke test PASSED.")


if __name__ == "__main__":
    _smoke_test()