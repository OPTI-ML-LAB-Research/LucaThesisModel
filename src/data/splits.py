"""Train / val / test splits for the AA Raman dataset.

Two strategies, both producing a ``SplitIndices`` dict and persisting to
JSON:

* :func:`split_A_vial_level` — Scheme A. Composition-OOD. The 54 vials
  (48 mixtures + 6 pure compounds) are partitioned 42 / 6 / 6 with a fixed
  random seed. Every spectrum from a vial goes entirely to one split. This
  is the **headline** OOD evaluation: at test time the model sees mixtures
  whose composition was never observed during training.
* :func:`split_A_sample_level` — Scheme A'. Random 60 / 20 / 20 across
  the 4378 individual spectra. Less rigorous (samples from the same vial
  can appear in train and test) but provides an upper bound and a sanity
  check.

Both functions return identical-shape outputs and use the same JSON
serialization, so downstream code can read either.

Reference: T07 in Chat-2 spec; PROJECT_REVISION_v2.md §3.1; the
prior thesis baseline (HNKHSV.pptx) used the same vial-level split.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np


# Canonical compound order — must match `AA_Data.csv` label columns and
# `engine/reference_spectra.npy` row order. Single source of truth.
COMPOUND_ORDER: list[str] = [
    "Alanine",
    "Asparagine",
    "Aspartic Acid",
    "Glutamic Acid",
    "Histidine",
    "Glucosamine",
]

# Pure-vial regex. Matches ANY of:
#   DL-alanine, L-alanine, L-asparagine, L-aspartic-acid, L-glutamic-acid,
#   L-histidine, D-glucosamine, D-glucosamine-HCl, etc.
# (See GROUNDWORK_SUMMARY §1.2 issue #8: real data uses DL-alanine,
# not L-alanine — accept all stereo prefixes.)
PURE_VIAL_REGEX = re.compile(
    r"^(?:[DLdl]+-)?(?:alanine|asparagine|aspartic[- ]?acid|"
    r"glutamic[- ]?acid|histidine|glucosamine)(?:[- ]hcl)?$",
    re.IGNORECASE,
)


@dataclass
class SplitIndices:
    """Train / val / test row-index lists into the spectra table."""

    train: list[int]
    val: list[int]
    test: list[int]
    scheme: str
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def n_total(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)

    def assert_valid(self, n_rows: int) -> None:
        """Raise if the split is malformed."""
        all_idx = list(self.train) + list(self.val) + list(self.test)
        if len(all_idx) != n_rows:
            raise ValueError(
                f"Split covers {len(all_idx)} rows but dataset has {n_rows}. "
                "Some rows are missing or duplicated across splits."
            )
        if len(set(all_idx)) != n_rows:
            raise ValueError("Train/val/test contain duplicate indices.")
        if any(i < 0 or i >= n_rows for i in all_idx):
            raise ValueError("Index out of range [0, n_rows).")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_pure_vial(vial_name: str) -> bool:
    """Return True if ``vial_name`` is one of the 6 pure-compound vials."""
    if not isinstance(vial_name, str):
        return False
    return bool(PURE_VIAL_REGEX.match(vial_name.strip()))


def _per_vial_rows(vial_ids: Sequence[str]) -> dict[str, list[int]]:
    """Group row indices by their vial id."""
    out: dict[str, list[int]] = {}
    for i, v in enumerate(vial_ids):
        out.setdefault(str(v), []).append(i)
    return out


def _check_compound_coverage(
    train_idx: Sequence[int],
    labels: np.ndarray,
    threshold: float = 1e-3,
) -> dict[str, int]:
    """For each compound, count training rows where ratio > threshold.

    Used to assert: *"mỗi compound xuất hiện ít nhất 1 lần trong train"*
    (T07 spec).
    """
    if labels.shape[1] != len(COMPOUND_ORDER):
        raise ValueError(
            f"labels has {labels.shape[1]} columns; expected {len(COMPOUND_ORDER)}."
        )
    train_labels = labels[list(train_idx)]
    counts = (train_labels > threshold).sum(axis=0)
    return {COMPOUND_ORDER[i]: int(counts[i]) for i in range(len(COMPOUND_ORDER))}


# ---------------------------------------------------------------------------
# Scheme A — vial-level (composition OOD)
# ---------------------------------------------------------------------------

def split_A_vial_level(
    vial_ids: Sequence[str],
    labels: np.ndarray,
    *,
    seed: int = 42,
    n_train_vials: int = 42,
    n_val_vials: int = 6,
    n_test_vials: int = 6,
    include_pure_in_train: bool = True,
) -> SplitIndices:
    """Split the dataset by *vial*: 42 / 6 / 6 of 54 unique vials.

    Parameters
    ----------
    vial_ids : Sequence[str]
        One vial id per row, length ``N``.
    labels : np.ndarray
        Shape ``(N, 6)`` composition labels (used only for coverage check).
    seed : int
        RNG seed.
    n_train_vials, n_val_vials, n_test_vials : int
        Number of vials in each split. Must sum to the number of unique
        vials in ``vial_ids``.
    include_pure_in_train : bool
        If True (default), force the 6 pure-compound vials into train so
        every compound is represented and the reconstruction module can
        learn against pure references. The remaining 48 mixture vials are
        split (n_train_vials - 6) / n_val_vials / n_test_vials. This
        matches the prior-thesis convention.

    Returns
    -------
    SplitIndices

    Raises
    ------
    ValueError
        If the vial counts are inconsistent or any compound is missing
        from the train split.
    """
    rng = np.random.default_rng(seed)
    vial_to_rows = _per_vial_rows(vial_ids)
    all_vials = sorted(vial_to_rows.keys())

    expected_total = n_train_vials + n_val_vials + n_test_vials
    if len(all_vials) != expected_total:
        raise ValueError(
            f"Found {len(all_vials)} unique vials but expected "
            f"{expected_total} ({n_train_vials} train + {n_val_vials} val "
            f"+ {n_test_vials} test). Check the vial column."
        )

    pure_vials = [v for v in all_vials if is_pure_vial(v)]
    mixture_vials = [v for v in all_vials if not is_pure_vial(v)]

    if include_pure_in_train:
        if len(pure_vials) > n_train_vials:
            raise ValueError(
                f"More pure vials ({len(pure_vials)}) than train slots "
                f"({n_train_vials})."
            )
        n_train_mix = n_train_vials - len(pure_vials)
        # Shuffle ONLY the mixture vials.
        shuffled = list(mixture_vials)
        rng.shuffle(shuffled)
        train_v = pure_vials + shuffled[:n_train_mix]
        val_v = shuffled[n_train_mix:n_train_mix + n_val_vials]
        test_v = shuffled[n_train_mix + n_val_vials:
                          n_train_mix + n_val_vials + n_test_vials]
    else:
        shuffled = list(all_vials)
        rng.shuffle(shuffled)
        train_v = shuffled[:n_train_vials]
        val_v = shuffled[n_train_vials:n_train_vials + n_val_vials]
        test_v = shuffled[n_train_vials + n_val_vials:expected_total]

    train_idx = sorted(i for v in train_v for i in vial_to_rows[v])
    val_idx = sorted(i for v in val_v for i in vial_to_rows[v])
    test_idx = sorted(i for v in test_v for i in vial_to_rows[v])

    split = SplitIndices(
        train=train_idx, val=val_idx, test=test_idx,
        scheme="A_vial_level", seed=seed,
    )
    split.assert_valid(n_rows=len(vial_ids))

    # Coverage check.
    coverage = _check_compound_coverage(train_idx, labels)
    missing = [c for c, n in coverage.items() if n == 0]
    if missing:
        raise ValueError(
            f"After Scheme-A split, no train rows contain compound(s): "
            f"{missing}. Coverage = {coverage}. Try a different seed or "
            f"set include_pure_in_train=True."
        )
    return split


# ---------------------------------------------------------------------------
# Scheme A' — sample-level (random)
# ---------------------------------------------------------------------------

def split_A_sample_level(
    n_rows: int,
    labels: np.ndarray,
    *,
    seed: int = 42,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
) -> SplitIndices:
    """Random per-spectrum split (60 / 20 / 20 by default).

    Parameters
    ----------
    n_rows : int
        Total number of spectra.
    labels : np.ndarray
        Shape ``(N, 6)`` (used for the coverage check).
    seed : int
        RNG seed.
    train_frac, val_frac, test_frac : float
        Must sum to ~1.0.

    Returns
    -------
    SplitIndices
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0, atol=1e-6):
        raise ValueError(
            f"Fractions must sum to 1.0; got "
            f"{train_frac} + {val_frac} + {test_frac} = "
            f"{train_frac + val_frac + test_frac}."
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows)
    n_train = int(round(train_frac * n_rows))
    n_val = int(round(val_frac * n_rows))
    # Test gets the remainder (handles rounding drift).
    train_idx = sorted(perm[:n_train].tolist())
    val_idx = sorted(perm[n_train:n_train + n_val].tolist())
    test_idx = sorted(perm[n_train + n_val:].tolist())

    split = SplitIndices(
        train=train_idx, val=val_idx, test=test_idx,
        scheme="A_prime_sample_level", seed=seed,
    )
    split.assert_valid(n_rows=n_rows)

    coverage = _check_compound_coverage(train_idx, labels)
    missing = [c for c, n in coverage.items() if n == 0]
    if missing:
        # Extremely unlikely with random 60% of 4378 rows, but defend anyway.
        raise ValueError(
            f"Random split happened to miss compound(s) {missing} in train. "
            f"Try a different seed."
        )
    return split


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def save_split(split: SplitIndices, path: str | Path) -> Path:
    """Persist a SplitIndices to JSON at ``path``.

    Format::

        {"train": [...], "val": [...], "test": [...],
         "scheme": "A_vial_level", "seed": 42}
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(split.to_dict(), f, indent=2)
    return path


def load_split(path: str | Path) -> SplitIndices:
    """Load a previously-saved SplitIndices."""
    with Path(path).open("r", encoding="utf-8") as f:
        d = json.load(f)
    return SplitIndices(
        train=list(d["train"]),
        val=list(d["val"]),
        test=list(d["test"]),
        scheme=d.get("scheme", "unknown"),
        seed=int(d.get("seed", -1)),
    )


__all__ = [
    "COMPOUND_ORDER",
    "SplitIndices",
    "is_pure_vial",
    "split_A_vial_level",
    "split_A_sample_level",
    "save_split",
    "load_split",
]
