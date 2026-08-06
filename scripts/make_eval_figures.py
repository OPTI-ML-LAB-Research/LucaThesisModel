"""Generate all evaluation figures from saved arrays.

Usage:
    python scripts/make_eval_figures.py \
        --results results/eval_arrays.npz \
        --benchmark results/benchmark_table.json \
        --out results/figures

The ``.npz`` may contain any subset of these arrays (missing ones are skipped):
    is_ood            (N,)    bool   - OOD ground truth
    ood_scores        (N,)    float  - OOD scores (higher = more OOD)
    ood_threshold     scalar  float  - decision threshold
    cosines           (M,)    float  - per-sample reconstruction cosine
    y_true, y_pred    (K, 6)  float  - true / predicted composition
    true_labels       (K,)    int    - identification ground-truth class index
    pred_labels       (K,)    int    - identification predicted class index
    compound_order    (6,)    str    - optional; defaults to the 6 amino acids

The ``--benchmark`` JSON is ``{metric: {model: value}}`` (e.g. the contents of
results/benchmark_table). Omit it to skip the comparison panel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.eval.eval_figures import make_all_from_arrays

DEFAULT_COMPOUNDS = ["Alanine", "Asparagine", "Aspartic Acid",
                     "Glutamic Acid", "Histidine", "Glucosamine"]


def _get(npz, key):
    return npz[key] if key in npz.files else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Make evaluation figures.")
    ap.add_argument("--results", required=True, help="Path to eval arrays .npz")
    ap.add_argument("--benchmark", default=None,
                    help="Optional benchmark JSON {metric:{model:value}}")
    ap.add_argument("--out", default="results/figures", help="Output directory")
    args = ap.parse_args()

    npz = np.load(args.results, allow_pickle=True)
    co = _get(npz, "compound_order")
    order = list(co) if co is not None else DEFAULT_COMPOUNDS
    thr = _get(npz, "ood_threshold")
    benchmark = None
    if args.benchmark:
        benchmark = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))

    # Pull AUROC from the canonical metric so the ROC label matches the table.
    auc_override = None
    is_ood, ood_scores = _get(npz, "is_ood"), _get(npz, "ood_scores")
    if is_ood is not None and ood_scores is not None:
        try:
            from src.eval.metrics import ood_auroc
            m = np.asarray(is_ood).astype(bool)
            auc_override = ood_auroc(np.asarray(ood_scores)[~m],
                                     np.asarray(ood_scores)[m])
        except Exception:  # noqa: BLE001
            auc_override = None

    made = make_all_from_arrays(
        args.out, compound_order=order,
        is_ood=is_ood, ood_scores=ood_scores,
        ood_threshold=(float(thr) if thr is not None else None),
        auc_override=auc_override,
        cosines=_get(npz, "cosines"),
        y_true=_get(npz, "y_true"), y_pred=_get(npz, "y_pred"),
        true_labels=_get(npz, "true_labels"),
        pred_labels=_get(npz, "pred_labels"),
        benchmark=benchmark,
    )
    print(f"Generated {len(made)} figure(s) in {args.out}:")
    for name, path in made.items():
        print(f"  {name:18} -> {path}")


if __name__ == "__main__":
    main()
