"""T25 — Comparison runner.

Evaluates all three models on the SAME test indices from
``data/splits/split_A_composition_ood.json`` and writes the comparison
deliverables in 4 formats:

* ``results/benchmark_table.md``       Markdown
* ``results/benchmark_table.csv``      Spreadsheet
* ``results/benchmark_table.json``     Machine-readable
* ``results/figures/benchmark_mae_bar.png``         Headline chart
* ``results/figures/benchmark_per_compound_mae.png`` Heatmap by compound
* ``results/figures/benchmark_pred_vs_true.png``    3 × 6 scatter grid

The "Our model" column re-uses the predictions saved by T17 at
``results/midcheckpoint_predictions.npz`` rather than re-running the
forward pass. This guarantees the comparison's "ours" numbers are
**bit-identical** to the T17 verdict — no drift between the gate
decision and the final benchmark.

PCA+SVM and ResNet-only are forwarded fresh from their saved
checkpoints.

Per CHAT4 §D.5: recon_cosine and OOD AUROC are only meaningful for
our model. They appear as "N/A" for baselines but are still in the
table so the differentiator story is visible.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.splits import load_split  # noqa: E402
from src.eval.metrics import (  # noqa: E402
    constraint_violation_rate,
    identification_accuracy,
    quantification_mae,
    reconstruction_cosine_similarity,
)
from src.models.baselines.pca_svm import (  # noqa: E402
    COMPOUND_ORDER, load_pca_svm, predict_pca_svm,
)


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def _eval_pca_svm(spectra: np.ndarray, labels: np.ndarray, test_idx: list[int],
                  ckpt_path: Path) -> dict:
    """Evaluate the saved PCA+SVM baseline on test rows."""
    payload = load_pca_svm(ckpt_path)
    X = spectra[test_idx]; y = labels[test_idx]
    t0 = time.perf_counter()
    y_pred = predict_pca_svm(X, payload["scaler"], payload["pca"], payload["model"])
    inf_seconds = time.perf_counter() - t0
    return {
        "name": "PCA+SVM",
        "y_true": y, "y_pred": y_pred,
        "s_input": X, "s_recon": None,            # no reconstruction module
        "metrics": _compute_metrics(y, y_pred, X, None),
        "inference_seconds": float(inf_seconds),
        "inference_per_sample_ms": float(inf_seconds * 1000 / max(1, X.shape[0])),
        "meta": payload.get("meta", {}),
    }


def _eval_resnet_only(spectra: np.ndarray, labels: np.ndarray, test_idx: list[int],
                      ckpt_path: Path, device: str = "cpu") -> dict:
    """Evaluate the saved ResNet-only baseline on test rows."""
    # Lazy torch import; T25 should still partially work if torch is missing.
    import torch  # noqa: F401
    from src.models.baselines.resnet_only import load_resnet_only, predict_resnet_only
    model, ck = load_resnet_only(ckpt_path, device=device)
    X = spectra[test_idx]; y = labels[test_idx]
    t0 = time.perf_counter()
    y_pred = predict_resnet_only(model, X, device=device, batch_size=128)
    inf_seconds = time.perf_counter() - t0
    return {
        "name": "ResNet-only (no physics)",
        "y_true": y, "y_pred": y_pred,
        "s_input": X, "s_recon": None,
        "metrics": _compute_metrics(y, y_pred, X, None),
        "inference_seconds": float(inf_seconds),
        "inference_per_sample_ms": float(inf_seconds * 1000 / max(1, X.shape[0])),
        "meta": {"epoch": ck.get("epoch"), "val_mae": ck.get("val_metrics", {}).get("val_mae")},
    }


def _eval_ours(predictions_npz: Path) -> dict:
    """Re-use the saved T17 predictions instead of re-running the model.

    Why: the T17 verdict (``midcheckpoint_report.md``) was computed on
    exactly this test split; re-running the forward pass would risk a
    one-bit drift if PyTorch / numpy versions differ. Loading the
    cached predictions guarantees identical headline numbers.
    """
    z = np.load(predictions_npz)
    y_true = z["y_true"]; y_pred = z["y_pred"]
    s_input = z["s_input"]; s_recon = z["s_recon"]
    return {
        "name": "Ours (physics-informed)",
        "y_true": y_true, "y_pred": y_pred,
        "s_input": s_input, "s_recon": s_recon,
        "metrics": _compute_metrics(y_true, y_pred, s_input, s_recon),
        # We don't have a clean inference time recorded inside the .npz, so
        # mark it explicitly as "see T17 logs" rather than fake a number.
        "inference_seconds": None,
        "inference_per_sample_ms": None,
        "meta": {"source": str(predictions_npz)},
    }


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     s_input: np.ndarray | None,
                     s_recon: np.ndarray | None) -> dict:
    """All 4 metrics, ``None`` slots for recon-dependent ones if no recon."""
    out = {
        "quant_mae": float(quantification_mae(y_true, y_pred)),
        "ident_accuracy": float(identification_accuracy(y_true, y_pred, threshold=0.05)),
        "recon_cosine_median": None,
        "constraint_violation_rate": None,
    }
    if s_recon is not None and s_input is not None:
        recon = reconstruction_cosine_similarity(s_input, s_recon)
        out["recon_cosine_median"] = float(recon["median"])
        out["constraint_violation_rate"] = float(
            constraint_violation_rate(s_input, s_recon, threshold=0.85)
        )
    # Per-compound MAE — same shape across all models, used by the heatmap.
    per_compound = np.mean(np.abs(y_true - y_pred), axis=0)
    out["per_compound_mae"] = [float(x) for x in per_compound]
    return out


# ---------------------------------------------------------------------------
# Output: Markdown / CSV / JSON
# ---------------------------------------------------------------------------

def _fmt(x, na="N/A", fmt="{:.4f}") -> str:
    return na if x is None else fmt.format(x)


def _bold_best(values: list, lower_is_better: bool) -> list[str]:
    """Return list of formatted strings, bolding the best non-None value."""
    nums = [(i, v) for i, v in enumerate(values) if v is not None]
    if not nums:
        return [_fmt(v) for v in values]
    best_i = min(nums, key=lambda t: t[1])[0] if lower_is_better else max(nums, key=lambda t: t[1])[0]
    return [f"**{_fmt(v)}**" if i == best_i and v is not None else _fmt(v)
            for i, v in enumerate(values)]


def write_markdown_table(results: list[dict], path: Path) -> None:
    """Write the headline 3-row comparison table."""
    names = [r["name"] for r in results]
    quant = [r["metrics"]["quant_mae"] for r in results]
    ident = [r["metrics"]["ident_accuracy"] for r in results]
    recon = [r["metrics"]["recon_cosine_median"] for r in results]
    cvr = [r["metrics"]["constraint_violation_rate"] for r in results]
    inf_ms = [r["inference_per_sample_ms"] for r in results]

    L = ["# Benchmark — T25", "",
         "*Three models, same test split (Scheme-A composition-OOD, 540 rows).*",
         "",
         "| Model | Quant MAE ↓ | Ident Acc ↑ | Recon cos median ↑ | CVR ↓ | Inference (ms/sample) |",
         "|---|---|---|---|---|---|"]

    quant_b = _bold_best(quant, lower_is_better=True)
    ident_b = _bold_best(ident, lower_is_better=False)
    recon_b = _bold_best(recon, lower_is_better=False)
    cvr_b = _bold_best(cvr, lower_is_better=True)

    for i, name in enumerate(names):
        infs = _fmt(inf_ms[i], fmt="{:.2f}")
        L.append(f"| {name} | {quant_b[i]} | {ident_b[i]} | {recon_b[i]} "
                 f"| {cvr_b[i]} | {infs} |")

    L += ["", "Notes:",
          "* Recon cosine and CVR are **only meaningful for our model** — "
          "the baselines have no reconstruction module. N/A entries are honest.",
          "* OOD AUROC is omitted from this table; it requires OOD samples not "
          "defined for the AA-only test split. Phase D will fill this in.",
          "* Inference time uses the same hardware for all rows (user's CPU). "
          "Our-model timing is recorded inside the T17 run, not re-measured here.",
          ""]

    # Per-compound breakdown
    L += ["## Per-compound MAE (lower = better)", "",
          "Useful for diagnosing whether the comparison's MAE gap is uniform across "
          "compounds, or driven by one hard compound (e.g. Glucosamine).",
          "",
          "| Model | " + " | ".join(COMPOUND_ORDER) + " |",
          "|---" + "|---" * len(COMPOUND_ORDER) + "|"]
    per_compound = np.array([r["metrics"]["per_compound_mae"] for r in results])
    # Bold the best (min) per column
    bolded = np.full(per_compound.shape, "", dtype=object)
    for col in range(per_compound.shape[1]):
        best_row = int(np.argmin(per_compound[:, col]))
        for row in range(per_compound.shape[0]):
            v = per_compound[row, col]
            cell = f"{v:.4f}"
            if row == best_row:
                cell = f"**{cell}**"
            bolded[row, col] = cell
    for i, name in enumerate(names):
        L.append(f"| {name} | " + " | ".join(bolded[i]) + " |")
    L.append("")

    path.write_text("\n".join(L), encoding="utf-8")


def write_csv_table(results: list[dict], path: Path) -> None:
    """Long-format CSV: one row per (model, metric)."""
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append({"model": r["name"], "metric": "quant_mae", "value": m["quant_mae"]})
        rows.append({"model": r["name"], "metric": "ident_accuracy", "value": m["ident_accuracy"]})
        rows.append({"model": r["name"], "metric": "recon_cosine_median",
                     "value": m["recon_cosine_median"]})
        rows.append({"model": r["name"], "metric": "constraint_violation_rate",
                     "value": m["constraint_violation_rate"]})
        rows.append({"model": r["name"], "metric": "inference_per_sample_ms",
                     "value": r["inference_per_sample_ms"]})
        for cmp_name, v in zip(COMPOUND_ORDER, m["per_compound_mae"]):
            rows.append({"model": r["name"], "metric": f"mae_{cmp_name}", "value": v})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "metric", "value"])
        w.writeheader()
        w.writerows(rows)


def write_json_table(results: list[dict], path: Path) -> None:
    """Compact JSON: one entry per model."""
    out = []
    for r in results:
        out.append({
            "name": r["name"],
            "metrics": r["metrics"],
            "inference_seconds": r["inference_seconds"],
            "inference_per_sample_ms": r["inference_per_sample_ms"],
            "meta": r["meta"],
        })
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_mae_bar(results: list[dict], path: Path) -> None:
    """Single bar chart: quant_mae per model, with target/floor lines."""
    import matplotlib.pyplot as plt
    names = [r["name"] for r in results]
    maes = [r["metrics"]["quant_mae"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, maes, color=["#888", "#4a8", "#e74"])
    for b, v in zip(bars, maes):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(0.020, ls="--", color="green", alpha=0.7, label="target 0.020")
    ax.axhline(0.025, ls=":", color="orange", alpha=0.7, label="floor 0.025")
    ax.set_ylabel("Quantification MAE (lower = better)")
    ax.set_title("Composition-OOD test set: MAE comparison")
    ax.set_ylim(0, max(maes) * 1.25)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=12, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_per_compound_heatmap(results: list[dict], path: Path) -> None:
    """Heatmap: rows = models, cols = compounds, cell = MAE."""
    import matplotlib.pyplot as plt
    mat = np.array([r["metrics"]["per_compound_mae"] for r in results])
    names = [r["name"] for r in results]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto", vmin=0.0,
                   vmax=max(0.08, mat.max() * 1.05))
    ax.set_xticks(range(len(COMPOUND_ORDER)))
    ax.set_xticklabels(COMPOUND_ORDER, rotation=20, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    # Annotate cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "white" if mat[i, j] > mat.max() * 0.55 else "black"
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                    color=color, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="MAE")
    ax.set_title("Per-compound MAE (rows: model; cols: compound)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_pred_vs_true(results: list[dict], path: Path) -> None:
    """3 rows × 6 cols scatter: per-model × per-compound, with y=x line."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=len(results), ncols=len(COMPOUND_ORDER),
                             figsize=(2.2 * len(COMPOUND_ORDER), 2.2 * len(results)),
                             sharex=True, sharey=True)
    if len(results) == 1:
        axes = axes[None, :]
    for i, r in enumerate(results):
        y_true = r["y_true"]; y_pred = r["y_pred"]
        for j in range(len(COMPOUND_ORDER)):
            ax = axes[i, j]
            ax.scatter(y_true[:, j], y_pred[:, j], s=4, alpha=0.4)
            ax.plot([0, 1], [0, 1], "r--", lw=0.8, alpha=0.6)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            if i == 0:
                ax.set_title(COMPOUND_ORDER[j], fontsize=10)
            if j == 0:
                ax.set_ylabel(r["name"].split()[0], fontsize=9)
            ax.grid(alpha=0.2)
    fig.suptitle("Predicted vs True composition (red dashed = identity)", fontsize=12)
    fig.supxlabel("True ratio", fontsize=10)
    fig.supylabel("Predicted ratio", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


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
    parser.add_argument("--pca-svm-ckpt", type=Path,
                        default=Path("checkpoints/baselines/pca_svm.pkl"))
    parser.add_argument("--resnet-ckpt", type=Path,
                        default=Path("checkpoints/baselines/resnet_only_best.pt"))
    parser.add_argument("--ours-predictions", type=Path,
                        default=Path("results/midcheckpoint_predictions.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--skip-resnet", action="store_true",
                        help="Skip the ResNet-only row (e.g. if torch missing).")
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading inputs ...")
    import torch
    spectra = torch.load(args.spectra, weights_only=False)
    labels = torch.load(args.labels, weights_only=False)
    if not isinstance(spectra, np.ndarray):
        spectra = spectra.numpy() if hasattr(spectra, "numpy") else np.asarray(spectra)
    if not isinstance(labels, np.ndarray):
        labels = labels.numpy() if hasattr(labels, "numpy") else np.asarray(labels)
    spectra = spectra.astype(np.float32); labels = labels.astype(np.float32)
    split = load_split(args.split)
    test_idx = split.test
    print(f"  spectra={spectra.shape}, labels={labels.shape}, |test|={len(test_idx)}")

    results: list[dict] = []

    print(f"[2/5] Evaluating PCA+SVM ...")
    if args.pca_svm_ckpt.exists():
        results.append(_eval_pca_svm(spectra, labels, test_idx, args.pca_svm_ckpt))
        print(f"  MAE={results[-1]['metrics']['quant_mae']:.4f}, "
              f"ident_acc={results[-1]['metrics']['ident_accuracy']:.4f}")
    else:
        print(f"  SKIP (checkpoint not found at {args.pca_svm_ckpt})")

    print(f"[3/5] Evaluating ResNet-only ...")
    if args.skip_resnet:
        print(f"  SKIP (--skip-resnet flag)")
    elif args.resnet_ckpt.exists():
        results.append(_eval_resnet_only(spectra, labels, test_idx,
                                         args.resnet_ckpt, device=args.device))
        print(f"  MAE={results[-1]['metrics']['quant_mae']:.4f}, "
              f"ident_acc={results[-1]['metrics']['ident_accuracy']:.4f}")
    else:
        print(f"  SKIP (checkpoint not found at {args.resnet_ckpt})")

    print(f"[4/5] Loading our-model predictions from T17 ...")
    if args.ours_predictions.exists():
        results.append(_eval_ours(args.ours_predictions))
        print(f"  MAE={results[-1]['metrics']['quant_mae']:.4f}, "
              f"ident_acc={results[-1]['metrics']['ident_accuracy']:.4f}, "
              f"recon_cos={results[-1]['metrics']['recon_cosine_median']:.4f}, "
              f"CVR={results[-1]['metrics']['constraint_violation_rate']:.4f}")
    else:
        print(f"  SKIP (predictions not found at {args.ours_predictions})")

    if not results:
        print("ERROR: no models evaluated. Run T23/T24/T17 first.", file=sys.stderr)
        return 1

    print(f"[5/5] Writing deliverables ...")
    md_path = out_dir / "benchmark_table.md"
    csv_path = out_dir / "benchmark_table.csv"
    json_path = out_dir / "benchmark_table.json"
    write_markdown_table(results, md_path); print(f"  {md_path}")
    write_csv_table(results, csv_path);     print(f"  {csv_path}")
    write_json_table(results, json_path);   print(f"  {json_path}")

    plot_mae_bar(results, fig_dir / "benchmark_mae_bar.png")
    print(f"  {fig_dir/'benchmark_mae_bar.png'}")
    plot_per_compound_heatmap(results, fig_dir / "benchmark_per_compound_mae.png")
    print(f"  {fig_dir/'benchmark_per_compound_mae.png'}")
    plot_pred_vs_true(results, fig_dir / "benchmark_pred_vs_true.png")
    print(f"  {fig_dir/'benchmark_pred_vs_true.png'}")

    print()
    print("====================== T25 SUMMARY ======================")
    for r in results:
        m = r["metrics"]
        print(f"  {r['name']:<30s} MAE={m['quant_mae']:.4f}  "
              f"ident={m['ident_accuracy']:.4f}  "
              f"recon={_fmt(m['recon_cosine_median'])}  "
              f"CVR={_fmt(m['constraint_violation_rate'])}")
    print("=========================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
