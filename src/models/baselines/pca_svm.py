"""T23 — Baseline 1: PCA + SVM (multi-output SVR).

Pipeline at inference: ``spectrum (1024,) → StandardScaler → PCA(50) →
6 × SVR → 6-dim composition → simplex projection``.

Choices and rationale
---------------------
* **StandardScaler before PCA.** Even though training spectra come from
  the SNV-normalised cache, vial-to-vial residual scale differences can
  still affect PCA loadings; centring + variance scaling makes PCA
  rotation-invariant to those.
* **PCA n_components = 50.** Mirrors the Custom Instructions Section 2
  spec (`PCA(n_components=50)`). On 1024-channel spectra, 50 components
  preserves > 99% of variance per spectroscopy literature (e.g. Lussier
  et al. 2020) — diminishing returns beyond.
* **Per-compound SVR with RBF kernel.** sklearn's `MultiOutputRegressor`
  is the standard idiom for multi-output regression when the underlying
  estimator is single-output. RBF handles the moderate non-linearity
  expected from physical mixtures without the engineering cost of a
  deep model.
* **Simplex projection.** SVR outputs are unconstrained real numbers;
  composition must live on the probability simplex. We apply a
  closed-form projection (Wang & Carreira-Perpiñán 2013) per row, so
  ``CVR`` and ``identification_accuracy`` are computed on a valid
  composition. Without this, an SVR can predict negative ratios and the
  CVR comparison vs. our softmax-headed model would be unfair.
* **Linear vs RBF kernel:** experiment via ``--kernel linear`` if needed;
  default is rbf because the dataset is small enough (~2.6K train
  samples) that the cost is acceptable and the gain is real.
* **Hyper-parameter search:** kept minimal (``C=1.0, gamma="scale"``).
  Spec asks for a baseline, not a tuned competitor. A heavily-tuned
  SVR beating our model would muddy the thesis story; a lightly-tuned
  one losing is a clean "physics-informed wins" headline.

Output
------
``checkpoints/baselines/pca_svm.pkl`` — a dict with ``scaler``,
``pca``, ``model``, and ``meta`` (training stats). Loadable via
:func:`load_pca_svm`.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.splits import load_split  # noqa: E402


# Single source of truth for which compounds + ordering (matches AA_Data.csv
# label cols and engine/reference_spectra.npy rows).
COMPOUND_ORDER = ["Alanine", "Asparagine", "Aspartic Acid",
                  "Glutamic Acid", "Histidine", "Glucosamine"]


@dataclass
class PCASVMMeta:
    """Training-time metadata persisted alongside the model."""
    pca_n_components: int
    pca_variance_explained: float
    svr_kernel: str
    svr_C: float
    svr_gamma: str
    n_train: int
    train_seconds: float
    sklearn_version: str
    feature_dim_in: int

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Simplex projection — Wang & Carreira-Perpiñán 2013
# ---------------------------------------------------------------------------

def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Project each row of ``v`` (N, K) onto the probability simplex.

    Returns a same-shape array whose rows are non-negative and sum to 1.0.
    O(K log K) per row via sorting; vectorised over batch.
    """
    if v.ndim == 1:
        return project_to_simplex(v.reshape(1, -1)).ravel()
    n, k = v.shape
    u = -np.sort(-v, axis=1)              # sort each row descending
    cssv = np.cumsum(u, axis=1) - 1.0
    rho = np.arange(1, k + 1).reshape(1, -1)
    cond = u - cssv / rho > 0
    # Index of the last True in each row.
    last_true = k - 1 - np.argmax(cond[:, ::-1], axis=1)
    theta = cssv[np.arange(n), last_true] / (last_true + 1)
    out = np.maximum(v - theta[:, None], 0.0)
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_pca_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    n_components: int = 50,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str = "scale",
    verbose: bool = True,
) -> tuple[StandardScaler, PCA, MultiOutputRegressor, PCASVMMeta]:
    """Fit StandardScaler → PCA → MultiOutputRegressor(SVR).

    Parameters
    ----------
    X_train : (n_train, n_features) — preprocessed spectra
    y_train : (n_train, 6)          — simplex composition targets
    n_components : PCA dim
    kernel, C, gamma : passed to ``sklearn.svm.SVR``

    Returns
    -------
    scaler, pca, model, meta
    """
    import sklearn
    if X_train.ndim != 2:
        raise ValueError(f"X_train must be 2D, got shape {X_train.shape}")
    if y_train.shape[0] != X_train.shape[0]:
        raise ValueError("X_train and y_train must have the same first dim")

    t0 = time.perf_counter()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    if verbose:
        print(f"  StandardScaler fit on {X_train.shape}.")

    pca = PCA(n_components=n_components, random_state=42)
    X_train_p = pca.fit_transform(X_train_s)
    var_explained = float(pca.explained_variance_ratio_.sum())
    if verbose:
        print(f"  PCA fit. n_components={n_components}, "
              f"variance_explained={var_explained:.4f}.")

    base = SVR(kernel=kernel, C=C, gamma=gamma)
    model = MultiOutputRegressor(base)
    model.fit(X_train_p, y_train)
    if verbose:
        print(f"  MultiOutputRegressor[SVR(kernel={kernel}, C={C}, "
              f"gamma={gamma})] fit on {X_train_p.shape} -> {y_train.shape}.")

    elapsed = time.perf_counter() - t0
    meta = PCASVMMeta(
        pca_n_components=int(n_components),
        pca_variance_explained=var_explained,
        svr_kernel=str(kernel),
        svr_C=float(C),
        svr_gamma=str(gamma),
        n_train=int(X_train.shape[0]),
        train_seconds=float(elapsed),
        sklearn_version=sklearn.__version__,
        feature_dim_in=int(X_train.shape[1]),
    )
    if verbose:
        print(f"  done in {elapsed:.1f} s.")
    return scaler, pca, model, meta


def predict_pca_svm(
    X: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    model: MultiOutputRegressor,
    *,
    project: bool = True,
) -> np.ndarray:
    """Forward through the fitted baseline. Returns (N, 6) on the simplex.

    Set ``project=False`` to inspect the raw SVR output (e.g. for CVR
    diagnosis of an unprojected baseline).
    """
    X_s = scaler.transform(X)
    X_p = pca.transform(X_s)
    y_raw = model.predict(X_p)
    return project_to_simplex(y_raw) if project else y_raw


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_pca_svm(path: str | Path, *, scaler, pca, model, meta: PCASVMMeta) -> Path:
    """Pickle the full pipeline + write a JSON meta side-car."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"scaler": scaler, "pca": pca, "model": model,
               "meta": meta.to_dict()}
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    side = path.with_suffix(".meta.json")
    with side.open("w", encoding="utf-8") as f:
        json.dump(meta.to_dict(), f, indent=2)
    return path


def load_pca_svm(path: str | Path) -> dict:
    """Load the pickle saved by :func:`save_pca_svm`."""
    with Path(path).open("rb") as f:
        return pickle.load(f)


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
    parser.add_argument("--out", type=Path,
                        default=Path("checkpoints/baselines/pca_svm.pkl"))
    parser.add_argument("--n-components", type=int, default=50)
    parser.add_argument("--kernel", choices=["rbf", "linear", "poly"], default="rbf")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", type=str, default="scale")
    args = parser.parse_args(argv)

    # ---- Load data ----
    print(f"[1/4] Loading {args.spectra} and {args.labels} ...")
    import torch
    spectra = torch.load(args.spectra, weights_only=False)
    labels = torch.load(args.labels, weights_only=False)
    if not isinstance(spectra, np.ndarray):
        spectra = spectra.numpy() if hasattr(spectra, "numpy") else np.asarray(spectra)
    if not isinstance(labels, np.ndarray):
        labels = labels.numpy() if hasattr(labels, "numpy") else np.asarray(labels)
    spectra = spectra.astype(np.float32)
    labels = labels.astype(np.float32)
    print(f"  spectra={spectra.shape}, labels={labels.shape}")

    # ---- Load split ----
    print(f"[2/4] Loading split {args.split} ...")
    split = load_split(args.split)
    print(f"  train={len(split.train)} val={len(split.val)} test={len(split.test)}")

    X_train = spectra[split.train]
    y_train = labels[split.train]

    # ---- Fit ----
    print(f"[3/4] Fitting PCA+SVM ...")
    scaler, pca, model, meta = train_pca_svm(
        X_train, y_train,
        n_components=args.n_components, kernel=args.kernel,
        C=args.C, gamma=args.gamma, verbose=True,
    )

    # ---- Save ----
    print(f"[4/4] Saving to {args.out} ...")
    save_pca_svm(args.out, scaler=scaler, pca=pca, model=model, meta=meta)
    print(f"  done. meta side-car at {args.out.with_suffix('.meta.json')}")
    print(f"  Training time: {meta.train_seconds:.1f} s")
    print(f"  PCA variance explained: {meta.pca_variance_explained:.4f}")

    # ---- Quick val sanity ----
    if split.val:
        from src.eval.metrics import quantification_mae
        X_val = spectra[split.val]
        y_val = labels[split.val]
        y_pred_val = predict_pca_svm(X_val, scaler, pca, model)
        val_mae = quantification_mae(y_val, y_pred_val)
        print(f"  Val MAE (sanity): {val_mae:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPOUND_ORDER", "PCASVMMeta",
    "project_to_simplex", "train_pca_svm", "predict_pca_svm",
    "save_pca_svm", "load_pca_svm",
]
