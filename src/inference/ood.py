"""Out-of-distribution (OOD) scoring for RamanPhysicsAI (T19).

Score formula (Custom Instructions Section 5):

    recon_err = 1 - cosine_similarity(s_input, s_recon)        # per spectrum
    pred_var  = composition_std.mean()                          # MC-Dropout, avg over compounds

    score = recon_weight * normalise(recon_err)
          + var_weight   * normalise(pred_var)

where ``normalise(x) = min(x / x_p95, 1.0)`` and ``x_p95`` is the 95th
percentile of that signal on a calibration set of in-distribution
validation samples. Threshold for "is OOD?" is the 95th percentile of
the COMBINED score on the same calibration set.

Why two signals?

* **Reconstruction error** alone is fooled by spectra that happen to
  decompose well into a fortuitous linear combo of pure references
  (e.g. a noisy blank).
* **Predictive variance** alone is fooled by inputs the model is
  confidently-wrong about (the dropout posterior collapses to a
  single mode that is not in the training set).

Combining the two -- weighted 0.6 / 0.4 by default per Custom
Instructions -- catches both failure modes.

Why percentile-normalisation?

Reconstruction error is on a different scale than composition std; if
you just summed them, one would dominate. Dividing each by its own
95th percentile makes them comparable and bounded in [0, 1] on the
training distribution.

Workflow:

    # 1. Calibrate ONCE after training:
    scorer = OODScorer(model, recon_weight=0.6, var_weight=0.4,
                       mc_samples=50)
    scorer.calibrate(val_loader)                # walks val set
    scorer.save("results/ood_calibration.json")

    # 2. Score new spectra:
    spectrum = ...  # (P,) or (1, P) tensor
    score = scorer.score(spectrum)              # scalar in ~[0, 1]
    is_ood = scorer.is_ood(spectrum)            # bool
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.uncertainty import predict_with_uncertainty


# ---------------------------------------------------------------------------
# Calibration container
# ---------------------------------------------------------------------------

@dataclass
class OODCalibration:
    """Stored statistics of the in-distribution score components.

    Attributes:
        recon_p95: 95th percentile of reconstruction error on cal set.
        var_p95: 95th percentile of mean compound std on cal set.
        score_p95: 95th percentile of combined normalised score on cal set.
            This is the default OOD threshold.
        n_calibration_samples: How many samples informed the percentiles.
        recon_weight / var_weight: Weights used to build score_p95 (must
            match the OODScorer instance using this calibration).
        mc_samples: Number of MC-Dropout passes used during calibration.
    """

    recon_p95: float
    var_p95: float
    score_p95: float
    n_calibration_samples: int
    recon_weight: float
    var_weight: float
    mc_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OODCalibration":
        # Tolerate extra keys for forward compat.
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Component computations
# ---------------------------------------------------------------------------

def compute_reconstruction_error(
    s_input: torch.Tensor, s_recon: torch.Tensor, eps: float = 1e-8,
) -> torch.Tensor:
    """Per-sample reconstruction error = 1 - cosine_similarity.

    Args:
        s_input: (B, P) or (B, 1, P) original spectrum.
        s_recon: (B, P) reconstruction (RamanPhysicsAI output shape).
        eps: Numerical floor for cosine denominator.

    Returns:
        Tensor of shape (B,), values in [0, 2]. Identical spectra -> 0.
    """
    if s_input.ndim == 3:
        s_input = s_input.squeeze(1)
    if s_recon.ndim == 3:
        s_recon = s_recon.squeeze(1)
    cos = F.cosine_similarity(s_input, s_recon, dim=-1, eps=eps)   # (B,)
    return 1.0 - cos


def compute_predictive_variance(mc_output: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Per-sample mean compound std from a ``predict_with_uncertainty`` output.

    Args:
        mc_output: Result dict from ``predict_with_uncertainty``.

    Returns:
        Tensor of shape (B,) -- average of ``composition_std`` over compounds.
    """
    return mc_output["mean_compound_std"]   # already (B,)


def _normalise_clipped(raw: torch.Tensor, p95: float) -> torch.Tensor:
    """``min(raw / p95, 1.0)``, clamping at 1 so OOD-ness saturates."""
    if p95 <= 0:
        # Degenerate calibration; fall back to identity to avoid div-by-zero.
        return raw.clamp_min(0.0)
    return (raw / p95).clamp(min=0.0, max=1.0)


# ---------------------------------------------------------------------------
# OOD scorer
# ---------------------------------------------------------------------------

class OODScorer:
    """Stateful OOD scorer wrapping a calibrated RamanPhysicsAI model.

    Args:
        model: Trained model (RamanPhysicsAI). Must be on the desired
            device; ``OODScorer`` does not move it.
        recon_weight: Weight on the reconstruction-error component
            (default 0.6 per Custom Instructions Section 5).
        var_weight: Weight on the predictive-variance component
            (default 0.4). The two weights need not sum to 1; the score
            is on the [0, recon_weight + var_weight] range pre-clipping.
        mc_samples: Default MC-Dropout passes per sample (default 50).
        threshold_percentile: Which percentile of calibration scores to
            use as OOD threshold (default 95).
        calibration: Optional pre-computed calibration; otherwise call
            ``.calibrate(loader)`` before scoring.

    Once calibrated, instance attributes ``recon_p95``, ``var_p95``,
    ``score_p95`` (= threshold) are populated.
    """

    def __init__(
        self,
        model: nn.Module,
        recon_weight: float = 0.6,
        var_weight: float = 0.4,
        mc_samples: int = 50,
        threshold_percentile: float = 95.0,
        calibration: Optional[OODCalibration] = None,
    ) -> None:
        if recon_weight < 0 or var_weight < 0:
            raise ValueError("Weights must be >= 0.")
        if not (0 < threshold_percentile < 100):
            raise ValueError(
                f"threshold_percentile must be in (0, 100), "
                f"got {threshold_percentile}"
            )
        if mc_samples < 2:
            raise ValueError(f"mc_samples must be >= 2, got {mc_samples}")

        self.model = model
        self.recon_weight = float(recon_weight)
        self.var_weight = float(var_weight)
        self.mc_samples = int(mc_samples)
        self.threshold_percentile = float(threshold_percentile)
        self.calibration: Optional[OODCalibration] = calibration

    # -- Calibration ------------------------------------------------------

    @torch.no_grad()
    def _collect_raw_signals(
        self,
        loader: Union[DataLoader, Iterable[Tuple[torch.Tensor, ...]]],
        max_batches: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Walk the loader and accumulate raw (recon_err, pred_var) signals."""
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        recon_errs: list = []
        pred_vars: list = []

        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            spectra = batch[0] if isinstance(batch, (tuple, list)) else batch
            spectra = spectra.to(device)
            # MC pass gives composition_std AND a stochastic reconstruction;
            # for the reconstruction *error* we want a deterministic
            # forward (dropout off) so the result is comparable across
            # calls. Use the model directly in eval() mode.
            self.model.eval()
            out_det = self.model(
                spectra if spectra.ndim == 3 else spectra.unsqueeze(1)
            )
            recon = out_det["reconstruction"]                  # (B, P)
            err = compute_reconstruction_error(spectra, recon)  # (B,)

            mc = predict_with_uncertainty(
                self.model, spectra, n_samples=self.mc_samples,
            )
            var = compute_predictive_variance(mc)               # (B,)

            recon_errs.append(err.cpu().numpy())
            pred_vars.append(var.cpu().numpy())

        if not recon_errs:
            # Empty loader -- nothing to concatenate. Raise with a
            # message the calibrate() caller can match on.
            raise ValueError(
                "Calibration loader yielded zero samples; pass a "
                "non-empty DataLoader of validation spectra."
            )
        return np.concatenate(recon_errs), np.concatenate(pred_vars)

    def calibrate(
        self,
        loader: Union[DataLoader, Iterable[Tuple[torch.Tensor, ...]]],
        max_batches: Optional[int] = None,
    ) -> OODCalibration:
        """Walk a (validation, in-distribution) loader and store percentiles.

        Args:
            loader: DataLoader yielding (spectrum, ...) tuples.
            max_batches: Optional cap (useful for quick tests).

        Returns:
            The freshly-fitted ``OODCalibration``; also stored on ``self``.
        """
        recon_arr, var_arr = self._collect_raw_signals(loader, max_batches)
        # _collect_raw_signals already raises ValueError on empty loader.

        recon_p95 = float(np.percentile(recon_arr, self.threshold_percentile))
        var_p95 = float(np.percentile(var_arr, self.threshold_percentile))

        # Build combined scores for the same calibration samples to get
        # the score threshold.
        recon_arr_t = torch.from_numpy(recon_arr)
        var_arr_t = torch.from_numpy(var_arr)
        scores = (
            self.recon_weight * _normalise_clipped(recon_arr_t, recon_p95)
            + self.var_weight * _normalise_clipped(var_arr_t, var_p95)
        ).numpy()
        score_p95 = float(np.percentile(scores, self.threshold_percentile))

        cal = OODCalibration(
            recon_p95=recon_p95,
            var_p95=var_p95,
            score_p95=score_p95,
            n_calibration_samples=int(len(recon_arr)),
            recon_weight=self.recon_weight,
            var_weight=self.var_weight,
            mc_samples=self.mc_samples,
        )
        self.calibration = cal
        return cal

    # -- Scoring ----------------------------------------------------------

    def _require_calibration(self) -> OODCalibration:
        if self.calibration is None:
            raise RuntimeError(
                "OODScorer not calibrated yet. Call .calibrate(val_loader) "
                "or supply a calibration to the constructor."
            )
        if (self.calibration.recon_weight != self.recon_weight
                or self.calibration.var_weight != self.var_weight):
            warnings.warn(
                "Calibration was fit with different weights than the "
                "current scorer. Threshold may be miscalibrated.",
                RuntimeWarning, stacklevel=2,
            )
        return self.calibration

    @torch.no_grad()
    def score_batch(
        self,
        spectra: Union[torch.Tensor, np.ndarray],
        return_components: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute OOD score(s) for a batch of spectra.

        Args:
            spectra: (B, P) or (B, 1, P) or (P,). Numpy or tensor.
            return_components: If True, also return raw + normalised
                recon_err and pred_var.

        Returns:
            If ``return_components`` is False: tensor (B,) of scores.
            Else: dict with keys ``score``, ``recon_err_raw``,
            ``recon_err_norm``, ``pred_var_raw``, ``pred_var_norm``,
            each (B,).
        """
        cal = self._require_calibration()

        # Coerce input.
        if isinstance(spectra, np.ndarray):
            spectra = torch.from_numpy(spectra).float()
        if spectra.ndim == 1:
            spectra = spectra.unsqueeze(0)               # (P,) -> (1, P)

        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        spectra = spectra.to(device)

        # Deterministic forward for reconstruction.
        self.model.eval()
        x_3d = spectra if spectra.ndim == 3 else spectra.unsqueeze(1)
        out_det = self.model(x_3d)
        recon_err = compute_reconstruction_error(spectra, out_det["reconstruction"])

        # MC pass for predictive variance.
        mc = predict_with_uncertainty(self.model, spectra,
                                      n_samples=self.mc_samples)
        pred_var = compute_predictive_variance(mc)

        recon_norm = _normalise_clipped(recon_err, cal.recon_p95)
        var_norm = _normalise_clipped(pred_var, cal.var_p95)

        score = self.recon_weight * recon_norm + self.var_weight * var_norm

        if return_components:
            return {
                "score": score,
                "recon_err_raw": recon_err,
                "recon_err_norm": recon_norm,
                "pred_var_raw": pred_var,
                "pred_var_norm": var_norm,
            }
        return score

    def score(self, spectrum: Union[torch.Tensor, np.ndarray]) -> float:
        """Single-spectrum convenience: returns a Python float."""
        s = self.score_batch(spectrum)
        if s.ndim == 0:
            return float(s.item())
        if s.shape == (1,):
            return float(s.item())
        raise ValueError(
            f"score() expects a single spectrum input; got batch of "
            f"{s.shape[0]}. Use score_batch() for batches."
        )

    def is_ood(self, spectrum: Union[torch.Tensor, np.ndarray]) -> bool:
        """True iff ``score(spectrum) > calibration.score_p95``."""
        cal = self._require_calibration()
        return self.score(spectrum) > cal.score_p95

    def is_ood_batch(
        self, spectra: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """Vectorised version of ``is_ood``. Returns (B,) bool tensor."""
        cal = self._require_calibration()
        scores = self.score_batch(spectra)
        return scores > cal.score_p95

    # -- Persistence ------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save calibration to JSON. (Model weights stay in checkpoint.)"""
        cal = self._require_calibration()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "calibration": cal.to_dict(),
                "threshold_percentile": self.threshold_percentile,
            }, f, indent=2)

    @classmethod
    def load_calibration(cls, path: Union[str, Path]) -> OODCalibration:
        """Load just the calibration object from a JSON file."""
        with open(path, "r") as f:
            payload = json.load(f)
        return OODCalibration.from_dict(payload["calibration"])

    @classmethod
    def from_file(
        cls, model: nn.Module, path: Union[str, Path], **kwargs: Any,
    ) -> "OODScorer":
        """Build an OODScorer with calibration loaded from disk.

        Args:
            model: Trained model.
            path: JSON saved by ``OODScorer.save``.
            **kwargs: Override ``recon_weight``, ``var_weight``, etc.
                Defaults read from the file.
        """
        with open(path, "r") as f:
            payload = json.load(f)
        cal = OODCalibration.from_dict(payload["calibration"])
        return cls(
            model=model,
            recon_weight=kwargs.pop("recon_weight", cal.recon_weight),
            var_weight=kwargs.pop("var_weight", cal.var_weight),
            mc_samples=kwargs.pop("mc_samples", cal.mc_samples),
            threshold_percentile=kwargs.pop(
                "threshold_percentile",
                payload.get("threshold_percentile", 95.0),
            ),
            calibration=cal,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Synthetic-OOD generators (for stretch test in __main__ and T22 demos)
# ---------------------------------------------------------------------------

def make_synthetic_ood(
    spectrum: torch.Tensor,
    mode: str = "spike",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Perturb an ID spectrum into a synthetic OOD example.

    Useful for the stretch-test in T18/T19 ("10 ID + 10 OOD discriminative")
    when real OOD samples (bacteria_ID, MoS2) are not yet in scope.

    Modes:
        "spike"  -- inject a narrow rectangular pulse not seen in training
                    (peaks at fixed-but-rare pixel locations).
        "noise"  -- add heavy Gaussian noise (sigma ~ 5x training noise).
        "mask"   -- zero out a large window of the fingerprint region.
        "scale"  -- non-physical extreme intensity scaling.

    Args:
        spectrum: (P,) or (B, P) input.
        mode: One of the four modes above.
        seed: Optional RNG seed for reproducibility.

    Returns:
        Perturbed spectrum, same shape as input.
    """
    rng = np.random.default_rng(seed)
    s = spectrum.clone()
    P = s.shape[-1]

    if mode == "spike":
        # Three narrow pulses at pixels that don't match any of the
        # discriminative peaks (Histidine 1003 -> pixel ~493,
        # Glucosamine 1080 -> pixel ~525 on the 1.7 cm-1/pixel AA grid).
        # Pick pixels well away from those.
        positions = [50, 200, 950]
        width = 5
        magnitude = float(s.abs().max().item()) * 5.0 + 1.0
        for pos in positions:
            lo = max(0, pos - width // 2)
            hi = min(P, pos + width // 2 + 1)
            s[..., lo:hi] = s[..., lo:hi] + magnitude
        return s

    if mode == "noise":
        # 5x the training augmentation sigma.
        noise = torch.from_numpy(
            rng.normal(0, 0.025, size=s.shape).astype(np.float32)
        )
        return s + noise

    if mode == "mask":
        # Zero out 30% of the fingerprint region.
        start = int(0.3 * P); stop = int(0.6 * P)
        s = s.clone()
        s[..., start:stop] = 0.0
        return s

    if mode == "scale":
        # Multiply by 10 -- very out of the SNV-normalised training distribution.
        return s * 10.0

    raise ValueError(
        f"Unknown mode {mode!r}. Expected one of "
        f"'spike', 'noise', 'mask', 'scale'."
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.models.full_model import RamanPhysicsAI

    torch.manual_seed(0)
    np.random.seed(0)

    P, K = 1024, 6
    refs = np.random.RandomState(0).randn(K, P).astype(np.float32) * 0.5
    model = RamanPhysicsAI(reference_spectra=refs, n_compounds=K,
                           spectrum_length=P, feature_dim=256).eval()

    # Build a tiny synthetic "val ID" calibration loader: 50 samples that
    # are mostly random combinations of the refs (mimicking training data).
    cal_X = []
    for _ in range(50):
        alpha = np.abs(np.random.randn(K))
        alpha /= alpha.sum()
        x = (alpha[:, None] * refs).sum(0)
        x = x + np.random.normal(0, 0.05, P).astype(np.float32)
        cal_X.append(x)
    cal_X = torch.from_numpy(np.stack(cal_X).astype(np.float32))     # (50, P)

    # Cheap loader: list of (x_batch,) tuples.
    cal_loader = [(cal_X,)]

    scorer = OODScorer(model, recon_weight=0.6, var_weight=0.4,
                       mc_samples=10)
    cal = scorer.calibrate(cal_loader)
    print(f"[T19] Calibration: recon_p95={cal.recon_p95:.4f} "
          f"var_p95={cal.var_p95:.4e} score_p95={cal.score_p95:.4f} "
          f"n={cal.n_calibration_samples}")
    assert cal.n_calibration_samples == 50

    # Score 10 ID samples
    id_scores = scorer.score_batch(cal_X[:10]).cpu().numpy()
    print(f"[T19] 10 ID scores: min={id_scores.min():.3f} "
          f"max={id_scores.max():.3f} mean={id_scores.mean():.3f}")

    # Build 10 synthetic OOD samples and score them
    ood_X = torch.stack([
        make_synthetic_ood(cal_X[i], mode="spike", seed=i)
        for i in range(10)
    ])
    ood_scores = scorer.score_batch(ood_X).cpu().numpy()
    print(f"[T19] 10 OOD (spike) scores: min={ood_scores.min():.3f} "
          f"max={ood_scores.max():.3f} mean={ood_scores.mean():.3f}")

    print(f"[T19] threshold = {cal.score_p95:.3f}")
    print(f"[T19] ID > threshold (false positive): "
          f"{(id_scores > cal.score_p95).sum()}/10")
    print(f"[T19] OOD > threshold (true positive): "
          f"{(ood_scores > cal.score_p95).sum()}/10")

    # Sanity: OOD scores should generally exceed ID scores
    assert ood_scores.mean() > id_scores.mean(), \
        "OOD mean score must exceed ID mean score"

    # is_ood() and is_ood_batch()
    assert scorer.is_ood_batch(ood_X).any().item(), \
        "At least one OOD sample should be flagged"

    # Component breakdown
    components = scorer.score_batch(cal_X[:3], return_components=True)
    print(f"[T19] component dict keys: {sorted(components.keys())}")
    for k in ["score", "recon_err_norm", "pred_var_norm"]:
        assert components[k].shape == (3,)

    # Save / load roundtrip
    import tempfile
    tmpdir = Path(tempfile.mkdtemp())
    save_path = tmpdir / "ood_calibration.json"
    scorer.save(save_path)
    loaded_cal = OODScorer.load_calibration(save_path)
    assert abs(loaded_cal.score_p95 - cal.score_p95) < 1e-9
    scorer2 = OODScorer.from_file(model, save_path)
    assert scorer2.calibration is not None
    assert abs(scorer2.score(cal_X[0]) - scorer.score(cal_X[0])) < 1e-6
    print(f"[T19] save/load roundtrip OK ({save_path})")

    print("[T19] All smoke tests PASSED")