"""MC-Dropout uncertainty estimation for RamanPhysicsAI (T18).

Monte Carlo Dropout (Gal & Ghahramani, 2016) approximates Bayesian model
posterior by running ``n_samples`` forward passes with dropout layers
active at inference time, and treating the resulting prediction
distribution as a sample from the predictive posterior.

Two API surfaces:

* Functional: ``predict_with_uncertainty(model, x, n_samples=50)`` --
  one-shot uncertainty estimate for either a single spectrum or a batch.
* Object-oriented: ``MCDropoutWrapper(model, n_samples=50)`` -- caches
  the dropout-mode toggle so repeated calls amortize the cost.

Critical correctness point:

    *** Only nn.Dropout modules go to train() mode. Everything else
        (BatchNorm, Conv1d, Linear) MUST stay in eval() mode. ***

If you set the whole model to train(), BatchNorm uses per-batch
statistics instead of the running estimates from training -- and at
inference time you usually have a single sample, so the batch statistics
are degenerate (variance = 0). This produces meaningless predictions and
even less meaningful variance. We avoid this by walking modules and
toggling only the dropout-bearing ones.

References:
    Gal & Ghahramani (2016), "Dropout as a Bayesian Approximation:
        Representing Model Uncertainty in Deep Learning", ICML.

Notes specific to this project:
    * The full model has dropout in two places: backbone (after global
      pool) and quantification_head (between FC1 and FC2). Both fire.
    * The reconstruction module has only the ``scale`` Parameter --
      no dropout -- so reconstruction uncertainty comes entirely from
      *propagated* composition uncertainty (each sampled composition
      drives a different reconstruction).
    * For the T17-checkpoint case (val_mae=0.0523), we expect higher
      variance on Glucosamine (r=-0.237 known issue, B7 in handover)
      and lower variance on Aspartic / Glutamic / Histidine
      (r > 0.7). T18 acts as both a feature AND a diagnostic.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Union

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Dropout-mode toggling
# ---------------------------------------------------------------------------

def _is_dropout_module(m: nn.Module) -> bool:
    """True for any nn.Dropout / Dropout1d / Dropout2d / Dropout3d / AlphaDropout."""
    return isinstance(
        m,
        (
            nn.Dropout,
            nn.Dropout1d,
            nn.Dropout2d,
            nn.Dropout3d,
            nn.AlphaDropout,
            nn.FeatureAlphaDropout,
        ),
    )


def enable_mc_dropout(model: nn.Module) -> int:
    """Force every ``nn.Dropout*`` submodule into training mode.

    The model itself should already be in ``eval()`` so BatchNorm and
    other non-dropout layers behave deterministically. This function
    only flips the dropout layers back on.

    Args:
        model: The (already ``.eval()``-ed) model.

    Returns:
        Number of dropout modules switched to train mode (sanity check;
        should be > 0 -- if 0, the model has no dropout, so MC Dropout
        will return zero variance).
    """
    n = 0
    for m in model.modules():
        if _is_dropout_module(m):
            m.train()
            n += 1
    return n


@contextmanager
def mc_dropout_mode(model: nn.Module) -> Iterator[int]:
    """Context manager: ``eval`` + dropout-only-train, restore on exit.

    Usage::

        with mc_dropout_mode(model) as n_dropouts:
            assert n_dropouts > 0, "model has no dropout"
            samples = [model(x) for _ in range(n_samples)]

    Yields:
        Number of dropout modules activated (0 means no dropout in model).
    """
    # Remember training state of every module so we can restore exactly.
    prior = {id(m): m.training for m in model.modules()}
    model.eval()
    n = enable_mc_dropout(model)
    try:
        yield n
    finally:
        # Restore every module's prior training flag.
        for m in model.modules():
            m.training = prior.get(id(m), False)


# ---------------------------------------------------------------------------
# Input coercion
# ---------------------------------------------------------------------------

def _coerce_spectrum_input(x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Accept (P,), (B, P), or (B, 1, P); always return (B, 1, P) float32.

    Args:
        x: Tensor or array of any of the three accepted shapes.

    Returns:
        Tensor of shape (B, 1, P), float32, on x's device.
    """
    if isinstance(x, np.ndarray):
        x = torch.as_tensor(x)
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            f"Expected torch.Tensor or numpy.ndarray, got {type(x).__name__}"
        )
    x = x.to(dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0).unsqueeze(0)            # (P,) -> (1, 1, P)
    elif x.ndim == 2:
        x = x.unsqueeze(1)                          # (B, P) -> (B, 1, P)
    elif x.ndim == 3:
        if x.shape[1] != 1:
            raise ValueError(
                f"3D input must have channel dim of 1, got shape "
                f"{tuple(x.shape)}."
            )
    else:
        raise ValueError(
            f"Expected 1D/2D/3D input, got {x.ndim}D shape {tuple(x.shape)}."
        )
    return x


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def predict_with_uncertainty(
    model: nn.Module,
    x: Union[torch.Tensor, np.ndarray],
    n_samples: int = 50,
    *,
    return_samples: bool = False,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """Run ``n_samples`` MC-Dropout forward passes and aggregate.

    Args:
        model: A trained ``RamanPhysicsAI`` (or compatible) model.
            Should be on the desired device already. Internal state
            mutation (eval + dropout-on) is fully restored on return.
        x: Spectrum input; shape (P,), (B, P), or (B, 1, P).
            Numpy arrays are auto-converted.
        n_samples: Number of stochastic forward passes (default 50,
            matching Custom Instructions Section 5).
        return_samples: If True, include the raw per-sample tensors
            for composition / reconstruction in the output. Costs
            ~n_samples * (K + P) * 4 bytes per spectrum in memory.
        eps: Floor used in entropy denominator.

    Returns:
        Dict with these tensors (all on x's device, batched over the
        leading dim if x was batched):

            composition_mean       (B, K)   -- mean over n_samples
            composition_std        (B, K)   -- sample std (n-1)
            reconstruction_mean    (B, P)
            reconstruction_std     (B, P)
            predictive_entropy     (B,)     -- entropy of composition_mean
            mean_compound_std      (B,)     -- composition_std.mean(dim=-1)
                                              -- single-number "uncertainty"
            composition_samples    (n_samples, B, K)  [if return_samples]
            reconstruction_samples (n_samples, B, P)  [if return_samples]
            n_samples              scalar int (Python int, not tensor)

    Notes:
        * The model is NOT modified on exit (dropout flags restored).
        * For a SINGLE spectrum, you still get a leading batch dim of 1
          in every output; squeeze it explicitly if your caller wants
          unbatched results. Doing this here would be guess-work.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    # Coerce + move to same device as model parameters.
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    x = _coerce_spectrum_input(x).to(device)

    # Eager dropout-presence scan: warn BEFORE entering the context
    # manager so the warning always reaches the caller (and so
    # pytest.warns can pick it up before any model forward runs).
    n_dropouts_precheck = sum(
        1 for m in model.modules() if _is_dropout_module(m)
    )
    if n_dropouts_precheck == 0:
        warnings.warn(
            "predict_with_uncertainty: model contains no nn.Dropout "
            "modules; all samples will be identical and variance "
            "will be 0.",
            RuntimeWarning,
            stacklevel=2,
        )

    comp_samples: list = []
    recon_samples: list = []

    with mc_dropout_mode(model):
        with torch.no_grad():
            for _ in range(n_samples):
                out = model(x)
                # ``out`` is a dict from RamanPhysicsAI.forward.
                comp_samples.append(out["composition"])
                recon_samples.append(out["reconstruction"])

    # (n_samples, B, K) / (n_samples, B, P)
    comp = torch.stack(comp_samples, dim=0)
    recon = torch.stack(recon_samples, dim=0)

    comp_mean = comp.mean(dim=0)                                 # (B, K)
    comp_std = comp.std(dim=0, unbiased=True) if n_samples > 1 \
        else torch.zeros_like(comp_mean)                          # (B, K)
    recon_mean = recon.mean(dim=0)                                # (B, P)
    recon_std = recon.std(dim=0, unbiased=True) if n_samples > 1 \
        else torch.zeros_like(recon_mean)                         # (B, P)

    # Entropy of the predictive composition (uses comp_mean):
    #   H = -sum p * log(p)
    # comp_mean is a valid simplex (softmax output, averaged). Shape (B, K).
    safe_mean = comp_mean.clamp_min(eps)
    predictive_entropy = -(safe_mean * safe_mean.log()).sum(dim=-1)  # (B,)

    # Mean compound std -- one number per spectrum, useful for OOD score.
    mean_compound_std = comp_std.mean(dim=-1)                       # (B,)

    result: Dict[str, Any] = {
        "composition_mean": comp_mean,
        "composition_std": comp_std,
        "reconstruction_mean": recon_mean,
        "reconstruction_std": recon_std,
        "predictive_entropy": predictive_entropy,
        "mean_compound_std": mean_compound_std,
        "n_samples": int(n_samples),
    }
    if return_samples:
        result["composition_samples"] = comp           # (n, B, K)
        result["reconstruction_samples"] = recon       # (n, B, P)
    return result


# ---------------------------------------------------------------------------
# OO wrapper
# ---------------------------------------------------------------------------

class MCDropoutWrapper(nn.Module):
    """Convenience wrapper that calls ``predict_with_uncertainty`` on every forward.

    Useful when MC-Dropout inference is the default behaviour (e.g. in
    the T19 OOD scorer and T20 ``predict()`` pipeline). Otherwise prefer
    the function form directly.

    Args:
        model: A trained ``RamanPhysicsAI`` (or compatible). Wrapper does
            not deep-copy; mutations to ``model`` are visible.
        n_samples: Default number of MC samples. Can be overridden per
            call via ``forward(x, n_samples=N)``.
        return_samples: Whether to return raw per-sample tensors by default.

    Forward signature:
        forward(x, *, n_samples=None, return_samples=None) -> dict
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 50,
        return_samples: bool = False,
    ) -> None:
        super().__init__()
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        self.model = model
        self.default_n_samples = int(n_samples)
        self.default_return_samples = bool(return_samples)

        # Diagnostic: count dropout modules at init, warn if 0.
        self._n_dropout_modules = sum(
            1 for m in model.modules() if _is_dropout_module(m)
        )
        if self._n_dropout_modules == 0:
            warnings.warn(
                f"MCDropoutWrapper: wrapped model has 0 dropout modules. "
                f"Variance estimates will be 0.",
                RuntimeWarning,
                stacklevel=2,
            )

    @property
    def n_dropout_modules(self) -> int:
        """How many ``nn.Dropout*`` layers the wrapped model exposes."""
        return self._n_dropout_modules

    def forward(
        self,
        x: Union[torch.Tensor, np.ndarray],
        *,
        n_samples: Optional[int] = None,
        return_samples: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        return predict_with_uncertainty(
            self.model,
            x,
            n_samples=n_samples if n_samples is not None else self.default_n_samples,
            return_samples=(
                return_samples
                if return_samples is not None
                else self.default_return_samples
            ),
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

    P = 1024
    K = 6
    # Synthetic refs (sandbox path -- no engine/reference_spectra.npy).
    refs = np.random.RandomState(0).randn(K, P).astype(np.float32)
    model = RamanPhysicsAI(reference_spectra=refs, n_compounds=K,
                           spectrum_length=P, feature_dim=256).eval()

    # ---- Single spectrum ----
    x = torch.randn(P)
    out = predict_with_uncertainty(model, x, n_samples=50)
    print("[T18] keys:", sorted(out.keys()))
    print(f"[T18] composition_mean shape: {tuple(out['composition_mean'].shape)} "
          f"(expected (1, {K}))")
    print(f"[T18] composition_std  shape: {tuple(out['composition_std'].shape)}")
    print(f"[T18] reconstruction_mean shape: {tuple(out['reconstruction_mean'].shape)}")
    print(f"[T18] predictive_entropy   : {out['predictive_entropy'].item():.4f}")
    print(f"[T18] mean_compound_std    : {out['mean_compound_std'].item():.4e}")
    print(f"[T18] n_samples            : {out['n_samples']}")

    # Sanity: composition_mean is still a simplex.
    s = out["composition_mean"].sum(-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5), \
        f"composition_mean rows must sum to 1, got {s}"

    # Sanity: std > 0 (we have dropout active).
    assert out["composition_std"].abs().sum() > 0, \
        "composition_std is exactly 0 -- dropout did not activate"

    # ---- Batched ----
    x_batch = torch.randn(4, P)
    out_b = predict_with_uncertainty(model, x_batch, n_samples=10)
    assert out_b["composition_mean"].shape == (4, K)
    print(f"[T18] batched: composition_mean shape "
          f"{tuple(out_b['composition_mean'].shape)} OK")

    # ---- 3D input ----
    x_3d = torch.randn(2, 1, P)
    out3 = predict_with_uncertainty(model, x_3d, n_samples=5)
    assert out3["composition_mean"].shape == (2, K)
    print("[T18] 3D input shape handled OK")

    # ---- Model NOT mutated on exit ----
    assert not model.training, "model.training must be False after exit"
    for m in model.modules():
        if _is_dropout_module(m):
            assert not m.training, "dropout module should be off after exit"
    print("[T18] dropout/BN state correctly restored after MC")

    # ---- return_samples ----
    out_s = predict_with_uncertainty(model, x, n_samples=7, return_samples=True)
    assert out_s["composition_samples"].shape == (7, 1, K)
    print(f"[T18] return_samples=True: composition_samples shape "
          f"{tuple(out_s['composition_samples'].shape)} OK")

    # ---- Wrapper ----
    wrapper = MCDropoutWrapper(model, n_samples=20)
    print(f"[T18] wrapper.n_dropout_modules = {wrapper.n_dropout_modules}")
    assert wrapper.n_dropout_modules > 0
    out_w = wrapper(x)
    assert out_w["n_samples"] == 20
    out_w2 = wrapper(x, n_samples=5)
    assert out_w2["n_samples"] == 5
    print("[T18] MCDropoutWrapper override n_samples OK")

    # ---- Determinism with seed ----
    torch.manual_seed(123)
    a = predict_with_uncertainty(model, x, n_samples=10)
    torch.manual_seed(123)
    b = predict_with_uncertainty(model, x, n_samples=10)
    assert torch.allclose(a["composition_mean"], b["composition_mean"]), \
        "Same seed must give identical MC results"
    print("[T18] seeded reproducibility OK")

    print("[T18] All smoke tests PASSED")