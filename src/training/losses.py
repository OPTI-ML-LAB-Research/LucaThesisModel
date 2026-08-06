"""Loss functions for the Raman Physics-Informed model.

Three pieces:

1. ``quantification_loss`` -- MAE between predicted and true composition.
   We use MAE (L1) instead of MSE because composition vectors are
   constrained to a simplex; L1 is the natural metric on the simplex
   and matches the metric used during evaluation (T09 ``quantification_mae``).

2. ``physics_loss`` -- a weighted sum of MSE and cosine-distance between
   the input spectrum ``s_input`` and the reconstruction ``s_recon``.
   The MSE term punishes amplitude mismatch; the cosine term punishes
   shape mismatch. Together they enforce that the predicted composition
   "explains" the spectrum as a linear mixture of pure components.

3. ``combined_loss`` -- alpha * quantification + beta * physics + gamma * L2.
   The L2 term is over the model parameters, NOT over the predictions.

Hyperparameter defaults match Custom Instructions Section 5:
    alpha=1.0, beta=0.5, gamma=0.01, lambda_cosine=0.3
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Component losses
# ---------------------------------------------------------------------------

def quantification_loss(
    y_true: torch.Tensor, y_pred: torch.Tensor,
) -> torch.Tensor:
    """Mean Absolute Error between true and predicted compositions.

    Args:
        y_true: True composition simplex, shape (B, n_compounds).
        y_pred: Predicted composition simplex, shape (B, n_compounds).

    Returns:
        Scalar tensor: mean over batch and compounds of |y_true - y_pred|.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true={tuple(y_true.shape)} vs "
            f"y_pred={tuple(y_pred.shape)}"
        )
    return F.l1_loss(y_pred, y_true, reduction="mean")


def quantification_loss_weighted(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Per-compound *weighted* MAE.

    Motivation
    ----------
    Standard MAE distributes gradient signal uniformly across compounds.
    When some compounds are intrinsically easier than others (Histidine
    has 4 strong discriminative peaks per Custom Instructions §8, while
    Aspartic Acid has only weak ones), the model finds a shortcut: fit
    Histidine perfectly and predict the marginal mean for the rest.
    T17 D3+D5 diagnostics confirmed this on the actual run
    (Histidine r=0.90 MAE=0.028 vs Aspartic Acid r=0.35 MAE=0.065).

    Per-compound weighting rebalances the gradient so each compound
    contributes in proportion to its current difficulty.

    Formula
    -------
    ``loss = mean_over_(B,K)( weights_k * |y_true_bk - y_pred_bk| )``

    Equivalent to MAE when ``weights == ones(K)``. To keep total
    gradient magnitude comparable to unweighted MAE, callers should
    pass weights that average ~1.0 (see
    :func:`compute_per_compound_weights`).

    Args:
        y_true: True composition simplex, shape (B, K).
        y_pred: Predicted composition simplex, shape (B, K).
        weights: Non-negative per-compound multipliers, shape (K,).

    Returns:
        Scalar tensor.

    Raises:
        ValueError: shape / negativity checks fail.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true={tuple(y_true.shape)} vs "
            f"y_pred={tuple(y_pred.shape)}"
        )
    if weights.ndim != 1:
        raise ValueError(
            f"weights must be 1-D, got shape {tuple(weights.shape)}"
        )
    if weights.shape[0] != y_pred.shape[1]:
        raise ValueError(
            f"weights length {weights.shape[0]} != n_compounds "
            f"{y_pred.shape[1]}"
        )
    if (weights < 0).any():
        raise ValueError(f"weights must be non-negative; got {weights.tolist()}")

    w = weights.to(device=y_pred.device, dtype=y_pred.dtype)
    abs_err = (y_pred - y_true).abs()          # (B, K)
    # Broadcasting: (B, K) * (1, K) -> (B, K); mean over B*K so the
    # result is on the same scale as unweighted MAE when mean(w) ~ 1.
    return (abs_err * w.unsqueeze(0)).mean()


def compute_per_compound_weights(
    per_compound_mae: Union[Sequence[float], torch.Tensor],
    floor: float = 0.1,
    ceiling: float = 5.0,
) -> torch.Tensor:
    """Convert observed per-compound MAE → balanced loss weights.

    Formula
    -------
    ``weights_k = K * mae_k / sum_j mae_j`` then clamped to ``[floor, ceiling]``.
    This guarantees ``mean(weights) == 1`` (before clamping) so total
    gradient magnitude matches unweighted MAE — protecting other
    hyperparameters (learning rate, alpha_quant) from drift.

    Args:
        per_compound_mae: Length-K iterable. Pass per-compound MAE from
            T17 D3 diagnostic, or from a quick baseline run.
        floor: Minimum allowed weight. Prevents zero-gradient for any
            compound even if its MAE happens to be tiny.
        ceiling: Maximum allowed weight. Prevents one outlier-hard
            compound from dominating the loss.

    Returns:
        Tensor of shape (K,), float32, mean ≈ 1.0.
    """
    mae = torch.as_tensor(per_compound_mae, dtype=torch.float32)
    if mae.ndim != 1:
        raise ValueError(f"per_compound_mae must be 1-D, got {tuple(mae.shape)}")
    if (mae < 0).any():
        raise ValueError(f"per_compound_mae must be non-negative; got {mae.tolist()}")
    K = mae.shape[0]
    total = mae.sum()
    if total <= 0:
        return torch.ones(K, dtype=torch.float32)
    weights = K * mae / total
    return weights.clamp(min=floor, max=ceiling)


def _flatten_to_2d(x: torch.Tensor) -> torch.Tensor:
    """Coerce a spectrum tensor to (B, P): drop a singleton channel dim if present.

    Accepts either (B, P) or (B, 1, P).
    """
    if x.ndim == 3:
        if x.shape[1] != 1:
            raise ValueError(
                f"Expected channel dim of 1, got shape {tuple(x.shape)}."
            )
        return x.squeeze(1)
    if x.ndim == 2:
        return x
    raise ValueError(
        f"Expected 2D or 3D tensor, got shape {tuple(x.shape)}."
    )


def physics_loss(
    s_input: torch.Tensor,
    s_recon: torch.Tensor,
    lambda_cosine: float = 0.3,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Reconstruction quality loss: amplitude (MSE) + shape (cosine distance).

    Loss = MSE(s_input, s_recon) + lambda_cosine * (1 - cosine_sim(s_input, s_recon))

    Args:
        s_input: Original spectrum, shape (B, P) or (B, 1, P).
        s_recon: Reconstructed spectrum, shape (B, P) or (B, 1, P).
        lambda_cosine: Weight on the cosine-distance term. Default 0.3
            (Custom Instructions Section 5).
        eps: Numerical floor for cosine-similarity denominator.

    Returns:
        Scalar tensor.
    """
    s_in = _flatten_to_2d(s_input)
    s_rc = _flatten_to_2d(s_recon)
    if s_in.shape != s_rc.shape:
        raise ValueError(
            f"After flattening, shapes still mismatch: "
            f"s_input={tuple(s_in.shape)} vs s_recon={tuple(s_rc.shape)}"
        )

    mse_term = F.mse_loss(s_rc, s_in, reduction="mean")

    # Cosine similarity per row, then mean.
    cos_sim = F.cosine_similarity(s_in, s_rc, dim=-1, eps=eps)   # (B,)
    cosine_term = (1.0 - cos_sim).mean()

    return mse_term + lambda_cosine * cosine_term


def l2_regularization(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    """Sum of squared norms of given parameters (no in-place mutation).

    Args:
        parameters: Iterable of nn.Parameter (typically ``model.parameters()``).

    Returns:
        Scalar tensor (sum_p ||p||_2^2).
    """
    total: Optional[torch.Tensor] = None
    for p in parameters:
        if not p.requires_grad:
            continue
        contrib = p.pow(2).sum()
        total = contrib if total is None else total + contrib
    if total is None:
        # No trainable params? Return a 0 on the appropriate device.
        return torch.tensor(0.0)
    return total


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

def combined_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    s_input: torch.Tensor,
    s_recon: torch.Tensor,
    model_parameters: Optional[Iterable[torch.nn.Parameter]] = None,
    *,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.01,
    lambda_cosine: float = 0.3,
    per_compound_weights: Optional[torch.Tensor] = None,
    return_components: bool = False,
):
    """Combined training loss = alpha * quant + beta * physics + gamma * L2.

    Args:
        y_true: True composition, shape (B, n_compounds).
        y_pred: Predicted composition, shape (B, n_compounds).
        s_input: Original spectrum, shape (B, P) or (B, 1, P).
        s_recon: Reconstructed spectrum, shape (B, P) or (B, 1, P).
        model_parameters: Iterable for L2 regularization. If None, the
            L2 term is skipped (gamma is then irrelevant).
        alpha: Weight on quantification loss (default 1.0).
        beta:  Weight on physics loss (default 0.5).
        gamma: Weight on L2 regularization (default 0.01).
        lambda_cosine: Inner weight for cosine inside physics_loss
            (default 0.3).
        per_compound_weights: Optional 1-D tensor (n_compounds,) of
            non-negative per-compound weights. When provided, the
            quantification term uses :func:`quantification_loss_weighted`
            instead of plain MAE. None (default) preserves legacy
            behaviour.
        return_components: If True, also return per-term breakdown
            for logging.

    Returns:
        If ``return_components`` is False: scalar tensor (the total loss).
        Else: a dict {"total", "quant", "physics", "l2"} of scalar tensors.
    """
    if per_compound_weights is None:
        quant = quantification_loss(y_true, y_pred)
    else:
        quant = quantification_loss_weighted(y_true, y_pred, per_compound_weights)
    phys = physics_loss(s_input, s_recon, lambda_cosine=lambda_cosine)

    if model_parameters is not None and gamma > 0:
        l2 = l2_regularization(model_parameters)
    else:
        l2 = torch.zeros((), device=quant.device, dtype=quant.dtype)

    total = alpha * quant + beta * phys + gamma * l2

    if return_components:
        return {
            "total": total,
            "quant": quant.detach(),
            "physics": phys.detach(),
            "l2": l2.detach(),
        }
    return total


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # ---- Test 1: quantification_loss known value ----
    y_true = torch.tensor([[0.5, 0.3, 0.2]])
    y_pred = torch.tensor([[0.6, 0.2, 0.2]])
    # |0.5-0.6| + |0.3-0.2| + |0.2-0.2| = 0.2; /3 = 0.0666...
    expected = (0.1 + 0.1 + 0.0) / 3
    val = quantification_loss(y_true, y_pred).item()
    print(f"[T13] quant: got {val:.6f}, expected {expected:.6f}")
    assert abs(val - expected) < 1e-6

    # ---- Test 2: physics_loss with identical spectra -> exactly 0 ----
    s = torch.randn(4, 1024)
    pl = physics_loss(s, s, lambda_cosine=0.3).item()
    print(f"[T13] physics(s, s) = {pl:.2e} (must be ~0)")
    assert pl < 1e-6

    # ---- Test 3: physics_loss handles (B, 1, P) input ----
    s3 = s.unsqueeze(1)         # (4, 1, 1024)
    pl_3d = physics_loss(s3, s3, lambda_cosine=0.3).item()
    assert pl_3d < 1e-6
    pl_mixed = physics_loss(s3, s, lambda_cosine=0.3).item()
    assert pl_mixed < 1e-6
    print(f"[T13] physics handles (B,1,P) and mixed dims -- OK")

    # ---- Test 4: physics_loss with negated spectrum ----
    pl_neg = physics_loss(s, -s, lambda_cosine=0.3).item()
    # MSE = 4 * mean(s^2); cosine_sim = -1 -> cosine_term = 2.
    expected_mse = 4 * s.pow(2).mean().item()
    expected_pl = expected_mse + 0.3 * 2.0
    print(f"[T13] physics(s, -s) = {pl_neg:.4f}, expected ~{expected_pl:.4f}")
    assert abs(pl_neg - expected_pl) < 1e-3

    # ---- Test 5: l2_regularization ----
    p = nn.Parameter(torch.tensor([3.0, 4.0]))
    l2 = l2_regularization([p]).item()
    print(f"[T13] L2([3,4]) = {l2} (expected 25.0)")
    assert abs(l2 - 25.0) < 1e-6

    # ---- Test 6: combined_loss returns dict when asked ----
    y_true = torch.rand(8, 6); y_true = y_true / y_true.sum(-1, keepdim=True)
    y_pred = torch.rand(8, 6); y_pred = y_pred / y_pred.sum(-1, keepdim=True)
    s_in = torch.randn(8, 1, 1024)
    s_rc = torch.randn(8, 1024)
    fake_param = nn.Parameter(torch.randn(10))
    out = combined_loss(
        y_true, y_pred, s_in, s_rc,
        model_parameters=[fake_param],
        return_components=True,
    )
    assert set(out.keys()) == {"total", "quant", "physics", "l2"}
    print(f"[T13] combined components: "
          f"quant={out['quant']:.4f}, physics={out['physics']:.4f}, "
          f"l2={out['l2']:.4f}, total={out['total']:.4f}")

    # ---- Test 7: gradients flow through combined_loss ----
    y_pred_g = y_pred.clone().requires_grad_(True)
    loss = combined_loss(y_true, y_pred_g, s_in, s_rc)
    loss.backward()
    assert y_pred_g.grad is not None and y_pred_g.grad.abs().sum() > 0
    print("[T13] gradient flows through combined_loss -- OK")

    # ---- Test 8: weighted quant matches unweighted when weights = 1 ----
    w_uniform = torch.ones(6)
    val_w = quantification_loss_weighted(y_true, y_pred, w_uniform).item()
    val_u = quantification_loss(y_true, y_pred).item()
    print(f"[T13] weighted(w=1) = {val_w:.6f}, unweighted = {val_u:.6f}")
    assert abs(val_w - val_u) < 1e-6

    # ---- Test 9: weighted quant amplifies one compound ----
    w_skewed = torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    val_skew = quantification_loss_weighted(y_true, y_pred, w_skewed).item()
    # Should be >= unweighted because we doubled one compound's contribution
    assert val_skew >= val_u
    print(f"[T13] weighted(w=[2,1,1,1,1,1]) = {val_skew:.6f} >= unweighted")

    # ---- Test 10: compute_per_compound_weights ----
    mae_obs = [0.0627, 0.0707, 0.0649, 0.0695, 0.0279, 0.0576]  # from T17 D3
    w = compute_per_compound_weights(mae_obs)
    print(f"[T13] computed weights: {w.tolist()}")
    assert w.shape == (6,)
    # mean should be ~1.0 (clamping only kicks in at extremes)
    assert abs(float(w.mean()) - 1.0) < 0.05, f"mean(w) = {float(w.mean()):.3f}"
    # Histidine (idx 4, lowest MAE) should have smallest weight
    assert int(w.argmin()) == 4

    # ---- Test 11: combined_loss with per_compound_weights ----
    out_w = combined_loss(
        y_true, y_pred, s_in, s_rc,
        per_compound_weights=w,
        return_components=True,
    )
    out_u = combined_loss(
        y_true, y_pred, s_in, s_rc,
        return_components=True,
    )
    # quant terms should differ; physics + l2 should be identical
    assert not torch.allclose(out_w["quant"], out_u["quant"])
    assert torch.allclose(out_w["physics"], out_u["physics"])
    print(f"[T13] combined: weighted quant = {out_w['quant']:.4f}, "
          f"unweighted = {out_u['quant']:.4f}")

    # ---- Test 12: weighted loss gradient flows ----
    y_pred_g2 = y_pred.clone().requires_grad_(True)
    loss_w = combined_loss(y_true, y_pred_g2, s_in, s_rc, per_compound_weights=w)
    loss_w.backward()
    assert y_pred_g2.grad is not None and y_pred_g2.grad.abs().sum() > 0
    print("[T13] gradient flows through weighted combined_loss -- OK")

    print("[T13] All smoke tests PASSED")