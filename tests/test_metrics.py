"""Unit tests for src/eval/metrics.py.

Each test uses a hand-computed expected value so a regression cannot
silently slip through.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.metrics import (
    constraint_violation_rate,
    identification_accuracy,
    ood_auroc,
    quantification_mae,
    reconstruction_cosine_similarity,
)


# ---------------------------------------------------------------------------
# quantification_mae
# ---------------------------------------------------------------------------

class TestQuantificationMAE:

    def test_perfect_prediction_is_zero(self):
        y = np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]])
        assert quantification_mae(y, y.copy()) == 0.0

    def test_known_answer(self):
        # y_true and y_pred differ by exactly 0.1 in every cell → MAE = 0.1
        y_true = np.full((4, 6), 0.5)
        y_pred = np.full((4, 6), 0.4)
        assert quantification_mae(y_true, y_pred) == pytest.approx(0.1)

    def test_handles_1d_input(self):
        y_true = np.array([0.2, 0.5, 0.3])
        y_pred = np.array([0.3, 0.4, 0.3])
        # |0.1| + |0.1| + |0| = 0.2; / 3 = 0.0666...
        assert quantification_mae(y_true, y_pred) == pytest.approx(0.2 / 3)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            quantification_mae(np.zeros((3, 6)), np.zeros((3, 5)))

    def test_negative_predictions_handled(self):
        # MAE is symmetric — negative values OK
        y_true = np.array([[0.0, 0.0]])
        y_pred = np.array([[-0.2, 0.3]])
        assert quantification_mae(y_true, y_pred) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# identification_accuracy
# ---------------------------------------------------------------------------

class TestIdentificationAccuracy:

    def test_perfect_match(self):
        y_true = np.array([[0.5, 0.5, 0.0]])
        y_pred = np.array([[0.45, 0.55, 0.0]])
        # both above 0.05 in cols 0&1, both below in col 2 → presence pattern matches
        assert identification_accuracy(y_true, y_pred) == 1.0

    def test_one_miss(self):
        # Sample 0: presence pattern matches.
        # Sample 1: pred says compound-2 absent (0.01 < 0.05) but true=0.4 → mismatch.
        y_true = np.array([[0.6, 0.4, 0.0],
                           [0.6, 0.0, 0.4]])
        y_pred = np.array([[0.55, 0.45, 0.0],
                           [0.55, 0.45, 0.01]])
        # Sample 0 correct, sample 1 wrong → 0.5
        assert identification_accuracy(y_true, y_pred, threshold=0.05) == 0.5

    def test_threshold_sensitivity(self):
        y_true = np.array([[0.10, 0.0, 0.90]])
        y_pred = np.array([[0.04, 0.0, 0.96]])
        # threshold 0.05: true sees compound-0 present, pred sees absent → mismatch
        assert identification_accuracy(y_true, y_pred, threshold=0.05) == 0.0
        # threshold 0.03: true sees compound-0 present, pred also sees present (0.04 > 0.03) → match
        assert identification_accuracy(y_true, y_pred, threshold=0.03) == 1.0

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            identification_accuracy(np.zeros((1, 3)), np.zeros((1, 3)), threshold=-0.1)


# ---------------------------------------------------------------------------
# ood_auroc
# ---------------------------------------------------------------------------

class TestOODAUROC:

    def test_perfect_separation(self):
        scores_id  = [0.0, 0.1, 0.2, 0.3]
        scores_ood = [0.7, 0.8, 0.9, 1.0]
        assert ood_auroc(scores_id, scores_ood) == pytest.approx(1.0)

    def test_no_separation(self):
        scores_id  = [0.5, 0.5, 0.5]
        scores_ood = [0.5, 0.5, 0.5]
        # Tied scores → 0.5 by convention (Mann-Whitney with ties)
        assert ood_auroc(scores_id, scores_ood) == pytest.approx(0.5)

    def test_inverted_separation(self):
        # If we mis-orient the score (lower = OOD by mistake), AUROC < 0.5
        scores_id  = [0.7, 0.8, 0.9]
        scores_ood = [0.0, 0.1, 0.2]
        assert ood_auroc(scores_id, scores_ood) == pytest.approx(0.0)

    def test_partial_separation_known_value(self):
        # scores_id = [0, 1], scores_ood = [0.5, 1.5]
        # Pairs (id, ood): (0, 0.5) → ood>id (1), (0, 1.5) → ood>id (1),
        #                  (1, 0.5) → ood<id (0), (1, 1.5) → ood>id (1)
        # AUROC = 3/4 = 0.75
        assert ood_auroc([0.0, 1.0], [0.5, 1.5]) == pytest.approx(0.75)

    def test_empty_inputs_raise(self):
        with pytest.raises(ValueError, match="non-empty"):
            ood_auroc([], [0.5])


# ---------------------------------------------------------------------------
# reconstruction_cosine_similarity
# ---------------------------------------------------------------------------

class TestReconstructionCosineSimilarity:

    def test_identical_spectra_give_unity(self):
        rng = np.random.default_rng(0)
        s = rng.uniform(0, 1, size=(5, 100))
        out = reconstruction_cosine_similarity(s, s.copy())
        assert out["median"] == pytest.approx(1.0)
        assert out["mean"] == pytest.approx(1.0)
        assert np.allclose(out["per_sample"], 1.0)

    def test_orthogonal_spectra_give_zero(self):
        # Build two orthogonal "spectra" of length 4
        s_in = np.array([[1.0, 0.0, 0.0, 0.0]])
        s_re = np.array([[0.0, 1.0, 0.0, 0.0]])
        out = reconstruction_cosine_similarity(s_in, s_re)
        assert out["per_sample"][0] == pytest.approx(0.0, abs=1e-9)

    def test_anti_correlated_spectra_give_minus_one(self):
        s_in = np.array([[1.0, 2.0, 3.0]])
        s_re = -s_in
        out = reconstruction_cosine_similarity(s_in, s_re)
        assert out["per_sample"][0] == pytest.approx(-1.0)

    def test_returns_full_distribution(self):
        rng = np.random.default_rng(1)
        s_in = rng.uniform(0, 1, size=(20, 100))
        s_re = s_in + 0.05 * rng.normal(size=s_in.shape)
        out = reconstruction_cosine_similarity(s_in, s_re)
        assert set(out.keys()) >= {"median", "mean", "p05", "p25", "p75", "p95", "per_sample"}
        assert out["p05"] <= out["p25"] <= out["median"] <= out["p75"] <= out["p95"]


# ---------------------------------------------------------------------------
# constraint_violation_rate
# ---------------------------------------------------------------------------

class TestConstraintViolationRate:

    def test_all_perfect_zero_cvr(self):
        rng = np.random.default_rng(2)
        s = rng.uniform(0, 1, size=(10, 50))
        assert constraint_violation_rate(s, s.copy(), threshold=0.85) == 0.0

    def test_all_violated(self):
        # Orthogonal pairs → cosine = 0 < 0.85 → CVR = 1.0
        s_in = np.tile([1.0, 0.0, 0.0, 0.0], (5, 1))
        s_re = np.tile([0.0, 1.0, 0.0, 0.0], (5, 1))
        assert constraint_violation_rate(s_in, s_re) == 1.0

    def test_mixed_known_value(self):
        # 3 perfect (cos=1) + 2 orthogonal (cos=0) → 2/5 = 0.4 violated at thr=0.85
        s_in = np.vstack([
            np.tile([1.0, 1.0, 1.0], (3, 1)),
            np.tile([1.0, 0.0, 0.0], (2, 1)),
        ])
        s_re = np.vstack([
            np.tile([1.0, 1.0, 1.0], (3, 1)),
            np.tile([0.0, 1.0, 0.0], (2, 1)),
        ])
        assert constraint_violation_rate(s_in, s_re, threshold=0.85) == pytest.approx(0.4)

    def test_threshold_extremes(self):
        s = np.array([[1.0, 0.0]])
        # cos(s, [0.5, 0.5]) = 1 / sqrt(2) ≈ 0.707
        s_re = np.array([[0.5, 0.5]])
        # threshold 1.0 → violated (0.707 < 1) → CVR = 1.0
        assert constraint_violation_rate(s, s_re, threshold=1.0) == 1.0
        # threshold 0.7 → not violated (0.707 > 0.7) → CVR = 0.0
        assert constraint_violation_rate(s, s_re, threshold=0.7) == 0.0

    def test_invalid_threshold(self):
        s = np.zeros((1, 3))
        with pytest.raises(ValueError, match="threshold"):
            constraint_violation_rate(s, s, threshold=2.0)
