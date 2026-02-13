"""Tests for core/model.py — POMDP state space definitions."""

import numpy as np
import pytest

from mindsphere.core.model import SphereModel, SKILL_FACTORS, SKILL_LEVEL_VALUES


@pytest.fixture
def model():
    return SphereModel()


class TestSphereModel:
    def test_all_factors_have_matrices(self, model):
        """Every factor should have A, B, C, D matrices."""
        for factor in model.get_all_factor_names():
            assert factor in model.A, f"Missing A matrix for {factor}"
            assert factor in model.B, f"Missing B matrix for {factor}"
            assert factor in model.C, f"Missing C matrix for {factor}"
            assert factor in model.D, f"Missing D matrix for {factor}"

    def test_a_matrices_are_valid(self, model):
        """A matrix columns should sum to 1 (valid likelihood)."""
        for factor, A in model.A.items():
            col_sums = A.sum(axis=0)
            np.testing.assert_allclose(
                col_sums, 1.0, atol=0.01,
                err_msg=f"A[{factor}] columns don't sum to 1: {col_sums}"
            )

    def test_b_matrices_are_valid(self, model):
        """B matrix columns should sum to 1 (valid transition)."""
        for factor, B in model.B.items():
            for a in range(B.shape[0]):
                col_sums = B[a].sum(axis=0)
                np.testing.assert_allclose(
                    col_sums, 1.0, atol=0.01,
                    err_msg=f"B[{factor}][action={a}] columns don't sum to 1"
                )

    def test_d_matrices_are_valid(self, model):
        """D vectors should sum to 1 (valid prior)."""
        for factor, D in model.D.items():
            np.testing.assert_allclose(
                D.sum(), 1.0, atol=0.01,
                err_msg=f"D[{factor}] doesn't sum to 1"
            )

    def test_initial_beliefs_match_d(self, model):
        """get_initial_beliefs should return copies of D vectors."""
        beliefs = model.get_initial_beliefs()
        for factor in model.get_all_factor_names():
            np.testing.assert_array_equal(beliefs[factor], model.D[factor])

    def test_skill_score_range(self, model):
        """Skill scores should be in [0, 100]."""
        # All probability on lowest level
        low_belief = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        assert model.get_skill_score(low_belief) == pytest.approx(10.0)

        # All probability on highest level
        high_belief = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        assert model.get_skill_score(high_belief) == pytest.approx(90.0)

    def test_eight_skill_factors(self, model):
        """Should have exactly 8 skill factors."""
        assert len(model.spec.skill_factors) == 8

    def test_get_all_skill_scores(self, model):
        """get_all_skill_scores should return dict with 8 entries."""
        beliefs = model.get_initial_beliefs()
        scores = model.get_all_skill_scores(beliefs)
        assert len(scores) == 8
        for skill in SKILL_FACTORS:
            assert skill in scores
            assert 0 <= scores[skill] <= 100
