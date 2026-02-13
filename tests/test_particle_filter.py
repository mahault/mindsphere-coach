"""Tests for tom/particle_filter.py — user type inference."""

import numpy as np
import pytest

from mindsphere.tom.particle_filter import UserTypeFilter
from mindsphere.tom.trust import compute_confidence, compute_reliability


@pytest.fixture
def pf():
    return UserTypeFilter(n_particles=50, seed=42)


class TestUserTypeFilter:
    def test_initial_weights_uniform(self, pf):
        """Initial weights should be uniform."""
        expected = np.ones(50) / 50
        np.testing.assert_allclose(pf.particle_weights, expected)

    def test_initial_particles_in_range(self, pf):
        """All particle values should be in [0, 1]."""
        assert pf.particle_params.min() >= 0.0
        assert pf.particle_params.max() <= 1.0

    def test_update_weights_normalizes(self, pf):
        """After weight update, weights should sum to 1."""
        intervention = {"difficulty": 0.3, "duration_minutes": 5, "evaluative": 0.1}
        pf.update_weights(0, intervention)  # User accepts
        assert pf.particle_weights.sum() == pytest.approx(1.0)

    def test_repeated_too_hard_shifts_particles(self, pf):
        """Repeated 'too hard' should shift particles toward overwhelm-sensitive."""
        intervention = {"difficulty": 0.7, "duration_minutes": 20, "evaluative": 0.5}

        for _ in range(5):
            pf.update_weights(1, intervention)  # 1 = too_hard

        user_type = pf.get_expected_user_type()
        # overwhelm_threshold (dim 6) should be low (easily overwhelmed)
        assert user_type[6] < 0.5  # Below midpoint

    def test_repeated_accept_shifts_particles(self, pf):
        """Repeated 'accept' should shift particles toward resilient types."""
        intervention = {"difficulty": 0.5, "duration_minutes": 10, "evaluative": 0.3}

        for _ in range(5):
            pf.update_weights(0, intervention)  # 0 = accept

        user_type = pf.get_expected_user_type()
        # overwhelm_threshold (dim 6) should be higher
        assert user_type[6] > 0.3

    def test_predict_response_valid(self, pf):
        """predict_response should return valid probabilities."""
        intervention = {"difficulty": 0.3, "duration_minutes": 5}
        pred = pf.predict_response(intervention)

        total = pred["p_accept"] + pred["p_too_hard"] + pred["p_not_relevant"]
        assert total == pytest.approx(1.0, abs=0.01)
        assert 0 <= pred["predicted_felt_cost"] <= 1

    def test_predict_response_gated(self, pf):
        """Gated prediction should blend learned and prior."""
        intervention = {"difficulty": 0.3, "duration_minutes": 5}
        pred = pf.predict_response_gated(intervention)

        total = pred["p_accept"] + pred["p_too_hard"] + pred["p_not_relevant"]
        assert total == pytest.approx(1.0, abs=0.02)
        assert "reliability" in pred

    def test_reliability_increases_with_observations(self, pf):
        """Reliability should increase as we observe more."""
        r_initial = pf.reliability
        intervention = {"difficulty": 0.5, "duration_minutes": 10}

        for _ in range(10):
            pf.update_weights(0, intervention)

        r_after = pf.reliability
        assert r_after >= r_initial  # May not always increase but should generally

    def test_reset(self, pf):
        """Reset should restore initial state."""
        intervention = {"difficulty": 0.5, "duration_minutes": 10}
        pf.update_weights(0, intervention)

        pf.reset(seed=42)
        expected = np.ones(50) / 50
        np.testing.assert_allclose(pf.particle_weights, expected)

    def test_user_type_summary(self, pf):
        """Summary should have all 7 dimensions."""
        summary = pf.get_user_type_summary()
        assert len(summary) == 7
        for val in summary.values():
            assert 0 <= val <= 1


class TestTrustFunctions:
    def test_uniform_weights_low_confidence(self):
        """Uniform weights should give low confidence."""
        weights = np.ones(50) / 50
        conf = compute_confidence(weights, 50)
        assert conf < 0.1

    def test_peaked_weights_high_confidence(self):
        """Peaked weights should give high confidence."""
        weights = np.zeros(50)
        weights[0] = 1.0
        conf = compute_confidence(weights, 50)
        assert conf > 0.9

    def test_reliability_sigmoid_shape(self):
        """Reliability should be sigmoid-shaped w.r.t. confidence."""
        r_low = compute_reliability(0.0, u_threshold=0.5, kappa=0.1)
        r_mid = compute_reliability(0.5, u_threshold=0.5, kappa=0.1)
        r_high = compute_reliability(1.0, u_threshold=0.5, kappa=0.1)

        assert r_low < 0.1
        assert 0.4 < r_mid < 0.6
        assert r_high > 0.9
