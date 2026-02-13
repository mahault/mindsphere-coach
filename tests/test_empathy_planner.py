"""Tests for tom/empathy_planner.py — empathic EFE blending."""

import numpy as np
import pytest

from mindsphere.tom.empathy_planner import EmpathyPlanner
from mindsphere.tom.particle_filter import UserTypeFilter


@pytest.fixture
def planner():
    return EmpathyPlanner(lambda_empathy=0.5, beta=4.0)


@pytest.fixture
def pf():
    return UserTypeFilter(n_particles=30, seed=42)


class TestEmpathyPlanner:
    def test_lambda_zero_ignores_user(self, planner):
        """With lambda=0, G_social should equal G_system."""
        planner.lambda_empathy = 0.0
        G = planner.compute_blended_efe(
            system_efe=-2.0,
            user_felt_cost=0.8,
            reliability=1.0,
        )
        assert G == pytest.approx(-2.0)

    def test_lambda_one_full_empathy(self, planner):
        """With lambda=1 and high reliability, user cost dominates."""
        planner.lambda_empathy = 1.0
        G = planner.compute_blended_efe(
            system_efe=-2.0,
            user_felt_cost=0.8,
            reliability=1.0,
        )
        # G_user = 0.8 * 5.0 = 4.0
        assert G == pytest.approx(4.0)

    def test_low_reliability_reduces_empathy(self, planner):
        """Low reliability should reduce effective lambda."""
        planner.lambda_empathy = 1.0
        G_high_rel = planner.compute_blended_efe(-2.0, 0.8, reliability=1.0)
        G_low_rel = planner.compute_blended_efe(-2.0, 0.8, reliability=0.1)

        # Low reliability → closer to G_system
        assert abs(G_low_rel - (-2.0)) < abs(G_high_rel - (-2.0))

    def test_counterfactual_returns_valid(self, planner, pf):
        """Counterfactual should return valid comparison data."""
        gentle = {"description": "Do 2 min", "difficulty": 0.1, "duration_minutes": 2}
        push = {"description": "Do 30 min", "difficulty": 0.7, "duration_minutes": 30}

        cf = planner.compute_counterfactual(gentle, push, pf)

        assert "gentle" in cf
        assert "push" in cf
        assert cf["recommendation"] in ("gentle", "push")
        assert 0 <= cf["gentle"]["p_completion"] <= 1
        assert 0 <= cf["push"]["p_completion"] <= 1

    def test_gentle_higher_completion(self, planner, pf):
        """Gentle intervention should generally have higher completion rate."""
        gentle = {"description": "tiny", "difficulty": 0.1, "duration_minutes": 2,
                  "evaluative": 0.0, "structured": 0.3}
        push = {"description": "big", "difficulty": 0.8, "duration_minutes": 30,
                "evaluative": 0.7, "structured": 0.8}

        cf = planner.compute_counterfactual(gentle, push, pf)
        assert cf["gentle"]["p_completion"] > cf["push"]["p_completion"]

    def test_format_counterfactual_text(self, planner, pf):
        """Formatted text should contain percentages."""
        gentle = {"description": "tiny", "difficulty": 0.1, "duration_minutes": 2}
        push = {"description": "big", "difficulty": 0.7, "duration_minutes": 30}

        cf = planner.compute_counterfactual(gentle, push, pf)
        text = planner.format_counterfactual_text(cf)

        assert "%" in text
        assert "Option A" in text
        assert "Option B" in text

    def test_select_empathic_action(self, planner, pf):
        """Action selection should return a valid action name."""
        system_efes = {"action_a": -2.0, "action_b": -1.5, "action_c": -0.5}
        interventions = {
            "action_a": {"difficulty": 0.1, "duration_minutes": 2},
            "action_b": {"difficulty": 0.4, "duration_minutes": 10},
            "action_c": {"difficulty": 0.8, "duration_minutes": 30},
        }

        best_action, info = planner.select_empathic_action(
            system_efes, interventions, pf
        )
        assert best_action in system_efes
        assert "action_probs" in info
        assert "reliability" in info
