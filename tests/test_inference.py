"""Tests for core/inference.py — belief updates and EFE computation."""

import numpy as np
import pytest

from mindsphere.core.model import SphereModel
from mindsphere.core.inference import (
    update_belief,
    compute_efe_single_factor,
    compute_information_gain,
    select_action,
)
from mindsphere.core.utils import normalize


@pytest.fixture
def model():
    return SphereModel()


class TestBeliefUpdate:
    def test_update_shifts_belief(self, model):
        """Observing low answer should shift belief toward low skill."""
        prior = model.D["focus"].copy()
        A = model.A["focus"]

        # Observe answer_0 (lowest)
        posterior = update_belief(prior, 0, A)

        # Posterior should have more weight on low skill levels
        assert posterior[0] > prior[0]
        assert posterior.sum() == pytest.approx(1.0)

    def test_update_with_high_answer(self, model):
        """Observing high answer should shift belief toward high skill."""
        prior = model.D["focus"].copy()
        A = model.A["focus"]

        # Observe answer_3 (highest)
        posterior = update_belief(prior, 3, A)

        # Posterior should have more weight on high skill levels
        assert posterior[4] > prior[4]

    def test_repeated_updates_converge(self, model):
        """Repeated same observations should converge belief."""
        belief = model.D["focus"].copy()
        A = model.A["focus"]

        for _ in range(10):
            belief = update_belief(belief, 3, A)  # Always highest

        # Should be strongly peaked at high skill
        assert belief[4] > 0.5
        assert belief.sum() == pytest.approx(1.0)

    def test_belief_stays_normalized(self, model):
        """Belief should always sum to 1 after update."""
        belief = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        A = model.A["focus"]

        for obs in range(4):
            updated = update_belief(belief, obs, A)
            assert updated.sum() == pytest.approx(1.0)


class TestEFE:
    def test_efe_returns_float(self, model):
        """EFE should return a finite float."""
        belief = model.D["focus"]
        A = model.A["focus"]
        B = model.B["focus"][0]  # First action
        C = model.C["focus"]

        G = compute_efe_single_factor(belief, A, B, C)
        assert isinstance(G, float)
        assert np.isfinite(G)

    def test_epistemic_drive(self, model):
        """Higher epistemic weight should increase the epistemic component."""
        uncertain = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        A = model.A["focus"]
        B = model.B["focus"][0]
        C = model.C["focus"]

        # Same belief, different epistemic weight — higher weight should lower G
        G_low_epist = compute_efe_single_factor(uncertain, A, B, C, lambda_epist=0.0)
        G_high_epist = compute_efe_single_factor(uncertain, A, B, C, lambda_epist=5.0)

        # More epistemic weight makes G more negative (better) for uncertain beliefs
        assert G_high_epist < G_low_epist


class TestInformationGain:
    def test_ig_higher_for_uncertain(self, model):
        """Uncertain beliefs should have higher information gain."""
        A = model.A["focus"]

        uncertain = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        certain = normalize(np.array([0.01, 0.01, 0.01, 0.01, 0.96]))

        ig_uncertain = compute_information_gain(uncertain, A)
        ig_certain = compute_information_gain(certain, A)

        assert ig_uncertain > ig_certain

    def test_ig_non_negative(self, model):
        """Information gain should be non-negative."""
        for skill in model.spec.skill_factors:
            ig = compute_information_gain(model.D[skill], model.A[skill])
            assert ig >= -1e-10  # Allow small numerical errors


class TestActionSelection:
    def test_selects_valid_action(self, model):
        """Selected action should be from the valid set."""
        beliefs = model.get_initial_beliefs()
        valid = [0, 1, 2]

        action, probs, efes = select_action(beliefs, model, valid)
        assert action in valid
        assert probs.sum() == pytest.approx(1.0)
        assert len(efes) == len(valid)
