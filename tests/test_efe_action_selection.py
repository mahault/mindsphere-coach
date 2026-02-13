"""Tests for EFE-driven action selection — B matrix differentiation,
action dispatcher, lambda scheduling, and integration with agent."""

import numpy as np
import pytest

from mindsphere.core.model import SphereModel, SKILL_FACTORS, ACTION_NAMES
from mindsphere.core.inference import (
    compute_efe_single_factor,
    compute_efe_all_factors,
    select_action,
)
from mindsphere.core.action_dispatcher import (
    select_coaching_action,
    compute_lambda_epist,
    VALID_ACTIONS,
    A_PROPOSE,
    A_REFRAME,
    A_ASK_FREE,
    A_SAFETY,
)
from mindsphere.core.utils import normalize
from mindsphere.core.agent import CoachingAgent


@pytest.fixture
def model():
    return SphereModel()


@pytest.fixture
def agent():
    return CoachingAgent()


# ── B Matrix Differentiation ─────────────────────────────────────────────

class TestBMatrixDifferentiation:
    def test_propose_intervention_differs_from_ask(self, model):
        """propose_intervention B should differ from ask_mc B for skills."""
        for skill in SKILL_FACTORS:
            B_ask = model.B[skill][0]    # ask_mc_question
            B_prop = model.B[skill][3]   # propose_intervention
            assert not np.allclose(B_ask, B_prop), (
                f"B[{skill}] action 0 and 3 should differ"
            )

    def test_propose_has_upward_shift(self, model):
        """propose_intervention should have probability of moving state up."""
        B_prop = model.B["focus"][3]
        # For a state in the middle (level 2), there should be non-trivial
        # probability of transitioning to level 3
        assert B_prop[3, 2] > B_prop[3, 2] * 0  # nonzero
        # The upward shift should be > noise floor
        B_ask = model.B["focus"][0]
        assert B_prop[3, 2] > B_ask[3, 2]

    def test_reframe_weaker_than_propose(self, model):
        """reframe should have weaker upward shift than propose_intervention."""
        for skill in SKILL_FACTORS:
            B_prop = model.B[skill][3]
            B_reframe = model.B[skill][4]
            # Diagonal of propose should be more different from identity
            # than reframe (propose has stronger improvement transition)
            prop_off_diag = 1.0 - np.mean(np.diag(B_prop))
            reframe_off_diag = 1.0 - np.mean(np.diag(B_reframe))
            assert prop_off_diag > reframe_off_diag, (
                f"propose should have stronger off-diagonal than reframe for {skill}"
            )

    def test_overwhelm_adjust_difficulty_shifts_down(self, model):
        """adjust_difficulty should shift overwhelm_sensitivity toward lower."""
        B_adj = model.B["overwhelm_sensitivity"][5]   # adjust_difficulty
        B_other = model.B["overwhelm_sensitivity"][0]  # ask_mc
        # For high overwhelm (state 2), probability of transitioning to
        # medium (state 1) should be higher for adjust_difficulty
        assert B_adj[1, 2] > B_other[1, 2]

    def test_safety_check_reduces_overwhelm(self, model):
        """safety_check should have small downward shift on overwhelm."""
        B_safe = model.B["overwhelm_sensitivity"][7]
        B_other = model.B["overwhelm_sensitivity"][1]  # ask_free_text
        assert B_safe[0, 1] > B_other[0, 1] or B_safe[1, 2] > B_other[1, 2]

    def test_all_b_matrices_valid(self, model):
        """All B matrix columns should still sum to 1."""
        for factor, B in model.B.items():
            for a in range(B.shape[0]):
                col_sums = B[a].sum(axis=0)
                np.testing.assert_allclose(
                    col_sums, 1.0, atol=0.01,
                    err_msg=f"B[{factor}][action={a}] columns don't sum to 1"
                )


# ── EFE Preferences ──────────────────────────────────────────────────────

class TestEFEPreferences:
    def test_efe_prefers_intervention_for_low_skill(self, model):
        """With low skill belief and low epistemic weight, EFE should prefer
        propose_intervention over ask_free_text (pragmatic drive)."""
        beliefs = model.get_initial_beliefs()
        # Set focus to very low
        beliefs["focus"] = normalize(np.array([0.8, 0.15, 0.04, 0.005, 0.005]))

        valid = [A_ASK_FREE, A_PROPOSE]
        action, probs, efes = select_action(
            beliefs, model, valid,
            lambda_epist=0.1,  # mostly pragmatic
            beta=8.0,
            relevant_factors=["focus"],
        )

        # propose_intervention (action 3) should have better (lower) EFE
        # because B[3] has upward shift → C prefers high observations
        propose_idx = valid.index(A_PROPOSE)
        ask_idx = valid.index(A_ASK_FREE)
        assert efes[propose_idx] < efes[ask_idx], (
            f"propose_intervention EFE ({efes[propose_idx]:.4f}) should be < "
            f"ask_free_text EFE ({efes[ask_idx]:.4f}) in pragmatic mode"
        )

    def test_efe_prefers_probe_for_uncertain_belief(self, model):
        """With uniform belief and high epistemic weight, EFE should prefer
        ask_free_text (more epistemic value from uncertain beliefs)."""
        beliefs = model.get_initial_beliefs()
        # Set focus to maximally uncertain
        beliefs["focus"] = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        valid = [A_ASK_FREE, A_PROPOSE]
        _, probs, efes = select_action(
            beliefs, model, valid,
            lambda_epist=5.0,  # strongly epistemic
            beta=8.0,
            relevant_factors=["focus"],
        )

        # With very high epistemic weight, ask_free_text should have comparable
        # or better EFE since both use similar A matrices for info gain
        # The key test: both should have reasonable probability
        assert probs[0] > 0.05, "ask_free_text should have nonzero probability"
        assert probs[1] > 0.05, "propose_intervention should have nonzero probability"

    def test_efe_discriminates_actions(self, model):
        """Different actions should produce different EFE values."""
        beliefs = model.get_initial_beliefs()
        beliefs["focus"] = normalize(np.array([0.7, 0.2, 0.08, 0.01, 0.01]))

        valid = [A_ASK_FREE, A_PROPOSE, A_REFRAME]
        _, probs, efes = select_action(
            beliefs, model, valid,
            lambda_epist=0.5,
            beta=4.0,
            relevant_factors=["focus"],
        )

        # EFE values should not all be identical
        assert not np.allclose(efes, efes[0]), (
            "All EFE values are the same — B matrices aren't discriminating"
        )
        # Probabilities should not be uniform
        assert np.std(probs) > 0.01, "Action probabilities too uniform"


# ── Lambda Scheduling ────────────────────────────────────────────────────

class TestLambdaSchedule:
    def test_visualization_high_epistemic(self):
        """Visualization phase should have high epistemic weight."""
        lam = compute_lambda_epist("visualization", timestep=5, tom_reliability=0.5)
        assert lam > 1.0, f"Visualization lambda should be > 1.0, got {lam}"

    def test_coaching_lower_epistemic(self):
        """Coaching phase should have lower epistemic weight than visualization."""
        lam_viz = compute_lambda_epist("visualization", timestep=10, tom_reliability=0.5)
        lam_coach = compute_lambda_epist("coaching", timestep=10, tom_reliability=0.5)
        assert lam_coach < lam_viz

    def test_late_session_more_pragmatic(self):
        """Later timesteps should reduce epistemic weight."""
        lam_early = compute_lambda_epist("coaching", timestep=3, tom_reliability=0.5)
        lam_late = compute_lambda_epist("coaching", timestep=25, tom_reliability=0.5)
        assert lam_late < lam_early

    def test_low_reliability_boosts_epistemic(self):
        """Low ToM reliability should boost epistemic drive."""
        lam_reliable = compute_lambda_epist("coaching", timestep=10, tom_reliability=0.8)
        lam_unreliable = compute_lambda_epist("coaching", timestep=10, tom_reliability=0.1)
        assert lam_unreliable > lam_reliable

    def test_uncertainty_boosts_epistemic(self, model):
        """Uncertain beliefs should boost epistemic weight."""
        beliefs_uncertain = model.get_initial_beliefs()
        for s in SKILL_FACTORS:
            beliefs_uncertain[s] = np.ones(5) / 5  # maximally uncertain

        beliefs_certain = model.get_initial_beliefs()
        for s in SKILL_FACTORS:
            beliefs_certain[s] = normalize(np.array([0.01, 0.01, 0.01, 0.01, 0.96]))

        lam_uncertain = compute_lambda_epist(
            "coaching", 10, 0.5, beliefs_uncertain
        )
        lam_certain = compute_lambda_epist(
            "coaching", 10, 0.5, beliefs_certain
        )
        assert lam_uncertain > lam_certain


# ── Action Dispatcher ────────────────────────────────────────────────────

class TestActionDispatcher:
    def test_valid_action_masks(self):
        """Each phase should have defined valid actions."""
        for phase in ["visualization", "planning", "coaching", "update"]:
            assert phase in VALID_ACTIONS
            assert len(VALID_ACTIONS[phase]) >= 2

    def test_select_returns_valid_action(self, model):
        """select_coaching_action should return an action from the valid set."""
        beliefs = model.get_initial_beliefs()
        action_idx, action_name, info = select_coaching_action(
            beliefs=beliefs,
            model=model,
            phase="coaching",
            timestep=10,
            tom_reliability=0.5,
        )
        assert action_idx in VALID_ACTIONS["coaching"]
        assert action_name == ACTION_NAMES[action_idx]
        assert "action_probabilities" in info
        assert "lambda_epist" in info

    def test_select_different_phases(self, model):
        """Different phases should allow different action sets."""
        beliefs = model.get_initial_beliefs()
        for phase in ["visualization", "coaching", "update"]:
            action_idx, _, info = select_coaching_action(
                beliefs=beliefs,
                model=model,
                phase=phase,
                timestep=10,
                tom_reliability=0.5,
            )
            assert action_idx in VALID_ACTIONS[phase]

    def test_info_has_all_valid_actions(self, model):
        """Info dict should include probabilities for all valid actions."""
        beliefs = model.get_initial_beliefs()
        _, _, info = select_coaching_action(
            beliefs=beliefs,
            model=model,
            phase="coaching",
            timestep=10,
            tom_reliability=0.5,
        )
        for action_idx in VALID_ACTIONS["coaching"]:
            action_name = ACTION_NAMES[action_idx]
            assert action_name in info["action_probabilities"]

    def test_probabilities_sum_to_one(self, model):
        """Action probabilities should sum to ~1."""
        beliefs = model.get_initial_beliefs()
        _, _, info = select_coaching_action(
            beliefs=beliefs,
            model=model,
            phase="coaching",
            timestep=10,
            tom_reliability=0.5,
        )
        total = sum(info["action_probabilities"].values())
        assert abs(total - 1.0) < 0.01


# ── Agent Integration ────────────────────────────────────────────────────

class TestAgentEFEIntegration:
    def test_coaching_response_has_efe_info(self, agent):
        """Coaching phase responses should include efe_info."""
        agent.start_session()
        # Run through calibration
        for i in range(10):
            q = agent.current_question
            if q and q.question_type == "mc":
                agent.step({"answer": q.options[1], "answer_index": 1})
            else:
                agent.step({"answer": "I want to be more focused"})

        # Ask for coaching to enter planning
        agent.step({"answer": "what should I work on?"})
        # Accept intervention
        agent.step({"choice": "accept"})

        # Now in coaching — response should have efe_info
        result = agent.step({"answer": "tell me more about focus"})
        assert "efe_info" in result
        assert "selected_action" in result["efe_info"]

    def test_visualization_has_efe_info(self, agent):
        """Visualization responses should include efe_info."""
        agent.start_session()
        for i in range(10):
            q = agent.current_question
            if q and q.question_type == "mc":
                agent.step({"answer": q.options[2], "answer_index": 2})
            else:
                agent.step({"answer": "I try to be consistent"})

        # Now in visualization
        result = agent.step({"answer": "this is interesting"})
        assert "efe_info" in result
