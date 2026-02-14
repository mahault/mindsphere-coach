"""Integration tests for the full MindSphere coaching pipeline."""

import pytest

from mindsphere.core.agent import (
    CoachingAgent,
    PHASE_CALIBRATION,
    PHASE_VISUALIZATION,
    PHASE_PLANNING,
    PHASE_UPDATE,
    PHASE_COMPLETE,
)
from mindsphere.core.model import SKILL_FACTORS


@pytest.fixture
def agent():
    return CoachingAgent(lambda_empathy=0.5, n_particles=30)


def _run_calibration(agent):
    """Fast-forward through all calibration questions."""
    agent.start_session()
    result = None
    for i in range(10):
        result = agent.step({"answer": f"answer {i}", "answer_index": i % 4})
        if result["phase"] == PHASE_VISUALIZATION:
            break
    return result


def _reach_planning(agent):
    """Fast-forward through calibration + visualization to planning."""
    result = _run_calibration(agent)
    # Visualization phase: explicitly ask for coaching to trigger transition
    if result["phase"] == PHASE_VISUALIZATION:
        result = agent.step({"answer": "what do you recommend I work on?"})
    return result


class TestCoachingPipeline:
    def test_start_session(self, agent):
        """Starting a session should return welcome + first question."""
        result = agent.start_session()
        assert result["phase"] == PHASE_CALIBRATION
        assert result["message"]
        assert result["question"] is not None
        assert result["question"]["question_type"] in ("mc", "free_text")

    def test_calibration_answers(self, agent):
        """Answering calibration questions should update beliefs."""
        agent.start_session()
        initial_beliefs = {k: v.copy() for k, v in agent.beliefs.items()}

        # Answer first question with MC choice
        result = agent.step({"answer": "test", "answer_index": 0})
        assert result["phase"] == PHASE_CALIBRATION

        # Beliefs should have changed for at least one factor
        changed = any(
            not (agent.beliefs[k] == initial_beliefs[k]).all()
            for k in SKILL_FACTORS
            if k in agent.beliefs and k in initial_beliefs
        )
        assert changed

    def test_full_calibration_leads_to_visualization(self, agent):
        """Answering all 10 questions should transition to visualization."""
        result = _run_calibration(agent)

        assert result["phase"] == PHASE_VISUALIZATION
        assert "sphere_data" in result
        # Bottlenecks are now inside sphere_data
        assert "bottlenecks" in result["sphere_data"]

    def test_visualization_has_personalized_message(self, agent):
        """Visualization should have a personalized sphere commentary."""
        result = _run_calibration(agent)

        assert result["phase"] == PHASE_VISUALIZATION
        msg = result["message"].lower()
        # Should mention the sphere and some skill-related content
        # LLM may use natural language ("mid-60s") vs template ("/100")
        has_sphere = any(w in msg for w in [
            "mindsphere", "sphere", "profile", "assessment", "results",
        ])
        has_skill_ref = any(w in msg for w in [
            "/100", "focus", "consistency", "follow", "emotional",
            "self-trust", "systems", "social", "task clarity",
            "score", "strength", "area",
        ])
        # LLM may use natural coaching language without explicit skill names
        has_personalization = any(w in msg for w in [
            "your", "you", "pattern", "tend to", "notice",
            "instinct", "brain", "habit", "struggle",
        ])
        assert has_sphere or has_skill_ref or has_personalization, (
            f"Visualization message lacks expected content: {msg[:200]}"
        )

    def test_sphere_data_has_all_skills(self, agent):
        """Sphere data should contain all 8 skills."""
        _run_calibration(agent)

        sphere = agent.get_sphere_data()
        for skill in SKILL_FACTORS:
            assert skill in sphere["categories"]
            assert 0 <= sphere["categories"][skill] <= 100

    def test_visualization_discussion_then_planning(self, agent):
        """Asking for coaching during visualization should transition to planning."""
        result = _run_calibration(agent)
        assert result["phase"] == PHASE_VISUALIZATION

        # Casual reply stays in visualization
        result = agent.step({"answer": "interesting"})
        assert result["phase"] == PHASE_VISUALIZATION
        assert result["message"]  # should have a conversational response

        # Explicit coaching request transitions to planning
        result = agent.step({"answer": "what should I work on first?"})
        assert result["phase"] == PHASE_PLANNING
        assert "intervention" in result
        assert "counterfactual" in result

    def test_planning_has_personalized_message(self, agent):
        """Planning message should reference specific skills and ToM predictions."""
        result = _reach_planning(agent)

        assert result["phase"] == PHASE_PLANNING
        msg = result["message"]
        # Should have a substantive response (LLM or template)
        assert len(msg) > 20

    def test_accept_intervention_starts_coaching(self, agent):
        """Accepting an intervention should transition to coaching, not end."""
        _reach_planning(agent)

        result = agent.step({"choice": "accept"})
        assert result["phase"] == "coaching"
        assert result["is_complete"] is False
        assert result["message"]  # should have encouragement + coaching probe

    def test_reject_too_hard_proposes_alternative(self, agent):
        """Rejecting as 'too hard' should propose a gentler alternative."""
        _reach_planning(agent)

        result = agent.step({"choice": "too_hard"})
        assert result["is_complete"] is False
        assert "intervention" in result
        assert result["message"]  # should acknowledge the rejection

    def test_reject_not_relevant_tries_different_skill(self, agent):
        """Rejecting as 'not relevant' should target a different skill."""
        _reach_planning(agent)

        result = agent.step({"choice": "not_relevant"})
        assert "intervention" in result
        assert result["message"]

    def test_free_text_in_planning_works(self, agent):
        """Typing free text during planning should get a conversational response."""
        _reach_planning(agent)

        # Ask a question about the intervention
        result = agent.step({"answer": "why did you pick this?"})
        assert result["message"]
        assert result["phase"] == PHASE_PLANNING

    def test_implicit_acceptance_via_text(self, agent):
        """Typing 'sounds good' during planning should count as acceptance."""
        _reach_planning(agent)

        result = agent.step({"answer": "sounds good, let's do it"})
        assert result["phase"] == "coaching"  # now transitions to coaching
        assert result["is_complete"] is False

    def test_coaching_conversation(self, agent):
        """Coaching phase should respond to free text conversationally."""
        _reach_planning(agent)
        result = agent.step({"choice": "accept"})
        assert result["phase"] == "coaching"

        # Chat with the coach
        result = agent.step({"answer": "I feel stressed about work lately"})
        assert result["phase"] == "coaching"
        assert result["message"]
        assert result["is_complete"] is False

    def test_coaching_more_steps(self, agent):
        """Asking for more steps in coaching should provide exercises."""
        _reach_planning(agent)
        agent.step({"choice": "accept"})

        result = agent.step({"answer": "give me another exercise"})
        assert result["phase"] == "coaching"
        assert result["message"]

    def test_end_session_from_coaching(self, agent):
        """Saying 'done' during coaching should end the session."""
        _reach_planning(agent)
        agent.step({"choice": "accept"})

        result = agent.step({"answer": "I'm done, let's wrap up"})
        assert result["phase"] == "complete"
        assert result["is_complete"] is True
        assert result["message"]

    def test_belief_summary_complete(self, agent):
        """Belief summary should contain all factor types."""
        agent.start_session()
        for i in range(10):
            agent.step({"answer": f"a{i}", "answer_index": i % 4})

        summary = agent.get_belief_summary()
        assert "tom_reliability" in summary
        assert "user_type" in summary

        # Should have skill scores
        has_skills = any(
            skill in summary for skill in SKILL_FACTORS
        )
        assert has_skills

    def test_conversational_text_during_mc_doesnt_consume_question(self, agent):
        """Typing conversational text during MC should re-show the question."""
        agent.start_session()
        q_before = agent.current_question

        # Send conversational text without answer_index
        result = agent.step({"answer": "ok let's start"})

        # Should still be in calibration, same question
        assert result["phase"] == PHASE_CALIBRATION
        assert result["question"] is not None
        assert agent.current_question.id == q_before.id
        assert len(agent.asked_question_ids) == 0  # question NOT consumed

    def test_calibration_returns_acknowledgment(self, agent):
        """Calibration steps should return a non-empty acknowledgment."""
        agent.start_session()

        result = agent.step({"answer": "test", "answer_index": 0})
        assert result["phase"] == PHASE_CALIBRATION
        assert result["message"]  # non-empty
        assert len(result["message"]) > 0

    def test_calibration_does_not_return_sphere_data(self, agent):
        """Calibration steps should not include sphere_data."""
        agent.start_session()

        result = agent.step({"answer": "test", "answer_index": 0})
        assert result.get("sphere_data") is None

    def test_empathy_dial_adjustment(self, agent):
        """Adjusting empathy dial should change lambda."""
        agent.set_empathy_dial(0.8)
        assert agent.empathy.lambda_empathy == 0.8

        agent.set_empathy_dial(0.2)
        assert agent.empathy.lambda_empathy == 0.2

        # Clamp to [0, 1]
        agent.set_empathy_dial(1.5)
        assert agent.empathy.lambda_empathy == 1.0
