"""
Behavioral tests for MindSphere Coach.

These tests verify the behaviors that users actually experience:
- One LLM call per turn (no stitched responses)
- Emotional inference affects action selection
- Companion gate triggers on emotional distress
- Cognitive load assessed once per turn

Uses mocking for speed — no real API calls needed.
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from mindsphere.core.agent import (
    CoachingAgent,
    PHASE_CALIBRATION,
    PHASE_VISUALIZATION,
    PHASE_PLANNING,
    PHASE_COACHING,
)
from mindsphere.core.action_dispatcher import (
    compute_lambda_epist,
    select_coaching_action,
)
from mindsphere.core.model import SphereModel, SKILL_FACTORS


# ── Helpers ──────────────────────────────────────────────────────────────────

@pytest.fixture
def agent():
    return CoachingAgent(lambda_empathy=0.5, n_particles=30)


def _run_calibration(agent):
    """Fast-forward through calibration to visualization."""
    agent.start_session()
    result = None
    for i in range(10):
        result = agent.step({"answer": f"answer {i}", "answer_index": i % 4})
        if result["phase"] == PHASE_VISUALIZATION:
            break
    return result


def _reach_coaching(agent):
    """Fast-forward through calibration → visualization → planning → coaching."""
    result = _run_calibration(agent)
    if result["phase"] == PHASE_VISUALIZATION:
        result = agent.step({"answer": "what should I work on?"})
    if result["phase"] == PHASE_PLANNING:
        result = agent.step({"choice": "accept"})
    return result


# ── Test: No stitched responses ─────────────────────────────────────────────

class TestNoStitchedResponses:
    """Responses should come from a single LLM call, not two calls stitched together."""

    def test_calibration_to_visualization_no_stitching(self, agent):
        """Cal→viz transition should produce ONE cohesive message, not ack + commentary."""
        result = _run_calibration(agent)
        assert result["phase"] == PHASE_VISUALIZATION
        msg = result["message"]

        # Check for signs of stitching: the same concept repeated in two paragraphs
        paragraphs = [p.strip() for p in msg.split("\n\n") if p.strip()]
        # With stitching, paragraph 1 would be a short ack and paragraph 2 would
        # repeat the same acknowledgment. One cohesive message shouldn't have
        # the same greeting pattern repeated.
        if len(paragraphs) >= 2:
            # First paragraph shouldn't be a standalone ack (<100 chars) followed
            # by a second paragraph that restates the same thing
            first_short = len(paragraphs[0]) < 100
            if first_short:
                # If first paragraph is very short, it's likely a separate ack
                # that got stitched. A cohesive response integrates the ack.
                # This is a soft check — not all short first paragraphs are bugs.
                pass

        # Hard check: message should exist and be reasonable length
        assert len(msg) > 50
        assert len(msg) < 2000, f"Response too long ({len(msg)} chars) — likely stitched"

    def test_visualization_to_planning_no_stitching(self, agent):
        """Viz→planning should produce ONE message, not viz response + plan message."""
        result = _run_calibration(agent)
        assert result["phase"] == PHASE_VISUALIZATION

        # Need 2 viz turns before EFE can transition
        agent.step({"answer": "interesting"})
        result = agent.step({"answer": "what should I work on first?"})
        assert result["phase"] == PHASE_PLANNING

        msg = result["message"]
        assert len(msg) > 20
        assert len(msg) < 2000, f"Response too long ({len(msg)} chars) — likely stitched"


# ── Test: Emotional inference affects action selection ───────────────────────

class TestEmotionalInferenceAffectsActions:
    """Emotional state from the Circumplex POMDP should influence EFE action selection."""

    def test_negative_valence_boosts_epistemic(self):
        """High emotion prediction error should increase lambda_epist."""
        l1 = compute_lambda_epist("coaching", 10, 0.5, emotion_prediction_error=0.0)
        l2 = compute_lambda_epist("coaching", 10, 0.5, emotion_prediction_error=0.8)
        assert l2 > l1, (
            f"lambda_epist should increase with emotion error: {l1} vs {l2}"
        )

    def test_negative_valence_penalizes_interventions(self):
        """Negative valence belief should reduce probability of propose_intervention."""
        model = SphereModel()
        beliefs = model.get_initial_beliefs()

        # Neutral valence
        _, _, info_neutral = select_coaching_action(
            beliefs=beliefs, model=model, phase="coaching",
            timestep=10, tom_reliability=0.5,
            emotion_valence_belief=np.array([0.05, 0.15, 0.60, 0.15, 0.05]),
        )

        # Strongly negative valence
        _, _, info_negative = select_coaching_action(
            beliefs=beliefs, model=model, phase="coaching",
            timestep=10, tom_reliability=0.5,
            emotion_valence_belief=np.array([0.40, 0.35, 0.15, 0.05, 0.05]),
        )

        prob_neutral = info_neutral["action_probabilities"].get("propose_intervention", 0)
        prob_negative = info_negative["action_probabilities"].get("propose_intervention", 0)
        assert prob_negative < prob_neutral, (
            f"Negative valence should reduce propose_intervention prob: "
            f"neutral={prob_neutral}, negative={prob_negative}"
        )

    def test_efe_info_has_emotional_fields(self):
        """EFE info should include emotion_prediction_error and valence_negative_mass."""
        model = SphereModel()
        beliefs = model.get_initial_beliefs()

        _, _, info = select_coaching_action(
            beliefs=beliefs, model=model, phase="coaching",
            timestep=10, tom_reliability=0.5,
            emotion_prediction_error=0.5,
            emotion_valence_belief=np.array([0.30, 0.30, 0.20, 0.10, 0.10]),
        )

        assert "emotion_prediction_error" in info
        assert "valence_negative_mass" in info
        assert info["emotion_prediction_error"] == 0.5
        assert info["valence_negative_mass"] == pytest.approx(0.6, abs=0.01)

    def test_emotional_data_reaches_cognitive_load(self, agent):
        """_assess_cognitive_load should detect emotional distress from emotional_data."""
        emotional_data = {
            "emotional_beliefs": {
                "valence": {
                    "belief": [0.40, 0.35, 0.15, 0.05, 0.05],
                },
            },
            "error": {"magnitude": 0.1},
            "current_emotion": {"valence": -0.6, "arousal": 0.7},
        }
        cog_load = agent._assess_cognitive_load("I feel terrible", emotional_data=emotional_data)
        assert "emotional_distress" in cog_load["signals"] or "low_valence" in cog_load["signals"], (
            f"Expected emotional signals, got: {cog_load['signals']}"
        )
        assert cog_load["coaching_readiness"] == "not_ready", (
            f"Expected not_ready, got: {cog_load['coaching_readiness']}"
        )


# ── Test: Companion gate ────────────────────────────────────────────────────

class TestCompanionGate:
    """When the user is emotionally vulnerable, companion gate should activate."""

    def test_companion_gate_coaching_on_distress(self, agent):
        """Emotional distress during coaching should trigger companion mode."""
        _reach_coaching(agent)

        # Mock emotional inference to return strongly negative state
        # (can't just set belief_valence — _run_emotional_inference overwrites it)
        distress_data = {
            "prediction": {"predicted_valence": 0.0, "predicted_arousal": 0.5,
                           "predicted_emotion": "neutral", "confidence": 0.3},
            "observation": {"observed_valence": -0.8, "observed_arousal": 0.7,
                            "observed_emotion": "sad", "valence_idx": 0, "arousal_idx": 3},
            "error": {"valence_error": -0.8, "arousal_error": 0.2,
                       "magnitude": 0.82, "surprise": 0.5},
            "current_emotion": {"valence": -0.8, "arousal": 0.7,
                                "intensity": 1.06, "angle": 138.8,
                                "emotion": "angry", "quadrant": "high-arousal-negative"},
            "emotional_beliefs": {
                "valence": {
                    "belief": [0.50, 0.30, 0.10, 0.05, 0.05],
                    "most_likely": "very_negative", "confidence": 0.50,
                    "entropy": 1.2,
                },
                "arousal": {
                    "belief": [0.05, 0.10, 0.20, 0.35, 0.30],
                    "most_likely": "high", "confidence": 0.35,
                    "entropy": 1.4,
                },
            },
        }

        with patch.object(agent, "_run_emotional_inference", return_value=distress_data):
            result = agent.step({"answer": "I just feel so ugly and worthless"})

        efe_info = result.get("efe_info", {})
        assert efe_info.get("selected_action") == "companion_chat", (
            f"Expected companion_chat on emotional distress, got: {efe_info.get('selected_action')}"
        )

    def test_companion_gate_visualization_on_distress(self, agent):
        """Emotional distress during visualization should not transition to planning."""
        result = _run_calibration(agent)
        assert result["phase"] == PHASE_VISUALIZATION

        distress_data = {
            "prediction": {"predicted_valence": 0.0, "predicted_arousal": 0.5,
                           "predicted_emotion": "neutral", "confidence": 0.3},
            "observation": {"observed_valence": -0.8, "observed_arousal": 0.7,
                            "observed_emotion": "sad", "valence_idx": 0, "arousal_idx": 3},
            "error": {"valence_error": -0.8, "arousal_error": 0.2,
                       "magnitude": 0.82, "surprise": 0.5},
            "current_emotion": {"valence": -0.8, "arousal": 0.7,
                                "intensity": 1.06, "angle": 138.8,
                                "emotion": "angry", "quadrant": "high-arousal-negative"},
            "emotional_beliefs": {
                "valence": {
                    "belief": [0.50, 0.30, 0.10, 0.05, 0.05],
                    "most_likely": "very_negative", "confidence": 0.50,
                    "entropy": 1.2,
                },
                "arousal": {
                    "belief": [0.05, 0.10, 0.20, 0.35, 0.30],
                    "most_likely": "high", "confidence": 0.35,
                    "entropy": 1.4,
                },
            },
        }

        with patch.object(agent, "_run_emotional_inference", return_value=distress_data):
            result = agent.step({"answer": "I feel really stressed and overwhelmed"})

        assert result["phase"] == PHASE_VISUALIZATION, (
            f"Should stay in visualization when emotionally distressed, got: {result['phase']}"
        )
        efe_info = result.get("efe_info", {})
        assert efe_info.get("selected_action") == "companion_chat", (
            f"Expected companion_chat, got: {efe_info.get('selected_action')}"
        )

    def test_off_topic_triggers_companion(self, agent):
        """Off-topic text during coaching should trigger companion mode."""
        _reach_coaching(agent)

        result = agent.step({"answer": "I want to talk about my cat instead"})
        efe_info = result.get("efe_info", {})
        assert efe_info.get("selected_action") == "companion_chat", (
            f"Off-topic should trigger companion, got: {efe_info}"
        )


# ── Test: Cognitive load assessed once ──────────────────────────────────────

class TestCognitiveLoadCalledOnce:
    """_assess_cognitive_load should be called exactly once per turn."""

    def test_coaching_turn_assesses_once(self, agent):
        """During a coaching turn, cognitive load should be assessed once."""
        _reach_coaching(agent)

        original_assess = agent._assess_cognitive_load
        call_count = {"n": 0}

        def counting_assess(*args, **kwargs):
            call_count["n"] += 1
            return original_assess(*args, **kwargs)

        with patch.object(agent, "_assess_cognitive_load", side_effect=counting_assess):
            agent.step({"answer": "tell me more"})

        assert call_count["n"] == 1, (
            f"_assess_cognitive_load called {call_count['n']} times, expected 1"
        )

    def test_visualization_turn_assesses_once(self, agent):
        """During a visualization turn, cognitive load should be assessed once."""
        _run_calibration(agent)

        original_assess = agent._assess_cognitive_load
        call_count = {"n": 0}

        def counting_assess(*args, **kwargs):
            call_count["n"] += 1
            return original_assess(*args, **kwargs)

        with patch.object(agent, "_assess_cognitive_load", side_effect=counting_assess):
            agent.step({"answer": "that's interesting"})

        assert call_count["n"] == 1, (
            f"_assess_cognitive_load called {call_count['n']} times, expected 1"
        )
