"""
Tests for the Circumplex Emotional State model.

Tests the POMDP-based emotion inference engine:
- Core computations (entropy, utility, valence)
- EmotionalState circumplex mapping
- A/B/D matrix validity
- Predict-observe-update loop
- Prediction error computation
- Emotional trajectory tracking
- Agent-level emotional inference integration
"""

import numpy as np
import pytest

from mindsphere.core.emotional_state import (
    compute_belief_entropy,
    compute_utility,
    compute_expected_utility,
    compute_valence,
    EmotionalState,
    EmotionEngine,
    EmotionalPrediction,
    EmotionalObservation,
    PredictionError,
    build_emotion_A_matrix,
    build_emotion_B_matrix,
    build_emotion_D,
    VALENCE_OBS_LEVELS,
    AROUSAL_OBS_LEVELS,
    EMOTION_SECTORS,
)


class TestCoreComputations:
    """Test the fundamental math from Pattisapu & Albarracin (2024)."""

    def test_entropy_uniform_is_max(self):
        """Uniform distribution should have maximum entropy."""
        uniform = np.ones(5) / 5
        peaked = np.array([0.01, 0.01, 0.96, 0.01, 0.01])
        assert compute_belief_entropy(uniform) > compute_belief_entropy(peaked)

    def test_entropy_peaked_is_low(self):
        """Peaked distribution should have low entropy."""
        peaked = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        entropy = compute_belief_entropy(peaked)
        assert entropy < 0.1

    def test_entropy_non_negative(self):
        """Entropy should always be non-negative."""
        for _ in range(20):
            qs = np.random.dirichlet(np.ones(5))
            assert compute_belief_entropy(qs) >= 0

    def test_utility_with_log_preferences(self):
        """Utility with log preferences (all <= 0) is just the value at the index."""
        C = np.array([-2.0, -1.0, -0.5, -0.1, 0.0])
        assert compute_utility(0, C) == -2.0
        assert compute_utility(4, C) == 0.0

    def test_valence_positive_for_good_surprise(self):
        """Valence should be positive when outcome is better than expected."""
        qs = np.array([0.4, 0.3, 0.2, 0.05, 0.05])  # expect negative
        A = build_emotion_A_matrix()
        C = np.array([-2.0, -0.5, 0.0, 0.5, 1.0])
        # Observe very positive (index 4) when expecting negative
        valence = compute_valence(4, qs, A, C)
        assert valence > 0, "Better-than-expected should give positive valence"

    def test_valence_negative_for_bad_surprise(self):
        """Valence should be negative when outcome is worse than expected."""
        qs = np.array([0.05, 0.05, 0.2, 0.3, 0.4])  # expect positive
        A = build_emotion_A_matrix()
        C = np.array([-2.0, -0.5, 0.0, 0.5, 1.0])
        # Observe very negative (index 0) when expecting positive
        valence = compute_valence(0, qs, A, C)
        assert valence < 0, "Worse-than-expected should give negative valence"


class TestEmotionalState:
    """Test the EmotionalState data class and circumplex mapping."""

    def test_relaxed_quadrant(self):
        """Positive valence, negative arousal → low-arousal-positive (relaxed) region."""
        state = EmotionalState(arousal=-0.2, valence=0.8)
        assert state.quadrant() == "low-arousal-positive"

    def test_angry_quadrant(self):
        """Negative valence, high arousal → angry/anxious region."""
        state = EmotionalState(arousal=0.8, valence=-0.5)
        assert state.quadrant() == "high-arousal-negative"

    def test_intensity_calculation(self):
        """Intensity should be sqrt(arousal^2 + valence^2)."""
        state = EmotionalState(arousal=0.3, valence=0.4)
        expected = np.sqrt(0.3**2 + 0.4**2)
        assert abs(state.intensity - expected) < 1e-6

    def test_to_dict_has_required_keys(self):
        """to_dict should contain all required keys."""
        state = EmotionalState(arousal=0.5, valence=0.3)
        d = state.to_dict()
        assert "arousal" in d
        assert "valence" in d
        assert "intensity" in d
        assert "angle" in d
        assert "emotion" in d
        assert "quadrant" in d

    def test_all_eight_emotions_reachable(self):
        """All 8 emotion labels should be reachable via different angles."""
        found = set()
        for angle_deg in range(0, 360, 5):
            rad = np.radians(angle_deg)
            v = np.cos(rad)
            a = np.sin(rad)
            state = EmotionalState(arousal=a, valence=v)
            found.add(state.emotion_label())
        assert len(found) == 8, f"Expected 8 emotions, got {len(found)}: {found}"


class TestPOMDPMatrices:
    """Test the POMDP matrices for emotional factors."""

    def test_A_matrix_columns_sum_to_one(self):
        """A-matrix columns should sum to 1 (valid probability distributions)."""
        A = build_emotion_A_matrix()
        for col in range(A.shape[1]):
            assert abs(A[:, col].sum() - 1.0) < 1e-10

    def test_A_matrix_shape(self):
        """A-matrix should be 5x5 (5 obs levels x 5 state levels)."""
        A = build_emotion_A_matrix()
        assert A.shape == (5, 5)

    def test_A_matrix_diagonal_dominant(self):
        """Diagonal should be the largest value in each column."""
        A = build_emotion_A_matrix()
        for col in range(A.shape[1]):
            assert A[col, col] == A[:, col].max()

    def test_B_matrix_columns_sum_to_one(self):
        """B-matrix columns should sum to 1."""
        B = build_emotion_B_matrix()
        for col in range(B.shape[1]):
            assert abs(B[:, col].sum() - 1.0) < 1e-10

    def test_B_matrix_inertia(self):
        """B-matrix diagonal should have the largest values (inertia)."""
        B = build_emotion_B_matrix()
        for s in range(B.shape[0]):
            assert B[s, s] == B[:, s].max()

    def test_D_prior_sums_to_one(self):
        """Initial prior D should sum to 1."""
        D = build_emotion_D()
        assert abs(D.sum() - 1.0) < 1e-10

    def test_D_prior_favors_neutral(self):
        """Initial prior should favor neutral state (index 2)."""
        D = build_emotion_D()
        assert D[2] == D.max()


class TestEmotionEngine:
    """Test the predict-observe-update engine."""

    @pytest.fixture
    def engine(self):
        return EmotionEngine()

    def test_predict_returns_prediction(self, engine):
        """Predict should return an EmotionalPrediction."""
        pred = engine.predict(
            belief_entropies={"focus": 1.0, "consistency": 0.5},
            tom_felt_cost=0.3,
            tom_p_accept=0.6,
            reliability=0.5,
        )
        assert isinstance(pred, EmotionalPrediction)
        assert pred.predicted_emotion in [
            "happy", "excited", "alert", "angry",
            "sad", "depressed", "calm", "relaxed",
        ]

    def test_predict_high_entropy_high_arousal(self, engine):
        """High belief entropy should produce higher arousal prediction."""
        pred_high = engine.predict(
            belief_entropies={"focus": 1.5, "consistency": 1.5},
            tom_felt_cost=0.3,
            tom_p_accept=0.5,
            reliability=0.8,
        )
        engine_low = EmotionEngine()
        pred_low = engine_low.predict(
            belief_entropies={"focus": 0.1, "consistency": 0.1},
            tom_felt_cost=0.3,
            tom_p_accept=0.5,
            reliability=0.8,
        )
        assert pred_high.predicted_arousal > pred_low.predicted_arousal

    def test_predict_low_cost_positive_valence(self, engine):
        """Low felt cost + high p_accept should produce positive valence."""
        pred = engine.predict(
            belief_entropies={"focus": 0.5},
            tom_felt_cost=0.1,
            tom_p_accept=0.9,
            reliability=0.8,
        )
        assert pred.predicted_valence > 0

    def test_predict_high_cost_negative_valence(self, engine):
        """High felt cost + low p_accept should produce negative valence."""
        pred = engine.predict(
            belief_entropies={"focus": 0.5},
            tom_felt_cost=0.9,
            tom_p_accept=0.1,
            reliability=0.8,
        )
        assert pred.predicted_valence < 0

    def test_observe_returns_observation(self, engine):
        """Observe should return an EmotionalObservation."""
        obs = engine.observe(valence_idx=3, arousal_idx=2)
        assert isinstance(obs, EmotionalObservation)
        assert obs.observed_valence_idx == 3
        assert obs.observed_arousal_idx == 2

    def test_observe_maps_indices_to_continuous(self, engine):
        """Observation indices should map to continuous values."""
        obs_neg = engine.observe(valence_idx=0, arousal_idx=0)
        obs_pos = engine.observe(valence_idx=4, arousal_idx=4)
        assert obs_neg.observed_valence < obs_pos.observed_valence
        assert obs_neg.observed_arousal < obs_pos.observed_arousal

    def test_update_returns_error(self, engine):
        """Update should return a PredictionError."""
        pred = engine.predict(
            belief_entropies={"focus": 0.5},
            tom_felt_cost=0.3,
            tom_p_accept=0.6,
            reliability=0.5,
        )
        obs = engine.observe(valence_idx=3, arousal_idx=2)
        error = engine.update(pred, obs)
        assert isinstance(error, PredictionError)

    def test_update_records_state(self, engine):
        """Update should record the emotional state."""
        pred = engine.predict(
            belief_entropies={"focus": 0.5},
            tom_felt_cost=0.3,
            tom_p_accept=0.6,
            reliability=0.5,
        )
        obs = engine.observe(valence_idx=3, arousal_idx=2)
        engine.update(pred, obs)
        assert len(engine.states) == 1
        assert engine.get_current_emotion() is not None

    def test_prediction_error_magnitude(self, engine):
        """Error magnitude should be sqrt(v_err^2 + a_err^2)."""
        error = PredictionError(valence_error=0.3, arousal_error=0.4)
        expected = np.sqrt(0.3**2 + 0.4**2)
        assert abs(error.magnitude - expected) < 1e-6

    def test_full_loop_updates_beliefs(self, engine):
        """Running predict-observe-update should change belief distributions."""
        initial_v = engine.belief_valence.copy()

        pred = engine.predict(
            belief_entropies={"focus": 0.5},
            tom_felt_cost=0.3,
            tom_p_accept=0.6,
            reliability=0.5,
        )
        obs = engine.observe(valence_idx=4, arousal_idx=4)  # Very positive, high arousal
        engine.update(pred, obs)

        # Beliefs should have shifted
        assert not np.allclose(engine.belief_valence, initial_v)

    def test_repeated_positive_shifts_belief_positive(self, engine):
        """Repeated positive observations should shift valence belief positive."""
        for _ in range(5):
            pred = engine.predict(
                belief_entropies={"focus": 0.5},
                tom_felt_cost=0.3,
                tom_p_accept=0.6,
                reliability=0.5,
            )
            obs = engine.observe(valence_idx=4, arousal_idx=2)  # Very positive
            engine.update(pred, obs)

        # Most likely valence state should be positive (index 3 or 4)
        most_likely = int(np.argmax(engine.belief_valence))
        assert most_likely >= 3, f"Expected positive state, got index {most_likely}"

    def test_get_belief_state(self, engine):
        """get_belief_state should return valid structure."""
        state = engine.get_belief_state()
        assert "valence" in state
        assert "arousal" in state
        assert "belief" in state["valence"]
        assert "most_likely" in state["valence"]
        assert "entropy" in state["valence"]

    def test_get_emotional_trajectory_empty(self, engine):
        """Trajectory should be empty when no observations."""
        traj = engine.get_emotional_trajectory()
        assert traj["states"] == []

    def test_get_emotional_trajectory_after_updates(self, engine):
        """Trajectory should contain states after updates."""
        for i in range(3):
            pred = engine.predict(
                belief_entropies={"focus": 0.5},
                tom_felt_cost=0.3,
                tom_p_accept=0.6,
                reliability=0.5,
            )
            obs = engine.observe(valence_idx=2 + i, arousal_idx=2)
            engine.update(pred, obs)

        traj = engine.get_emotional_trajectory()
        assert len(traj["states"]) == 3
        assert len(traj["predictions"]) == 3
        assert len(traj["errors"]) == 3
        assert "avg_prediction_error" in traj

    def test_reset_clears_all_state(self, engine):
        """Reset should clear all beliefs and history."""
        pred = engine.predict(
            belief_entropies={"focus": 0.5},
            tom_felt_cost=0.3,
            tom_p_accept=0.6,
            reliability=0.5,
        )
        obs = engine.observe(valence_idx=4, arousal_idx=4)
        engine.update(pred, obs)

        engine.reset()
        assert len(engine.states) == 0
        assert len(engine.predictions) == 0
        assert len(engine.errors) == 0
        assert engine.get_current_emotion() is None


class TestAgentEmotionalIntegration:
    """Test emotional inference integration in CoachingAgent."""

    @pytest.fixture
    def agent(self):
        from mindsphere.core.agent import CoachingAgent
        agent = CoachingAgent()
        agent.start_session()
        return agent

    def test_agent_has_emotion_engine(self, agent):
        """Agent should have an EmotionEngine."""
        assert hasattr(agent, "emotion")
        assert isinstance(agent.emotion, EmotionEngine)

    def test_heuristic_classifier_positive(self, agent):
        """Heuristic classifier should detect positive text."""
        result = agent._classify_emotion_heuristic("I feel great and excited!")
        assert result["valence"] in ("positive", "very_positive")
        assert result["valence_idx"] >= 3

    def test_heuristic_classifier_negative(self, agent):
        """Heuristic classifier should detect negative text."""
        result = agent._classify_emotion_heuristic("I'm stressed and overwhelmed")
        assert result["valence"] in ("negative", "very_negative")
        assert result["valence_idx"] <= 1

    def test_heuristic_classifier_neutral(self, agent):
        """Heuristic classifier should handle neutral text."""
        result = agent._classify_emotion_heuristic("The meeting is at 3pm")
        assert result["valence"] == "neutral"
        assert result["valence_idx"] == 2

    def test_heuristic_classifier_high_arousal(self, agent):
        """Heuristic classifier should detect high arousal."""
        result = agent._classify_emotion_heuristic("I can't believe this! I'm so angry!")
        assert result["arousal"] in ("high", "very_high")
        assert result["arousal_idx"] >= 3

    def test_heuristic_classifier_low_arousal(self, agent):
        """Heuristic classifier should detect low arousal."""
        result = agent._classify_emotion_heuristic("I'm bored and tired")
        assert result["arousal"] in ("low", "very_low")
        assert result["arousal_idx"] <= 1

    def test_heuristic_returns_required_keys(self, agent):
        """Heuristic classifier must return all required keys."""
        result = agent._classify_emotion_heuristic("hello")
        required = {"valence", "arousal", "valence_idx", "arousal_idx",
                     "primary_emotion", "confidence", "emotional_cues"}
        assert required.issubset(set(result.keys()))

    def test_predict_emotion_runs(self, agent):
        """Agent should be able to predict emotion."""
        pred = agent._predict_emotion()
        assert isinstance(pred, EmotionalPrediction)

    def test_observe_emotion_runs(self, agent):
        """Agent should be able to observe emotion from text."""
        obs = agent._observe_emotion("I'm feeling great today!")
        assert isinstance(obs, EmotionalObservation)

    def test_full_emotional_inference_loop(self, agent):
        """Full predict-observe-update loop should run and return valid data."""
        result = agent._run_emotional_inference("I'm excited about this!")
        assert "prediction" in result
        assert "observation" in result
        assert "error" in result
        assert "current_emotion" in result

    def test_emotional_inference_during_coaching(self, agent):
        """Emotional inference should run during coaching phase steps."""
        # Complete calibration
        for i in range(10):
            q = agent.current_question
            if q and q.question_type == "mc":
                agent.step({"answer": q.options[1], "answer_index": 1})
            else:
                agent.step({"answer": "I want to be more focused and productive"})

        # Explicitly ask for coaching to trigger planning transition
        agent.step({"answer": "what do you recommend I work on?"})

        # Accept intervention
        agent.step({"choice": "accept"})

        # Now in coaching phase — emotional inference should run
        result = agent.step({"answer": "I'm feeling stressed about work"})
        assert result["phase"] == "coaching"
        # The emotional_state key should be present
        assert "emotional_state" in result

    def test_emotion_reset_on_session_start(self, agent):
        """Emotion engine should be reset when session starts."""
        agent._run_emotional_inference("test")
        assert len(agent.emotion.states) > 0

        agent.start_session()
        assert len(agent.emotion.states) == 0
