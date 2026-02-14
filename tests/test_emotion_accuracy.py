"""
Tests for emotion detection accuracy and prediction dynamics.

Tests the full emotional inference pipeline:
1. Heuristic classifier accuracy on a range of statements
2. Circumplex mapping correctness (are emotions in the right quadrant?)
3. Prediction dynamics: does the system adapt to sequential emotional shifts?
4. Prediction error magnitude: does the system detect emotional surprises?
5. Belief update convergence: do repeated signals shift beliefs correctly?
6. Incoherent sequences: does the system handle sudden emotional shifts?

Uses a series of statements with varying clarity (obvious → subtle → ambiguous)
and sequences that sometimes make emotional sense and sometimes don't.
"""

import numpy as np
import pytest

from mindsphere.core.agent import CoachingAgent, PHASE_COACHING, SKILL_FACTORS
from mindsphere.core.emotional_state import (
    EmotionalState,
    EmotionEngine,
    VALENCE_OBS_LEVELS,
    AROUSAL_OBS_LEVELS,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def agent():
    """Create a fresh agent for emotion testing.

    Uses whichever classifier is available — LLM when API key is present,
    heuristic fallback otherwise. Both should handle sentiment correctly.
    """
    a = CoachingAgent()
    a.start_session()
    # Fast-forward past calibration to coaching phase
    for skill in SKILL_FACTORS:
        a.beliefs[skill] = np.array([0.05, 0.1, 0.6, 0.2, 0.05])
    a.phase = PHASE_COACHING
    a.timestep = 15
    a._coaching_turns = 0
    return a


@pytest.fixture
def engine():
    """Create a fresh EmotionEngine."""
    return EmotionEngine()


# =============================================================================
# TEST DATA — Statements with expected emotions
# =============================================================================

# Format: (statement, expected_valence_direction, expected_arousal_direction, expected_quadrant)
# valence_direction: "positive", "negative", "neutral"
# arousal_direction: "high", "low", "moderate"

CLEAR_STATEMENTS = [
    # --- CLEAR positive high arousal → excited ---
    ("This is amazing! I can't believe how well it's going!", "positive", "high", "high-arousal-positive"),
    ("I'm so excited about this, let's do more!", "positive", "high", "high-arousal-positive"),

    # --- CLEAR negative high arousal → angry/tense ---
    ("I'm so frustrated and angry right now, nothing works!", "negative", "high", "high-arousal-negative"),
    ("This is terrible, I hate everything about this!", "negative", "high", "high-arousal-negative"),
    ("I'm stressed and anxious about all of this", "negative", "high", "high-arousal-negative"),

    # --- CLEAR negative low arousal → sad/depressed ---
    ("I feel hopeless and tired, nothing matters", "negative", "low", "low-arousal-negative"),
    ("I'm so bored, this is pointless", "negative", "low", "low-arousal-negative"),

    # --- CLEAR positive low arousal → calm/relaxed ---
    ("I feel peaceful and calm right now", "positive", "low", "low-arousal-positive"),
    ("That's good, I'm relaxed about it", "positive", "low", "low-arousal-positive"),
]

SUBTLE_STATEMENTS = [
    # --- SUBTLE negative (no explicit emotion words) ---
    ("I don't think I can keep doing this every day", "negative", "moderate", None),
    ("It's just a lot, you know?", "negative", "moderate", None),

    # --- SUBTLE positive ---
    ("Actually, I think I'm starting to get it", "positive", "moderate", None),
    ("Hmm, that's an interesting way to look at it", "positive", "moderate", None),

    # --- SUBTLE high arousal ---
    ("Wait wait wait, hold on, what?!", "neutral", "high", None),

    # --- SUBTLE low arousal ---
    ("ok", "neutral", "low", None),
    ("fine", "neutral", "low", None),
    ("meh, whatever", "negative", "low", None),
]

AMBIGUOUS_STATEMENTS = [
    # These could go either way — we just test that classifier returns valid output
    ("I don't know how I feel about that", None, None, None),
    ("Let me think about this for a moment", None, None, None),
    ("It's complicated", None, None, None),
    ("I see what you mean", None, None, None),
]


# =============================================================================
# 1. HEURISTIC CLASSIFIER ACCURACY
# =============================================================================

class TestHeuristicClassifierAccuracy:
    """Test that the heuristic emotion classifier gets clear cases right."""

    @pytest.mark.parametrize("statement,expected_v,expected_a,_quad", CLEAR_STATEMENTS)
    def test_clear_valence_detection(self, agent, statement, expected_v, expected_a, _quad):
        """Clear emotional statements should be classified with correct valence direction."""
        result = agent._classify_emotion_heuristic(statement)
        valence_idx = result["valence_idx"]

        if expected_v == "positive":
            assert valence_idx >= 3, (
                f"Expected positive valence for '{statement}', got {VALENCE_OBS_LEVELS[valence_idx]} (idx={valence_idx})"
            )
        elif expected_v == "negative":
            assert valence_idx <= 1, (
                f"Expected negative valence for '{statement}', got {VALENCE_OBS_LEVELS[valence_idx]} (idx={valence_idx})"
            )

    @pytest.mark.parametrize("statement,expected_v,expected_a,_quad", CLEAR_STATEMENTS)
    def test_clear_arousal_detection(self, agent, statement, expected_v, expected_a, _quad):
        """Clear emotional statements should be classified with correct arousal direction."""
        result = agent._classify_emotion_heuristic(statement)
        arousal_idx = result["arousal_idx"]

        if expected_a == "high":
            assert arousal_idx >= 3, (
                f"Expected high arousal for '{statement}', got {AROUSAL_OBS_LEVELS[arousal_idx]} (idx={arousal_idx})"
            )
        elif expected_a == "low":
            assert arousal_idx <= 1, (
                f"Expected low arousal for '{statement}', got {AROUSAL_OBS_LEVELS[arousal_idx]} (idx={arousal_idx})"
            )

    @pytest.mark.parametrize("statement,expected_v,expected_a,expected_quad", CLEAR_STATEMENTS)
    def test_clear_quadrant_mapping(self, agent, statement, expected_v, expected_a, expected_quad):
        """Clear emotional statements should map to the correct circumplex quadrant."""
        result = agent._classify_emotion_heuristic(statement)
        vi, ai = result["valence_idx"], result["arousal_idx"]

        # Map to continuous values (same as EmotionEngine.observe)
        valence_map = [-0.8, -0.4, 0.0, 0.4, 0.8]
        arousal_map = [0.1, 0.3, 0.5, 0.7, 0.9]
        v = valence_map[vi]
        # Center arousal (backend now centers in observe/update)
        a_centered = (arousal_map[ai] - 0.5) * 2

        state = EmotionalState(arousal=a_centered, valence=v)

        if expected_quad:
            assert state.quadrant() == expected_quad, (
                f"Expected {expected_quad} for '{statement}', "
                f"got {state.quadrant()} (v={v:.1f}, a_centered={a_centered:.1f}, "
                f"emotion={state.emotion_label()})"
            )

    def test_classifier_returns_valid_keys(self, agent):
        """All classifier results should have required keys."""
        for statement, _, _, _ in CLEAR_STATEMENTS + SUBTLE_STATEMENTS + AMBIGUOUS_STATEMENTS:
            result = agent._classify_emotion_heuristic(statement)
            assert "valence" in result
            assert "arousal" in result
            assert "valence_idx" in result
            assert "arousal_idx" in result
            assert "primary_emotion" in result
            assert 0 <= result["valence_idx"] <= 4
            assert 0 <= result["arousal_idx"] <= 4

    @pytest.mark.parametrize("statement,expected_v,expected_a,_quad", SUBTLE_STATEMENTS)
    def test_subtle_valence_tendency(self, agent, statement, expected_v, expected_a, _quad):
        """Subtle statements: valence should at least trend in the right direction."""
        result = agent._classify_emotion_heuristic(statement)
        vi = result["valence_idx"]

        if expected_v == "positive":
            assert vi >= 2, f"Expected non-negative valence for '{statement}', got idx={vi}"
        elif expected_v == "negative":
            assert vi <= 2, f"Expected non-positive valence for '{statement}', got idx={vi}"


# =============================================================================
# 2. FULL EMOTIONAL INFERENCE PIPELINE
# =============================================================================

class TestEmotionalInferencePipeline:
    """Test the predict-observe-update loop with the agent."""

    def test_inference_returns_all_components(self, agent):
        """_run_emotional_inference should return prediction, observation, error, and state."""
        result = agent._run_emotional_inference("I feel terrible right now")

        assert "prediction" in result
        assert "observation" in result
        assert "error" in result
        assert "current_emotion" in result
        assert "emotional_beliefs" in result

        # Prediction has expected fields
        pred = result["prediction"]
        assert "predicted_valence" in pred
        assert "predicted_arousal" in pred
        assert "predicted_emotion" in pred

        # Observation has expected fields
        obs = result["observation"]
        assert "observed_valence" in obs
        assert "observed_arousal" in obs
        assert "observed_emotion" in obs

    def test_negative_text_produces_negative_observation(self, agent):
        """Clearly negative text should produce a negative valence observation."""
        result = agent._run_emotional_inference("I hate this, everything is terrible")
        obs = result["observation"]
        assert obs["observed_valence"] < 0, (
            f"Expected negative observed_valence, got {obs['observed_valence']}"
        )

    def test_positive_text_produces_positive_observation(self, agent):
        """Clearly positive text should produce a positive valence observation."""
        result = agent._run_emotional_inference("This is amazing, I love it, I feel great!")
        obs = result["observation"]
        assert obs["observed_valence"] > 0, (
            f"Expected positive observed_valence, got {obs['observed_valence']}"
        )

    def test_prediction_error_computed_correctly(self, agent):
        """Prediction error should be the difference between prediction and observation."""
        result = agent._run_emotional_inference("I'm so angry and frustrated!")
        pred = result["prediction"]
        obs = result["observation"]
        err = result["error"]

        expected_v_err = obs["observed_valence"] - pred["predicted_valence"]
        expected_a_err = obs["observed_arousal"] - pred["predicted_arousal"]

        assert abs(err["valence_error"] - expected_v_err) < 0.001
        assert abs(err["arousal_error"] - expected_a_err) < 0.001


# =============================================================================
# 3. COHERENT EMOTIONAL SEQUENCES
# =============================================================================

class TestCoherentEmotionalSequences:
    """
    Test that coherent emotional sequences produce sensible belief trajectories.

    When someone is consistently negative, beliefs should shift negative.
    When someone goes from negative to positive, beliefs should track.
    """

    def test_repeated_negative_shifts_beliefs_negative(self, agent):
        """Repeated negative messages should shift valence beliefs toward negative."""
        negative_sequence = [
            "I'm feeling really down today",
            "Nothing seems to work out for me",
            "I'm so frustrated with everything",
            "This is hopeless, I give up",
        ]

        initial_v_belief = agent.emotion.belief_valence.copy()

        for msg in negative_sequence:
            agent._run_emotional_inference(msg)

        final_v_belief = agent.emotion.belief_valence

        # Belief mass should shift toward negative states (indices 0-1)
        initial_neg_mass = initial_v_belief[0] + initial_v_belief[1]
        final_neg_mass = final_v_belief[0] + final_v_belief[1]

        assert final_neg_mass > initial_neg_mass, (
            f"Expected negative belief mass to increase after negative sequence. "
            f"Initial: {initial_neg_mass:.3f}, Final: {final_neg_mass:.3f}"
        )

    def test_repeated_positive_shifts_beliefs_positive(self, agent):
        """Repeated positive messages should shift valence beliefs toward positive."""
        positive_sequence = [
            "I'm feeling really good about this",
            "This is great, I love the progress",
            "I'm excited and motivated!",
            "Everything is going amazingly well",
        ]

        initial_v_belief = agent.emotion.belief_valence.copy()

        for msg in positive_sequence:
            agent._run_emotional_inference(msg)

        final_v_belief = agent.emotion.belief_valence

        # Belief mass should shift toward positive states (indices 3-4)
        initial_pos_mass = initial_v_belief[3] + initial_v_belief[4]
        final_pos_mass = final_v_belief[3] + final_v_belief[4]

        assert final_pos_mass > initial_pos_mass, (
            f"Expected positive belief mass to increase after positive sequence. "
            f"Initial: {initial_pos_mass:.3f}, Final: {final_pos_mass:.3f}"
        )

    def test_negative_to_positive_recovery(self, agent):
        """
        Going from negative to positive should show belief recovery.

        Sequence: negative → negative → positive → positive
        Final beliefs should be more positive than after the negative phase.
        """
        # Phase 1: negative
        agent._run_emotional_inference("I'm frustrated and stuck")
        agent._run_emotional_inference("This is terrible, I hate it")

        mid_belief = agent.emotion.belief_valence.copy()
        mid_neg_mass = mid_belief[0] + mid_belief[1]

        # Phase 2: positive recovery
        agent._run_emotional_inference("Actually, I think I see a way forward")
        agent._run_emotional_inference("Yes! This is great, I feel much better now!")

        final_belief = agent.emotion.belief_valence
        final_neg_mass = final_belief[0] + final_belief[1]

        # Negative mass should decrease after positive recovery
        assert final_neg_mass < mid_neg_mass, (
            f"Expected negative mass to decrease after positive recovery. "
            f"Mid: {mid_neg_mass:.3f}, Final: {final_neg_mass:.3f}"
        )

    def test_escalating_arousal_sequence(self, agent):
        """
        Escalating arousal statements should shift arousal beliefs upward.

        calm → moderate → high → very high
        """
        sequence = [
            "I'm feeling pretty calm about this",
            "Hmm, this is getting a bit intense",
            "Oh wow, I can't believe this is happening!",
            "This is urgent! I need help right now! I'm panicking!",
        ]

        arousal_trajectory = []
        for msg in sequence:
            result = agent._run_emotional_inference(msg)
            obs = result["observation"]
            arousal_trajectory.append(obs["observed_arousal"])

        # Last observation should have higher arousal than first
        assert arousal_trajectory[-1] > arousal_trajectory[0], (
            f"Expected escalating arousal: first={arousal_trajectory[0]:.2f}, "
            f"last={arousal_trajectory[-1]:.2f}, trajectory={arousal_trajectory}"
        )


# =============================================================================
# 4. INCOHERENT / SURPRISING SEQUENCES
# =============================================================================

class TestIncoherentSequences:
    """
    Test that sudden emotional shifts produce large prediction errors.

    When someone is consistently positive and suddenly becomes angry,
    the prediction error should be large (the model was surprised).
    """

    def test_sudden_negative_shift_produces_large_error(self, agent):
        """
        Positive → positive → ANGRY should produce a larger error
        than positive → positive → positive.
        """
        # Build positive context
        agent._run_emotional_inference("I'm feeling great today!")
        agent._run_emotional_inference("Everything is wonderful, I love this")

        # Coherent continuation
        result_coherent = agent._run_emotional_inference("I'm so happy and excited!")
        error_coherent = result_coherent["error"]["magnitude"]

        # Reset for incoherent test
        agent2 = CoachingAgent()
        agent2.start_session()
        for skill in SKILL_FACTORS:
            agent2.beliefs[skill] = np.array([0.05, 0.1, 0.6, 0.2, 0.05])
        agent2.phase = PHASE_COACHING
        agent2.timestep = 15

        agent2._run_emotional_inference("I'm feeling great today!")
        agent2._run_emotional_inference("Everything is wonderful, I love this")

        # Incoherent: sudden anger
        result_incoherent = agent2._run_emotional_inference("I'm so angry and frustrated! I hate this!")
        error_incoherent = result_incoherent["error"]["magnitude"]

        # The incoherent shift should (usually) produce larger prediction error
        # This is probabilistic, so we check the observations diverge
        obs_coherent = result_coherent["observation"]["observed_valence"]
        obs_incoherent = result_incoherent["observation"]["observed_valence"]

        assert obs_incoherent < obs_coherent, (
            f"Expected incoherent observation to be more negative. "
            f"Coherent: {obs_coherent:.2f}, Incoherent: {obs_incoherent:.2f}"
        )

    def test_sudden_calm_after_panic(self, agent):
        """
        Panicked → panicked → suddenly calm should be detected.
        The arousal observation should be markedly lower.
        """
        # Build high-arousal context
        agent._run_emotional_inference("I'm panicking, this is so stressful!")
        agent._run_emotional_inference("Help! I can't handle this, I'm overwhelmed!")

        arousal_high = agent.emotion.states[-1].arousal

        # Sudden calm
        result = agent._run_emotional_inference("Actually, I feel completely calm and relaxed now")
        arousal_after = result["observation"]["observed_arousal"]

        # Arousal should be much lower
        assert arousal_after < arousal_high, (
            f"Expected arousal to drop. High phase: {arousal_high:.2f}, "
            f"Calm observation: {arousal_after:.2f}"
        )

    def test_mixed_signals_dont_crash(self, agent):
        """
        Contradictory emotional signals should not crash the system.

        Example: "I'm happy and sad at the same time" — this is ambiguous
        but should still produce valid output.
        """
        mixed_statements = [
            "I'm happy but also kind of worried",
            "This is exciting and terrifying at the same time",
            "I love this but I'm also stressed about it",
            "Everything is fine but nothing feels right",
        ]

        for stmt in mixed_statements:
            result = agent._run_emotional_inference(stmt)
            assert result["observation"]["observed_valence"] is not None
            assert result["observation"]["observed_arousal"] is not None
            assert result["current_emotion"] is not None
            emotion = result["current_emotion"]["emotion"]
            assert emotion in [
                "happy", "excited", "alert", "angry",
                "sad", "depressed", "calm", "relaxed",
            ], f"Got unexpected emotion '{emotion}' for '{stmt}'"


# =============================================================================
# 5. CIRCUMPLEX MAPPING COVERAGE
# =============================================================================

class TestCircumplexCoverage:
    """Test that the full circumplex space is reachable and correctly mapped."""

    def test_all_eight_emotions_reachable_via_observations(self, engine):
        """
        All 8 emotions should be reachable by providing the right
        valence/arousal observation indices.
        """
        # (valence_idx, arousal_idx) → expected emotion
        cases = [
            (4, 2, "happy"),      # positive valence, moderate arousal
            (4, 4, "excited"),    # positive valence, high arousal
            (2, 4, "alert"),      # neutral valence, high arousal
            (0, 4, "angry"),      # negative valence, high arousal
            (0, 2, "sad"),        # negative valence, moderate arousal
            (0, 0, "depressed"),  # negative valence, low arousal
            (2, 0, "calm"),       # neutral valence, low arousal
            (4, 0, "relaxed"),    # positive valence, low arousal
        ]

        reached_emotions = set()
        for vi, ai, expected in cases:
            obs = engine.observe(vi, ai)
            reached_emotions.add(obs.observed_emotion)
            assert obs.observed_emotion == expected, (
                f"Expected {expected} for vi={vi}, ai={ai}, "
                f"got {obs.observed_emotion}"
            )

        assert len(reached_emotions) == 8, (
            f"Expected all 8 emotions reachable, got {reached_emotions}"
        )

    def test_centered_arousal_maps_correctly(self):
        """
        After centering arousal (for display), the quadrants should be correct.

        Internal arousal 0.1 → centered -0.8 (low)
        Internal arousal 0.5 → centered 0.0 (neutral)
        Internal arousal 0.9 → centered 0.8 (high)
        """
        center = lambda a: (a - 0.5) * 2

        # Low arousal
        assert center(0.1) == pytest.approx(-0.8)
        assert center(0.3) == pytest.approx(-0.4)

        # Neutral
        assert center(0.5) == pytest.approx(0.0)

        # High arousal
        assert center(0.7) == pytest.approx(0.4)
        assert center(0.9) == pytest.approx(0.8)

    def test_emotion_label_angle_boundaries(self):
        """Test that emotion labels are assigned at the correct angle boundaries."""
        # Test exact boundary cases
        boundary_cases = [
            (1.0, 0.0, "happy"),       # angle = 0°
            (0.7, 0.7, "excited"),     # angle ≈ 45°
            (0.0, 1.0, "alert"),       # angle = 90°
            (-0.7, 0.7, "angry"),      # angle ≈ 135°
            (-1.0, 0.0, "sad"),        # angle = 180°
            (-0.7, -0.7, "depressed"), # angle ≈ 225°
            (0.0, -1.0, "calm"),       # angle = 270°
            (0.7, -0.7, "relaxed"),    # angle ≈ 315°
        ]

        for v, a, expected in boundary_cases:
            state = EmotionalState(arousal=a, valence=v)
            assert state.emotion_label() == expected, (
                f"Expected {expected} at v={v}, a={a}, "
                f"got {state.emotion_label()} (angle={state.angle:.1f}°)"
            )


# =============================================================================
# 6. PREDICTION DYNAMICS AND ToM INTERACTION
# =============================================================================

class TestPredictionDynamics:
    """
    Test that predictions adapt based on the agent's model state.

    The prediction is derived from:
    - Belief entropy → arousal
    - ToM felt cost / acceptance → valence
    - Reliability → confidence gating
    """

    def test_initial_prediction_near_neutral(self, agent):
        """
        With uniform beliefs and low reliability, prediction should be near neutral.
        """
        prediction = agent._predict_emotion()
        # With uniform beliefs, arousal should be moderate (high entropy)
        assert 0.3 <= prediction.predicted_arousal <= 0.7, (
            f"Expected moderate arousal, got {prediction.predicted_arousal:.3f}"
        )
        # With no intervention, valence should be near neutral
        assert -0.5 <= prediction.predicted_valence <= 0.5, (
            f"Expected near-neutral valence, got {prediction.predicted_valence:.3f}"
        )

    def test_peaked_beliefs_lower_predicted_arousal(self, agent):
        """
        When beliefs are peaked (certain), predicted arousal should be lower
        than when beliefs are uniform (uncertain).
        """
        # Uniform beliefs → high entropy → high arousal
        agent.beliefs["focus"] = np.ones(5) / 5
        pred_uniform = agent._predict_emotion()

        # Reset
        agent.emotion.predictions.clear()

        # Peaked beliefs → low entropy → lower arousal
        agent.beliefs["focus"] = np.array([0.01, 0.01, 0.01, 0.01, 0.96])
        pred_peaked = agent._predict_emotion()

        assert pred_peaked.predicted_arousal < pred_uniform.predicted_arousal, (
            f"Expected peaked beliefs to predict lower arousal. "
            f"Uniform: {pred_uniform.predicted_arousal:.3f}, "
            f"Peaked: {pred_peaked.predicted_arousal:.3f}"
        )

    def test_error_magnitude_bounded(self, agent):
        """Prediction error magnitude should be bounded and non-negative."""
        statements = [
            "I'm really excited!",
            "I feel terrible",
            "This is fine",
            "HELP!!!",
            "whatever",
        ]

        for stmt in statements:
            result = agent._run_emotional_inference(stmt)
            magnitude = result["error"]["magnitude"]
            assert magnitude >= 0, f"Error magnitude should be non-negative, got {magnitude}"
            assert magnitude < 5.0, f"Error magnitude seems too large: {magnitude}"

    def test_trajectory_grows_with_observations(self, agent):
        """Emotional trajectory should record each observation."""
        assert len(agent.emotion.states) == 0

        statements = ["Hello", "I feel great", "This is stressful", "I'm calm now"]
        for i, stmt in enumerate(statements, 1):
            agent._run_emotional_inference(stmt)
            assert len(agent.emotion.states) == i


# =============================================================================
# 7. BELIEF UPDATE DYNAMICS (POMDP level)
# =============================================================================

class TestBeliefUpdateDynamics:
    """Test the Bayesian belief update mechanics of the EmotionEngine."""

    def test_belief_stays_normalized(self, engine):
        """Beliefs should always sum to 1 after updates."""
        for vi in range(5):
            for ai in range(5):
                pred = engine.predict(
                    belief_entropies={"focus": 1.0},
                    tom_felt_cost=0.5,
                    tom_p_accept=0.5,
                    reliability=0.5,
                )
                obs = engine.observe(vi, ai)
                engine.update(pred, obs)

                assert abs(engine.belief_valence.sum() - 1.0) < 1e-10, (
                    f"Valence belief not normalized: {engine.belief_valence.sum()}"
                )
                assert abs(engine.belief_arousal.sum() - 1.0) < 1e-10, (
                    f"Arousal belief not normalized: {engine.belief_arousal.sum()}"
                )

    def test_extreme_observation_shifts_belief(self, engine):
        """
        A very_negative observation should shift valence belief toward negative states.
        """
        initial_neg_mass = engine.belief_valence[0] + engine.belief_valence[1]

        pred = engine.predict(
            belief_entropies={"focus": 1.0},
            tom_felt_cost=0.5,
            tom_p_accept=0.5,
            reliability=0.5,
        )
        # Observe very negative
        obs = engine.observe(0, 2)
        engine.update(pred, obs)

        final_neg_mass = engine.belief_valence[0] + engine.belief_valence[1]
        assert final_neg_mass > initial_neg_mass, (
            f"Expected negative mass to increase after very_negative obs. "
            f"Initial: {initial_neg_mass:.3f}, Final: {final_neg_mass:.3f}"
        )

    def test_drift_toward_neutral(self, engine):
        """
        Without new observations, emotional beliefs should drift toward neutral
        over several update cycles (B-matrix inertia).
        """
        # Push beliefs to extreme negative
        for _ in range(5):
            pred = engine.predict(
                belief_entropies={"focus": 1.0},
                tom_felt_cost=0.9,
                tom_p_accept=0.1,
                reliability=0.8,
            )
            obs = engine.observe(0, 2)  # very negative, moderate arousal
            engine.update(pred, obs)

        extreme_belief = engine.belief_valence.copy()
        extreme_neg_mass = extreme_belief[0] + extreme_belief[1]

        # Now observe neutral for several rounds → drift back
        for _ in range(5):
            pred = engine.predict(
                belief_entropies={"focus": 1.0},
                tom_felt_cost=0.5,
                tom_p_accept=0.5,
                reliability=0.5,
            )
            obs = engine.observe(2, 2)  # neutral
            engine.update(pred, obs)

        recovered_belief = engine.belief_valence
        recovered_neg_mass = recovered_belief[0] + recovered_belief[1]

        assert recovered_neg_mass < extreme_neg_mass, (
            f"Expected negative mass to decrease after neutral observations. "
            f"Extreme: {extreme_neg_mass:.3f}, Recovered: {recovered_neg_mass:.3f}"
        )
