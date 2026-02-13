"""
Circumplex Model of Emotion for MindSphere Coach.

Implements the Active Inference formulation from:
    Pattisapu, Verbelen, Pitliya, Kiefer & Albarracin (2024)
    "Free Energy in a Circumplex Model of Emotion"

Two-dimensional emotional space:
    Arousal = H[Q(s|o)] = entropy of posterior beliefs
        High entropy → high uncertainty → high arousal (alert, anxious)
        Low entropy → high certainty → low arousal (calm, relaxed)

    Valence = Utility - Expected Utility = log P(o|C) - E[log P(o|C)]
        Positive → "better than expected" outcome (happy, excited)
        Negative → "worse than expected" outcome (sad, angry)

The key innovation in MindSphere: the agent PREDICTS the user's emotional
state via ToM, then the LLM generates OBSERVATIONS by classifying user text.
Prediction error drives both emotional belief updates AND ToM learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# CORE COMPUTATION — from Pattisapu et al. (2024)
# =============================================================================

def compute_belief_entropy(qs: np.ndarray, eps: float = 1e-16) -> float:
    """
    Compute entropy of posterior beliefs H[Q(s|o)].

    This is the arousal signal in the Circumplex Model:
    - High entropy = high uncertainty = high arousal
    - Low entropy = high certainty = low arousal
    """
    qs_safe = np.clip(qs, eps, 1.0)
    qs_safe = qs_safe / qs_safe.sum()
    entropy = -np.sum(qs_safe * np.log(qs_safe))
    return float(entropy)


def compute_utility(obs_idx: int, C: np.ndarray, eps: float = 1e-16) -> float:
    """
    Compute utility of an observation given preferences.

    U = log P(o|C)  or  C[obs] if C is already log-preferences.
    """
    c_val = C[obs_idx]
    if np.all(C <= 0):
        return float(c_val)
    else:
        return float(np.log(max(c_val, eps)))


def compute_expected_utility(
    qs: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    eps: float = 1e-16,
) -> float:
    """
    Compute expected utility before seeing observation.

    EU = E_Q(o|s) [log P(o|C)] = sum_o P(o|qs) * log P(o|C)
    """
    obs_dist = A @ qs
    obs_dist = np.clip(obs_dist, eps, 1.0)
    obs_dist = obs_dist / obs_dist.sum()

    if np.all(C <= 0):
        log_C = C
    else:
        log_C = np.log(np.clip(C, eps, 1.0))

    eu = np.sum(obs_dist * log_C)
    return float(eu)


def compute_valence(
    obs_idx: int,
    qs_prior: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    eps: float = 1e-16,
) -> float:
    """
    Compute valence as reward prediction error.

    Valence = Utility - Expected Utility
            = log P(o|C) - E[log P(o|C)]

    Positive valence = "better than expected"
    Negative valence = "worse than expected"
    """
    u = compute_utility(obs_idx, C, eps)
    eu = compute_expected_utility(qs_prior, A, C, eps)
    return u - eu


# =============================================================================
# CIRCUMPLEX MAPPING
# =============================================================================

# 8 emotion sectors (45 degrees each)
EMOTION_SECTORS = {
    "happy":    (337.5, 22.5),
    "excited":  (22.5, 67.5),
    "alert":    (67.5, 112.5),
    "angry":    (112.5, 157.5),
    "sad":      (157.5, 202.5),
    "depressed": (202.5, 247.5),
    "calm":     (247.5, 292.5),
    "relaxed":  (292.5, 337.5),
}


@dataclass
class EmotionalState:
    """
    Single emotional state in the Circumplex Model.

    arousal: H[Q(s|o)] = posterior entropy (uncertainty)
    valence: U - EU = reward prediction error
    intensity: sqrt(arousal^2 + valence^2)
    angle: position on the circumplex (degrees)
    """
    arousal: float
    valence: float
    intensity: float = 0.0
    angle: float = 0.0
    timestep: int = 0

    def __post_init__(self):
        self.intensity = float(np.sqrt(self.arousal**2 + self.valence**2))
        self.angle = float(np.degrees(np.arctan2(self.arousal, self.valence)))
        if self.angle < 0:
            self.angle += 360

    def emotion_label(self) -> str:
        """Get discrete emotion label based on angle in circumplex."""
        angle = self.angle % 360
        if 337.5 <= angle or angle < 22.5:
            return "happy"
        elif 22.5 <= angle < 67.5:
            return "excited"
        elif 67.5 <= angle < 112.5:
            return "alert"
        elif 112.5 <= angle < 157.5:
            return "angry"
        elif 157.5 <= angle < 202.5:
            return "sad"
        elif 202.5 <= angle < 247.5:
            return "depressed"
        elif 247.5 <= angle < 292.5:
            return "calm"
        else:
            return "relaxed"

    def quadrant(self) -> str:
        """Get quadrant description."""
        if self.valence > 0 and self.arousal > 0:
            return "high-arousal-positive"
        elif self.valence <= 0 and self.arousal > 0:
            return "high-arousal-negative"
        elif self.valence > 0 and self.arousal <= 0:
            return "low-arousal-positive"
        else:
            return "low-arousal-negative"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "arousal": round(self.arousal, 3),
            "valence": round(self.valence, 3),
            "intensity": round(self.intensity, 3),
            "angle": round(self.angle, 1),
            "emotion": self.emotion_label(),
            "quadrant": self.quadrant(),
        }


# =============================================================================
# OBSERVATION MODEL — mapping LLM-classified text to valence/arousal
# =============================================================================

# Observation levels for LLM-classified emotional signals
VALENCE_OBS_LEVELS = ["very_negative", "negative", "neutral", "positive", "very_positive"]
AROUSAL_OBS_LEVELS = ["very_low", "low", "moderate", "high", "very_high"]

# Valence state levels (hidden)
VALENCE_STATES = ["very_negative", "negative", "neutral", "positive", "very_positive"]
# Arousal state levels (hidden)
AROUSAL_STATES = ["very_low", "low", "moderate", "high", "very_high"]


def build_emotion_A_matrix(n_obs: int = 5, n_states: int = 5) -> np.ndarray:
    """
    Build observation model A: p(obs | state) for emotional factors.

    LLM-classified text is the observation. The hidden state is the
    user's actual emotional state. The A-matrix encodes how likely
    each LLM classification is given the true state.

    Higher diagonal = LLM classification is reliable.
    """
    # High reliability on diagonal, with noise for uncertainty
    A = np.array([
        [0.60, 0.20, 0.05, 0.02, 0.01],  # obs=very_negative
        [0.25, 0.50, 0.15, 0.05, 0.02],  # obs=negative
        [0.10, 0.20, 0.55, 0.20, 0.10],  # obs=neutral
        [0.03, 0.07, 0.15, 0.50, 0.25],  # obs=positive
        [0.02, 0.03, 0.10, 0.23, 0.62],  # obs=very_positive
    ], dtype=np.float64)
    # Normalize columns
    for col in range(n_states):
        A[:, col] = A[:, col] / A[:, col].sum()
    return A


def build_emotion_B_matrix(n_states: int = 5) -> np.ndarray:
    """
    Build transition model B: p(state' | state) for emotional factors.

    Emotional states have inertia but drift toward neutral over time.
    This captures the psychological reality that extreme emotions
    don't persist indefinitely without continued stimuli.
    """
    # Each state has 70% inertia, 20% drift toward neutral, 10% noise
    B = np.zeros((n_states, n_states), dtype=np.float64)
    neutral_idx = n_states // 2  # index 2

    for s in range(n_states):
        B[s, s] = 0.70  # Inertia

        # Drift toward neutral
        if s < neutral_idx:
            B[s + 1, s] = 0.20  # Move toward neutral
        elif s > neutral_idx:
            B[s - 1, s] = 0.20  # Move toward neutral
        else:
            B[s, s] += 0.20  # Stay at neutral

        # Small noise to all states
        B[:, s] += 0.10 / n_states

    # Normalize columns
    for col in range(n_states):
        B[:, col] = B[:, col] / B[:, col].sum()
    return B


def build_emotion_D(n_states: int = 5) -> np.ndarray:
    """
    Build initial prior D for emotional factors.

    Start with a mild assumption of neutral emotional state.
    """
    D = np.array([0.05, 0.15, 0.60, 0.15, 0.05], dtype=np.float64)
    return D / D.sum()


# =============================================================================
# PREDICT-OBSERVE-UPDATE ENGINE
# =============================================================================

@dataclass
class EmotionalPrediction:
    """
    A prediction of the user's emotional state, made BEFORE observing their text.

    This is the "predict" half of the predict-observe-update loop.
    """
    predicted_valence: float      # From ToM + preference models
    predicted_arousal: float      # From belief entropy
    predicted_emotion: str        # Circumplex label
    confidence: float             # How confident we are in the prediction
    source: str = "tom"           # Where the prediction came from

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_valence": round(self.predicted_valence, 3),
            "predicted_arousal": round(self.predicted_arousal, 3),
            "predicted_emotion": self.predicted_emotion,
            "confidence": round(self.confidence, 3),
            "source": self.source,
        }


@dataclass
class EmotionalObservation:
    """
    An observation of the user's emotional state from LLM classification.

    This is the "observe" half of the predict-observe-update loop.
    """
    observed_valence_idx: int     # 0-4 index into VALENCE_OBS_LEVELS
    observed_arousal_idx: int     # 0-4 index into AROUSAL_OBS_LEVELS
    observed_valence: float       # Continuous value [-1, 1]
    observed_arousal: float       # Continuous value [0, 1]
    observed_emotion: str         # Circumplex label
    raw_classification: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observed_valence": round(self.observed_valence, 3),
            "observed_arousal": round(self.observed_arousal, 3),
            "observed_emotion": self.observed_emotion,
            "valence_idx": self.observed_valence_idx,
            "arousal_idx": self.observed_arousal_idx,
        }


@dataclass
class PredictionError:
    """
    Prediction error between predicted and observed emotional state.

    This drives learning:
    - Large errors → the ToM model is wrong → update particles more
    - Small errors → the model is accurate → increase confidence
    """
    valence_error: float          # observed - predicted
    arousal_error: float          # observed - predicted
    magnitude: float = 0.0       # sqrt(v_err^2 + a_err^2)
    surprise: float = 0.0        # - log p(observed | predicted)

    def __post_init__(self):
        self.magnitude = float(np.sqrt(
            self.valence_error**2 + self.arousal_error**2
        ))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valence_error": round(self.valence_error, 3),
            "arousal_error": round(self.arousal_error, 3),
            "magnitude": round(self.magnitude, 3),
            "surprise": round(self.surprise, 3),
        }


class EmotionEngine:
    """
    Circumplex emotion inference engine with predict-observe-update loop.

    Architecture:
    1. PREDICT: Before observing user text, predict their emotional state
       - Arousal from belief entropy (how uncertain is our model?)
       - Valence from ToM (how does the user feel about what happened?)

    2. OBSERVE: LLM classifies user text into valence/arousal observations
       - These become formal POMDP observations

    3. UPDATE: Compare prediction vs observation
       - Update emotional belief via Bayesian update (A-matrix)
       - Use prediction error to soft-update ToM particles
       - Large errors → ToM is miscalibrated → increase learning rate
       - Small errors → ToM is accurate → increase confidence

    4. RECORD: Store emotional trajectory for context
    """

    def __init__(self):
        # POMDP matrices for emotional factors
        self.A_valence = build_emotion_A_matrix()
        self.A_arousal = build_emotion_A_matrix()
        self.B_valence = build_emotion_B_matrix()
        self.B_arousal = build_emotion_B_matrix()

        # Beliefs over emotional states
        self.belief_valence = build_emotion_D()
        self.belief_arousal = build_emotion_D()

        # Preferences: agent prefers user in positive, calm state
        self.C_valence = np.array([-2.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
        self.C_arousal = np.array([0.0, 0.5, 0.0, -0.5, -1.0], dtype=np.float64)

        # History
        self.predictions: List[EmotionalPrediction] = []
        self.observations: List[EmotionalObservation] = []
        self.errors: List[PredictionError] = []
        self.states: List[EmotionalState] = []

        # Normalization scales (from Pattisapu et al.)
        self.arousal_scale = 1.6  # log(5) for 5-state factors
        self.valence_scale = 3.0

    def predict(
        self,
        belief_entropies: Dict[str, float],
        tom_felt_cost: float,
        tom_p_accept: float,
        reliability: float,
    ) -> EmotionalPrediction:
        """
        PREDICT the user's emotional state before observing their text.

        Arousal: derived from the entropy of the agent's beliefs about the user.
        If the agent is uncertain about the user → the user is likely in an
        uncertain situation → high arousal.

        Valence: derived from ToM predictions.
        - Low felt_cost + high p_accept → things going well → positive valence
        - High felt_cost + low p_accept → things going badly → negative valence

        This follows the Active Inference pattern: valence = U - EU.
        The "utility" for the user is low felt cost and acceptance.
        """
        # Arousal from belief entropy (average across skill factors)
        if belief_entropies:
            avg_entropy = np.mean(list(belief_entropies.values()))
            arousal = avg_entropy / self.arousal_scale
        else:
            arousal = 0.5

        # Valence from ToM predictions
        # User's "utility" is inversely related to felt cost
        # p_accept is a proxy for how well things are going
        valence_raw = (1.0 - tom_felt_cost) * 0.5 + tom_p_accept * 0.5 - 0.5
        valence = float(np.tanh(valence_raw / 0.3))

        # Scale by reliability — less confident predictions are closer to neutral
        arousal = arousal * reliability + 0.5 * (1 - reliability)
        valence = valence * reliability

        # Map to circumplex
        state = EmotionalState(arousal=arousal, valence=valence)

        prediction = EmotionalPrediction(
            predicted_valence=valence,
            predicted_arousal=arousal,
            predicted_emotion=state.emotion_label(),
            confidence=reliability,
        )
        self.predictions.append(prediction)
        return prediction

    def observe(
        self,
        valence_idx: int,
        arousal_idx: int,
        raw_classification: Optional[Dict] = None,
    ) -> EmotionalObservation:
        """
        OBSERVE the user's emotional state from LLM classification.

        The LLM classifies user text into discrete valence (0-4) and
        arousal (0-4) levels. These are formal POMDP observations.
        """
        # Map indices to continuous values
        valence_map = [-0.8, -0.4, 0.0, 0.4, 0.8]
        arousal_map = [0.1, 0.3, 0.5, 0.7, 0.9]

        v = valence_map[min(valence_idx, 4)]
        a = arousal_map[min(arousal_idx, 4)]

        state = EmotionalState(arousal=a, valence=v)

        observation = EmotionalObservation(
            observed_valence_idx=valence_idx,
            observed_arousal_idx=arousal_idx,
            observed_valence=v,
            observed_arousal=a,
            observed_emotion=state.emotion_label(),
            raw_classification=raw_classification or {},
        )
        self.observations.append(observation)
        return observation

    def update(
        self,
        prediction: EmotionalPrediction,
        observation: EmotionalObservation,
    ) -> PredictionError:
        """
        UPDATE beliefs by comparing prediction against observation.

        This is the core of the predict-observe-update loop:
        1. Bayesian update of emotional beliefs using A-matrix
        2. Compute prediction error for ToM learning
        3. Record emotional state

        Returns the prediction error, which the caller uses to
        soft-update ToM particles.
        """
        # --- Bayesian belief update ---
        # Valence belief: p(state | obs) ∝ p(obs | state) * p(state)
        likelihood_v = self.A_valence[observation.observed_valence_idx, :]
        self.belief_valence = likelihood_v * self.belief_valence
        bv_sum = self.belief_valence.sum()
        if bv_sum > 0:
            self.belief_valence /= bv_sum
        else:
            self.belief_valence = build_emotion_D()

        # Arousal belief
        likelihood_a = self.A_arousal[observation.observed_arousal_idx, :]
        self.belief_arousal = likelihood_a * self.belief_arousal
        ba_sum = self.belief_arousal.sum()
        if ba_sum > 0:
            self.belief_arousal /= ba_sum
        else:
            self.belief_arousal = build_emotion_D()

        # --- Apply transition dynamics (drift toward neutral) ---
        self.belief_valence = self.B_valence @ self.belief_valence
        self.belief_arousal = self.B_arousal @ self.belief_arousal

        # --- Compute prediction error ---
        v_error = observation.observed_valence - prediction.predicted_valence
        a_error = observation.observed_arousal - prediction.predicted_arousal

        # Surprise: how unlikely was this observation given the prediction?
        # Use the belief before update to compute this
        surprise_v = -np.log(max(likelihood_v.max(), 1e-12))
        surprise_a = -np.log(max(likelihood_a.max(), 1e-12))
        surprise = float(surprise_v + surprise_a) / 2

        error = PredictionError(
            valence_error=v_error,
            arousal_error=a_error,
            surprise=surprise,
        )
        self.errors.append(error)

        # --- Record emotional state ---
        state = EmotionalState(
            arousal=observation.observed_arousal,
            valence=observation.observed_valence,
            timestep=len(self.states),
        )
        self.states.append(state)

        return error

    def get_current_emotion(self) -> Optional[EmotionalState]:
        """Get the most recent emotional state."""
        return self.states[-1] if self.states else None

    def get_belief_state(self) -> Dict[str, Any]:
        """Get current emotional beliefs for the POMDP."""
        # Most likely valence/arousal states
        v_idx = int(np.argmax(self.belief_valence))
        a_idx = int(np.argmax(self.belief_arousal))

        return {
            "valence": {
                "belief": self.belief_valence.tolist(),
                "most_likely": VALENCE_STATES[v_idx],
                "confidence": float(np.max(self.belief_valence)),
                "entropy": compute_belief_entropy(self.belief_valence),
            },
            "arousal": {
                "belief": self.belief_arousal.tolist(),
                "most_likely": AROUSAL_STATES[a_idx],
                "confidence": float(np.max(self.belief_arousal)),
                "entropy": compute_belief_entropy(self.belief_arousal),
            },
        }

    def get_emotional_trajectory(self) -> Dict[str, Any]:
        """Get the emotional trajectory summary."""
        if not self.states:
            return {"states": [], "emotions": [], "errors": []}

        return {
            "states": [s.to_dict() for s in self.states[-5:]],
            "emotions": [s.emotion_label() for s in self.states[-5:]],
            "predictions": [p.to_dict() for p in self.predictions[-5:]],
            "errors": [e.to_dict() for e in self.errors[-5:]],
            "current": self.states[-1].to_dict(),
            "avg_prediction_error": float(np.mean(
                [e.magnitude for e in self.errors[-10:]]
            )) if self.errors else 0.0,
        }

    def reset(self) -> None:
        """Reset all state."""
        self.belief_valence = build_emotion_D()
        self.belief_arousal = build_emotion_D()
        self.predictions = []
        self.observations = []
        self.errors = []
        self.states = []
