"""
CoachingAgent: main orchestrator for MindSphere coaching sessions.

Manages the full lifecycle:
    calibration → visualization → planning → coaching → complete

Integrates the POMDP model, ToM particle filter, empathy planner,
and LLM layers.  Post-calibration responses are routed through Mistral
LLM for natural, companion-style conversation — with belief context,
ToM predictions, and cognitive load inference injected dynamically.

Falls back to template-based responses when no API key is available.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .model import SphereModel, SKILL_FACTORS, SKILL_LEVEL_VALUES, ACTION_NAMES
from .inference import (
    update_belief,
    update_all_beliefs,
    compute_efe_all_factors,
    select_action,
    compute_information_gain,
)
from .action_dispatcher import select_coaching_action
from .dependency_graph import DependencyGraph
from .utils import normalize, softmax
from ..tom.particle_filter import UserTypeFilter
from ..tom.empathy_planner import EmpathyPlanner
from ..content.question_bank import (
    QUESTION_BANK,
    CalibrationQuestion,
    get_adaptive_question_order,
)
from ..content.interventions import (
    get_gentle_push_pair,
    get_interventions_for_skill,
    Intervention,
)
from ..content.templates import (
    WELCOME_MESSAGE, SPHERE_INTRO, PLAN_INTRO,
    COACHING_PROBES, COACHING_EXERCISES,
)
from .emotional_state import (
    EmotionEngine,
    EmotionalPrediction,
    EmotionalObservation,
    PredictionError,
    compute_belief_entropy,
)
from .learning import ModelLearner
from .user_profile import UserProfile

logger = logging.getLogger(__name__)

# Phase constants
PHASE_CALIBRATION = "calibration"
PHASE_VISUALIZATION = "visualization"
PHASE_PLANNING = "planning"
PHASE_UPDATE = "update"
PHASE_COACHING = "coaching"
PHASE_COMPLETE = "complete"

# Human-readable skill names
SKILL_LABELS = {
    "focus": "Focus",
    "follow_through": "Follow-through",
    "social_courage": "Social Courage",
    "emotional_reg": "Emotional Regulation",
    "systems_thinking": "Systems Thinking",
    "self_trust": "Self-Trust",
    "task_clarity": "Task Clarity",
    "consistency": "Consistency",
}


class CoachingAgent:
    """
    Main orchestrator for a MindSphere coaching session.

    When a Mistral API key is available, all post-calibration responses
    are generated via LLM with dynamic belief/ToM context injection.
    Otherwise, falls back to template-based responses.

    Usage:
        agent = CoachingAgent()
        result = agent.start_session()  # returns welcome + first question

        # Phase 1: Calibration loop
        result = agent.step(user_answer_dict)
        # ... repeat for ~10 questions ...

        # Phase 2+: All responses routed through LLM
        result = agent.step(user_choice_dict)
    """

    def __init__(
        self,
        lambda_empathy: float = 0.5,
        n_particles: int = 50,
        beta: float = 4.0,
    ):
        self.model = SphereModel()
        self.dep_graph = DependencyGraph()
        self.tom = UserTypeFilter(n_particles=n_particles)
        self.empathy = EmpathyPlanner(lambda_empathy=lambda_empathy, beta=beta)

        self.beliefs: Dict[str, np.ndarray] = {}
        self.phase: str = PHASE_CALIBRATION
        self.timestep: int = 0
        self.asked_question_ids: List[str] = []
        self.history: List[Dict[str, Any]] = []
        self.current_question: Optional[CalibrationQuestion] = None
        self.current_intervention: Optional[Intervention] = None
        self.target_skill: Optional[str] = None

        # Conversation history for LLM context (role + content pairs)
        self.conversation_history: List[Dict[str, str]] = []

        # Tracks turns within each post-calibration phase
        self._viz_turns: int = 0
        self._planning_turns: int = 0
        self._coaching_turns: int = 0
        self._probes_asked: List[str] = []  # track which probing questions we've used
        self._exercises_given: List[str] = []  # track which exercises we've suggested
        self._accepted_interventions: List[Dict[str, Any]] = []

        # Cognitive load tracking
        self._recent_sentiments: List[str] = []  # last N sentiment signals

        self._max_calibration_questions = 10

        # POMDP parameter learning (Dirichlet concentration parameters)
        self.learner = ModelLearner(self.model)

        # Circumplex emotion engine (Pattisapu & Albarracin 2024)
        self.emotion = EmotionEngine()
        self._last_prediction: Optional[EmotionalPrediction] = None
        self._last_observation: Optional[EmotionalObservation] = None
        self._last_error: Optional[PredictionError] = None

        # Semantic user profile with Bayesian network causal model
        self.profile = UserProfile()

        # LLM generator + classifier — initialized lazily
        self._generator = None
        self._generator_initialized = False
        self._classifier = None
        self._classifier_initialized = False

    @property
    def generator(self):
        """Lazy-initialize the LLM generator."""
        if not self._generator_initialized:
            self._generator_initialized = True
            try:
                from ..llm.client import MistralClient
                from ..llm.generator import CoachGenerator
                client = MistralClient()
                self._generator = CoachGenerator(client=client)
                if self._generator.is_available:
                    logger.info("LLM generator available — using Mistral for conversation")
                else:
                    logger.warning("LLM generator created but not available (no API key?) — using template responses")
                    self._generator = None
            except Exception as e:
                logger.warning(f"LLM generator init failed: {e} — using template responses")
                self._generator = None
        return self._generator

    @property
    def classifier(self):
        """Lazy-initialize the LLM classifier."""
        if not self._classifier_initialized:
            self._classifier_initialized = True
            try:
                from ..llm.client import MistralClient
                from ..llm.classifier import SphereClassifier
                client = MistralClient()
                if client.is_available:
                    self._classifier = SphereClassifier(client=client)
                    logger.info("LLM classifier available for emotion observations")
                else:
                    logger.warning("LLM classifier not available (no API key?)")
            except Exception as e:
                logger.warning(f"LLM classifier init failed: {e}")
                self._classifier = None
        return self._classifier

    def _llm_generate(self, user_message: str) -> str:
        """
        Try to generate a response via LLM. Returns empty string if unavailable.
        The caller should fall back to template responses when this returns "".
        """
        gen = self.generator
        if gen is None:
            logger.info(f"[LLM] Generator is None — using template fallback (phase={self.phase})")
            return ""

        intervention_dict = None
        if self.current_intervention:
            intervention_dict = {
                "description": self.current_intervention.description,
                "target_skill": self.current_intervention.target_skill,
                "duration_minutes": self.current_intervention.duration_minutes,
                "difficulty": self.current_intervention.difficulty,
            }

        # Build enriched cognitive load with inferred user state + emotional data
        cognitive_load = self._assess_cognitive_load(user_message)
        user_state = self.get_inferred_user_state()
        cognitive_load["inferred_emotions"] = user_state.get("recent_emotions", [])
        cognitive_load["inferred_topics"] = user_state.get("recent_topics", [])
        cognitive_load["engagement_level"] = user_state.get("engagement_level", "moderate")

        # Add circumplex emotional state
        current_emotion = self.emotion.get_current_emotion()
        if current_emotion:
            cognitive_load["circumplex_emotion"] = current_emotion.emotion_label()
            cognitive_load["circumplex_valence"] = round(current_emotion.valence, 2)
            cognitive_load["circumplex_arousal"] = round(current_emotion.arousal, 2)

        # Add prediction error (how well did we predict their emotion?)
        if self._last_error:
            cognitive_load["emotion_prediction_error"] = round(self._last_error.magnitude, 3)
            if self._last_prediction and self._last_observation:
                cognitive_load["predicted_emotion"] = self._last_prediction.predicted_emotion
                cognitive_load["observed_emotion"] = self._last_observation.observed_emotion

        # Profile section for system prompt (Bayesian network causal model)
        profile_section = self.profile.format_for_prompt()

        logger.info(f"[LLM] Generating response (phase={self.phase}, msg_len={len(user_message)}, history_len={len(self.conversation_history)}, profile_facts={len(self.profile.facts)})")
        result = gen.generate(
            user_message=user_message,
            conversation_history=self.conversation_history,
            phase=self.phase,
            belief_summary=self.get_belief_summary(),
            tom_summary=self.tom.get_user_type_summary(),
            cognitive_load=cognitive_load,
            target_skill=self.target_skill,
            current_intervention=intervention_dict,
            accepted_interventions=self._accepted_interventions,
            profile_section=profile_section,
        )
        if result:
            logger.info(f"[LLM] Got response ({len(result)} chars)")
        else:
            logger.warning(f"[LLM] Empty response — falling back to template (phase={self.phase})")
        return result

    def _assess_cognitive_load(self, user_message: str = "") -> Dict[str, Any]:
        """
        Infer cognitive load from ToM dimensions + conversation signals.

        Uses:
        - ToM overwhelm_threshold (low = easily overwhelmed)
        - Recent sentiment signals from conversation
        - Message length and engagement patterns
        """
        # ToM-based assessment
        tom_type = self.tom.get_user_type_summary()
        overwhelm_threshold = tom_type.get("overwhelm_threshold", 0.5)
        autonomy = tom_type.get("autonomy_sensitivity", 0.5)

        # Conversation-based signals
        lower = user_message.lower()
        signals = []

        # Detect disengagement
        disengaged_words = [
            "bored", "boring", "meh", "whatever", "don't care",
            "not interested", "idk", "dunno",
        ]
        if any(w in lower for w in disengaged_words):
            signals.append("disengaged")

        # Detect overwhelm
        overwhelm_words = [
            "overwhelmed", "too much", "can't handle", "stressed",
            "anxious", "burned out", "exhausted", "tired",
        ]
        if any(w in lower for w in overwhelm_words):
            signals.append("overwhelmed")

        # Detect positive engagement
        engaged_words = [
            "interesting", "tell me more", "curious", "that makes sense",
            "let's do it", "i want to", "excited", "ready",
        ]
        if any(w in lower for w in engaged_words):
            signals.append("engaged")

        # Detect off-topic intent
        offtopic_words = [
            "by the way", "random question", "off topic", "unrelated",
            "can you", "do you know", "what about", "have you heard",
            "what's going on", "what do you think about",
        ]
        if any(w in lower for w in offtopic_words):
            signals.append("off_topic")

        # Detect deflection / topic change
        deflection_words = [
            "don't want to talk about", "not want to talk",
            "change the subject", "change topic", "something else",
            "let's talk about", "can we talk about", "i want to talk about",
            "anyway", "never mind", "forget it", "moving on",
            "i'd rather", "not right now", "not now",
            "but what about", "what about", "how about",
        ]
        if any(w in lower for w in deflection_words):
            signals.append("deflection")

        # Short messages may indicate low engagement
        if len(user_message.strip()) < 10 and user_message.strip():
            signals.append("low_effort")

        # Track recent sentiments
        self._recent_sentiments.append(
            "disengaged" if "disengaged" in signals
            else "overwhelmed" if "overwhelmed" in signals
            else "engaged" if "engaged" in signals
            else "neutral"
        )
        # Keep last 5
        self._recent_sentiments = self._recent_sentiments[-5:]

        # Determine overall load level
        recent_disengaged = sum(1 for s in self._recent_sentiments if s == "disengaged")
        recent_overwhelmed = sum(1 for s in self._recent_sentiments if s == "overwhelmed")

        if recent_overwhelmed >= 2 or (overwhelm_threshold < 0.3 and "overwhelmed" in signals):
            load_level = "high"
            coaching_readiness = "not_ready"
        elif recent_disengaged >= 2 or "disengaged" in signals:
            load_level = "low_engagement"
            coaching_readiness = "not_ready"
        elif "off_topic" in signals or "deflection" in signals:
            load_level = "redirected"
            coaching_readiness = "not_ready"
        elif "engaged" in signals:
            load_level = "optimal"
            coaching_readiness = "ready"
        else:
            load_level = "moderate"
            coaching_readiness = "open"

        return {
            "level": load_level,
            "coaching_readiness": coaching_readiness,
            "signals": signals,
            "overwhelm_threshold": overwhelm_threshold,
            "autonomy_sensitivity": autonomy,
        }

    def _track_conversation(self, role: str, content: str) -> None:
        """Add a message to conversation history for LLM context."""
        if content:
            self.conversation_history.append({"role": role, "content": content})

    # -------------------------------------------------------------------------
    # PREDICT-OBSERVE-UPDATE — Circumplex Emotional Inference
    # -------------------------------------------------------------------------

    def _predict_emotion(self) -> EmotionalPrediction:
        """
        PREDICT the user's emotional state before observing their text.

        Uses:
        - Belief entropy → arousal (how uncertain is the model about the user?)
        - ToM felt cost + acceptance probability → valence
        - Reliability → confidence gating

        This follows Pattisapu & Albarracin (2024):
        - Arousal = H[Q(s|o)] = posterior entropy
        - Valence = U - EU = reward prediction error
        """
        # Compute belief entropies across skill factors
        belief_entropies = {}
        for skill in SKILL_FACTORS:
            if skill in self.beliefs:
                belief_entropies[skill] = compute_belief_entropy(self.beliefs[skill])

        # Get ToM predictions for felt cost
        if self.current_intervention:
            pred = self.tom.predict_response_gated(self.current_intervention.to_dict())
            tom_felt_cost = pred.get("predicted_felt_cost", 0.3)
            tom_p_accept = pred.get("p_accept", 0.5)
        else:
            tom_felt_cost = 0.3
            tom_p_accept = 0.5

        reliability = self.tom.reliability

        prediction = self.emotion.predict(
            belief_entropies=belief_entropies,
            tom_felt_cost=tom_felt_cost,
            tom_p_accept=tom_p_accept,
            reliability=reliability,
        )
        self._last_prediction = prediction
        return prediction

    def _observe_emotion(self, user_text: str) -> EmotionalObservation:
        """
        OBSERVE the user's emotional state by classifying their text.

        Uses LLM classification when available, falls back to heuristic.
        The classification becomes a formal POMDP observation.
        """
        classifier = self.classifier
        if classifier is not None:
            # Use LLM for accurate emotional observation
            context = f"Phase: {self.phase}"
            if self.target_skill:
                context += f", Target skill: {self.target_skill}"
            if self.current_intervention:
                context += f", Current suggestion: {self.current_intervention.description}"

            try:
                result = classifier.classify_emotion(user_text, context=context)
            except Exception:
                result = classifier.classify_emotion_heuristic(user_text)
        else:
            # Heuristic fallback — no LLM available
            result = self._classify_emotion_heuristic(user_text)

        observation = self.emotion.observe(
            valence_idx=result.get("valence_idx", 2),
            arousal_idx=result.get("arousal_idx", 2),
            raw_classification=result,
        )
        self._last_observation = observation
        return observation

    def _update_from_emotion_error(
        self,
        prediction: EmotionalPrediction,
        observation: EmotionalObservation,
    ) -> PredictionError:
        """
        UPDATE beliefs using the prediction error between predicted
        and observed emotional state.

        This is the core of the predict-observe-update loop:
        1. Bayesian update of emotional beliefs (A-matrix in EmotionEngine)
        2. Prediction error used to soft-update ToM particles
           - Large errors → ToM was wrong → adjust user type particles
           - Small errors → ToM is accurate → increase confidence

        From the empathy project:
        - Prediction error is the accuracy gate signal
        - When ToM predicts "calm" but user is "angry" → ToM needs calibration
        """
        error = self.emotion.update(prediction, observation)
        self._last_error = error

        # Use prediction error to soft-update ToM particles
        # Large valence error → our model of what makes them feel good/bad is wrong
        # Large arousal error → our model of their uncertainty/engagement is wrong
        self._update_tom_from_emotion_error(error, observation)

        return error

    def _update_tom_from_emotion_error(
        self,
        error: PredictionError,
        observation: EmotionalObservation,
    ) -> None:
        """
        Use emotional prediction error to update ToM particle filter.

        When our emotional prediction is wrong, it tells us something
        about the user's type:

        - We predicted calm but they're stressed → overwhelm_threshold lower than thought
        - We predicted positive but they're frustrated → autonomy higher than thought
        - We predicted engaged but they're bored → novelty_seeking higher than thought

        The update strength scales with the magnitude of the error.
        """
        if error.magnitude < 0.1:
            # Small error — prediction was close, no update needed
            return

        # Scale update strength by error magnitude (capped at 0.2)
        strength = min(error.magnitude * 0.15, 0.2)

        # Determine which ToM dimensions to adjust based on the emotional observation
        observed_emotion = observation.observed_emotion
        valence_error = error.valence_error  # positive = user happier than predicted
        arousal_error = error.arousal_error  # positive = user more aroused than predicted

        # Dimension indices in particle space:
        # 0: avoids_evaluation, 1: hates_long_tasks, 2: novelty_seeking
        # 3: structure_preference, 4: external_validation, 5: autonomy_sensitivity
        # 6: overwhelm_threshold

        biases = []

        # Negative valence surprise: user is more negative than predicted
        if valence_error < -0.2:
            biases.append((6, -1, strength))   # Lower overwhelm threshold
            biases.append((0, 1, strength * 0.5))  # Higher evaluation avoidance

        # Positive valence surprise: user is happier than predicted
        if valence_error > 0.2:
            biases.append((6, 1, strength * 0.5))  # Higher overwhelm threshold

        # High arousal surprise: user is more activated than predicted
        if arousal_error > 0.2:
            if observation.observed_valence < 0:
                # Negative + high arousal = angry/anxious → autonomy sensitivity
                biases.append((5, 1, strength))
            else:
                # Positive + high arousal = excited → novelty seeking
                biases.append((2, 1, strength * 0.5))

        # Low arousal surprise: user is less activated than predicted
        if arousal_error < -0.2:
            if observation.observed_valence < 0:
                # Negative + low arousal = bored/depressed → novelty seeking
                biases.append((2, 1, strength))
                biases.append((1, 1, strength * 0.5))  # hates long tasks
            else:
                # Positive + low arousal = calm/relaxed → structure preference
                biases.append((3, 1, strength * 0.3))

        # Apply biases to particles
        for dim_idx, direction, bias_strength in biases:
            for j in range(self.tom.n_particles):
                particle_val = self.tom.particle_params[j, dim_idx]
                if direction > 0:
                    likelihood = 0.5 + bias_strength * particle_val
                else:
                    likelihood = 0.5 + bias_strength * (1.0 - particle_val)
                self.tom.particle_weights[j] *= likelihood

            # Renormalize
            total = np.sum(self.tom.particle_weights)
            if total > 0:
                self.tom.particle_weights /= total
            else:
                self.tom.particle_weights = np.ones(self.tom.n_particles) / self.tom.n_particles

        # Invalidate caches
        self.tom._reliability_cache = None
        self.tom._confidence_cache = None

    def _classify_emotion_heuristic(self, user_text: str) -> Dict[str, Any]:
        """
        Heuristic emotion classification — fallback when LLM is unavailable.
        Uses keyword matching to estimate valence and arousal.
        """
        lower = user_text.lower()

        neg_words = [
            "stressed", "anxious", "worried", "frustrated", "stuck",
            "hopeless", "lost", "confused", "scared", "tired",
            "burned out", "overwhelmed", "sad", "angry", "annoyed",
            "hate", "terrible", "awful", "bad", "depressed",
            "bored", "boring", "meh", "ugh",
        ]
        pos_words = [
            "good", "great", "happy", "excited", "curious",
            "interesting", "love", "amazing", "better", "hopeful",
            "motivated", "ready", "clear", "peaceful", "calm",
            "relaxed", "grateful", "proud", "confident",
        ]
        high_arousal = [
            "!", "stressed", "anxious", "excited", "angry",
            "can't believe", "urgent", "help", "panic", "scared",
            "amazing", "overwhelmed",
        ]
        low_arousal = [
            "bored", "tired", "calm", "peaceful", "relaxed",
            "meh", "whatever", "ok", "fine", "sleepy",
        ]

        neg_c = sum(1 for w in neg_words if w in lower)
        pos_c = sum(1 for w in pos_words if w in lower)
        hi_c = sum(1 for w in high_arousal if w in lower)
        lo_c = sum(1 for w in low_arousal if w in lower)

        if len(user_text.strip()) < 10:
            lo_c += 1

        # Valence
        if neg_c > pos_c + 1:
            v, vi = "very_negative", 0
        elif neg_c > pos_c:
            v, vi = "negative", 1
        elif pos_c > neg_c + 1:
            v, vi = "very_positive", 4
        elif pos_c > neg_c:
            v, vi = "positive", 3
        else:
            v, vi = "neutral", 2

        # Arousal
        if hi_c > lo_c + 1:
            a, ai = "very_high", 4
        elif hi_c > lo_c:
            a, ai = "high", 3
        elif lo_c > hi_c + 1:
            a, ai = "very_low", 0
        elif lo_c > hi_c:
            a, ai = "low", 1
        else:
            a, ai = "moderate", 2

        # Emotion label
        if vi >= 3 and ai >= 3:
            emotion = "excited"
        elif vi >= 3 and ai <= 1:
            emotion = "relaxed"
        elif vi <= 1 and ai >= 3:
            emotion = "angry"
        elif vi <= 1 and ai <= 1:
            emotion = "depressed"
        elif vi >= 3:
            emotion = "happy"
        elif vi <= 1:
            emotion = "sad"
        elif ai >= 3:
            emotion = "alert"
        else:
            emotion = "calm"

        return {
            "valence": v, "arousal": a,
            "valence_idx": vi, "arousal_idx": ai,
            "primary_emotion": emotion,
            "confidence": "medium", "emotional_cues": [],
        }

    def _run_emotional_inference(self, user_text: str) -> Dict[str, Any]:
        """
        Run the full predict-observe-update loop for emotional inference.

        Returns a dict with prediction, observation, error, and current state.
        This is called each turn during post-calibration phases.
        """
        # 1. PREDICT — before seeing the text
        prediction = self._predict_emotion()

        # 2. OBSERVE — classify the text
        observation = self._observe_emotion(user_text)

        # 3. UPDATE — compare and learn
        error = self._update_from_emotion_error(prediction, observation)

        return {
            "prediction": prediction.to_dict(),
            "observation": observation.to_dict(),
            "error": error.to_dict(),
            "current_emotion": self.emotion.get_current_emotion().to_dict()
                if self.emotion.get_current_emotion() else None,
            "emotional_beliefs": self.emotion.get_belief_state(),
        }

    def start_session(self) -> Dict[str, Any]:
        """Initialize and return welcome message + first question."""
        self.beliefs = self.model.get_initial_beliefs()
        self.phase = PHASE_CALIBRATION
        self.timestep = 0
        self.asked_question_ids = []
        self.history = []
        self.conversation_history = []
        self.tom.reset()
        self.emotion.reset()
        self._viz_turns = 0
        self._planning_turns = 0
        self._recent_sentiments = []
        self._last_prediction = None
        self._last_observation = None
        self._last_error = None

        # Get first question (adaptive ordering)
        next_q = self._get_next_question()
        self.current_question = next_q

        # Track in conversation history
        self._track_conversation("assistant", WELCOME_MESSAGE)

        return {
            "phase": self.phase,
            "message": WELCOME_MESSAGE,
            "question": self._format_question(next_q) if next_q else None,
            "is_complete": False,
        }

    def step(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process one step of the coaching session.

        Args:
            user_input: Dict with keys depending on phase:
                - calibration: {"answer": str, "answer_index": int (optional)}
                - planning/update: {"choice": "accept"|"too_hard"|"not_relevant"}
                - any phase: {"answer": str} for free-text chat

        Returns:
            Result dict with message, phase, question/intervention data, etc.
        """
        self.timestep += 1

        if self.phase == PHASE_CALIBRATION:
            return self._step_calibration(user_input)
        elif self.phase == PHASE_VISUALIZATION:
            return self._step_visualization(user_input)
        elif self.phase == PHASE_PLANNING:
            return self._step_planning(user_input)
        elif self.phase == PHASE_UPDATE:
            return self._step_update(user_input)
        elif self.phase == PHASE_COACHING:
            return self._step_coaching(user_input)
        else:
            return {"phase": PHASE_COMPLETE, "message": "Session complete.", "is_complete": True}

    # -------------------------------------------------------------------------
    # CALIBRATION PHASE
    # -------------------------------------------------------------------------

    # Acknowledgment messages cycled during calibration
    _CALIBRATION_ACKS = [
        "Got it.",
        "Thanks for sharing that.",
        "Understood.",
        "Okay, noted.",
        "That's helpful to know.",
        "Interesting — thanks.",
        "Appreciate your honesty.",
        "Good to know.",
    ]

    def _step_calibration(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process a calibration answer and return next question or transition."""
        if self.current_question is None:
            return self._transition_to_visualization()

        q = self.current_question
        message_type = user_input.get("message_type", "text")

        # --- Handle MC questions ---
        if q.question_type == "mc":
            answer_idx = user_input.get("answer_index")
            if answer_idx is None:
                # Try to match text to option
                answer_text = user_input.get("answer", "")
                answer_idx = self._match_mc_answer(answer_text, q.options)

            # If we still have no valid answer, treat as conversational text
            if answer_idx is None:
                return {
                    "phase": PHASE_CALIBRATION,
                    "message": "No worries — just pick whichever option fits best.",
                    "question": self._format_question(q),
                    "progress": len(self.asked_question_ids) / self._max_calibration_questions,
                    "is_complete": False,
                }

            if q.a_matrix is not None:
                # Update belief for this skill factor
                self.beliefs[q.category] = update_belief(
                    self.beliefs[q.category],
                    answer_idx,
                    q.a_matrix,
                )
                # Learn: refine the A-matrix from this observation
                self.learner.learn_from_observation(
                    q.category, answer_idx, self.beliefs[q.category]
                )
        else:
            # Free text — update multiple factors via classification result
            classified = user_input.get("classified", {})
            skill_signals = classified.get("skill_signals", {})
            for skill, signal in skill_signals.items():
                if skill in self.beliefs:
                    signal_map = {"very_low": 0, "low": 1, "medium": 2, "high": 3, "very_high": 4}
                    obs_idx = signal_map.get(signal, 2)
                    if skill in self.model.A:
                        self.beliefs[skill] = update_belief(
                            self.beliefs[skill], obs_idx, self.model.A[skill]
                        )
                        # Learn: refine A-matrix
                        self.learner.learn_from_observation(
                            skill, obs_idx, self.beliefs[skill]
                        )

            # Update friction beliefs from classification
            overwhelm = classified.get("overwhelm_signal", "medium")
            overwhelm_map = {"low": 0, "medium": 1, "high": 2}
            if "overwhelm_sensitivity" in self.model.A:
                obs_idx = overwhelm_map.get(overwhelm, 1)
                self.beliefs["overwhelm_sensitivity"] = update_belief(
                    self.beliefs["overwhelm_sensitivity"],
                    obs_idx,
                    self.model.A["overwhelm_sensitivity"],
                )
                # Learn: refine friction observation model
                self.learner.learn_from_observation(
                    "overwhelm_sensitivity", obs_idx, self.beliefs["overwhelm_sensitivity"]
                )

        self.asked_question_ids.append(q.id)

        # Record in history
        self.history.append({
            "timestep": self.timestep,
            "phase": PHASE_CALIBRATION,
            "question_id": q.id,
            "user_input": user_input,
        })

        # Extract profile facts from calibration answers (especially free-text)
        cal_text = user_input.get("answer", "")
        if cal_text and len(cal_text) > 10:  # Only extract from substantive answers
            self.profile.extract_and_store(
                user_text=cal_text,
                turn=self.timestep,
                classifier=self.classifier,
                context=f"Answering calibration Q: {q.question_text[:80]}",
            )

        # Check if calibration is complete
        if len(self.asked_question_ids) >= self._max_calibration_questions:
            # Generate ack for the LAST answer before transitioning
            user_text = user_input.get("answer", "")
            last_ack = self._generate_calibration_ack(user_text, q)
            self._track_conversation("user", user_text)
            self._track_conversation("assistant", last_ack)
            return self._transition_to_visualization(last_ack)

        # Get next question
        next_q = self._get_next_question()
        if next_q is None:
            user_text = user_input.get("answer", "")
            last_ack = self._generate_calibration_ack(user_text, q)
            self._track_conversation("user", user_text)
            self._track_conversation("assistant", last_ack)
            return self._transition_to_visualization(last_ack)

        self.current_question = next_q

        # Generate acknowledgment — LLM first, template fallback
        user_text = user_input.get("answer", "")
        ack = self._generate_calibration_ack(user_text, q)

        # Track in conversation history so LLM has context
        self._track_conversation("user", user_text)
        self._track_conversation("assistant", ack)

        return {
            "phase": PHASE_CALIBRATION,
            "message": ack,
            "question": self._format_question(next_q),
            "progress": len(self.asked_question_ids) / self._max_calibration_questions,
            "is_complete": False,
        }

    def _generate_calibration_ack(self, user_text: str, question: CalibrationQuestion) -> str:
        """
        Generate a natural acknowledgment of the user's calibration answer.

        Uses LLM when available for context-aware responses (e.g., empathetic
        when user shares something emotional). Falls back to template cycling.
        """
        gen = self.generator
        if gen is not None:
            logger.info(f"[LLM] Generating calibration ack for Q{len(self.asked_question_ids)} (user said: '{user_text[:50]}')")
            try:
                # Import here to avoid circular imports at module level
                from ..llm.generator import build_system_prompt
                system_prompt = build_system_prompt(
                    phase="calibration",
                    belief_summary=self.get_belief_summary(),
                )

                # Give the LLM context about the question that was asked
                context_msg = f'[The user was asked: "{question.question_text}"]'
                if question.question_type == "mc" and question.options:
                    context_msg += f'\n[Options were: {", ".join(question.options)}]'
                context_msg += f'\n[This is question {len(self.asked_question_ids)} of {self._max_calibration_questions}]'

                messages = [{"role": "system", "content": system_prompt}]

                # Add recent conversation for continuity
                for msg in self.conversation_history[-4:]:
                    messages.append({"role": msg["role"], "content": msg["content"]})

                messages.append({"role": "assistant", "content": context_msg})
                messages.append({"role": "user", "content": user_text})

                response = gen.client.chat_completion(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=80,  # Keep acks short
                    model_override="mistral-medium-latest",
                )
                if response and response.strip():
                    logger.info(f"[LLM] Calibration ack: '{response.strip()[:60]}...'")
                    return response.strip()
                logger.warning("[LLM] Empty calibration ack from Mistral")
            except Exception as e:
                logger.warning(f"[LLM] Calibration ack failed: {e}")
        else:
            logger.info("[LLM] No generator — using template calibration ack")

        # Template fallback
        ack_idx = (len(self.asked_question_ids) - 1) % len(self._CALIBRATION_ACKS)
        logger.info(f"[LLM] Template fallback ack: '{self._CALIBRATION_ACKS[ack_idx]}'")
        return self._CALIBRATION_ACKS[ack_idx]

    # -------------------------------------------------------------------------
    # VISUALIZATION PHASE
    # -------------------------------------------------------------------------

    def _transition_to_visualization(self, last_ack: str = "") -> Dict[str, Any]:
        """Transition from calibration to visualization with personalized commentary."""
        self.phase = PHASE_VISUALIZATION
        self._viz_turns = 0
        sphere_data = self.get_sphere_data()

        # Generate personalized sphere commentary
        commentary = self._generate_sphere_commentary()

        # Prepend the ack for the last calibration answer so it doesn't get lost
        if last_ack:
            full_message = last_ack + "\n\n" + commentary
        else:
            full_message = commentary

        self._track_conversation("assistant", full_message)

        return {
            "phase": PHASE_VISUALIZATION,
            "message": full_message,
            "sphere_data": sphere_data,
            "belief_summary": self.get_belief_summary(),
            "is_complete": False,
        }

    def _step_visualization(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user messages during sphere discussion, transition when ready."""
        self._viz_turns += 1
        user_text = user_input.get("answer", "").strip()

        # Run emotional inference
        emotional_data = self._run_emotional_inference(user_text)

        # Assess cognitive load / intent
        cog_load = self._assess_cognitive_load(user_text)

        # Update cognitive model
        self._update_user_model_from_text(user_text)

        # Record history
        self.history.append({
            "timestep": self.timestep,
            "phase": PHASE_VISUALIZATION,
            "user_input": user_input,
            "emotional_inference": emotional_data,
        })

        # === Keyword overrides: explicit coaching request always honored ===
        lower = user_text.lower()
        wants_coaching = any(w in lower for w in [
            "suggest", "what should", "help me", "what can i do",
            "let's work", "what next", "show me", "i want to improve",
            "what do you recommend", "first step",
        ])

        # Try LLM first — it handles off-topic naturally
        llm_response = self._llm_generate(user_text)
        self._track_conversation("user", user_text)
        response = llm_response or self._respond_to_sphere_reaction(user_text)

        if wants_coaching:
            result = self._transition_to_planning_with_bridge(response)
            result["efe_info"] = {"selected_action": "propose_intervention", "override": "explicit_request"}
            self._track_conversation("assistant", result["message"])
            return result

        # === EFE-driven action selection ===
        action_idx, action_name, efe_info = select_coaching_action(
            beliefs=self.beliefs,
            model=self.model,
            phase="visualization",
            timestep=self.timestep,
            tom_reliability=self.tom.reliability,
            empathy_planner=self.empathy,
            tom_filter=self.tom,
            target_skill=self.target_skill,
            current_intervention=self.current_intervention,
        )

        # Only transition when EFE strongly favors it AND we've had some discussion
        propose_prob = efe_info["action_probabilities"].get("propose_intervention", 0)
        if action_name == "propose_intervention" and propose_prob > 0.45 and self._viz_turns >= 2:
            # EFE confidently says transition to planning
            result = self._transition_to_planning_with_bridge(response)
            result["efe_info"] = efe_info
            self._track_conversation("assistant", result["message"])
            return result

        # ask_free_text or show_sphere (or propose_intervention below threshold): stay in visualization
        self._track_conversation("assistant", response)
        return {
            "phase": PHASE_VISUALIZATION,
            "message": response,
            "sphere_data": self.get_sphere_data(),
            "belief_summary": self.get_belief_summary(),
            "efe_info": efe_info,
            "is_complete": False,
        }

    # -------------------------------------------------------------------------
    # PLANNING PHASE
    # -------------------------------------------------------------------------

    def _transition_to_planning_with_bridge(self, bridge_text: str) -> Dict[str, Any]:
        """Transition to planning with a conversational bridge from the discussion."""
        planning_result = self._transition_to_planning()
        # Prepend the bridge response so the user sees both the reply
        # to their message AND the planning introduction
        planning_result["message"] = bridge_text + "\n\n" + planning_result["message"]
        return planning_result

    def _transition_to_planning(self) -> Dict[str, Any]:
        """Transition to planning: propose first intervention using EFE to select skill."""
        self.phase = PHASE_PLANNING
        self._planning_turns = 0

        # Use EFE to select the best skill to target from top candidates
        impact_ranking = self.dep_graph.compute_impact_ranking(
            {k: v for k, v in self.beliefs.items() if k in SKILL_FACTORS}
        )
        top_skills = [s for s, _ in impact_ranking[:3]] if impact_ranking else SKILL_FACTORS[:1]

        # Evaluate EFE of propose_intervention targeting each candidate skill
        best_skill = top_skills[0]
        best_G = float("inf")
        for skill in top_skills:
            G = compute_efe_all_factors(
                self.beliefs, self.model, 3,  # action = propose_intervention
                relevant_factors=[skill],
                lambda_epist=0.5,
            )
            # Blend with empathy: predict felt cost for gentle intervention
            gentle_list = get_interventions_for_skill(skill, "gentle")
            if gentle_list:
                pred = self.tom.predict_response_gated(gentle_list[0].to_dict())
                G_social = self.empathy.compute_blended_efe(
                    G, pred["predicted_felt_cost"], self.tom.reliability
                )
            else:
                G_social = G
            if G_social < best_G:
                best_G = G_social
                best_skill = skill

        self.target_skill = best_skill

        # Get gentle/push pair for counterfactual
        gentle, push = get_gentle_push_pair(self.target_skill)

        # Compute counterfactual
        counterfactual = self.empathy.compute_counterfactual(
            gentle.to_dict(), push.to_dict(), self.tom
        )

        # Select the recommended intervention
        gentle_pred = self.tom.predict_response_gated(gentle.to_dict())
        push_pred = self.tom.predict_response_gated(push.to_dict())

        if gentle_pred["p_accept"] >= push_pred["p_accept"]:
            self.current_intervention = gentle
        else:
            self.current_intervention = push

        # Generate personalized planning message
        plan_message = self._generate_plan_message()
        self._track_conversation("assistant", plan_message)

        return {
            "phase": PHASE_PLANNING,
            "message": plan_message,
            "intervention": {
                "description": self.current_intervention.description,
                "target_skill": self.current_intervention.target_skill,
                "duration_minutes": self.current_intervention.duration_minutes,
                "difficulty": self.current_intervention.difficulty,
            },
            "counterfactual": counterfactual,
            "target_skill": self.target_skill,
            "sphere_data": self.get_sphere_data(),
            "belief_summary": self.get_belief_summary(),
            "is_complete": False,
        }

    def _step_planning(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """User responds to proposed intervention — either a choice or free text."""
        self._planning_turns += 1
        choice = user_input.get("choice")
        user_text = user_input.get("answer", "").strip()

        # If they clicked a choice button, handle it
        if choice:
            self._track_conversation("user", choice)
            result = self._process_choice(choice)
            self._track_conversation("assistant", result.get("message", ""))
            return result

        # Otherwise it's free text — have a conversation
        # NOTE: _respond_to_planning_chat may call _llm_generate,
        # so track user message AFTER to avoid duplicate user messages
        result = self._respond_to_planning_chat(user_text)
        self._track_conversation("user", user_text)
        self._track_conversation("assistant", result.get("message", ""))
        return result

    # -------------------------------------------------------------------------
    # UPDATE PHASE
    # -------------------------------------------------------------------------

    def _step_update(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process subsequent user choices or free-text in the update loop."""
        choice = user_input.get("choice")
        user_text = user_input.get("answer", "").strip()

        if choice:
            self._track_conversation("user", choice)
            result = self._process_choice(choice)
            self._track_conversation("assistant", result.get("message", ""))
            return result

        # Free text during update phase
        # NOTE: _respond_to_update_chat calls _llm_generate,
        # so track user message AFTER to avoid duplicate user messages
        result = self._respond_to_update_chat(user_text)
        self._track_conversation("user", user_text)
        self._track_conversation("assistant", result.get("message", ""))
        return result

    def _process_choice(self, choice: str) -> Dict[str, Any]:
        """Process a user choice and update beliefs."""
        self.phase = PHASE_UPDATE

        # Map choice to observation index
        choice_map = {"accept": 0, "too_hard": 1, "not_relevant": 2}
        choice_idx = choice_map.get(choice, 0)

        # Update ToM particle filter
        intervention_dict = (
            self.current_intervention.to_dict()
            if self.current_intervention
            else {"difficulty": 0.3, "duration_minutes": 5}
        )
        tom_stats = self.tom.update_weights(choice_idx, intervention_dict)

        # Update friction beliefs
        if "overwhelm_sensitivity" in self.model.A:
            belief_before = self.beliefs["overwhelm_sensitivity"].copy()
            self.beliefs["overwhelm_sensitivity"] = update_belief(
                self.beliefs["overwhelm_sensitivity"],
                choice_idx,
                self.model.A["overwhelm_sensitivity"],
            )
            # Learn: refine friction observation model + transition model
            self.learner.learn_from_observation(
                "overwhelm_sensitivity", choice_idx, self.beliefs["overwhelm_sensitivity"]
            )
            self.learner.learn_from_transition(
                "overwhelm_sensitivity", choice_idx, belief_before, self.beliefs["overwhelm_sensitivity"]
            )

        # Record history
        self.history.append({
            "timestep": self.timestep,
            "phase": PHASE_UPDATE,
            "choice": choice,
            "tom_stats": tom_stats,
        })

        if choice == "accept":
            # Track the accepted intervention
            if self.current_intervention:
                self._accepted_interventions.append(self.current_intervention.to_dict())

            # Generate personalized encouragement and transition to coaching
            message = self._generate_acceptance_message()
            return self._transition_to_coaching(message, tom_stats)
        else:
            # User rejected — adapt and propose alternative
            return self._propose_alternative(choice, tom_stats)

    def _propose_alternative(
        self, rejection_reason: str, tom_stats: Dict
    ) -> Dict[str, Any]:
        """Propose an adjusted intervention after rejection, using EFE+ToM to select."""
        if self.target_skill is None:
            self.target_skill = SKILL_FACTORS[0]

        # Use EFE to find the best alternative across candidate (skill, intervention) pairs
        impact_ranking = self.dep_graph.compute_impact_ranking(
            {k: v for k, v in self.beliefs.items() if k in SKILL_FACTORS}
        )

        candidates = []
        for skill, _ in (impact_ranking[:4] if impact_ranking else [(SKILL_FACTORS[0], 0)]):
            # For "too_hard", always use gentle; for "not_relevant", try different skills
            if rejection_reason == "too_hard" and skill == self.target_skill:
                ivs = get_interventions_for_skill(skill, "gentle")
            elif rejection_reason == "not_relevant" and skill == self.target_skill:
                continue  # skip the rejected skill
            else:
                ivs = get_interventions_for_skill(skill, "gentle")

            for iv in ivs[:2]:  # consider up to 2 interventions per skill
                G = compute_efe_all_factors(
                    self.beliefs, self.model, 3,
                    relevant_factors=[skill], lambda_epist=0.5,
                )
                pred = self.tom.predict_response_gated(iv.to_dict())
                G_social = self.empathy.compute_blended_efe(
                    G, pred["predicted_felt_cost"], self.tom.reliability
                )
                candidates.append((skill, iv, G_social))

        if candidates:
            candidates.sort(key=lambda x: x[2])
            best_skill, best_iv, _ = candidates[0]
            self.target_skill = best_skill
            self.current_intervention = best_iv
        else:
            # Fallback: original hardcoded logic
            if rejection_reason == "too_hard":
                ivs = get_interventions_for_skill(self.target_skill, "gentle")
                if ivs:
                    self.current_intervention = ivs[0]

        # Compute new counterfactual
        gentle, push = get_gentle_push_pair(self.target_skill)
        counterfactual = self.empathy.compute_counterfactual(
            gentle.to_dict(), push.to_dict(), self.tom
        )

        # Generate personalized adaptation message
        adaptation_message = self._generate_adaptation_message(rejection_reason)

        return {
            "phase": PHASE_UPDATE,
            "message": adaptation_message,
            "intervention": {
                "description": self.current_intervention.description if self.current_intervention else "",
                "target_skill": self.target_skill,
                "duration_minutes": self.current_intervention.duration_minutes if self.current_intervention else 2,
                "difficulty": self.current_intervention.difficulty if self.current_intervention else 0.1,
            },
            "counterfactual": counterfactual,
            "tom_stats": tom_stats,
            "user_type_summary": self.tom.get_user_type_summary(),
            "sphere_data": self.get_sphere_data(),
            "belief_summary": self.get_belief_summary(),
            "is_complete": False,
        }

    # -------------------------------------------------------------------------
    # COACHING PHASE
    # -------------------------------------------------------------------------

    def _transition_to_coaching(self, acceptance_message: str, tom_stats: Dict = None) -> Dict[str, Any]:
        """Transition to ongoing coaching after intervention acceptance."""
        self.phase = PHASE_COACHING
        self._coaching_turns = 0

        # Generate a coaching follow-up — a probing question about the target skill
        probe = self._get_next_probe()
        if probe:
            coaching_message = acceptance_message + "\n\n" + probe
        else:
            coaching_message = acceptance_message + (
                "\n\nNow that you have a concrete step, let's dig a bit deeper. "
                "Tell me more about what's going on for you right now."
            )

        result = {
            "phase": PHASE_COACHING,
            "message": coaching_message,
            "sphere_data": self.get_sphere_data(),
            "belief_summary": self.get_belief_summary(),
            "is_complete": False,
        }
        if tom_stats:
            result["tom_stats"] = tom_stats
        return result

    def _step_coaching(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the ongoing coaching conversation using EFE-driven action selection."""
        self._coaching_turns += 1
        user_text = user_input.get("answer", "").strip()
        choice = user_input.get("choice")

        # === EMOTIONAL INFERENCE (Circumplex Model) ===
        emotional_data = self._run_emotional_inference(user_text)

        # Update the cognitive model from this message
        self._update_user_model_from_text(user_text)

        # Record history with emotional data
        self.history.append({
            "timestep": self.timestep,
            "phase": PHASE_COACHING,
            "user_input": user_input,
            "emotional_inference": emotional_data,
        })

        # === Keyword overrides: explicit stop always honored ===
        lower = user_text.lower()
        wants_to_stop = any(w in lower for w in [
            "done", "that's enough", "let's stop", "end session",
            "i'm good", "goodbye", "wrap up", "finish",
        ])
        if wants_to_stop:
            self._track_conversation("user", user_text)
            result = self._end_session()
            result["efe_info"] = {"selected_action": "end_session", "override": "explicit_request"}
            self._track_conversation("assistant", result.get("message", ""))
            return result

        # === Override: explicit request for more action ===
        wants_more_action = any(w in lower for w in [
            "another step", "next step", "what else",
            "give me something", "another exercise", "what now",
        ])
        cog_load = self._assess_cognitive_load(user_text)
        if wants_more_action and cog_load["coaching_readiness"] != "not_ready":
            self._track_conversation("user", user_text)
            result = self._propose_next_coaching_step()
            result["efe_info"] = {"selected_action": "propose_intervention", "override": "explicit_request"}
            result["emotional_state"] = emotional_data
            self._track_conversation("assistant", result.get("message", ""))
            return result

        # === Companion mode: respect off-topic and deflection ===
        if cog_load["coaching_readiness"] == "not_ready":
            # User is off-topic, deflecting, or disengaged — be a companion
            logger.info(f"[Coaching] Companion mode triggered (signals={cog_load['signals']}, readiness={cog_load['coaching_readiness']})")
            # NOTE: _llm_generate MUST be called BEFORE _track_conversation
            # because the generator also appends user_message to the messages list.
            # Tracking first would cause the user message to appear twice.
            llm_response = self._llm_generate(user_text)
            self._track_conversation("user", user_text)
            response = llm_response or self._generate_companion_response(user_text)
            logger.info(f"[Coaching] Companion response (llm={'yes' if llm_response else 'template'}, len={len(response)})")
            self._track_conversation("assistant", response)
            return {
                "phase": PHASE_COACHING,
                "message": response,
                "sphere_data": self.get_sphere_data(),
                "belief_summary": self.get_belief_summary(),
                "emotional_state": emotional_data,
                "efe_info": {"selected_action": "companion_chat", "override": "not_coaching_ready"},
                "is_complete": False,
            }

        # === EFE-driven action selection ===
        action_idx, action_name, efe_info = select_coaching_action(
            beliefs=self.beliefs,
            model=self.model,
            phase="coaching",
            timestep=self.timestep,
            tom_reliability=self.tom.reliability,
            empathy_planner=self.empathy,
            tom_filter=self.tom,
            target_skill=self.target_skill,
            current_intervention=self.current_intervention,
        )

        # Dispatch based on EFE-selected action
        logger.info(f"[Coaching] EFE selected: {action_name} (probs={efe_info.get('action_probabilities', {})})")
        if action_name == "propose_intervention":
            self._track_conversation("user", user_text)
            result = self._propose_next_coaching_step()
            result["efe_info"] = efe_info
            result["emotional_state"] = emotional_data
            self._track_conversation("assistant", result.get("message", ""))
            return result
        elif action_name == "end_session":
            self._track_conversation("user", user_text)
            result = self._end_session()
            result["efe_info"] = efe_info
            result["emotional_state"] = emotional_data
            self._track_conversation("assistant", result.get("message", ""))
            return result
        elif action_name == "safety_check":
            self._track_conversation("user", user_text)
            response = (
                "I want to check in — how are you feeling about all this? "
                "Sometimes coaching conversations bring up a lot, and I want to make sure "
                "we're going at the right pace for you."
            )
            self._track_conversation("assistant", response)
            return {
                "phase": PHASE_COACHING,
                "message": response,
                "sphere_data": self.get_sphere_data(),
                "belief_summary": self.get_belief_summary(),
                "emotional_state": emotional_data,
                "efe_info": efe_info,
                "is_complete": False,
            }
        elif action_name == "show_counterfactual":
            self._track_conversation("user", user_text)
            # Show counterfactual comparison for the current target skill
            gentle, push = get_gentle_push_pair(self.target_skill or SKILL_FACTORS[0])
            cf = self.empathy.compute_counterfactual(
                gentle.to_dict(), push.to_dict(), self.tom
            )
            cf_text = self.empathy.format_counterfactual_text(cf)
            llm_response = self._llm_generate(
                f"[SYSTEM: Present this counterfactual naturally: {cf_text}]"
            )
            response = llm_response or (
                f"Here's what my model predicts for two approaches:\n\n{cf_text}\n\n"
                "What feels more realistic for you?"
            )
            self._track_conversation("assistant", response)
            return {
                "phase": PHASE_COACHING,
                "message": response,
                "counterfactual": cf,
                "sphere_data": self.get_sphere_data(),
                "belief_summary": self.get_belief_summary(),
                "emotional_state": emotional_data,
                "efe_info": efe_info,
                "is_complete": False,
            }
        elif action_name == "reframe":
            # Reframing: use LLM with reframe context
            llm_response = self._llm_generate(user_text)
            self._track_conversation("user", user_text)
            if not llm_response:
                # Template reframe
                probe = self._get_next_probe()
                llm_response = (
                    "Let me offer a different way to look at this. "
                    "What you're describing isn't a flaw — it's information about "
                    "where the friction is. "
                )
                if probe:
                    llm_response += probe
            self._track_conversation("assistant", llm_response)
            return {
                "phase": PHASE_COACHING,
                "message": llm_response,
                "sphere_data": self.get_sphere_data(),
                "belief_summary": self.get_belief_summary(),
                "emotional_state": emotional_data,
                "efe_info": efe_info,
                "is_complete": False,
            }
        else:
            # Default: ask_free_text / adjust_difficulty — conversational response
            llm_response = self._llm_generate(user_text)
            self._track_conversation("user", user_text)
            response = llm_response or self._generate_coaching_response(user_text)
            self._track_conversation("assistant", response)
            return {
                "phase": PHASE_COACHING,
                "message": response,
                "sphere_data": self.get_sphere_data(),
                "belief_summary": self.get_belief_summary(),
                "emotional_state": emotional_data,
                "efe_info": efe_info,
                "is_complete": False,
            }

    def _get_next_probe(self) -> Optional[str]:
        """Get a probing question for the target skill that hasn't been asked yet."""
        skill = self.target_skill or "focus"
        probes = COACHING_PROBES.get(skill, [])

        for probe in probes:
            if probe not in self._probes_asked:
                self._probes_asked.append(probe)
                return probe

        # Fall back to a different skill's probe
        sorted_skills = self._get_skill_scores_sorted()
        for skill_name, _ in sorted_skills:
            probes = COACHING_PROBES.get(skill_name, [])
            for probe in probes:
                if probe not in self._probes_asked:
                    self._probes_asked.append(probe)
                    return probe

        return None

    def _get_next_exercise(self) -> Optional[str]:
        """Get a coaching exercise for the target skill."""
        skill = self.target_skill or "focus"
        exercises = COACHING_EXERCISES.get(skill, [])

        for ex in exercises:
            if ex not in self._exercises_given:
                self._exercises_given.append(ex)
                return ex

        # Fall back to a different skill
        sorted_skills = self._get_skill_scores_sorted()
        for skill_name, _ in sorted_skills:
            exercises = COACHING_EXERCISES.get(skill_name, [])
            for ex in exercises:
                if ex not in self._exercises_given:
                    self._exercises_given.append(ex)
                    return ex

        return None

    def _propose_next_coaching_step(self) -> Dict[str, Any]:
        """Propose the next coaching step, using EFE to decide between exercise, probe, or sphere."""
        # Use EFE to decide what type of next step to offer
        action_idx, action_name, efe_info = select_coaching_action(
            beliefs=self.beliefs,
            model=self.model,
            phase="coaching",
            timestep=self.timestep,
            tom_reliability=self.tom.reliability,
            empathy_planner=self.empathy,
            tom_filter=self.tom,
            target_skill=self.target_skill,
            current_intervention=self.current_intervention,
        )

        if action_name in ("ask_free_text", "reframe"):
            # EFE says: probe the user more (epistemic drive)
            probe = self._get_next_probe()
            if probe:
                message = (
                    f"Before we add more steps, let me understand something better. "
                    f"{probe}"
                )
            else:
                message = (
                    "We've covered a lot of ground. What feels like the most important thing "
                    "you're taking away from this conversation?"
                )
        else:
            # EFE says: propose an exercise (pragmatic drive)
            exercise = self._get_next_exercise()
            if exercise:
                sorted_skills = self._get_skill_scores_sorted()
                next_skill = sorted_skills[0][0]
                for skill_name, score in sorted_skills:
                    if skill_name != self.target_skill:
                        next_skill = skill_name
                        break
                skill_label = self._label(next_skill)
                message = (
                    f"Here's something else to try — this one targets your {skill_label}: "
                    f"{exercise}\n\n"
                    f"How does that land?"
                )
            else:
                # Exhausted exercises — probe instead
                probe = self._get_next_probe()
                if probe:
                    message = (
                        f"Before we add more steps, let me understand something better. "
                        f"{probe}"
                    )
                else:
                    message = (
                        "We've covered a lot of ground. What feels like the most important thing "
                        "you're taking away from this conversation?"
                    )

        return {
            "phase": PHASE_COACHING,
            "message": message,
            "sphere_data": self.get_sphere_data(),
            "belief_summary": self.get_belief_summary(),
            "is_complete": False,
        }

    def _generate_coaching_response(self, user_text: str) -> str:
        """Generate a coaching response that probes deeper or offers insight."""
        lower = user_text.lower()

        # Detect what the user is talking about
        is_emotional = any(w in lower for w in [
            "stressed", "anxious", "worried", "frustrated", "stuck",
            "hopeless", "lost", "confused", "scared", "tired",
            "burned out", "overwhelmed", "depressed", "sad",
        ])
        is_reflective = any(w in lower for w in [
            "i think", "i realize", "i notice", "interesting",
            "never thought", "makes sense", "you're right",
        ])
        is_asking = any(w in lower for w in [
            "why", "how", "what should", "what do", "can you",
            "tell me", "explain",
        ])
        mentions_work = any(w in lower for w in [
            "work", "job", "career", "boss", "colleague", "deadline",
            "project", "meeting",
        ])
        mentions_personal = any(w in lower for w in [
            "family", "relationship", "friend", "partner", "home",
            "health", "sleep", "exercise",
        ])

        if is_emotional:
            # Empathize first, then probe gently
            probe = self._get_next_probe()
            response = (
                "I hear that, and I don't want to brush past it. What you're feeling "
                "is information — it tells us something about where the friction is. "
            )
            if probe:
                response += f"Let me ask you this: {probe}"
            else:
                response += "What do you think is the root of that feeling?"
            return response

        elif is_reflective:
            # Reinforce the insight and build on it
            response = (
                "That's a really useful observation. The ability to see your own patterns "
                "is exactly what makes change possible. "
            )
            exercise = self._get_next_exercise()
            if exercise:
                response += (
                    f"Building on that insight, here's something you could try: {exercise}"
                )
            else:
                probe = self._get_next_probe()
                if probe:
                    response += f"Let's go deeper. {probe}"
            return response

        elif is_asking:
            # Answer based on the sphere data and ToM
            sorted_skills = self._get_skill_scores_sorted()
            weakest = sorted_skills[0]
            strongest = sorted_skills[-1]

            return (
                f"Based on your patterns, your biggest opportunity is in "
                f"{self._label(weakest[0])} ({round(weakest[1])}/100). But your "
                f"{self._label(strongest[0])} ({round(strongest[1])}/100) shows you already "
                f"have real capacity. The question isn't whether you can change — you can. "
                f"It's about finding the right entry point and making it small enough to actually stick. "
                f"What specifically would you like to know more about?"
            )

        elif mentions_work:
            # Probe into the work context
            probe = self._get_next_probe()
            response = (
                "Work is often where these patterns show up most clearly. "
                "The stakes are higher, the structure is external, and the pressure is real. "
            )
            if probe:
                response += probe
            else:
                response += (
                    "Can you tell me about a specific situation at work where you felt stuck or frustrated?"
                )
            return response

        elif mentions_personal:
            # Acknowledge and explore
            response = (
                "The personal side matters a lot — these patterns don't stay at work. "
                "They show up in relationships, health, everything. "
            )
            probe = self._get_next_probe()
            if probe:
                response += probe
            else:
                response += "What's the connection you're seeing between this and your personal life?"
            return response

        else:
            # General coaching response — probe or exercise
            if self._coaching_turns % 3 == 0:
                # Every third turn, offer an exercise
                exercise = self._get_next_exercise()
                if exercise:
                    return (
                        f"Thanks for sharing that. Here's something concrete you can do with that: "
                        f"{exercise}\n\nBut I also want to understand you better. "
                        f"What would change in your life if you got this right?"
                    )

            # Default: ask a probing question
            probe = self._get_next_probe()
            if probe:
                return (
                    f"That helps me understand where you're coming from. "
                    f"Let me dig a bit deeper: {probe}"
                )

            return (
                "I appreciate you sharing that. Let's keep exploring — "
                "what feels like the most important thing right now? Is there something "
                "specific you'd like to work on, or should I suggest another step?"
            )

    def _generate_companion_response(self, user_text: str) -> str:
        """Template fallback when user is off-topic and LLM is unavailable.

        Instead of forcing coaching, acknowledges what the user said and
        follows their lead — companion first, coach second.
        """
        lower = user_text.lower()

        # Detect explicit deflection from coaching
        deflecting = any(w in lower for w in [
            "don't want to talk about", "not want to talk",
            "something else", "never mind", "forget it",
            "not right now", "not now", "moving on",
        ])
        if deflecting:
            return (
                "No pressure at all — we can talk about whatever you want. "
                "What's on your mind?"
            )

        # Detect overwhelm / disengagement
        overwhelmed = any(w in lower for w in [
            "overwhelmed", "too much", "can't handle", "exhausted",
        ])
        if overwhelmed:
            return (
                "Hey, let's slow down. We don't have to push through anything right now. "
                "What would feel good to talk about instead?"
            )

        # User brought up a non-coaching topic — engage with it
        return (
            "I'm happy to chat about that! "
            "We can always come back to the other stuff whenever — or not. "
            "Tell me more."
        )

    def _end_session(self) -> Dict[str, Any]:
        """End the coaching session with a summary."""
        sorted_skills = self._get_skill_scores_sorted()
        weakest = self._label(sorted_skills[0][0])
        strongest = self._label(sorted_skills[-1][0])

        # Try LLM for a natural wrap-up
        llm_context = (
            f"[SYSTEM: The user wants to end the session. "
            f"Strongest area: {strongest}. Growth area: {weakest}. "
        )
        if self._accepted_interventions:
            last_step = self._accepted_interventions[-1].get("description", "")
            pred = self.tom.predict_response_gated(self._accepted_interventions[-1])
            p_complete = pred.get("p_accept", 0.5)
            llm_context += (
                f"They committed to: \"{last_step}\" "
                f"(predicted {round(p_complete * 100)}% follow-through). "
            )
        llm_context += "Wrap up warmly and briefly. Reference things from the conversation.]"

        llm_response = self._llm_generate(llm_context)

        if not llm_response:
            # Template fallback
            parts = ["Let's wrap up. Here's what I'm taking away from our conversation:"]
            parts.append(
                f"Your biggest strength is {strongest} — that's your foundation. "
                f"Your biggest growth area is {weakest}."
            )
            if self._accepted_interventions:
                last_step = self._accepted_interventions[-1].get("description", "")
                pred = self.tom.predict_response_gated(self._accepted_interventions[-1])
                p_complete = pred.get("p_accept", 0.5)
                parts.append(
                    f"Your committed step: \"{last_step}\" — "
                    f"I predict a {round(p_complete * 100)}% chance you'll follow through."
                )
            parts.append(
                "The fact that you showed up and went through this process tells me something important "
                "about you. You're not just thinking about change — you're doing something about it. "
                "Come back anytime to check in."
            )
            llm_response = " ".join(parts)

        return {
            "phase": PHASE_COMPLETE,
            "message": llm_response,
            "sphere_data": self.get_sphere_data(),
            "belief_summary": self.get_belief_summary(),
            "user_type_summary": self.tom.get_user_type_summary(),
            "is_complete": True,
        }

    # -------------------------------------------------------------------------
    # CONVERSATIONAL RESPONSE GENERATION
    # -------------------------------------------------------------------------

    def _get_skill_scores_sorted(self) -> List[Tuple[str, float]]:
        """Get skill scores sorted low→high."""
        scores = self.model.get_all_skill_scores(self.beliefs)
        return sorted(scores.items(), key=lambda x: x[1])

    def _label(self, skill: str) -> str:
        """Human-readable skill name."""
        return SKILL_LABELS.get(skill, skill.replace("_", " ").title())

    def _generate_sphere_commentary(self) -> str:
        """Generate personalized commentary about the user's sphere."""
        sorted_skills = self._get_skill_scores_sorted()
        weakest = sorted_skills[:2]
        strongest = sorted_skills[-2:]

        bottlenecks = self.dep_graph.find_bottlenecks(
            {k: v for k, v in self.beliefs.items() if k in SKILL_FACTORS}
        )

        # Try LLM first — give it the full sphere data to work with
        gen = self.generator
        logger.info(f"[LLM] Generating sphere commentary (generator={'available' if gen else 'None'})")
        if gen is not None:
            try:
                from ..llm.generator import build_system_prompt

                # Build a rich data summary for the LLM
                skill_lines = []
                for skill_name, score in sorted_skills:
                    skill_lines.append(f"  - {self._label(skill_name)}: {round(score)}/100")

                sphere_context = "Here are the user's skill scores:\n" + "\n".join(skill_lines)

                if bottlenecks:
                    bn = bottlenecks[0]
                    blocker = self._label(bn["blocker"])
                    blocked_names = [self._label(b) for b in bn["blocked"][:2]]
                    sphere_context += (
                        f"\n\nKey insight: {blocker} is a bottleneck — "
                        f"it's holding back {' and '.join(blocked_names)}."
                    )

                system_prompt = build_system_prompt(
                    phase="sphere_commentary",
                    belief_summary=self.get_belief_summary(),
                    tom_summary=self.tom.get_user_type_summary(),
                )

                messages = [{"role": "system", "content": system_prompt}]

                # Include conversation history from calibration
                for msg in self.conversation_history[-6:]:
                    messages.append({"role": msg["role"], "content": msg["content"]})

                messages.append({
                    "role": "user",
                    "content": (
                        f"[SYSTEM: Generate the sphere commentary. {sphere_context}]\n\n"
                        f"Present these results to the user warmly and conversationally."
                    ),
                })

                response = gen.client.chat_completion(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=300,
                    model_override="mistral-medium-latest",
                )
                if response and response.strip():
                    logger.info(f"[LLM] Sphere commentary generated ({len(response)} chars)")
                    return response.strip()
                logger.warning("[LLM] Empty sphere commentary from Mistral")
            except Exception as e:
                logger.warning(f"[LLM] Sphere commentary failed: {e}")

        logger.info("[LLM] Using template sphere commentary")
        # Template fallback
        parts = [
            "Here's your MindSphere — a snapshot of your patterns across eight areas."
        ]

        s1_name, s1_score = strongest[-1]
        s2_name, s2_score = strongest[-2]
        parts.append(
            f"Your strongest areas are {self._label(s1_name)} ({round(s1_score)}/100) "
            f"and {self._label(s2_name)} ({round(s2_score)}/100) — that's real foundation to build on."
        )

        w1_name, w1_score = weakest[0]
        w2_name, w2_score = weakest[1]
        parts.append(
            f"The biggest dents are in {self._label(w1_name)} ({round(w1_score)}/100) "
            f"and {self._label(w2_name)} ({round(w2_score)}/100)."
        )

        if bottlenecks:
            bn = bottlenecks[0]
            blocker = self._label(bn["blocker"])
            blocked_names = [self._label(b) for b in bn["blocked"][:2]]
            parts.append(
                f"Interestingly, {blocker} is acting as a bottleneck — "
                f"it's holding back your {' and '.join(blocked_names)}. "
                f"That means improving it would have a ripple effect."
            )

        parts.append(
            "Does this match how you see yourself? I'm curious what stands out to you."
        )

        return " ".join(parts)

    def _respond_to_sphere_reaction(self, user_text: str) -> str:
        """Respond to the user's reaction to their sphere."""
        lower = user_text.lower()

        # Detect sentiment/intent
        agrees = any(w in lower for w in [
            "yes", "yeah", "accurate", "right", "true", "makes sense",
            "spot on", "correct", "agree", "that's me", "sounds right",
        ])
        disagrees = any(w in lower for w in [
            "no", "don't think", "disagree", "wrong", "not really",
            "doesn't seem", "inaccurate", "off", "surprised",
        ])
        asks_why = any(w in lower for w in [
            "why", "how come", "explain", "what does", "what do you mean",
        ])
        mentions_stress = any(w in lower for w in [
            "stress", "anxious", "overwhelm", "burned", "tired",
            "exhausted", "struggling",
        ])
        mentions_focus = any(w in lower for w in [
            "focus", "distract", "attention", "concentrate",
        ])

        sorted_skills = self._get_skill_scores_sorted()
        weakest_name = self._label(sorted_skills[0][0])
        weakest_score = round(sorted_skills[0][1])

        if agrees:
            return (
                f"Good — that self-awareness is genuinely useful. The fact that you can see "
                f"these patterns clearly means you're already ahead of where most people start. "
                f"Let me show you what I think would make the biggest difference right now."
            )
        elif disagrees:
            # Acknowledge and show openness to being wrong
            return (
                f"That's important feedback — my model is built from your answers, but you know "
                f"yourself better than ten questions can capture. Which part feels off? "
                f"I can adjust my understanding as we talk."
            )
        elif asks_why:
            bottlenecks = self.dep_graph.find_bottlenecks(
                {k: v for k, v in self.beliefs.items() if k in SKILL_FACTORS}
            )
            if bottlenecks:
                bn = bottlenecks[0]
                blocker = self._label(bn["blocker"])
                blocked = [self._label(b) for b in bn["blocked"][:2]]
                return (
                    f"The scores come from how you answered the calibration questions — "
                    f"each answer shifts my estimate of where you sit on each skill. "
                    f"The dependency analysis shows that {blocker} is a leverage point because "
                    f"it feeds into {' and '.join(blocked)}. "
                    f"Think of it like a supply chain — a bottleneck upstream affects everything downstream."
                )
            return (
                f"The scores come from your calibration answers — each one shifts my estimate "
                f"of where you sit across these eight dimensions. The dents show where "
                f"there's the most room for movement."
            )
        elif mentions_stress:
            return (
                f"Stress is real, and it touches a lot of these dimensions — especially "
                f"Emotional Regulation and Focus. The good news is that working on even one "
                f"of these can take pressure off the others. You mentioned being stressed before, "
                f"and I factored that into my model of you. Let me suggest something small "
                f"that might help."
            )
        elif mentions_focus:
            focus_score = round(self.model.get_skill_score(self.beliefs.get("focus", self.model.D["focus"])))
            return (
                f"Focus came in at {focus_score}/100 — and from what you told me, that tracks. "
                f"The interesting thing is that focus isn't just about willpower. It often connects "
                f"to other patterns — like how clear your tasks are, or how you handle interruptions. "
                f"Let me show you what I think would help most."
            )
        else:
            # Generic thoughtful response
            return (
                f"Thanks for sharing that. Based on everything you've told me, "
                f"I think the most impactful place to start is {weakest_name} — "
                f"it's currently at {weakest_score}/100, and improving it would "
                f"unlock progress in other areas too. Want me to suggest a concrete first step?"
            )

    def _generate_plan_message(self) -> str:
        """Generate personalized planning message for the first intervention."""
        if not self.current_intervention or not self.target_skill:
            return PLAN_INTRO

        skill_label = self._label(self.target_skill)
        score = round(self.model.get_skill_score(
            self.beliefs.get(self.target_skill, self.model.D.get(self.target_skill, np.array([0.2]*5)))
        ))

        # Check ToM predictions to calibrate tone
        pred = self.tom.predict_response_gated(self.current_intervention.to_dict())
        p_accept = pred.get("p_accept", 0.5)

        # Check if there's a dependency explanation
        bottlenecks = self.dep_graph.find_bottlenecks(
            {k: v for k, v in self.beliefs.items() if k in SKILL_FACTORS}
        )

        # Try LLM first
        llm_response = self._llm_generate(
            f"[SYSTEM: Propose this intervention naturally: "
            f"Target skill: {skill_label} ({score}/100). "
            f"Suggestion: \"{self.current_intervention.description}\" "
            f"({self.current_intervention.duration_minutes} min, "
            f"difficulty {self.current_intervention.difficulty}). "
            f"Predicted acceptance: {round(p_accept * 100)}%. "
            f"Weave it into conversation naturally.]"
        )
        if llm_response:
            return llm_response

        # Template fallback
        parts = []

        if bottlenecks and bottlenecks[0]["blocker"] == self.target_skill:
            blocked = [self._label(b) for b in bottlenecks[0]["blocked"][:2]]
            parts.append(
                f"I'm starting with {skill_label} because it's your biggest leverage point right now — "
                f"at {score}/100, it's holding back your {' and '.join(blocked)}."
            )
        else:
            parts.append(
                f"I'm starting with {skill_label} ({score}/100) because "
                f"I think it's where a small change would make the biggest difference."
            )

        parts.append(
            f'Here\'s what I have in mind: "{self.current_intervention.description}"'
        )

        if p_accept < 0.4:
            parts.append(
                "I know this might feel like a stretch — and that's okay. "
                "It's designed to be small enough that you can try it without committing to anything big."
            )
        elif p_accept > 0.7:
            parts.append(
                "I think this is right in your sweet spot — challenging enough to matter, "
                "small enough to actually happen."
            )

        parts.append(
            "What do you think? You can also tell me if it feels too much or not relevant, "
            "and I'll adjust."
        )

        return " ".join(parts)

    def _respond_to_planning_chat(self, user_text: str) -> Dict[str, Any]:
        """Respond to free-text during the planning phase."""
        lower = user_text.lower()

        # Update cognitive model
        self._update_user_model_from_text(user_text)

        # Detect implicit acceptance or rejection
        positive = any(w in lower for w in [
            "sure", "ok", "okay", "sounds good", "let's do it", "i'll try",
            "let's go", "yes", "yeah", "i can do that", "worth a shot",
        ])
        negative = any(w in lower for w in [
            "too hard", "too much", "can't", "won't work", "not for me",
            "impossible", "no way", "overwhelming",
        ])
        irrelevant = any(w in lower for w in [
            "not relevant", "doesn't apply", "not my issue", "wrong area",
            "not the problem",
        ])

        if positive:
            return self._process_choice("accept")
        elif negative:
            return self._process_choice("too_hard")
        elif irrelevant:
            return self._process_choice("not_relevant")

        # For everything else (questions, off-topic, general chat) — use LLM
        llm_response = self._llm_generate(user_text)
        if llm_response:
            return {
                "phase": PHASE_PLANNING,
                "message": llm_response,
                "sphere_data": self.get_sphere_data(),
                "belief_summary": self.get_belief_summary(),
                "is_complete": False,
            }

        # Template fallback
        asks_question = any(w in lower for w in [
            "why", "how", "what if", "explain", "tell me more",
        ])
        if asks_question:
            response = self._explain_intervention()
        else:
            response = self._respond_general_chat(user_text, PHASE_PLANNING)

        return {
            "phase": PHASE_PLANNING,
            "message": response,
            "sphere_data": self.get_sphere_data(),
            "belief_summary": self.get_belief_summary(),
            "is_complete": False,
        }

    def _respond_to_update_chat(self, user_text: str) -> Dict[str, Any]:
        """Respond to free-text during the update phase."""
        lower = user_text.lower()

        # Update cognitive model
        self._update_user_model_from_text(user_text)

        positive = any(w in lower for w in [
            "sure", "ok", "okay", "sounds good", "i'll try", "yes", "yeah",
        ])
        negative = any(w in lower for w in [
            "too hard", "too much", "can't", "no",
        ])
        irrelevant = any(w in lower for w in [
            "not relevant", "doesn't apply", "wrong area",
        ])

        if positive:
            return self._process_choice("accept")
        elif negative:
            return self._process_choice("too_hard")
        elif irrelevant:
            return self._process_choice("not_relevant")

        # Try LLM for natural conversation
        llm_response = self._llm_generate(user_text)
        response = llm_response or self._respond_general_chat(user_text, PHASE_UPDATE)
        return {
            "phase": PHASE_UPDATE,
            "message": response,
            "sphere_data": self.get_sphere_data(),
            "belief_summary": self.get_belief_summary(),
            "is_complete": False,
        }

    def _explain_intervention(self) -> str:
        """Explain why the current intervention was chosen."""
        if not self.current_intervention or not self.target_skill:
            return "I chose this based on where I think a small change would make the biggest impact."

        skill_label = self._label(self.target_skill)
        score = round(self.model.get_skill_score(
            self.beliefs.get(self.target_skill, self.model.D.get(self.target_skill, np.array([0.2]*5)))
        ))

        # Check dependency
        bottlenecks = self.dep_graph.find_bottlenecks(
            {k: v for k, v in self.beliefs.items() if k in SKILL_FACTORS}
        )

        parts = [
            f"{skill_label} is at {score}/100 in your sphere."
        ]

        if bottlenecks and bottlenecks[0]["blocker"] == self.target_skill:
            blocked = [self._label(b) for b in bottlenecks[0]["blocked"][:2]]
            parts.append(
                f"It's also a bottleneck — it's limiting your {' and '.join(blocked)}. "
                f"So improving it has a multiplier effect."
            )

        # ToM insight
        pred = self.tom.predict_response_gated(self.current_intervention.to_dict())
        p_accept = pred.get("p_accept", 0.5)
        parts.append(
            f"I picked this specific step because my model predicts about a "
            f"{round(p_accept * 100)}% chance you'll actually follow through on it — "
            f"and that matters more than ambition."
        )

        parts.append("Does that make sense?")

        return " ".join(parts)

    def _generate_acceptance_message(self) -> str:
        """Generate encouragement when user accepts an intervention."""
        if not self.current_intervention:
            return "Great choice. Let's see how it goes."

        skill_label = self._label(self.current_intervention.target_skill)
        duration = self.current_intervention.duration_minutes

        pred = self.tom.predict_response_gated(self.current_intervention.to_dict())
        p_complete = pred.get("p_accept", 0.6)

        # Try LLM first
        llm_response = self._llm_generate(
            f"[SYSTEM: The user just accepted this step: "
            f"\"{self.current_intervention.description}\" "
            f"({round(duration)} min). "
            f"Predicted follow-through: {round(p_complete * 100)}%. "
            f"Encourage them naturally and transition into coaching conversation.]"
        )
        if llm_response:
            return llm_response

        # Template fallback
        parts = [f"Good. {self.current_intervention.description}"]

        if duration <= 5:
            parts.append(
                f"It's only {round(duration)} minutes — the point isn't to change everything, "
                f"it's to prove to yourself that you can shift the pattern."
            )
        else:
            parts.append(
                f"Block out {round(duration)} minutes for this. "
                f"You don't need to do it perfectly, just do it."
            )

        parts.append(
            f"Based on what I know about you, I predict a {round(p_complete * 100)}% chance "
            f"you'll follow through. Let's check in next time and see how it went."
        )

        return " ".join(parts)

    def _generate_adaptation_message(self, rejection_reason: str) -> str:
        """Generate personalized message when adapting after rejection."""
        # Try LLM first
        if self.current_intervention:
            llm_response = self._llm_generate(
                f"[SYSTEM: The user rejected the previous suggestion as '{rejection_reason}'. "
                f"New suggestion: \"{self.current_intervention.description}\" "
                f"({round(self.current_intervention.duration_minutes)} min). "
                f"Acknowledge their feedback warmly and present the alternative naturally.]"
            )
            if llm_response:
                return llm_response

        # Template fallback
        if rejection_reason == "too_hard":
            if self.current_intervention:
                return (
                    f"I hear you — that was too much. That's useful information for me. "
                    f"Let me try something smaller: \"{self.current_intervention.description}\" "
                    f"This should take about {round(self.current_intervention.duration_minutes)} minutes. "
                    f"How does that feel?"
                )
            return (
                "I hear you — that was too much. Let me find something smaller "
                "that still moves the needle."
            )
        elif rejection_reason == "not_relevant":
            skill_label = self._label(self.target_skill) if self.target_skill else "a different area"
            if self.current_intervention:
                return (
                    f"Fair enough — let me look at {skill_label} instead. "
                    f"Here's what I have in mind: \"{self.current_intervention.description}\" "
                    f"Does this connect better to what you're actually dealing with?"
                )
            return (
                f"Fair enough. Let me find something that connects better to what "
                f"actually matters to you."
            )
        return "Let me try a different approach."

    # -------------------------------------------------------------------------
    # COGNITIVE MODELING — continuous user model updates from conversation
    # -------------------------------------------------------------------------

    def _update_user_model_from_text(self, user_text: str) -> None:
        """
        Update the cognitive model of the user from conversational signals.

        This goes beyond explicit choices — it infers emotional state, interests,
        engagement patterns, and updates the ToM particle filter accordingly.
        The agent is always building a richer picture of who this person is.

        Also extracts structured profile facts and updates the Bayesian network.
        """
        # Extract profile facts, causal links, and progress signals
        extraction = self.profile.extract_and_store(
            user_text=user_text,
            turn=self.timestep,
            classifier=self.classifier,
            context=self._get_recent_context(),
        )

        # Apply profile signals and progress to POMDP skill beliefs
        self._apply_skill_signals(extraction)

        lower = user_text.lower()

        # --- Infer emotional state ---
        emotional_signals = {
            "stressed": ["stressed", "stress", "anxious", "anxiety", "worried"],
            "frustrated": ["frustrated", "annoying", "annoyed", "ugh", "hate"],
            "sad": ["sad", "depressed", "hopeless", "lonely", "lost"],
            "excited": ["excited", "pumped", "can't wait", "stoked"],
            "calm": ["calm", "peaceful", "relaxed", "good"],
            "bored": ["bored", "boring", "meh", "whatever"],
            "overwhelmed": ["overwhelmed", "too much", "drowning", "buried"],
            "curious": ["curious", "wonder", "interesting", "tell me"],
        }

        detected_emotions = []
        for emotion, keywords in emotional_signals.items():
            if any(w in lower for w in keywords):
                detected_emotions.append(emotion)

        # --- Infer topics of interest ---
        topic_signals = {
            "work": ["work", "job", "career", "boss", "colleague", "office", "meeting", "deadline", "project"],
            "relationships": ["partner", "friend", "family", "relationship", "dating", "marriage"],
            "health": ["health", "exercise", "gym", "diet", "sleep", "energy", "body"],
            "creativity": ["creative", "art", "music", "writing", "design", "ideas"],
            "learning": ["learn", "study", "book", "course", "skill", "education"],
            "finances": ["money", "budget", "savings", "debt", "financial", "income"],
            "identity": ["who am i", "purpose", "meaning", "values", "identity", "authentic"],
        }

        detected_topics = []
        for topic, keywords in topic_signals.items():
            if any(w in lower for w in keywords):
                detected_topics.append(topic)

        # --- Update particle filter from conversational signals ---
        # The particle filter normally only updates from explicit choices.
        # Here we do soft updates based on inferred signals to expand the model.

        if detected_emotions:
            self._soft_update_tom_from_emotions(detected_emotions)

        # --- Store inferred state in history ---
        if detected_emotions or detected_topics:
            self.history.append({
                "timestep": self.timestep,
                "type": "cognitive_model_update",
                "emotions": detected_emotions,
                "topics": detected_topics,
                "user_text_snippet": user_text[:100],
            })

    def _apply_skill_signals(self, extraction: Dict[str, Any]) -> None:
        """
        Apply profile and progress signals to update POMDP skill beliefs.

        This closes the feedback loop: facts extracted from conversation
        (breakup, progress reports, challenges) actually shift the skill
        beliefs, so the model evolves throughout the session.

        Two sources of signals:
        1. Profile Bayesian network — inferred states affect skill beliefs
           (e.g., breakup → emotional_stress → emotional_reg impaired)
        2. Progress signals — user reports improvement or regression
           (e.g., "I've been focusing better" → focus belief shifts up)
        """
        # --- 1. Apply Bayesian network skill impacts ---
        bn_impacts = self.profile.bayes_net.get_skill_impacts()
        for skill, impact in bn_impacts.items():
            if skill not in self.beliefs or skill not in SKILL_FACTORS:
                continue
            if abs(impact) < 0.03:
                continue  # Too small to matter

            belief = self.beliefs[skill]
            n_levels = len(belief)

            # Shift belief: positive impact → shift toward higher levels,
            # negative impact → shift toward lower levels
            shift_strength = min(abs(impact), 0.3)  # Cap the shift
            if impact > 0:
                # Shift probability mass toward higher skill levels
                target = np.zeros(n_levels)
                target[-1] = 0.5  # Weight toward high
                target[-2] = 0.3
                target[n_levels // 2] = 0.2
            else:
                # Shift probability mass toward lower skill levels
                target = np.zeros(n_levels)
                target[0] = 0.5  # Weight toward low
                target[1] = 0.3
                target[n_levels // 2] = 0.2
            target = normalize(target)

            # Soft blend: belief = (1 - alpha) * belief + alpha * target
            alpha = shift_strength * 0.15  # Gentle: max ~4.5% blend per message
            self.beliefs[skill] = normalize(
                (1.0 - alpha) * belief + alpha * target
            )

        # --- 2. Apply progress signals (user-reported improvement/regression) ---
        progress_signals = extraction.get("progress_signals", [])
        for signal in progress_signals:
            skill = signal.get("skill")
            direction = signal.get("direction")
            magnitude = signal.get("magnitude", 0.2)

            if skill not in self.beliefs or skill not in SKILL_FACTORS:
                continue

            belief = self.beliefs[skill]
            n_levels = len(belief)

            # Progress signals are stronger than passive BN impacts —
            # user is explicitly reporting behavioral change
            shift_strength = min(magnitude, 0.5)

            if direction == "improvement":
                # Shift toward higher levels
                target = np.zeros(n_levels)
                for i in range(n_levels):
                    target[i] = (i + 1) / n_levels  # Linear ramp up
            else:
                # Shift toward lower levels
                target = np.zeros(n_levels)
                for i in range(n_levels):
                    target[i] = (n_levels - i) / n_levels  # Linear ramp down
            target = normalize(target)

            # Stronger blend for explicit progress reports
            alpha = shift_strength * 0.25  # Up to 12.5% blend
            self.beliefs[skill] = normalize(
                (1.0 - alpha) * belief + alpha * target
            )

            logger.info(
                f"[Learning] Skill '{skill}' belief shifted ({direction}, "
                f"magnitude={magnitude:.2f}): {signal.get('evidence', '')}"
            )

    def _soft_update_tom_from_emotions(self, emotions: List[str]) -> None:
        """
        Soft-update the ToM particle filter based on emotional signals.

        Instead of a hard Bayesian update (which requires explicit choice observations),
        this applies a gentle bias to the particle weights based on what emotions
        tell us about the user's type dimensions.

        For example: "overwhelmed" suggests low overwhelm_threshold (dimension 6).
        "bored" with structured tasks suggests high novelty_seeking (dimension 2).
        """
        # Map emotions to particle dimension biases
        # Each entry: (dimension_index, direction, strength)
        # direction > 0 means "this emotion suggests HIGH value on that dimension"
        # direction < 0 means "this emotion suggests LOW value"
        emotion_dim_map = {
            "overwhelmed": [(6, -1, 0.15)],     # low overwhelm_threshold
            "stressed": [(6, -1, 0.10)],          # low overwhelm_threshold
            "bored": [(2, 1, 0.10)],              # high novelty_seeking
            "frustrated": [(5, 1, 0.08)],         # high autonomy_sensitivity
            "excited": [(2, 1, 0.05), (6, 1, 0.05)],  # novelty + high threshold
            "curious": [(2, 1, 0.05)],            # novelty_seeking
        }

        for emotion in emotions:
            biases = emotion_dim_map.get(emotion, [])
            for dim_idx, direction, strength in biases:
                # Apply soft weight update: particles that match the signal get upweighted
                for j in range(self.tom.n_particles):
                    particle_val = self.tom.particle_params[j, dim_idx]
                    if direction > 0:
                        # Upweight particles with high value on this dimension
                        likelihood = 0.5 + strength * particle_val
                    else:
                        # Upweight particles with low value on this dimension
                        likelihood = 0.5 + strength * (1.0 - particle_val)
                    self.tom.particle_weights[j] *= likelihood

                # Renormalize
                total = np.sum(self.tom.particle_weights)
                if total > 0:
                    self.tom.particle_weights /= total
                else:
                    self.tom.particle_weights = np.ones(self.tom.n_particles) / self.tom.n_particles

                # Invalidate reliability cache
                self.tom._reliability_cache = None
                self.tom._confidence_cache = None

    def _get_recent_context(self) -> str:
        """Get recent conversation context for profile extraction."""
        if not self.conversation_history:
            return ""
        recent = self.conversation_history[-4:]
        return " | ".join(f"{m['role']}: {m['content'][:80]}" for m in recent)

    def get_inferred_user_state(self) -> Dict[str, Any]:
        """
        Get the current inferred cognitive/emotional state of the user.

        Aggregates recent conversation signals into a summary the LLM can use.
        """
        # Get recent emotion/topic history
        recent_updates = [
            h for h in self.history[-10:]
            if h.get("type") == "cognitive_model_update"
        ]

        recent_emotions = []
        recent_topics = []
        for update in recent_updates:
            recent_emotions.extend(update.get("emotions", []))
            recent_topics.extend(update.get("topics", []))

        # Deduplicate and get most recent
        unique_emotions = list(dict.fromkeys(reversed(recent_emotions)))[:3]
        unique_topics = list(dict.fromkeys(reversed(recent_topics)))[:3]

        # Assess overall engagement from sentiment tracking
        engagement = "moderate"
        if self._recent_sentiments:
            recent = self._recent_sentiments[-3:]
            if all(s == "engaged" for s in recent):
                engagement = "high"
            elif any(s == "disengaged" for s in recent):
                engagement = "low"
            elif any(s == "overwhelmed" for s in recent):
                engagement = "strained"

        return {
            "recent_emotions": unique_emotions,
            "recent_topics": unique_topics,
            "engagement_level": engagement,
            "turns_in_phase": self._coaching_turns if self.phase == PHASE_COACHING else 0,
            "tom_type": self.tom.get_user_type_summary(),
            "tom_reliability": self.tom.reliability,
        }

    def _respond_general_chat(self, user_text: str, phase: str) -> str:
        """Generate a response to general chat that relates back to coaching."""
        lower = user_text.lower()

        # Detect emotional content
        is_emotional = any(w in lower for w in [
            "stressed", "anxious", "worried", "frustrated", "stuck",
            "hopeless", "lost", "confused", "scared", "tired",
            "burned out", "overwhelmed",
        ])
        is_positive = any(w in lower for w in [
            "better", "good", "happy", "excited", "motivated",
            "hopeful", "ready", "clear",
        ])

        if is_emotional:
            return (
                f"I hear that. And I want you to know — what you're feeling makes sense "
                f"given what I'm seeing in your patterns. This isn't about fixing something "
                f"broken. It's about finding the smallest adjustment that creates the most relief. "
                f"That's what I'm trying to do here."
            )
        elif is_positive:
            return (
                f"That energy is worth channeling. Based on your sphere, the thing that would "
                f"compound most right now is working on your {self._label(self.target_skill or 'focus')}. "
                f"Want to commit to the step I suggested?"
            )
        else:
            # Acknowledge and gently steer back
            if phase == PHASE_PLANNING:
                return (
                    f"Thanks for sharing that — it helps me understand you better. "
                    f"Coming back to the step I suggested: does it feel doable, "
                    f"or would you like me to adjust it?"
                )
            return (
                f"I appreciate you telling me that. It gives me more context for "
                f"how to help. What would be most useful for you right now?"
            )

    # -------------------------------------------------------------------------
    # DATA ACCESSORS
    # -------------------------------------------------------------------------

    def get_sphere_data(self) -> Dict[str, Any]:
        """Get radar chart data: skill scores + bottlenecks + edges."""
        skill_beliefs = {
            k: v for k, v in self.beliefs.items() if k in SKILL_FACTORS
        }
        scores = self.model.get_all_skill_scores(self.beliefs)
        bottlenecks = self.dep_graph.find_bottlenecks(skill_beliefs)

        return {
            "categories": scores,
            "bottlenecks": bottlenecks,
            "dependency_edges": self.dep_graph.get_all_edges(),
        }

    def get_belief_summary(self) -> Dict[str, Any]:
        """Get human-readable summary of all beliefs."""
        summary = {}

        # Skill scores
        for skill in SKILL_FACTORS:
            if skill in self.beliefs:
                score = self.model.get_skill_score(self.beliefs[skill])
                uncertainty = float(np.std(self.beliefs[skill] * SKILL_LEVEL_VALUES))
                summary[skill] = {"score": round(score, 1), "uncertainty": round(uncertainty, 1)}

        # Preferences
        for factor, levels in self.model.spec.preference_factors.items():
            if factor in self.beliefs:
                best_idx = int(np.argmax(self.beliefs[factor]))
                summary[factor] = {
                    "inferred": levels[best_idx],
                    "confidence": float(np.max(self.beliefs[factor])),
                }

        # Friction
        for factor, levels in self.model.spec.friction_factors.items():
            if factor in self.beliefs:
                best_idx = int(np.argmax(self.beliefs[factor]))
                summary[factor] = {
                    "inferred": levels[best_idx],
                    "confidence": float(np.max(self.beliefs[factor])),
                }

        # ToM reliability
        summary["tom_reliability"] = self.tom.reliability
        summary["user_type"] = self.tom.get_user_type_summary()

        # Learning progress
        summary["learning"] = self.learner.get_learning_summary()

        # Semantic profile + Bayesian network
        summary["profile"] = self.profile.get_summary()

        return summary

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    def _get_next_question(self) -> Optional[CalibrationQuestion]:
        """Get the next question using adaptive ordering."""
        ordered = get_adaptive_question_order(self.beliefs, self.asked_question_ids)
        return ordered[0] if ordered else None

    def _format_question(self, q: CalibrationQuestion) -> Dict[str, Any]:
        """Format a question for the frontend."""
        result = {
            "id": q.id,
            "category": q.category,
            "question_text": q.question_text,
            "question_type": q.question_type,
        }
        if q.question_type == "mc":
            result["options"] = q.options
        return result

    def _match_mc_answer(self, answer_text: str, options: List[str]) -> Optional[int]:
        """Simple text matching for MC answers. Returns None if no match."""
        answer_lower = answer_text.strip().lower()
        if not answer_lower:
            return None

        # Check for exact index (e.g., "0", "1", "2", "3")
        if answer_lower.isdigit():
            idx = int(answer_lower)
            if 0 <= idx < len(options):
                return idx

        # Check for letter (a, b, c, d)
        letter_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        if answer_lower in letter_map:
            idx = letter_map[answer_lower]
            if idx < len(options):
                return idx

        # Require substantial overlap for substring match (>50% of option text)
        for i, opt in enumerate(options):
            opt_lower = opt.lower()
            if opt_lower == answer_lower:
                return i
            # Only match if the user's text contains most of the option
            if len(answer_lower) > 10 and opt_lower in answer_lower:
                return i

        return None  # No confident match — treat as conversational

    def set_empathy_dial(self, lambda_value: float) -> None:
        """Adjust the empathy dial (0=challenging, 1=gentle)."""
        self.empathy.lambda_empathy = max(0.0, min(1.0, lambda_value))
