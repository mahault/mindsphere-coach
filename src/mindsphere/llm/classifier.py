"""
SphereClassifier: converts user text into structured observations for the POMDP.

Adapted from NEXT-prototype's MistralClassifier pattern.
Three classification modes: MC answers, free text, and user choices.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .client import MistralClient, MistralAPIError

CLASSIFIER_SYSTEM_PROMPT = """\
You are an observation classifier for a personal coaching system called MindSphere Coach.
Your job is to analyze a user's response and classify it into structured observations.

You will be given the context (what type of input to classify) in the user message.

Always respond with a valid JSON object, no other text."""

MC_CLASSIFY_PROMPT = """\
The user answered a multiple-choice question about "{category}".

Question: {question}
Options:
{options_text}

User's answer: "{answer}"

Classify into JSON:
{{
    "answer_index": <int 0-{max_idx}>,
    "confidence": "low" | "medium" | "high",
    "engagement_signal": "low" | "medium" | "high"
}}"""

FREE_TEXT_CLASSIFY_PROMPT = """\
The user gave a free-text response to the coaching question: "{question}"

User's response: "{answer}"

Classify into JSON:
{{
    "primary_categories": [list of relevant skills from: "focus", "follow_through", "social_courage", "emotional_reg", "systems_thinking", "self_trust", "task_clarity", "consistency"],
    "skill_signals": {{category: "very_low" | "low" | "medium" | "high" | "very_high"}},
    "emotional_tone": "resistant" | "neutral" | "engaged" | "enthusiastic",
    "friction_signal": "avoidant" | "neutral" | "eager",
    "autonomy_signal": "compliant" | "neutral" | "assertive",
    "overwhelm_signal": "low" | "medium" | "high"
}}"""

CHOICE_CLASSIFY_PROMPT = """\
The user was presented with a coaching micro-step and responded.

Proposed step: "{intervention_description}"
User's response: "{answer}"

Classify into JSON:
{{
    "choice": "accept" | "too_hard" | "not_relevant",
    "engagement": "low" | "medium" | "high",
    "resistance_type": "none" | "overwhelm" | "boredom" | "uncertainty" | "self_doubt" | "resentment"
}}"""

EMOTION_CLASSIFY_PROMPT = """\
Analyze the emotional content of this message using the Circumplex Model of Emotion \
(two dimensions: valence and arousal).

User's message: "{answer}"

Context: {context}

Classify into JSON:
{{
    "valence": "very_negative" | "negative" | "neutral" | "positive" | "very_positive",
    "arousal": "very_low" | "low" | "moderate" | "high" | "very_high",
    "primary_emotion": "happy" | "excited" | "alert" | "angry" | "sad" | "depressed" | "calm" | "relaxed",
    "confidence": "low" | "medium" | "high",
    "emotional_cues": [list of specific words/phrases that signal the emotion]
}}

Guidelines:
- Valence = how positive or negative the person feels (prediction error: better or worse than expected)
- Arousal = how activated or calm the person is (uncertainty/alertness level)
- "I'm bored" = negative valence + low arousal = depressed quadrant
- "This is exciting!" = positive valence + high arousal = excited quadrant
- "I'm stressed about work" = negative valence + high arousal = angry/anxious quadrant
- "I feel peaceful today" = positive valence + low arousal = relaxed quadrant
- Short, flat responses suggest low arousal
- Exclamation marks, urgent language suggest high arousal
- Complaints, frustration, sadness suggest negative valence
- Curiosity, enthusiasm, gratitude suggest positive valence"""


class SphereClassifier:
    """
    Converts user text into structured observations for the POMDP.

    Acts as the "sensor" in the active inference loop.
    Falls back to defaults on API errors to keep the session running.

    Includes emotional classification for the Circumplex Model:
    classifies user text into valence/arousal observations that
    become formal POMDP observations for emotional state inference.
    """

    DEFAULT_MC_RESULT = {
        "answer_index": 1,
        "confidence": "medium",
        "engagement_signal": "medium",
    }

    DEFAULT_FREE_TEXT_RESULT = {
        "primary_categories": [],
        "skill_signals": {},
        "emotional_tone": "neutral",
        "friction_signal": "neutral",
        "autonomy_signal": "neutral",
        "overwhelm_signal": "medium",
    }

    DEFAULT_CHOICE_RESULT = {
        "choice": "accept",
        "engagement": "medium",
        "resistance_type": "none",
    }

    DEFAULT_EMOTION_RESULT = {
        "valence": "neutral",
        "arousal": "moderate",
        "primary_emotion": "calm",
        "confidence": "medium",
        "emotional_cues": [],
    }

    # Maps for converting string levels to observation indices
    VALENCE_INDEX_MAP = {
        "very_negative": 0, "negative": 1, "neutral": 2,
        "positive": 3, "very_positive": 4,
    }
    AROUSAL_INDEX_MAP = {
        "very_low": 0, "low": 1, "moderate": 2,
        "high": 3, "very_high": 4,
    }

    def __init__(self, client: MistralClient):
        self.client = client

    def classify_mc_answer(
        self,
        answer_text: str,
        question: str,
        category: str,
        options: List[str],
    ) -> Dict[str, Any]:
        """Classify a multiple-choice answer."""
        options_text = "\n".join(f"  {i}. {opt}" for i, opt in enumerate(options))
        prompt = MC_CLASSIFY_PROMPT.format(
            category=category,
            question=question,
            options_text=options_text,
            answer=answer_text,
            max_idx=len(options) - 1,
        )
        return self._call_llm(prompt, self.DEFAULT_MC_RESULT)

    def classify_free_text(
        self,
        answer_text: str,
        question: str,
    ) -> Dict[str, Any]:
        """Classify a free-text response."""
        prompt = FREE_TEXT_CLASSIFY_PROMPT.format(
            question=question,
            answer=answer_text,
        )
        return self._call_llm(prompt, self.DEFAULT_FREE_TEXT_RESULT)

    def classify_user_choice(
        self,
        answer_text: str,
        intervention_description: str,
    ) -> Dict[str, Any]:
        """Classify a user's response to a proposed intervention."""
        prompt = CHOICE_CLASSIFY_PROMPT.format(
            intervention_description=intervention_description,
            answer=answer_text,
        )
        return self._call_llm(prompt, self.DEFAULT_CHOICE_RESULT)

    def classify_emotion(
        self,
        answer_text: str,
        context: str = "general coaching conversation",
    ) -> Dict[str, Any]:
        """
        Classify user text into circumplex emotional observations.

        Returns valence (0-4) and arousal (0-4) indices plus emotion label.
        These become formal POMDP observations for the emotional state model.
        """
        prompt = EMOTION_CLASSIFY_PROMPT.format(
            answer=answer_text,
            context=context,
        )
        result = self._call_llm(prompt, self.DEFAULT_EMOTION_RESULT)

        # Convert string levels to indices
        result["valence_idx"] = self.VALENCE_INDEX_MAP.get(
            result.get("valence", "neutral"), 2
        )
        result["arousal_idx"] = self.AROUSAL_INDEX_MAP.get(
            result.get("arousal", "moderate"), 2
        )
        return result

    def classify_emotion_heuristic(
        self,
        answer_text: str,
    ) -> Dict[str, Any]:
        """
        Heuristic emotion classification — fallback when LLM is unavailable.

        Uses keyword matching to estimate valence and arousal.
        Less accurate than LLM classification but always available.
        """
        lower = answer_text.lower()

        # Valence detection
        negative_words = [
            "stressed", "anxious", "worried", "frustrated", "stuck",
            "hopeless", "lost", "confused", "scared", "tired",
            "burned out", "overwhelmed", "sad", "angry", "annoyed",
            "hate", "terrible", "awful", "bad", "depressed",
            "bored", "boring", "meh", "ugh",
        ]
        positive_words = [
            "good", "great", "happy", "excited", "curious",
            "interesting", "love", "amazing", "better", "hopeful",
            "motivated", "ready", "clear", "peaceful", "calm",
            "relaxed", "grateful", "proud", "confident",
        ]

        neg_count = sum(1 for w in negative_words if w in lower)
        pos_count = sum(1 for w in positive_words if w in lower)

        if neg_count > pos_count + 1:
            valence = "very_negative"
        elif neg_count > pos_count:
            valence = "negative"
        elif pos_count > neg_count + 1:
            valence = "very_positive"
        elif pos_count > neg_count:
            valence = "positive"
        else:
            valence = "neutral"

        # Arousal detection
        high_arousal_words = [
            "!", "stressed", "anxious", "excited", "angry",
            "can't believe", "urgent", "help", "panic", "scared",
            "amazing", "overwhelmed",
        ]
        low_arousal_words = [
            "bored", "tired", "calm", "peaceful", "relaxed",
            "meh", "whatever", "ok", "fine", "sleepy",
        ]

        high_count = sum(1 for w in high_arousal_words if w in lower)
        low_count = sum(1 for w in low_arousal_words if w in lower)

        # Short messages suggest low arousal
        if len(answer_text.strip()) < 10:
            low_count += 1

        if high_count > low_count + 1:
            arousal = "very_high"
        elif high_count > low_count:
            arousal = "high"
        elif low_count > high_count + 1:
            arousal = "very_low"
        elif low_count > high_count:
            arousal = "low"
        else:
            arousal = "moderate"

        # Map to emotion label
        v_idx = self.VALENCE_INDEX_MAP[valence]
        a_idx = self.AROUSAL_INDEX_MAP[arousal]

        # Simple circumplex mapping from indices
        if v_idx >= 3 and a_idx >= 3:
            emotion = "excited"
        elif v_idx >= 3 and a_idx <= 1:
            emotion = "relaxed"
        elif v_idx <= 1 and a_idx >= 3:
            emotion = "angry"
        elif v_idx <= 1 and a_idx <= 1:
            emotion = "depressed"
        elif v_idx >= 3:
            emotion = "happy"
        elif v_idx <= 1:
            emotion = "sad"
        elif a_idx >= 3:
            emotion = "alert"
        else:
            emotion = "calm"

        return {
            "valence": valence,
            "arousal": arousal,
            "valence_idx": v_idx,
            "arousal_idx": a_idx,
            "primary_emotion": emotion,
            "confidence": "medium",
            "emotional_cues": [],
        }

    def _call_llm(self, user_prompt: str, default: Dict) -> Dict[str, Any]:
        """Call the LLM and parse JSON response, with fallback."""
        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.05,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            return self._validate_and_repair(json.loads(response), default)
        except (MistralAPIError, json.JSONDecodeError, KeyError):
            return default.copy()

    def _validate_and_repair(
        self, result: Dict, default: Dict
    ) -> Dict[str, Any]:
        """Ensure result has all required keys, fill missing with defaults."""
        repaired = default.copy()
        repaired.update({k: v for k, v in result.items() if k in default})
        return repaired
