"""
CoachGenerator: LLM-powered conversational engine for MindSphere Coach.

Routes ALL post-calibration conversation through Mistral, with dynamic
belief context, ToM predictions, and cognitive load awareness injected
into the system prompt.

The agent is a warm companion first, a coach second. It follows the
user's lead, talks about anything, and returns to coaching naturally
when the user is ready — as inferred through ToM.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .client import MistralClient, MistralAPIError

logger = logging.getLogger(__name__)


# The model to use for conversation (needs web_search support)
CONVERSATION_MODEL = "mistral-medium-latest"

# Web search tool definition for Mistral
# NOTE: web_search format varies by Mistral API version.
# Set to None to disable if the API returns 422 errors.
WEB_SEARCH_TOOL = None


def build_system_prompt(
    phase: str,
    belief_summary: Optional[Dict[str, Any]] = None,
    tom_summary: Optional[Dict[str, float]] = None,
    cognitive_load: Optional[Dict[str, Any]] = None,
    target_skill: Optional[str] = None,
    current_intervention: Optional[Dict[str, Any]] = None,
    accepted_interventions: Optional[List[Dict]] = None,
    profile_section: Optional[str] = None,
) -> str:
    """
    Build a dynamic system prompt that includes the user's current state.

    This is the core of making the agent feel alive — the LLM sees
    the full picture of what we know about the user and adapts naturally.
    """
    parts = [COMPANION_CORE_PROMPT]

    # Inject current phase context
    parts.append(f"\n\n## Current Session State\nPhase: {phase}")

    # Inject belief context if available
    if belief_summary:
        skill_lines = []
        for key, val in belief_summary.items():
            if key in ("tom_reliability", "user_type"):
                continue
            if isinstance(val, dict) and "score" in val:
                skill_lines.append(f"  - {key.replace('_', ' ').title()}: {val['score']}/100")
            elif isinstance(val, dict) and "inferred" in val:
                skill_lines.append(f"  - {key.replace('_', ' ').title()}: {val['inferred']} (confidence: {round(val.get('confidence', 0) * 100)}%)")

        if skill_lines:
            parts.append("\n\n## User's Skill Profile (from calibration)")
            parts.append("\n".join(skill_lines))

    # Inject ToM predictions
    if tom_summary:
        tom_lines = []
        dim_labels = {
            "avoids_evaluation": "Avoids evaluative tasks",
            "hates_long_tasks": "Prefers short tasks",
            "novelty_seeking": "Drawn to novelty",
            "structure_preference": "Prefers structure",
            "external_validation": "Needs external validation",
            "autonomy_sensitivity": "Values autonomy",
            "overwhelm_threshold": "Overwhelm threshold",
        }
        for dim, val in tom_summary.items():
            label = dim_labels.get(dim, dim.replace("_", " ").title())
            level = "low" if val < 0.35 else "medium" if val < 0.65 else "high"
            tom_lines.append(f"  - {label}: {level} ({val:.2f})")

        if tom_lines:
            parts.append("\n\n## Theory of Mind — Inferred User Type")
            parts.append("\n".join(tom_lines))

    # Inject cognitive load assessment
    if cognitive_load:
        load_level = cognitive_load.get("level", "unknown")
        readiness = cognitive_load.get("coaching_readiness", "unknown")
        parts.append(f"\n\n## Cognitive Load Assessment")
        parts.append(f"  - Current load: {load_level}")
        parts.append(f"  - Coaching readiness: {readiness}")
        if cognitive_load.get("signals"):
            parts.append(f"  - Signals: {', '.join(cognitive_load['signals'])}")
        if cognitive_load.get("inferred_emotions"):
            parts.append(f"  - Recent emotions detected: {', '.join(cognitive_load['inferred_emotions'])}")
        if cognitive_load.get("inferred_topics"):
            parts.append(f"  - Topics they care about: {', '.join(cognitive_load['inferred_topics'])}")
        if cognitive_load.get("engagement_level"):
            parts.append(f"  - Engagement level: {cognitive_load['engagement_level']}")

        # Circumplex emotional state (from Pattisapu & Albarracin 2024)
        if cognitive_load.get("circumplex_emotion"):
            parts.append(f"\n\n## Emotional State (Circumplex Model)")
            parts.append(f"  - Current emotion: {cognitive_load['circumplex_emotion']}")
            parts.append(f"  - Valence: {cognitive_load.get('circumplex_valence', 0)} (negative=bad, positive=good)")
            parts.append(f"  - Arousal: {cognitive_load.get('circumplex_arousal', 0.5)} (low=calm, high=activated)")

            if cognitive_load.get("predicted_emotion") and cognitive_load.get("observed_emotion"):
                pred = cognitive_load["predicted_emotion"]
                obs = cognitive_load["observed_emotion"]
                err = cognitive_load.get("emotion_prediction_error", 0)
                if pred != obs:
                    parts.append(f"  - PREDICTION ERROR: I predicted '{pred}' but observed '{obs}' (error={err})")
                    parts.append(f"    This means my model of this person needs updating. Adjust your response accordingly.")
                else:
                    parts.append(f"  - Prediction validated: predicted '{pred}', observed '{obs}' (error={err})")
                    parts.append(f"    My model of this person is tracking well.")

    # Inject coaching context if in coaching-relevant phases
    if phase in ("planning", "update", "coaching"):
        if target_skill:
            parts.append(f"\n\n## Current Coaching Focus")
            parts.append(f"  Target skill: {target_skill.replace('_', ' ').title()}")

        if current_intervention:
            parts.append(f"  Current suggestion: \"{current_intervention.get('description', '')}\"")
            parts.append(f"  Duration: {current_intervention.get('duration_minutes', '?')} minutes")

        if accepted_interventions:
            steps = [i.get("description", "?") for i in accepted_interventions[-3:]]
            parts.append(f"  Accepted steps so far: {'; '.join(steps)}")

    # Inject user profile and causal model (Bayesian network)
    if profile_section:
        parts.append(profile_section)

    # Phase-specific behavioral guidance
    if phase == "calibration":
        parts.append(PHASE_GUIDANCE_CALIBRATION)
    elif phase == "sphere_commentary":
        parts.append(PHASE_GUIDANCE_SPHERE)
    elif phase == "visualization":
        parts.append(PHASE_GUIDANCE_VIZ)
    elif phase == "planning":
        parts.append(PHASE_GUIDANCE_PLANNING)
    elif phase == "coaching":
        parts.append(PHASE_GUIDANCE_COACHING)
    elif phase == "complete":
        parts.append(PHASE_GUIDANCE_COMPLETE)

    return "\n".join(parts)


COMPANION_CORE_PROMPT = """\
You are MindSphere Coach — a warm, perceptive companion powered by an Active Inference engine with Theory of Mind.

## Your Core Identity
You are a COMPANION who also coaches. You genuinely care about the person you're talking to. You can talk about anything — their day, their interests, what's on their mind, the news, random curiosities. Coaching happens naturally, woven into conversation when opportunities arise — not announced or forced.

## Communication Style
- Warm but not saccharine — genuine, not performative
- Direct but not blunt — clear, respectful
- Curious, not interrogating — you're interested, not testing
- Brief — keep responses to 2-4 sentences MAX. Say ONE thing, not three variations of the same thing. If you ask a question, ask ONE question and stop. Resist the urge to offer multiple options — pick the best one and commit.
- Never use jargon, buzzwords, or therapeutic clichés
- Match the user's energy — if they're playful, be playful; if they're heavy, be present
- NEVER say "Thanks for sharing that" followed by steering to coaching. That's what a bad chatbot does.
- When the user wants to change the subject, FULLY change the subject. Do not sneak coaching back in. Do not add "by the way" coaching at the end. Just go where they want to go.

## How Coaching Works
You don't ask "want me to coach you?" or "ready for a suggestion?" — that's clunky. Instead:
- You notice patterns in what they say and gently reflect them back
- You make connections between what they're telling you and what the model shows
- You drop small, actionable insights when the moment is right — mid-conversation, not as a separate "coaching phase"
- If they're talking about procrastinating, and the model shows low Task Clarity, you might say "I wonder if the issue isn't motivation but knowing what 'done' looks like"
- If they're venting about being overwhelmed, and ToM shows high autonomy sensitivity, you don't add more structure — you help them find their own path
- Coaching is invisible when done well. It's just being a really perceptive friend.

## Key Behavioral Rules
1. **Follow the user's lead.** If they want to talk about Jeffrey Epstein, talk about Jeffrey Epstein. If they want to talk about their cat, talk about their cat. Be a real conversationalist.
2. **Coach through the conversation, not at it.** Don't announce coaching. Don't ask permission. Just be perceptive and helpful in the flow of whatever you're discussing.
3. **Infer cognitive load.** Use the ToM data to sense when the user is overwhelmed, disengaged, or not in the headspace for insight. Back off naturally. Drop coaching seeds when they're receptive.
4. **Be transparent about your models when asked.** You're powered by computational models. If someone asks how you know something, explain. But don't volunteer "my model says..." unprompted — it breaks the flow.
5. **Use web search** when the user asks about current events, facts, or anything you're not sure about. Don't make things up.
6. **Remember context.** Reference things the user said earlier. Build on the conversation. Feel like a real person, not a stateless bot.

## What You Know
You have access to an Active Inference model that has computed:
- The user's skill profile across 8 dimensions (0-100 each)
- Their behavioral type (via particle filter Theory of Mind)
- Their predicted responses to different coaching approaches
- Skill dependencies (which skills are bottlenecks for others)

This information is injected below when available. Use it to inform your responses — not to recite numbers, but to be genuinely perceptive about what's going on for this person."""

PHASE_GUIDANCE_VIZ = """

## Phase Guidance: Visualization
The user just saw their skill sphere. They may be curious, surprised, defensive, dismissive — or they may want to talk about something else entirely. Your job:
- If they react to the sphere, discuss it naturally — what resonates, what doesn't
- If they bring up something unrelated, go with it. The sphere is just a conversation starter, not a cage.
- Be open to being wrong — the model is built from 10 questions, it's imperfect
- Don't ask "want me to suggest something?" — if a coaching moment arises naturally, take it. Otherwise, just be present."""

PHASE_GUIDANCE_PLANNING = """

## Phase Guidance: Planning
A coaching intervention has been identified by the system. You don't need to present it formally — weave it in naturally.
- If there's a natural opening in conversation, share the insight or suggestion casually
- If the user is talking about something else, keep talking about that — the coaching can wait
- If they seem receptive, go deeper. If not, back off without drawing attention to it.
- Never say "so what do you think about the step?" — that's a chatbot move"""

PHASE_GUIDANCE_COACHING = """

## Phase Guidance: Coaching
The user is in an ongoing conversation. Coach through the conversation itself:
- Be a companion. Chat about anything they want to talk about.
- Notice patterns in what they say and reflect them back when it feels right
- Offer exercises, reframes, or insights when the moment is natural — not on a schedule
- If they're processing emotions, be present — don't rush to solutions
- If they want to go off-topic, go off-topic COMPLETELY. You're a person, not a protocol. Don't loop back to coaching in the same message. Stay on THEIR topic until THEY bring coaching back.
- Only end the session if the user says they want to stop
- CRITICAL: Keep it short. One thought, one question, done. Never offer multiple options or backup suggestions in the same message."""

PHASE_GUIDANCE_COMPLETE = """

## Phase Guidance: Complete
The session is winding down. Keep being a good conversationalist.
- Be warm and reflective
- Don't re-coach unless they ask
- Answer questions, chat, be helpful"""

PHASE_GUIDANCE_CALIBRATION = """

## Phase Guidance: Calibration
You're getting to know this person through calibration questions. Your job right now:
- Acknowledge what they shared naturally and briefly (1 sentence, maybe 2)
- If they share something emotional or personal, respond with genuine warmth — don't brush past it
- NEVER say generic things like "Got it." or "Thanks for sharing that." or "Good to know." — those are chatbot clichés
- Match the emotional weight of what they said: light answer gets a light ack, heavy answer gets a warmer response
- Don't coach yet — you're still learning about them
- Don't ask follow-up questions — the next calibration question is coming automatically
- NEVER add meta-text like "[Next question loading...]" or "[Next calibration question loading...]" — that breaks immersion. Just respond to what they said and stop.
- Keep it SHORT — one sentence is perfect, two max
- Just write your acknowledgment. Nothing else. No brackets, no loading messages, no system notes."""

PHASE_GUIDANCE_SPHERE = """

## Phase Guidance: Sphere Commentary
You're presenting the user's MindSphere results for the first time. Your job:
- Present the sphere warmly and conversationally — not as a clinical readout
- Mention their strongest and weakest areas naturally, with scores
- If there's a bottleneck, explain the dependency in plain language
- Be curious about whether it resonates — invite their reaction
- Don't overwhelm with numbers. Lead with patterns and insights.
- Keep it to 3-5 sentences"""


class CoachGenerator:
    """
    LLM-powered conversation engine for MindSphere Coach.

    Routes all post-calibration conversation through Mistral with
    dynamic context injection. Falls back to template responses
    when the LLM is unavailable.
    """

    def __init__(self, client: Optional[MistralClient] = None):
        self.client = client
        self._available: Optional[bool] = None

    @property
    def is_available(self) -> bool:
        """Check if LLM generation is available."""
        if self._available is not None:
            return self._available
        if self.client is None:
            self._available = False
            return False
        self._available = self.client.is_available
        return self._available

    def generate(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        phase: str,
        belief_summary: Optional[Dict[str, Any]] = None,
        tom_summary: Optional[Dict[str, float]] = None,
        cognitive_load: Optional[Dict[str, Any]] = None,
        target_skill: Optional[str] = None,
        current_intervention: Optional[Dict[str, Any]] = None,
        accepted_interventions: Optional[List[Dict]] = None,
        enable_web_search: bool = True,
        profile_section: Optional[str] = None,
    ) -> str:
        """
        Generate a conversational response using Mistral LLM.

        The system prompt is dynamically built from the user's current
        belief state, ToM predictions, cognitive load assessment, and
        semantic profile with Bayesian network causal model.

        Args:
            user_message: The user's most recent message
            conversation_history: Prior conversation for context
            phase: Current coaching phase
            belief_summary: Skill scores + preference/friction beliefs
            tom_summary: ToM user type dimensions
            cognitive_load: Inferred cognitive load assessment
            target_skill: Current coaching target skill
            current_intervention: Current proposed intervention
            accepted_interventions: Previously accepted steps
            enable_web_search: Whether to enable Mistral web search
            profile_section: Formatted user profile + causal model for prompt

        Returns:
            Natural language response string
        """
        if not self.is_available:
            logger.warning("[CoachGenerator] Not available — no API key")
            return ""  # Caller should fall back to templates

        system_prompt = build_system_prompt(
            phase=phase,
            belief_summary=belief_summary,
            tom_summary=tom_summary,
            cognitive_load=cognitive_load,
            target_skill=target_skill,
            current_intervention=current_intervention,
            accepted_interventions=accepted_interventions,
            profile_section=profile_section,
        )

        # Build messages — system + recent conversation + current message
        messages = [{"role": "system", "content": system_prompt}]

        # Include recent conversation history (last 10 turns for context)
        for msg in conversation_history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Add the current user message
        messages.append({"role": "user", "content": user_message})

        # Log message structure for debugging
        roles = [m["role"] for m in messages]
        logger.info(f"[CoachGenerator] Sending {len(messages)} messages to Mistral (roles: {roles[-5:]})")

        # Web search tool (disabled if WEB_SEARCH_TOOL is None)
        tools = [WEB_SEARCH_TOOL] if (enable_web_search and WEB_SEARCH_TOOL) else None

        try:
            response = self.client.chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=300,
                tools=tools,
                model_override=CONVERSATION_MODEL,
            )
            if response and response.strip():
                logger.info(f"[CoachGenerator] Mistral responded ({len(response)} chars)")
                return response.strip()
            logger.warning("[CoachGenerator] Empty response from Mistral")
            return ""
        except Exception as e:
            logger.warning(f"[CoachGenerator] Exception: {e}")
            return ""  # Fall back to templates
