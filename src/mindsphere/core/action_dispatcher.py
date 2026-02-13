"""
EFE-driven action selection dispatcher.

Bridges the POMDP inference (select_action, compute_efe_all_factors) to
actual agent behavior.  Handles:
  - Valid action masks per phase
  - Dynamic lambda_epist scheduling (epistemic ↔ pragmatic balance)
  - Empathy blending for intervention-type actions
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .model import SphereModel, ACTION_NAMES, SKILL_FACTORS
from .inference import compute_efe_all_factors, select_action
from .utils import entropy

logger = logging.getLogger(__name__)

# ── Action indices (convenience aliases) ────────────────────────────────────
A_ASK_MC = 0
A_ASK_FREE = 1
A_SHOW_SPHERE = 2
A_PROPOSE = 3
A_REFRAME = 4
A_ADJUST = 5
A_COUNTERFACTUAL = 6
A_SAFETY = 7
A_END = 8

# ── Valid action masks per phase ────────────────────────────────────────────
VALID_ACTIONS: Dict[str, List[int]] = {
    "visualization": [A_ASK_FREE, A_SHOW_SPHERE, A_PROPOSE],
    "planning":      [A_ASK_FREE, A_PROPOSE, A_REFRAME, A_COUNTERFACTUAL, A_SAFETY, A_END],
    "coaching":      [A_ASK_FREE, A_PROPOSE, A_REFRAME, A_ADJUST, A_COUNTERFACTUAL, A_SAFETY, A_END],
    "update":        [A_ASK_FREE, A_PROPOSE, A_REFRAME, A_ADJUST, A_COUNTERFACTUAL, A_SAFETY],
}


# ── Dynamic lambda_epist ───────────────────────────────────────────────────

def compute_lambda_epist(
    phase: str,
    timestep: int,
    tom_reliability: float,
    beliefs: Optional[Dict[str, np.ndarray]] = None,
    emotion_prediction_error: float = 0.0,
) -> float:
    """
    Compute the epistemic drive weight, balancing exploration vs exploitation.

    Starts high (explore the user) and decays toward pragmatic (propose actions).
    Boosted when ToM reliability is low, beliefs are uncertain, or emotional
    prediction error is high (model is wrong about user's emotional state).
    """
    # Phase-dependent base value
    phase_base = {
        "visualization": 2.0,
        "planning": 0.8,
        "coaching": 0.5,
        "update": 0.6,
    }.get(phase, 1.0)

    # Temporal decay: linear from 1.0 → 0.3 over first 30 timesteps
    temporal_factor = max(0.3, 1.0 - 0.7 * min(timestep / 30.0, 1.0))

    # Uncertainty boost: average belief entropy / max possible entropy
    uncertainty_factor = 1.0
    if beliefs:
        skill_beliefs = [beliefs[s] for s in SKILL_FACTORS if s in beliefs]
        if skill_beliefs:
            max_ent = np.log(len(skill_beliefs[0]))  # log(5) for 5 levels
            avg_ent = float(np.mean([entropy(b) for b in skill_beliefs]))
            uncertainty_factor = 1.0 + 0.5 * (avg_ent / max(max_ent, 1e-12))

    # Reliability discount: boost epistemic when ToM is unreliable
    reliability_factor = 1.0 + max(0.0, 0.5 - tom_reliability)

    # Emotion prediction error boost: when our emotional model is wrong,
    # increase exploration (ask questions, don't push interventions)
    emotion_factor = 1.0
    if emotion_prediction_error > 0.2:
        emotion_factor = 1.0 + 0.8 * min(emotion_prediction_error, 1.0)

    result = phase_base * temporal_factor * uncertainty_factor * reliability_factor * emotion_factor
    return result


# ── Main action selector ───────────────────────────────────────────────────

def select_coaching_action(
    beliefs: Dict[str, np.ndarray],
    model: SphereModel,
    phase: str,
    timestep: int,
    tom_reliability: float,
    empathy_planner=None,
    tom_filter=None,
    target_skill: Optional[str] = None,
    current_intervention=None,
    beta: float = 4.0,
    emotion_prediction_error: float = 0.0,
    emotion_valence_belief: Optional[np.ndarray] = None,
) -> Tuple[int, str, Dict]:
    """
    Full EFE-driven action selection with empathy blending.

    Args:
        beliefs: Current beliefs per factor
        model: The POMDP model
        phase: Current phase (visualization, planning, coaching, update)
        timestep: Current session timestep
        tom_reliability: ToM particle filter reliability [0, 1]
        empathy_planner: EmpathyPlanner for G_social blending (optional)
        tom_filter: UserTypeFilter for user predictions (optional)
        target_skill: Currently targeted skill (optional)
        current_intervention: Current intervention dict (optional)
        beta: Inverse temperature for softmax
        emotion_prediction_error: Magnitude of emotional prediction error [0, ~1.4]
        emotion_valence_belief: 5-element belief over valence states (optional)

    Returns:
        (action_idx, action_name, efe_info_dict)
    """
    valid = VALID_ACTIONS.get(phase, [A_ASK_FREE, A_PROPOSE])

    lambda_epist = compute_lambda_epist(
        phase, timestep, tom_reliability, beliefs,
        emotion_prediction_error=emotion_prediction_error,
    )

    # Determine which factors are relevant
    relevant_factors = None
    if target_skill and target_skill in SKILL_FACTORS:
        relevant_factors = [target_skill]
    # If no target, use all skill factors (default in select_action)

    # Step 1: Compute raw system EFE and select
    action_idx, action_probs, efe_values = select_action(
        beliefs=beliefs,
        model=model,
        valid_actions=valid,
        lambda_epist=lambda_epist,
        beta=beta,
        relevant_factors=relevant_factors,
    )

    # Step 2: Blend with empathy for intervention-type actions
    # Only applies when we have empathy planner + ToM + an intervention context
    intervention_actions = {A_PROPOSE, A_REFRAME, A_ADJUST}
    if empathy_planner is not None and tom_filter is not None:
        # Compute G_social for each valid action
        blended_values = np.zeros(len(valid), dtype=np.float64)
        for i, v in enumerate(valid):
            if v in intervention_actions and current_intervention is not None:
                # Get ToM prediction for this intervention
                iv_dict = (
                    current_intervention.to_dict()
                    if hasattr(current_intervention, "to_dict")
                    else current_intervention
                    if isinstance(current_intervention, dict)
                    else {"difficulty": 0.3, "duration_minutes": 5}
                )
                prediction = tom_filter.predict_response_gated(iv_dict)
                blended_values[i] = empathy_planner.compute_blended_efe(
                    system_efe=efe_values[i],
                    user_felt_cost=prediction["predicted_felt_cost"],
                    reliability=tom_reliability,
                )
            else:
                # Non-intervention actions: use system EFE directly
                blended_values[i] = efe_values[i]

        # Re-select based on blended values
        from .utils import softmax as _softmax
        q_values = -blended_values
        blended_probs = _softmax(q_values, temperature=1.0 / max(beta, 0.01))
        best_idx = int(np.argmax(blended_probs))
        action_idx = valid[best_idx]
        action_probs = blended_probs
        efe_values = blended_values

    # Step 3: Emotional valence penalty on intervention actions
    # When the user is in a negative emotional state, penalize pushing
    # coaching interventions — prefer softer actions (ask, safety_check)
    valence_negative_mass = None
    if emotion_valence_belief is not None and len(emotion_valence_belief) >= 5:
        valence_negative_mass = float(emotion_valence_belief[0] + emotion_valence_belief[1])
        if valence_negative_mass > 0.4:
            penalty = 0.5 * valence_negative_mass
            for i, v in enumerate(valid):
                if v in intervention_actions:
                    efe_values[i] += penalty  # Higher G = worse action
            # Re-select with penalty applied
            from .utils import softmax as _softmax
            q_values = -efe_values
            penalized_probs = _softmax(q_values, temperature=1.0 / max(beta, 0.01))
            best_idx = int(np.argmax(penalized_probs))
            action_idx = valid[best_idx]
            action_probs = penalized_probs
            logger.info(
                f"[EFE] Emotional valence penalty applied (neg_mass={valence_negative_mass:.2f}, "
                f"penalty={penalty:.2f})"
            )

    action_name = ACTION_NAMES[action_idx]

    info = {
        "selected_action": action_name,
        "action_probabilities": {
            ACTION_NAMES[v]: round(float(p), 4)
            for v, p in zip(valid, action_probs)
        },
        "efe_values": {
            ACTION_NAMES[v]: round(float(e), 4)
            for v, e in zip(valid, efe_values)
        },
        "lambda_epist": round(lambda_epist, 3),
        "phase": phase,
        "emotion_prediction_error": round(emotion_prediction_error, 3),
        "valence_negative_mass": round(valence_negative_mass, 3) if valence_negative_mass is not None else None,
    }

    logger.info(
        f"[EFE] Phase={phase} → {action_name} "
        f"(λ_epist={lambda_epist:.2f}, probs={info['action_probabilities']})"
    )

    return action_idx, action_name, info
