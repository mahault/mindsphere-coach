"""
Empathic planning: G_social blending and counterfactual computation.

Implements the core empathic EFE formula:
    G_social(a) = (1 - lambda_eff) * G_system(a) + lambda_eff * E[G_user(a)]

Where:
    - G_system: progress toward rounder sphere (closing dents)
    - G_user: predicted felt cost (overwhelm, aversiveness)
    - lambda_eff = lambda_base * reliability
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .particle_filter import UserTypeFilter
from ..core.utils import softmax


class EmpathyPlanner:
    """
    Empathy-aware action selection that balances system goals with user welfare.

    The empathy dial (lambda) controls how much the planner weighs
    the user's predicted felt cost vs. coaching progress.

    - lambda=0: pure system optimization (max progress, may overwhelm)
    - lambda=1: pure user comfort (minimal friction, may stall)
    - lambda=0.5: balanced (default)
    """

    def __init__(self, lambda_empathy: float = 0.5, beta: float = 4.0):
        self.lambda_empathy = lambda_empathy
        self.beta = beta

    def compute_blended_efe(
        self,
        system_efe: float,
        user_felt_cost: float,
        reliability: float,
    ) -> float:
        """
        Compute empathy-blended EFE for an action.

        G_social = (1 - lambda_eff) * G_system + lambda_eff * G_user
        lambda_eff = lambda_base * reliability

        When reliability is low, lambda_eff shrinks → more conservative
        (system-only planning). When reliability is high, empathy fully engaged.

        Args:
            system_efe: EFE from core inference (lower = better for system)
            user_felt_cost: Predicted felt cost [0,1] (higher = worse for user)
            reliability: Current ToM reliability [0,1]

        Returns:
            G_social: Blended EFE (lower = better action)
        """
        lambda_eff = self.lambda_empathy * reliability

        # G_user: convert felt cost to EFE-like quantity
        # Higher felt cost → higher G_user (worse)
        G_user = user_felt_cost * 5.0  # Scale to match G_system magnitude

        G_social = (1 - lambda_eff) * system_efe + lambda_eff * G_user
        return G_social

    def select_empathic_action(
        self,
        system_efes: Dict[str, float],
        interventions: Dict[str, Dict],
        user_filter: UserTypeFilter,
    ) -> Tuple[str, Dict]:
        """
        Select the best action considering both system progress and user welfare.

        Args:
            system_efes: Dict mapping action_name -> system EFE
            interventions: Dict mapping action_name -> intervention details
            user_filter: Particle filter for predicting user responses

        Returns:
            (best_action_name, info_dict)
        """
        reliability = user_filter.reliability
        action_names = list(system_efes.keys())
        G_social_values = np.zeros(len(action_names))

        action_details = {}
        for i, action_name in enumerate(action_names):
            intervention = interventions.get(action_name, {})
            prediction = user_filter.predict_response_gated(intervention)

            G_social_values[i] = self.compute_blended_efe(
                system_efes[action_name],
                prediction["predicted_felt_cost"],
                reliability,
            )

            action_details[action_name] = {
                "G_system": system_efes[action_name],
                "G_user": prediction["predicted_felt_cost"],
                "G_social": float(G_social_values[i]),
                "p_accept": prediction["p_accept"],
                "p_too_hard": prediction["p_too_hard"],
            }

        # Select via softmax over negative G_social
        q_values = -G_social_values
        probs = softmax(q_values, temperature=1.0 / max(self.beta, 0.01))

        best_idx = int(np.argmax(probs))
        best_action = action_names[best_idx]

        return best_action, {
            "selected": best_action,
            "action_probs": {n: float(p) for n, p in zip(action_names, probs)},
            "details": action_details,
            "lambda_effective": self.lambda_empathy * reliability,
            "reliability": reliability,
        }

    def compute_counterfactual(
        self,
        gentle_intervention: Dict,
        push_intervention: Dict,
        user_filter: UserTypeFilter,
    ) -> Dict:
        """
        Compute counterfactual comparison between gentle and push approaches.

        This is the "wow" display:
        "If I push hard → 70% dropout; 2-min version → 85% completion"

        Args:
            gentle_intervention: Low-friction intervention details
            push_intervention: High-friction intervention details
            user_filter: Particle filter for user predictions

        Returns:
            Counterfactual comparison dict
        """
        gentle_pred = user_filter.predict_response_gated(gentle_intervention)
        push_pred = user_filter.predict_response_gated(push_intervention)

        return {
            "gentle": {
                "description": gentle_intervention.get("description", ""),
                "duration_minutes": gentle_intervention.get("duration_minutes", 2),
                "p_completion": gentle_pred["p_accept"],
                "p_dropout": gentle_pred["p_too_hard"],
                "felt_cost": gentle_pred["predicted_felt_cost"],
            },
            "push": {
                "description": push_intervention.get("description", ""),
                "duration_minutes": push_intervention.get("duration_minutes", 30),
                "p_completion": push_pred["p_accept"],
                "p_dropout": push_pred["p_too_hard"],
                "felt_cost": push_pred["predicted_felt_cost"],
            },
            "recommendation": "gentle" if gentle_pred["p_accept"] > push_pred["p_accept"] else "push",
            "confidence": user_filter.reliability,
        }

    def format_counterfactual_text(self, cf: Dict) -> str:
        """Format counterfactual data into human-readable text."""
        gentle = cf["gentle"]
        push = cf["push"]

        g_pct = int(gentle["p_completion"] * 100)
        p_pct = int(push["p_completion"] * 100)
        g_drop = int(gentle["p_dropout"] * 100)
        p_drop = int(push["p_dropout"] * 100)

        text = (
            f"Option A ({gentle['duration_minutes']}min, gentle): "
            f"{g_pct}% predicted completion, {g_drop}% dropout risk\n"
            f"Option B ({push['duration_minutes']}min, challenging): "
            f"{p_pct}% predicted completion, {p_drop}% dropout risk"
        )

        if cf["confidence"] < 0.3:
            text += "\n(Note: These predictions are preliminary - I'm still learning your patterns.)"

        return text
