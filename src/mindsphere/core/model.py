"""
Factored POMDP Model for MindSphere Coaching.

State space is factored into independent factors (mean-field approximation)
to avoid combinatorial explosion. Each factor has its own A/B/C/D matrices.

Adapted from NEXT-prototype's onboarding_model_v0.py patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .utils import normalize


# =============================================================================
# STATE SPACE DEFINITIONS
# =============================================================================

SKILL_FACTORS = [
    "focus",
    "follow_through",
    "social_courage",
    "emotional_reg",
    "systems_thinking",
    "self_trust",
    "task_clarity",
    "consistency",
]

SKILL_LEVELS = ["very_low", "low", "medium", "high", "very_high"]
SKILL_LEVEL_VALUES = np.array([10.0, 30.0, 50.0, 70.0, 90.0])  # 0-100 scale

PREFERENCE_FACTORS = {
    "coaching_style": ["gentle", "balanced", "challenging"],
    "feedback_mode": ["visual", "narrative", "action_oriented"],
}

FRICTION_FACTORS = {
    "overwhelm_sensitivity": ["low", "medium", "high"],
    "autonomy_need": ["low", "medium", "high"],
}

# Actions the coaching agent can take
ACTION_NAMES = [
    "ask_mc_question",
    "ask_free_text",
    "show_sphere",
    "propose_intervention",
    "reframe",
    "adjust_difficulty",
    "show_counterfactual",
    "safety_check",
    "end_session",
]

# Observation modalities
OBS_MC_LEVELS = ["answer_0", "answer_1", "answer_2", "answer_3"]
OBS_ENGAGEMENT_LEVELS = ["low", "medium", "high"]
OBS_CHOICE_LEVELS = ["accept", "too_hard", "not_relevant"]


# =============================================================================
# MODEL SPECIFICATION
# =============================================================================

@dataclass
class SphereModelSpec:
    """Specification for the factored POMDP model."""
    skill_factors: List[str] = field(default_factory=lambda: list(SKILL_FACTORS))
    n_skill_levels: int = 5
    preference_factors: Dict[str, List[str]] = field(
        default_factory=lambda: dict(PREFERENCE_FACTORS)
    )
    friction_factors: Dict[str, List[str]] = field(
        default_factory=lambda: dict(FRICTION_FACTORS)
    )
    action_names: List[str] = field(default_factory=lambda: list(ACTION_NAMES))
    n_actions: int = 9
    n_mc_options: int = 4


# =============================================================================
# POMDP MODEL
# =============================================================================

class SphereModel:
    """
    Factored POMDP model for MindSphere coaching.

    Each factor has independent:
    - A matrix: p(observation | state)  [n_obs x n_states]
    - B matrix: p(state' | state, action) [n_actions x n_states x n_states]
    - C vector: preference over observations [n_obs]
    - D vector: initial prior over states [n_states]

    This avoids the combinatorial explosion of a full joint state space.
    """

    def __init__(self, spec: Optional[SphereModelSpec] = None):
        self.spec = spec or SphereModelSpec()
        self.A: Dict[str, np.ndarray] = {}
        self.B: Dict[str, np.ndarray] = {}
        self.C: Dict[str, np.ndarray] = {}
        self.D: Dict[str, np.ndarray] = {}
        self._build()

    def _build(self) -> None:
        """Populate all A/B/C/D matrices."""
        self._build_skill_matrices()
        self._build_preference_matrices()
        self._build_friction_matrices()

    # -------------------------------------------------------------------------
    # SKILL FACTOR MATRICES
    # -------------------------------------------------------------------------

    def _build_skill_matrices(self) -> None:
        """Build A/B/C/D for each of the 8 skill factors."""
        n_s = self.spec.n_skill_levels  # 5
        n_o = self.spec.n_mc_options     # 4

        for skill in self.spec.skill_factors:
            # A matrix: p(answer | skill_level) [n_obs x n_states]
            # Higher skill → higher-numbered answer is more likely
            A = np.array([
                [0.55, 0.30, 0.10, 0.04, 0.01],  # answer_0 (lowest)
                [0.30, 0.40, 0.25, 0.10, 0.05],  # answer_1
                [0.10, 0.20, 0.40, 0.36, 0.14],  # answer_2
                [0.05, 0.10, 0.25, 0.50, 0.80],  # answer_3 (highest)
            ], dtype=np.float64)
            # Normalize columns
            for col in range(n_s):
                A[:, col] = normalize(A[:, col])
            self.A[skill] = A

            # B matrix: p(s' | s, action) - skills mostly stable during session
            # [n_actions x n_states x n_states]
            B = np.zeros((self.spec.n_actions, n_s, n_s), dtype=np.float64)
            identity = np.eye(n_s, dtype=np.float64)
            for a in range(self.spec.n_actions):
                B[a] = identity * 0.98 + np.ones((n_s, n_s)) * 0.02 / n_s
                # Normalize columns
                for col in range(n_s):
                    B[a, :, col] = normalize(B[a, :, col])
            self.B[skill] = B

            # C vector: prefer high-skill observations
            self.C[skill] = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float64)

            # D vector: mildly pessimistic prior
            self.D[skill] = normalize(
                np.array([0.12, 0.23, 0.32, 0.22, 0.11], dtype=np.float64)
            )

    # -------------------------------------------------------------------------
    # PREFERENCE FACTOR MATRICES
    # -------------------------------------------------------------------------

    def _build_preference_matrices(self) -> None:
        """Build A/B/C/D for preference factors."""
        for factor_name, levels in self.spec.preference_factors.items():
            n_levels = len(levels)

            # A: uniform observation model (preferences inferred from free text)
            A = np.ones((n_levels, n_levels), dtype=np.float64)
            for col in range(n_levels):
                A[:, col] = normalize(A[:, col])
            self.A[factor_name] = A

            # B: stable (preferences don't change during session)
            B = np.zeros(
                (self.spec.n_actions, n_levels, n_levels), dtype=np.float64
            )
            for a in range(self.spec.n_actions):
                B[a] = np.eye(n_levels, dtype=np.float64)
            self.B[factor_name] = B

            # C: no strong preference
            self.C[factor_name] = np.zeros(n_levels, dtype=np.float64)

            # D: uniform prior
            self.D[factor_name] = np.ones(n_levels, dtype=np.float64) / n_levels

    # -------------------------------------------------------------------------
    # FRICTION FACTOR MATRICES
    # -------------------------------------------------------------------------

    def _build_friction_matrices(self) -> None:
        """Build A/B/C/D for friction factors."""
        for factor_name, levels in self.spec.friction_factors.items():
            n_levels = len(levels)

            # A: inferred from user choices and engagement
            # For overwhelm_sensitivity:
            # "too_hard" response more likely when sensitivity is high
            if factor_name == "overwhelm_sensitivity":
                A = np.array([
                    [0.60, 0.30, 0.10],  # "accept" - more likely if low sensitivity
                    [0.25, 0.40, 0.55],  # "too_hard" - more likely if high
                    [0.15, 0.30, 0.35],  # "not_relevant"
                ], dtype=np.float64)
            elif factor_name == "autonomy_need":
                A = np.array([
                    [0.50, 0.35, 0.15],  # "accept" structured task
                    [0.20, 0.30, 0.35],  # "too_hard"
                    [0.30, 0.35, 0.50],  # "not_relevant" - high autonomy rejects
                ], dtype=np.float64)
            else:
                A = np.ones((n_levels, n_levels), dtype=np.float64)

            for col in range(n_levels):
                A[:, col] = normalize(A[:, col])
            self.A[factor_name] = A

            # B: friction can shift based on coaching actions
            B = np.zeros(
                (self.spec.n_actions, n_levels, n_levels), dtype=np.float64
            )
            for a in range(self.spec.n_actions):
                B[a] = np.eye(n_levels, dtype=np.float64) * 0.9 + \
                       np.ones((n_levels, n_levels)) * 0.1 / n_levels
                for col in range(n_levels):
                    B[a, :, col] = normalize(B[a, :, col])
            self.B[factor_name] = B

            # C: prefer low overwhelm, moderate autonomy
            if factor_name == "overwhelm_sensitivity":
                self.C[factor_name] = np.array([1.0, 0.0, -1.5], dtype=np.float64)
            else:
                self.C[factor_name] = np.zeros(n_levels, dtype=np.float64)

            # D: most people are medium
            self.D[factor_name] = normalize(
                np.array([0.25, 0.50, 0.25], dtype=np.float64)
            )

    # -------------------------------------------------------------------------
    # ACCESSORS
    # -------------------------------------------------------------------------

    def get_all_factor_names(self) -> List[str]:
        """Get all factor names (skills + preferences + friction)."""
        factors = list(self.spec.skill_factors)
        factors.extend(self.spec.preference_factors.keys())
        factors.extend(self.spec.friction_factors.keys())
        return factors

    def get_initial_beliefs(self) -> Dict[str, np.ndarray]:
        """Get initial belief distributions for all factors."""
        return {name: self.D[name].copy() for name in self.get_all_factor_names()}

    def get_skill_score(self, belief: np.ndarray) -> float:
        """Convert a 5-level skill belief into a 0-100 score."""
        return float(np.dot(belief, SKILL_LEVEL_VALUES))

    def get_all_skill_scores(
        self, beliefs: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Get 0-100 scores for all 8 skill factors."""
        return {
            skill: self.get_skill_score(beliefs[skill])
            for skill in self.spec.skill_factors
        }
