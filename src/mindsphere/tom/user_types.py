"""
User type definitions and particle initialization.

A user type is a 7-dimensional vector capturing behavioral tendencies
that determine how a user responds to coaching interventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


# User type dimensions (each in [0, 1])
USER_TYPE_DIMS = [
    "avoids_evaluation",       # Tendency to avoid tasks that feel evaluative
    "hates_long_tasks",        # Preference for short bursts over deep work
    "novelty_seeking",         # Attraction to new/playful approaches
    "structure_preference",    # Preference for structured vs open-ended
    "external_validation",     # Need for external accountability/feedback
    "autonomy_sensitivity",    # Resistance to feeling controlled
    "overwhelm_threshold",     # How easily overwhelmed (low = easily)
]

N_USER_DIMS = len(USER_TYPE_DIMS)


@dataclass
class UserTypeArchetype:
    """A named archetype to seed the particle filter."""
    name: str
    params: np.ndarray    # [N_USER_DIMS] values in [0, 1]
    weight: float = 1.0   # Relative frequency in prior


# Pre-defined archetypes based on common coaching patterns
ARCHETYPES = [
    UserTypeArchetype(
        name="perfectionist",
        params=np.array([0.8, 0.3, 0.2, 0.9, 0.7, 0.3, 0.3]),
    ),
    UserTypeArchetype(
        name="novelty_explorer",
        params=np.array([0.2, 0.6, 0.9, 0.2, 0.3, 0.7, 0.6]),
    ),
    UserTypeArchetype(
        name="overwhelmed_achiever",
        params=np.array([0.5, 0.7, 0.3, 0.6, 0.5, 0.4, 0.2]),
    ),
    UserTypeArchetype(
        name="autonomous_thinker",
        params=np.array([0.3, 0.4, 0.5, 0.4, 0.2, 0.9, 0.7]),
    ),
    UserTypeArchetype(
        name="structure_seeker",
        params=np.array([0.4, 0.5, 0.3, 0.9, 0.8, 0.2, 0.5]),
    ),
    UserTypeArchetype(
        name="avoidant",
        params=np.array([0.9, 0.8, 0.4, 0.5, 0.3, 0.6, 0.2]),
    ),
]


def initialize_particles(
    n_particles: int = 50,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Initialize particle set for user type inference.

    Strategy: sample near archetypes with Gaussian noise, plus some
    uniform random particles for coverage.

    Args:
        n_particles: Number of particles
        rng: Random number generator

    Returns:
        particles: [n_particles, N_USER_DIMS] array in [0, 1]
    """
    if rng is None:
        rng = np.random.default_rng(42)

    particles = np.zeros((n_particles, N_USER_DIMS), dtype=np.float64)

    # Allocate ~70% to archetypes, ~30% uniform
    n_archetype = int(0.7 * n_particles)
    n_uniform = n_particles - n_archetype

    # Sample near archetypes
    archetype_count = len(ARCHETYPES)
    per_archetype = n_archetype // archetype_count
    idx = 0
    for arch in ARCHETYPES:
        for _ in range(per_archetype):
            if idx >= n_archetype:
                break
            noise = rng.normal(0, 0.1, size=N_USER_DIMS)
            particles[idx] = np.clip(arch.params + noise, 0.0, 1.0)
            idx += 1
    # Fill remaining archetype slots
    while idx < n_archetype:
        arch = ARCHETYPES[idx % archetype_count]
        noise = rng.normal(0, 0.15, size=N_USER_DIMS)
        particles[idx] = np.clip(arch.params + noise, 0.0, 1.0)
        idx += 1

    # Uniform random particles for coverage
    particles[n_archetype:] = rng.uniform(0, 1, size=(n_uniform, N_USER_DIMS))

    return particles


def compute_response_probability(
    user_type: np.ndarray,
    intervention: Dict,
) -> np.ndarray:
    """
    Predict user response distribution given their type and an intervention.

    Returns p = [p_accept, p_too_hard, p_not_relevant]

    The model encodes how user type dimensions interact with intervention features.
    """
    # Intervention features
    difficulty = intervention.get("difficulty", 0.5)       # 0-1
    duration = intervention.get("duration_minutes", 5) / 30  # normalized
    evaluative = intervention.get("evaluative", 0.3)       # 0-1
    structured = intervention.get("structured", 0.5)       # 0-1

    # User type dimensions
    avoids_eval = user_type[0]
    hates_long = user_type[1]
    novelty = user_type[2]
    structure_pref = user_type[3]
    external_val = user_type[4]
    autonomy = user_type[5]
    overwhelm_thresh = user_type[6]

    # Score for "too hard"
    # Higher if: high difficulty + low overwhelm threshold + long duration + evaluative
    too_hard_score = (
        0.4 * difficulty * (1 - overwhelm_thresh) +
        0.3 * duration * hates_long +
        0.2 * evaluative * avoids_eval +
        0.1 * max(0, difficulty - overwhelm_thresh)
    )

    # Score for "not relevant"
    # Higher if: high autonomy + mismatch with structure preference
    structure_mismatch = abs(structured - structure_pref)
    not_relevant_score = (
        0.4 * autonomy * 0.5 +
        0.3 * structure_mismatch +
        0.2 * (1 - external_val) * 0.5 +
        0.1 * novelty * (1 - intervention.get("playful", 0.3))
    )

    # Score for "accept"
    accept_score = max(0.1, 1.0 - too_hard_score - not_relevant_score)

    # Normalize to probabilities
    scores = np.array([accept_score, too_hard_score, not_relevant_score])
    scores = np.maximum(scores, 0.01)
    return scores / np.sum(scores)


def compute_felt_cost(
    user_type: np.ndarray,
    intervention: Dict,
) -> float:
    """
    Predict the felt cost (overwhelm, aversion) of an intervention for this user type.

    Returns a value in [0, 1]:
    - 0: intervention feels easy and natural
    - 1: intervention feels maximally aversive

    This becomes G_user in the empathic planning formula.
    """
    difficulty = intervention.get("difficulty", 0.5)
    duration = intervention.get("duration_minutes", 5) / 30
    evaluative = intervention.get("evaluative", 0.3)
    structured = intervention.get("structured", 0.5)

    overwhelm_thresh = user_type[6]
    avoids_eval = user_type[0]
    hates_long = user_type[1]
    autonomy = user_type[5]

    # Overwhelm component
    overwhelm = max(0.0, difficulty - overwhelm_thresh) * 0.4

    # Aversion components
    eval_aversion = evaluative * avoids_eval * 0.2
    time_aversion = duration * hates_long * 0.2
    control_aversion = structured * autonomy * 0.1

    # Loss of autonomy
    autonomy_loss = max(0.0, structured - 0.5) * autonomy * 0.1

    felt_cost = overwhelm + eval_aversion + time_aversion + control_aversion + autonomy_loss
    return float(np.clip(felt_cost, 0.0, 1.0))
