"""
Particle filter over user types for Theory of Mind inference.

Adapted from empathy-prisonner-dilemma/src/empathy/clean_up/agent/social/particle_filter.py.
Simplified for coaching domain: particles represent user behavioral type vectors
instead of Dirichlet parameters.

Key pattern:
    w_j^new ∝ w_j^old * p(choice | user_type_j, context)

With reliability gating:
    q_gated = r * q_learned + (1 - r) * q_prior
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .trust import compute_confidence, compute_reliability, compute_weight_entropy
from .user_types import (
    N_USER_DIMS,
    initialize_particles,
    compute_response_probability,
    compute_felt_cost,
)


class UserTypeFilter:
    """
    Particle-based inference over user behavioral types.

    Maintains a weighted set of hypotheses about what kind of user
    we're coaching, and updates based on their responses.

    Follows the ParticleFilter pattern from the empathy project:
    - Bayesian weight update from observations
    - Entropy-based reliability gating
    - Systematic resampling when effective sample size drops
    """

    def __init__(
        self,
        n_particles: int = 50,
        u_threshold: float = 0.05,
        kappa: float = 0.05,
        resample_threshold: float = 0.5,
        diffusion_noise: float = 0.05,
        seed: int = 42,
    ):
        self.n_particles = n_particles
        self.u_threshold = u_threshold
        self.kappa = kappa
        self.resample_threshold = resample_threshold
        self.diffusion_noise = diffusion_noise
        self.rng = np.random.default_rng(seed)

        self.particle_params = initialize_particles(n_particles, self.rng)
        self.particle_weights = np.ones(n_particles, dtype=np.float64) / n_particles

        self._reliability_cache: Optional[float] = None
        self._confidence_cache: Optional[float] = None

    def update_weights(
        self,
        observed_choice: int,
        intervention: Dict,
    ) -> Dict[str, float]:
        """
        Bayesian weight update from an observed user choice.

        w_j^new ∝ w_j^old * p(choice | user_type_j, intervention)

        Args:
            observed_choice: 0=accept, 1=too_hard, 2=not_relevant
            intervention: Dict describing the proposed intervention

        Returns:
            stats dict with entropy, confidence, reliability
        """
        # Compute statistics BEFORE updating (pre-resampling)
        stats = self.compute_statistics()

        # Update weights
        for j in range(self.n_particles):
            response_probs = compute_response_probability(
                self.particle_params[j], intervention
            )
            likelihood = max(response_probs[observed_choice], 1e-12)
            self.particle_weights[j] *= likelihood

        # Normalize
        total = np.sum(self.particle_weights)
        if total <= 0.0:
            self.particle_weights = np.ones(self.n_particles) / self.n_particles
        else:
            self.particle_weights /= total

        # Recompute statistics after update
        stats_post = self.compute_statistics()

        # Resample if needed
        resample_info = self._resample_if_needed()

        return {
            **stats_post,
            **resample_info,
        }

    def _resample_if_needed(self) -> Dict[str, float]:
        """Systematic resampling when effective sample size is low."""
        n_eff = self._effective_sample_size()
        is_resampled = 0.0

        if n_eff < self.resample_threshold * self.n_particles:
            self._systematic_resample()
            self._add_diffusion_noise()
            is_resampled = 1.0

        return {
            "n_eff": float(n_eff),
            "is_resampled": is_resampled,
        }

    def _effective_sample_size(self) -> float:
        """N_eff = 1 / sum(w_j^2)"""
        return 1.0 / max(np.sum(self.particle_weights ** 2), 1e-12)

    def _systematic_resample(self) -> None:
        """Systematic resampling: evenly spaced selection."""
        cumsum = np.cumsum(self.particle_weights)
        u = self.rng.uniform(0, 1.0 / self.n_particles)
        positions = u + np.arange(self.n_particles) / self.n_particles

        indices = np.zeros(self.n_particles, dtype=int)
        i, j = 0, 0
        while i < self.n_particles:
            if positions[i] < cumsum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1

        self.particle_params = self.particle_params[indices].copy()
        self.particle_weights = np.ones(self.n_particles) / self.n_particles

    def _add_diffusion_noise(self) -> None:
        """Add small noise after resampling to maintain diversity."""
        noise = self.rng.normal(0, self.diffusion_noise, self.particle_params.shape)
        self.particle_params = np.clip(self.particle_params + noise, 0.0, 1.0)

    def compute_statistics(self) -> Dict[str, float]:
        """
        Compute entropy, confidence, and reliability.

        Must be called BEFORE resampling for accurate reliability.
        """
        weight_entropy = compute_weight_entropy(self.particle_weights)
        confidence = compute_confidence(self.particle_weights, self.n_particles)
        reliability = compute_reliability(confidence, self.u_threshold, self.kappa)

        self._confidence_cache = confidence
        self._reliability_cache = reliability

        return {
            "weight_entropy": weight_entropy,
            "confidence": confidence,
            "reliability": reliability,
        }

    @property
    def reliability(self) -> float:
        """Current reliability value."""
        if self._reliability_cache is None:
            self.compute_statistics()
        return self._reliability_cache  # type: ignore

    def predict_response(
        self, intervention: Dict
    ) -> Dict[str, float]:
        """
        Predict user response distribution via particle-weighted expectation.

        q_learned(choice) = sum_j w_j * p(choice | type_j, intervention)

        Returns:
            Dict with p_accept, p_too_hard, p_not_relevant, predicted_felt_cost
        """
        response_dist = np.zeros(3, dtype=np.float64)
        felt_cost_total = 0.0

        for j in range(self.n_particles):
            probs = compute_response_probability(
                self.particle_params[j], intervention
            )
            response_dist += self.particle_weights[j] * probs
            felt_cost_total += self.particle_weights[j] * compute_felt_cost(
                self.particle_params[j], intervention
            )

        return {
            "p_accept": float(response_dist[0]),
            "p_too_hard": float(response_dist[1]),
            "p_not_relevant": float(response_dist[2]),
            "predicted_felt_cost": float(felt_cost_total),
        }

    def predict_response_gated(
        self,
        intervention: Dict,
        prior: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Reliability-gated prediction blending learned and prior.

        q_gated = r * q_learned + (1-r) * q_prior

        Args:
            intervention: Proposed intervention details
            prior: Prior response distribution [3]. Defaults to uniform-ish.

        Returns:
            Gated prediction dict
        """
        if prior is None:
            prior = np.array([0.5, 0.3, 0.2])

        learned = self.predict_response(intervention)
        r = self.reliability
        q_learned = np.array([
            learned["p_accept"], learned["p_too_hard"], learned["p_not_relevant"]
        ])

        q_gated = r * q_learned + (1 - r) * prior

        return {
            "p_accept": float(q_gated[0]),
            "p_too_hard": float(q_gated[1]),
            "p_not_relevant": float(q_gated[2]),
            "predicted_felt_cost": learned["predicted_felt_cost"],
            "reliability": float(r),
        }

    def get_expected_user_type(self) -> np.ndarray:
        """Get particle-weighted expected user type vector."""
        return np.sum(
            self.particle_weights[:, None] * self.particle_params,
            axis=0,
        )

    def get_user_type_summary(self) -> Dict[str, float]:
        """Get named summary of inferred user type."""
        from .user_types import USER_TYPE_DIMS
        expected = self.get_expected_user_type()
        return {
            dim: float(val)
            for dim, val in zip(USER_TYPE_DIMS, expected)
        }

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset particles and weights to initial state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.particle_params = initialize_particles(self.n_particles, self.rng)
        self.particle_weights = np.ones(self.n_particles) / self.n_particles
        self._reliability_cache = None
        self._confidence_cache = None
