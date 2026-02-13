"""
POMDP Parameter Learning via Dirichlet Concentration Parameters.

Standard Active Inference learning: instead of fixed A/B matrices,
store Dirichlet concentration parameters (pseudo-counts). Each
observation increments counts, and the actual matrices are derived
by normalizing the columns.

This gives the model the ability to learn:
- A-matrices: "how reliable are my observations?" (observation model)
- B-matrices: "how do actions change states?" (transition model)

The learning rate controls how fast new evidence overrides the prior.
High concentration (many pseudo-counts) = slow learning, stable model.
Low concentration = fast learning, plastic model.

Reference: Friston et al. (2016) "Active Inference and Learning"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .utils import normalize


class DirichletLearner:
    """
    Learns POMDP matrices via Dirichlet concentration parameter updates.

    The key insight: a matrix column A[:, s] = Dir(alpha[:, s]).
    The expected A-matrix is just normalize(alpha).
    Each observation (obs, state) pair increments alpha[obs, state].

    This is Bayesian parameter learning — the concentration parameters
    encode both the prior structure AND accumulated evidence.

    Usage:
        learner = DirichletLearner(prior_A, learning_rate=1.0)
        # After observing obs=2 when belief peaks at state=3:
        learner.update(obs_idx=2, state_belief=beliefs)
        # Get the learned matrix:
        A_learned = learner.get_matrix()
    """

    def __init__(
        self,
        prior_matrix: np.ndarray,
        prior_strength: float = 10.0,
        learning_rate: float = 1.0,
    ):
        """
        Args:
            prior_matrix: Initial matrix to learn from (columns should sum to 1).
                Shape: [n_obs x n_states] for A, [n_states x n_states] for B.
            prior_strength: How many pseudo-counts the prior is worth.
                Higher = more prior-dominated, slower to change.
                Lower = more plastic, faster adaptation.
            learning_rate: How much each observation contributes.
                1.0 = standard Bayesian update.
                < 1.0 = discounted (for non-stationary environments).
        """
        # Convert prior matrix to concentration parameters
        # alpha = prior_strength * prior_matrix
        self.alpha = prior_matrix.copy() * prior_strength
        self.learning_rate = learning_rate
        self.n_updates = 0

    def update(
        self,
        obs_idx: int,
        state_belief: np.ndarray,
    ) -> None:
        """
        Update concentration parameters given an observation.

        Since we don't know the true state, we weight the update
        by the current belief over states (soft assignment).

        alpha[obs, s] += lr * q(s)  for all s

        This is the standard "expected sufficient statistics" update
        for Dirichlet-Categorical models.

        Args:
            obs_idx: Which observation was seen (row index)
            state_belief: Current belief over states p(s) [n_states]
        """
        self.alpha[obs_idx, :] += self.learning_rate * state_belief
        self.n_updates += 1

    def update_transition(
        self,
        state_belief_before: np.ndarray,
        state_belief_after: np.ndarray,
    ) -> None:
        """
        Update transition model concentration parameters.

        For B-matrices: we observe the state before and after.
        The outer product of beliefs gives the expected transition count.

        alpha[s', s] += lr * q(s') * q(s)  for all s, s'

        Args:
            state_belief_before: Belief over states before transition
            state_belief_after: Belief over states after transition
        """
        # Outer product: expected transition counts
        transition_counts = np.outer(state_belief_after, state_belief_before)
        self.alpha += self.learning_rate * transition_counts
        self.n_updates += 1

    def get_matrix(self) -> np.ndarray:
        """
        Get the current learned matrix by normalizing concentrations.

        A[:, s] = alpha[:, s] / sum(alpha[:, s])
        """
        matrix = self.alpha.copy()
        for col in range(matrix.shape[1]):
            col_sum = matrix[:, col].sum()
            if col_sum > 0:
                matrix[:, col] /= col_sum
            else:
                matrix[:, col] = 1.0 / matrix.shape[0]
        return matrix

    def get_confidence(self) -> float:
        """
        Get confidence in the learned parameters.

        Higher total concentration = more confident.
        Returns the average concentration per column (how many
        effective observations per state).
        """
        avg_concentration = np.mean(np.sum(self.alpha, axis=0))
        return float(avg_concentration)

    def get_learning_progress(self) -> Dict[str, Any]:
        """Get summary of learning state."""
        matrix = self.get_matrix()
        return {
            "n_updates": self.n_updates,
            "avg_concentration": float(np.mean(np.sum(self.alpha, axis=0))),
            "max_diagonal": float(np.max(np.diag(matrix))) if matrix.shape[0] == matrix.shape[1] else None,
            "learned_matrix": matrix.tolist(),
        }

    def reset(self, prior_matrix: np.ndarray, prior_strength: float = 10.0) -> None:
        """Reset to a new prior."""
        self.alpha = prior_matrix.copy() * prior_strength
        self.n_updates = 0


class ModelLearner:
    """
    Manages learning for all POMDP factors in the SphereModel.

    Wraps DirichletLearner instances for each factor's A-matrix
    and optionally B-matrices. Provides a clean interface for
    the CoachingAgent to call on each observation.
    """

    def __init__(
        self,
        model,
        a_learning_rate: float = 1.0,
        b_learning_rate: float = 0.5,
        a_prior_strength: float = 10.0,
        b_prior_strength: float = 20.0,
    ):
        """
        Args:
            model: SphereModel instance
            a_learning_rate: Learning rate for observation models
            b_learning_rate: Learning rate for transition models
            a_prior_strength: Prior strength for A-matrices
            b_prior_strength: Prior strength for B-matrices (higher = more stable)
        """
        self.model = model

        # A-matrix learners (one per factor)
        self.a_learners: Dict[str, DirichletLearner] = {}
        for factor_name, A in model.A.items():
            self.a_learners[factor_name] = DirichletLearner(
                prior_matrix=A,
                prior_strength=a_prior_strength,
                learning_rate=a_learning_rate,
            )

        # B-matrix learners (one per factor per action)
        # Only learn B for friction factors — skill transitions are too
        # stable within a single session to learn meaningfully
        self.b_learners: Dict[str, Dict[int, DirichletLearner]] = {}
        friction_factors = list(model.spec.friction_factors.keys())
        for factor_name in friction_factors:
            if factor_name in model.B:
                self.b_learners[factor_name] = {}
                n_actions = model.B[factor_name].shape[0]
                for action_idx in range(n_actions):
                    self.b_learners[factor_name][action_idx] = DirichletLearner(
                        prior_matrix=model.B[factor_name][action_idx],
                        prior_strength=b_prior_strength,
                        learning_rate=b_learning_rate,
                    )

    def learn_from_observation(
        self,
        factor_name: str,
        obs_idx: int,
        state_belief: np.ndarray,
    ) -> None:
        """
        Update A-matrix for a factor given an observation.

        Called after each belief update to refine the observation model.
        """
        if factor_name in self.a_learners:
            self.a_learners[factor_name].update(obs_idx, state_belief)
            # Update the model's A-matrix in place
            self.model.A[factor_name] = self.a_learners[factor_name].get_matrix()

    def learn_from_transition(
        self,
        factor_name: str,
        action_idx: int,
        belief_before: np.ndarray,
        belief_after: np.ndarray,
    ) -> None:
        """
        Update B-matrix for a factor given a state transition.

        Called when we observe how the user's state changed after an action.
        """
        if factor_name in self.b_learners and action_idx in self.b_learners[factor_name]:
            learner = self.b_learners[factor_name][action_idx]
            learner.update_transition(belief_before, belief_after)
            # Update the model's B-matrix in place
            self.model.B[factor_name][action_idx] = learner.get_matrix()

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of all learning progress."""
        summary = {}
        for factor_name, learner in self.a_learners.items():
            progress = learner.get_learning_progress()
            summary[f"A_{factor_name}"] = {
                "n_updates": progress["n_updates"],
                "avg_concentration": progress["avg_concentration"],
            }
        for factor_name, action_learners in self.b_learners.items():
            total_updates = sum(l.n_updates for l in action_learners.values())
            summary[f"B_{factor_name}"] = {
                "n_updates": total_updates,
            }
        return summary

    def reset(self) -> None:
        """Reset all learners to their priors."""
        for learner in self.a_learners.values():
            learner.reset(learner.alpha / max(learner.get_confidence(), 1.0))
        for action_learners in self.b_learners.values():
            for learner in action_learners.values():
                learner.reset(learner.alpha / max(learner.get_confidence(), 1.0))
