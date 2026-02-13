"""
Belief update and Expected Free Energy (EFE) computation.

Implements per-factor Bayesian belief updates and EFE-based action selection
for the factored POMDP. NumPy only, adapted from empathy project patterns.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import normalize, softmax, entropy, expected_value
from .model import SphereModel, SKILL_FACTORS


def update_belief(
    prior: np.ndarray,
    observation: int,
    A_matrix: np.ndarray,
) -> np.ndarray:
    """
    Bayesian belief update for a single factor.

    q(s) ∝ prior(s) * p(o | s)

    Args:
        prior: Prior belief over states [n_states]
        observation: Observed outcome index
        A_matrix: Observation likelihood [n_obs x n_states]

    Returns:
        posterior: Updated belief [n_states]
    """
    likelihood = A_matrix[observation, :]
    posterior = prior * likelihood
    return normalize(posterior)


def update_all_beliefs(
    beliefs: Dict[str, np.ndarray],
    observations: Dict[str, int],
    model: SphereModel,
) -> Dict[str, np.ndarray]:
    """
    Update beliefs for all observed factors.

    Args:
        beliefs: Current beliefs per factor
        observations: Dict mapping factor_name -> observation_index
        model: The POMDP model

    Returns:
        Updated beliefs dict
    """
    updated = {k: v.copy() for k, v in beliefs.items()}
    for factor_name, obs_idx in observations.items():
        if factor_name in model.A and obs_idx >= 0:
            updated[factor_name] = update_belief(
                updated[factor_name], obs_idx, model.A[factor_name]
            )
    return updated


def compute_efe_single_factor(
    belief: np.ndarray,
    A_matrix: np.ndarray,
    B_action: np.ndarray,
    C_vector: np.ndarray,
    lambda_epist: float = 0.5,
) -> float:
    """
    Compute Expected Free Energy for one factor under one action.

    G(a) = -pragmatic - lambda * epistemic

    Pragmatic: E_q[log p(o | C)] = expected preference satisfaction
    Epistemic: E_q[H[p(o|s)] - H[p(o)]] = expected information gain

    Args:
        belief: Current belief over states [n_states]
        A_matrix: Observation likelihood [n_obs x n_states]
        B_action: Transition matrix for this action [n_states x n_states]
        C_vector: Preferred observations [n_obs]
        lambda_epist: Weight on epistemic drive (0 = pure exploitation)

    Returns:
        G: Expected free energy (lower = better)
    """
    n_obs, n_states = A_matrix.shape

    # Predicted state after transition: q(s') = B @ q(s)
    predicted_state = B_action @ belief
    predicted_state = normalize(predicted_state)

    # Predicted observation: q(o) = A @ q(s')
    predicted_obs = A_matrix @ predicted_state
    predicted_obs = normalize(predicted_obs)

    # Pragmatic value: E_q(o)[C(o)] = how much we like predicted observations
    pragmatic = float(np.dot(predicted_obs, C_vector))

    # Epistemic value: expected information gain
    # = H[q(o)] - E_q(s')[H[p(o|s')]]
    H_predicted = entropy(predicted_obs)
    H_conditional = 0.0
    for s in range(n_states):
        if predicted_state[s] > 1e-12:
            H_conditional += predicted_state[s] * entropy(A_matrix[:, s])
    epistemic = H_predicted - H_conditional

    # G = -(pragmatic + lambda * epistemic)
    # Lower G = better action
    return -(pragmatic + lambda_epist * epistemic)


def compute_efe_all_factors(
    beliefs: Dict[str, np.ndarray],
    model: SphereModel,
    action_idx: int,
    relevant_factors: Optional[List[str]] = None,
    lambda_epist: float = 0.5,
) -> float:
    """
    Compute total EFE across all factors for one action.

    Args:
        beliefs: Current beliefs per factor
        model: The POMDP model
        action_idx: Which action to evaluate
        relevant_factors: Subset of factors to consider (None = all skills)
        lambda_epist: Epistemic drive weight

    Returns:
        Total G (sum across factors)
    """
    factors = relevant_factors or list(SKILL_FACTORS)
    total_G = 0.0

    for factor in factors:
        if factor not in model.A or factor not in beliefs:
            continue
        G_f = compute_efe_single_factor(
            beliefs[factor],
            model.A[factor],
            model.B[factor][action_idx],
            model.C[factor],
            lambda_epist,
        )
        total_G += G_f

    return total_G


def select_action(
    beliefs: Dict[str, np.ndarray],
    model: SphereModel,
    valid_actions: List[int],
    lambda_epist: float = 0.5,
    beta: float = 4.0,
    relevant_factors: Optional[List[str]] = None,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Select action by computing EFE for each valid action and applying softmax.

    Args:
        beliefs: Current beliefs per factor
        model: The POMDP model
        valid_actions: List of action indices to consider
        lambda_epist: Epistemic drive weight
        beta: Inverse temperature (higher = more deterministic)
        relevant_factors: Subset of factors to evaluate

    Returns:
        (selected_action_idx, action_probabilities, efe_values)
    """
    efe_values = np.zeros(len(valid_actions), dtype=np.float64)

    for i, action_idx in enumerate(valid_actions):
        efe_values[i] = compute_efe_all_factors(
            beliefs, model, action_idx, relevant_factors, lambda_epist
        )

    # Action probabilities via softmax over negative EFE (lower G → higher prob)
    q_values = -efe_values
    action_probs = softmax(q_values, temperature=1.0 / max(beta, 0.01))

    selected_idx = int(np.argmax(action_probs))
    selected_action = valid_actions[selected_idx]

    return selected_action, action_probs, efe_values


def compute_information_gain(
    belief: np.ndarray,
    A_matrix: np.ndarray,
) -> float:
    """
    Compute expected information gain for a factor (pure epistemic value).

    IG = H[q(o)] - E_q(s)[H[p(o|s)]]

    Higher IG means asking about this factor would be most informative.

    Args:
        belief: Current belief over states
        A_matrix: Observation likelihood matrix

    Returns:
        Expected information gain (bits)
    """
    predicted_obs = A_matrix @ belief
    predicted_obs = normalize(predicted_obs)

    H_predicted = entropy(predicted_obs)
    H_conditional = 0.0
    n_states = len(belief)
    for s in range(n_states):
        if belief[s] > 1e-12:
            H_conditional += belief[s] * entropy(A_matrix[:, s])

    return H_predicted - H_conditional
