"""
Reliability gate: entropy-based confidence and sigmoid gating.

Adapted from empathy-prisonner-dilemma/src/empathy/clean_up/agent/social/trust.py.
Pure numpy, no numba dependency.

Mathematical Foundation:
    Confidence: u_t = 1 - H(w) / log(N_p)
    Reliability: r_t = sigmoid((u_t - u_0) / kappa)
"""

from __future__ import annotations

import numpy as np


def compute_weight_entropy(weights: np.ndarray) -> float:
    """
    Compute entropy of particle weights.

    H(w) = -sum_j w_j log w_j

    - H = 0: all weight on one particle (max confidence)
    - H = log(N): uniform weights (max uncertainty)

    Must be computed BEFORE resampling.
    """
    h = 0.0
    for w in weights:
        if w > 1e-12:
            h -= w * np.log(w)
    return h


def compute_confidence(weights: np.ndarray, n_particles: int) -> float:
    """
    Compute normalized confidence from particle weights.

    u_t = 1 - H(w) / log(N_p)

    Returns value in [0, 1]:
    - 0: uniform weights, maximum uncertainty
    - 1: all weight on one particle, maximum confidence
    """
    if n_particles <= 1:
        return 1.0
    max_entropy = np.log(n_particles)
    if max_entropy <= 0.0:
        return 1.0
    h = compute_weight_entropy(weights)
    confidence = 1.0 - h / max_entropy
    return float(max(0.0, min(confidence, 1.0)))


def compute_reliability(
    confidence: float,
    u_threshold: float = 0.05,
    kappa: float = 0.05,
) -> float:
    """
    Compute soft reliability gate from confidence.

    r_t = sigmoid((u_t - u_0) / kappa)

    - r ~ 0: ToM uncertain, fall back to prior
    - r ~ 1: ToM confident, trust learned model
    """
    if kappa <= 0.0:
        raise ValueError("kappa must be positive.")
    x = (confidence - u_threshold) / kappa
    x = np.clip(x, -500, 500)
    return float(1.0 / (1.0 + np.exp(-x)))
