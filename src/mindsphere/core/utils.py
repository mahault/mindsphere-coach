"""
Numerical utilities for Active Inference computations.

Adapted from empathy-prisonner-dilemma inference utilities.
Pure numpy, no numba dependency.
"""

from __future__ import annotations

import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize array to sum to 1 (probability distribution)."""
    total = np.sum(x)
    if total <= 0.0:
        return np.ones_like(x) / x.size
    return x / total


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax with temperature scaling. Lower temp = more deterministic."""
    if temperature <= 0.0:
        out = np.zeros_like(x, dtype=np.float64)
        out[np.argmax(x)] = 1.0
        return out
    scaled = x / temperature
    shifted = scaled - np.max(scaled)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals)


def entropy(p: np.ndarray) -> float:
    """Shannon entropy H(p) = -sum(p * log(p))."""
    p_safe = p[p > 1e-12]
    return -float(np.sum(p_safe * np.log(p_safe)))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence D_KL(p || q) = sum(p * log(p/q))."""
    mask = p > 1e-12
    q_safe = np.maximum(q[mask], 1e-12)
    return float(np.sum(p[mask] * np.log(p[mask] / q_safe)))


def sigmoid(x: float, center: float = 0.0, scale: float = 1.0) -> float:
    """Sigmoid function with configurable center and scale."""
    z = (x - center) / max(scale, 1e-12)
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def expected_value(belief: np.ndarray, values: np.ndarray) -> float:
    """Compute expected value under a belief distribution."""
    return float(np.dot(belief, values))
