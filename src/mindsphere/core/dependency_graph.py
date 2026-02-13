"""
Skill Dependency DAG: models how low scores in one skill block progress in another.

Hand-designed for MVP, encodes domain knowledge about coaching interdependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class DependencyEdge:
    """An edge in the skill dependency graph."""
    source: str       # Upstream skill (the blocker)
    target: str       # Downstream skill (the blocked)
    weight: float     # Blocking strength [0, 1]


# Hand-designed dependency edges for MVP
DEFAULT_EDGES = [
    DependencyEdge("emotional_reg",  "consistency",      0.4),
    DependencyEdge("emotional_reg",  "focus",            0.2),
    DependencyEdge("task_clarity",   "follow_through",   0.5),
    DependencyEdge("task_clarity",   "consistency",      0.3),
    DependencyEdge("self_trust",     "social_courage",   0.3),
    DependencyEdge("focus",          "systems_thinking", 0.4),
]


class DependencyGraph:
    """
    DAG of skill dependencies with blocking/bottleneck analysis.

    Used to identify which upstream "dents" in the sphere block downstream
    improvement, and to prioritize interventions at the root cause.
    """

    def __init__(self, edges: List[DependencyEdge] | None = None):
        self.edges = edges or DEFAULT_EDGES
        self._adjacency: Dict[str, List[Tuple[str, float]]] = {}
        self._reverse: Dict[str, List[Tuple[str, float]]] = {}
        self._build()

    def _build(self) -> None:
        self._adjacency.clear()
        self._reverse.clear()
        for edge in self.edges:
            self._adjacency.setdefault(edge.source, []).append(
                (edge.target, edge.weight)
            )
            self._reverse.setdefault(edge.target, []).append(
                (edge.source, edge.weight)
            )

    def find_bottlenecks(
        self,
        beliefs: Dict[str, np.ndarray],
        low_threshold: float = 0.4,
    ) -> List[Dict]:
        """
        Identify skills that are low AND block other skills.

        A bottleneck occurs when:
        - source skill's expected score is below low_threshold
        - at least one downstream skill exists

        Args:
            beliefs: Dict mapping skill name -> belief vector (5 levels)
            low_threshold: Normalized score threshold (0-1) below which
                           a skill is considered "low"

        Returns:
            List of bottleneck dicts sorted by impact (highest first):
            [{"blocker": str, "blocked": [str], "score": float, "impact": float}]
        """
        level_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        bottlenecks = []

        for source, targets in self._adjacency.items():
            if source not in beliefs:
                continue
            score = float(np.dot(beliefs[source], level_values))
            if score < low_threshold:
                blocked_skills = []
                total_impact = 0.0
                for target, weight in targets:
                    blocked_skills.append(target)
                    total_impact += weight * (low_threshold - score)
                if blocked_skills:
                    bottlenecks.append({
                        "blocker": source,
                        "blocked": blocked_skills,
                        "score": score,
                        "impact": total_impact,
                    })

        bottlenecks.sort(key=lambda b: b["impact"], reverse=True)
        return bottlenecks

    def compute_impact_ranking(
        self, beliefs: Dict[str, np.ndarray]
    ) -> List[Tuple[str, float]]:
        """
        Rank all skills by improvement impact, considering downstream effects.

        Impact of improving skill s = direct deficit + sum of downstream unblocking.

        Returns:
            List of (skill_name, impact_score) sorted by impact descending.
        """
        level_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        impacts = []

        for skill_name, belief in beliefs.items():
            score = float(np.dot(belief, level_values))
            direct_deficit = max(0.0, 0.5 - score)

            downstream_impact = 0.0
            for target, weight in self._adjacency.get(skill_name, []):
                downstream_impact += weight * direct_deficit

            total = direct_deficit + downstream_impact
            impacts.append((skill_name, total))

        impacts.sort(key=lambda x: x[1], reverse=True)
        return impacts

    def get_blockers_for(self, skill: str) -> List[Tuple[str, float]]:
        """Get upstream skills that block a given skill."""
        return self._reverse.get(skill, [])

    def get_blocked_by(self, skill: str) -> List[Tuple[str, float]]:
        """Get downstream skills blocked by a given skill."""
        return self._adjacency.get(skill, [])

    def get_explanation(self, blocker: str, blocked: str) -> str:
        """Generate a human-readable explanation of a blocking relationship."""
        explanations = {
            ("emotional_reg", "consistency"):
                "When emotions are hard to manage, maintaining routines becomes much harder.",
            ("emotional_reg", "focus"):
                "Emotional turbulence steals attention and makes focus difficult.",
            ("task_clarity", "follow_through"):
                "Without a clear picture of what 'done' looks like, follow-through stalls.",
            ("task_clarity", "consistency"):
                "Ambiguity about tasks makes it hard to build consistent habits.",
            ("self_trust", "social_courage"):
                "When you doubt your own judgment, speaking up feels riskier.",
            ("focus", "systems_thinking"):
                "Systems thinking needs sustained attention to hold multiple pieces together.",
        }
        return explanations.get(
            (blocker, blocked),
            f"Low {blocker.replace('_', ' ')} tends to limit {blocked.replace('_', ' ')}.",
        )

    def get_all_edges(self) -> List[Dict]:
        """Get all edges as dicts for visualization."""
        return [
            {"source": e.source, "target": e.target, "weight": e.weight}
            for e in self.edges
        ]
