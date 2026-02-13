"""
Dependency graph visualization helpers.

Generates data structures for rendering the skill dependency
graph alongside the radar chart.
"""

from __future__ import annotations

from typing import Dict, List


def format_bottleneck_summary(
    bottlenecks: List[Dict],
    categories: Dict[str, float],
) -> str:
    """
    Format bottleneck data into readable text for display.

    Args:
        bottlenecks: List of bottleneck dicts from DependencyGraph
        categories: Skill scores (0-100)

    Returns:
        Formatted text string
    """
    if not bottlenecks:
        return "No significant bottlenecks detected. Your sphere is fairly balanced."

    lines = []
    for i, bn in enumerate(bottlenecks[:3]):  # Top 3
        blocker = bn["blocker"].replace("_", " ").title()
        score = int(bn["score"] * 100)
        blocked = [s.replace("_", " ").title() for s in bn["blocked"]]

        lines.append(
            f"{i+1}. **{blocker}** (score: {score}/100) "
            f"is limiting: {', '.join(blocked)}"
        )

    return "Key bottlenecks:\n" + "\n".join(lines)


def format_dependency_explanation(
    dependency_edges: List[Dict],
    categories: Dict[str, float],
) -> List[Dict]:
    """
    Create annotated dependency edges for the frontend.

    Returns list of edges with human-readable descriptions
    and visual properties (color, thickness based on blocking severity).
    """
    annotated = []
    for edge in dependency_edges:
        source = edge["source"]
        target = edge["target"]
        weight = edge["weight"]

        source_score = categories.get(source, 50)
        is_active = source_score < 40  # Blocking is active if source is low

        annotated.append({
            "from": source.replace("_", " ").title(),
            "to": target.replace("_", " ").title(),
            "strength": weight,
            "is_active": is_active,
            "color": "#E74C3C" if is_active else "#BDC3C7",
            "thickness": max(1, int(weight * 5)) if is_active else 1,
            "label": f"{'Blocking' if is_active else 'Linked'}: "
                     f"{source.replace('_', ' ')} → {target.replace('_', ' ')}",
        })

    return annotated
