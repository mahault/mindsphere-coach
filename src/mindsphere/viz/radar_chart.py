"""
Plotly radar chart generation for the "dented sphere" visualization.

Returns Plotly JSON for client-side rendering.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import plotly.graph_objects as go


# Human-readable labels for skill factors
SKILL_LABELS = {
    "focus": "Focus",
    "follow_through": "Follow-through",
    "social_courage": "Social Courage",
    "emotional_reg": "Emotional Regulation",
    "systems_thinking": "Systems Thinking",
    "self_trust": "Self-Trust",
    "task_clarity": "Task Clarity",
    "consistency": "Consistency",
}


def create_radar_chart(
    categories: Dict[str, float],
    bottlenecks: List[Dict] | None = None,
    title: str = "Your MindSphere",
) -> str:
    """
    Create an interactive Plotly radar chart.

    Args:
        categories: Dict mapping skill_name -> score (0-100)
        bottlenecks: List of bottleneck dicts with "blocker" and "blocked" keys
        title: Chart title

    Returns:
        JSON string for Plotly.js rendering
    """
    # Order and label the categories
    ordered_skills = [
        "focus", "follow_through", "social_courage", "emotional_reg",
        "systems_thinking", "self_trust", "task_clarity", "consistency",
    ]

    labels = []
    values = []
    for skill in ordered_skills:
        if skill in categories:
            labels.append(SKILL_LABELS.get(skill, skill))
            values.append(categories[skill])

    # Close the polygon
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()

    # Main sphere trace
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        name="Your Sphere",
        line=dict(color="#4A90D9", width=2),
        fillcolor="rgba(74, 144, 217, 0.25)",
        hovertemplate="%{theta}: %{r:.0f}/100<extra></extra>",
    ))

    # Ideal sphere (100 on all axes) as reference
    ideal_values = [100] * len(labels) + [100]
    fig.add_trace(go.Scatterpolar(
        r=ideal_values,
        theta=labels_closed,
        fill=None,
        name="Ideal",
        line=dict(color="rgba(200, 200, 200, 0.3)", width=1, dash="dot"),
        hoverinfo="skip",
    ))

    # Highlight bottleneck skills in red
    if bottlenecks:
        blocker_skills = set()
        for bn in bottlenecks:
            blocker_skills.add(bn.get("blocker", ""))

        bn_labels = []
        bn_values = []
        for skill in ordered_skills:
            if skill in blocker_skills and skill in categories:
                bn_labels.append(SKILL_LABELS.get(skill, skill))
                bn_values.append(categories[skill])

        if bn_labels:
            fig.add_trace(go.Scatterpolar(
                r=bn_values,
                theta=bn_labels,
                mode="markers",
                name="Bottlenecks",
                marker=dict(
                    color="#E74C3C",
                    size=12,
                    symbol="diamond",
                ),
                hovertemplate="%{theta}: %{r:.0f} (bottleneck)<extra></extra>",
            ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                ticktext=["20", "40", "60", "80", "100"],
                gridcolor="rgba(200, 200, 200, 0.3)",
            ),
            angularaxis=dict(
                gridcolor="rgba(200, 200, 200, 0.3)",
            ),
            bgcolor="rgba(0, 0, 0, 0)",
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
        title=dict(text=title, x=0.5, font=dict(size=16)),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(t=60, b=60, l=60, r=60),
        height=450,
        width=500,
    )

    return fig.to_json()


def create_dependency_overlay(
    categories: Dict[str, float],
    dependency_edges: List[Dict],
) -> List[Dict]:
    """
    Create dependency edge data for frontend overlay.

    Returns list of edge objects with source/target positions
    for drawing arrows on top of the radar chart.
    """
    ordered_skills = [
        "focus", "follow_through", "social_courage", "emotional_reg",
        "systems_thinking", "self_trust", "task_clarity", "consistency",
    ]

    edges_data = []
    for edge in dependency_edges:
        source = edge.get("source", "")
        target = edge.get("target", "")
        weight = edge.get("weight", 0.3)

        if source in ordered_skills and target in ordered_skills:
            source_score = categories.get(source, 50)
            target_score = categories.get(target, 50)

            edges_data.append({
                "source": SKILL_LABELS.get(source, source),
                "target": SKILL_LABELS.get(target, target),
                "source_score": source_score,
                "target_score": target_score,
                "weight": weight,
                "is_blocking": source_score < 40,
            })

    return edges_data
