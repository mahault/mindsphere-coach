"""
Calibration question bank for MindSphere Coach.

10 questions (8 MC targeting individual skills, 2 free-text targeting multiple).
Each question includes an A-matrix row for Bayesian belief update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class CalibrationQuestion:
    """A calibration question with observation model."""
    id: str
    category: str                          # Primary skill factor targeted
    question_text: str
    question_type: str                     # "mc" or "free_text"
    options: List[str] = field(default_factory=list)
    # A matrix: p(answer | skill_level) for MC questions
    # Shape: [n_options, 5_skill_levels] — columns must sum to 1
    a_matrix: Optional[np.ndarray] = None
    information_gain_weight: float = 1.0   # Relative priority for adaptive ordering


QUESTION_BANK: List[CalibrationQuestion] = [
    # Q1: Focus
    CalibrationQuestion(
        id="focus_1",
        category="focus",
        question_text="When you sit down to do something important, how often do you get pulled away within the first 10 minutes?",
        question_type="mc",
        options=[
            "Almost always — I can barely start",
            "Often — more than half the time",
            "Sometimes — maybe a third of the time",
            "Rarely — I can usually stay locked in",
        ],
        a_matrix=np.array([
            [0.60, 0.30, 0.08, 0.02, 0.00],
            [0.25, 0.40, 0.30, 0.08, 0.02],
            [0.10, 0.20, 0.40, 0.35, 0.18],
            [0.05, 0.10, 0.22, 0.55, 0.80],
        ], dtype=np.float64),
    ),

    # Q2: Follow-through
    CalibrationQuestion(
        id="follow_through_1",
        category="follow_through",
        question_text="When you start a new project or habit, how far do you typically get before dropping it?",
        question_type="mc",
        options=[
            "A day or two at most",
            "About a week",
            "A few weeks, then it fades",
            "I usually see things through",
        ],
        a_matrix=np.array([
            [0.65, 0.30, 0.05, 0.02, 0.00],
            [0.20, 0.40, 0.30, 0.10, 0.03],
            [0.10, 0.20, 0.40, 0.33, 0.17],
            [0.05, 0.10, 0.25, 0.55, 0.80],
        ], dtype=np.float64),
    ),

    # Q3: Social Courage
    CalibrationQuestion(
        id="social_courage_1",
        category="social_courage",
        question_text="If you disagree with someone in a meeting or group setting, what do you usually do?",
        question_type="mc",
        options=[
            "Stay quiet and go along",
            "Mention it privately afterward",
            "Speak up but soften it a lot",
            "Say what I think directly",
        ],
        a_matrix=np.array([
            [0.60, 0.35, 0.10, 0.03, 0.01],
            [0.25, 0.35, 0.30, 0.12, 0.05],
            [0.10, 0.20, 0.38, 0.40, 0.24],
            [0.05, 0.10, 0.22, 0.45, 0.70],
        ], dtype=np.float64),
    ),

    # Q4: Emotional Regulation
    CalibrationQuestion(
        id="emotional_reg_1",
        category="emotional_reg",
        question_text="When something unexpected goes wrong, how quickly do you recover and refocus?",
        question_type="mc",
        options=[
            "It can derail my whole day",
            "Takes me a few hours",
            "I'm rattled but recover within the hour",
            "I bounce back pretty quickly",
        ],
        a_matrix=np.array([
            [0.60, 0.30, 0.08, 0.02, 0.01],
            [0.25, 0.38, 0.27, 0.10, 0.04],
            [0.10, 0.22, 0.40, 0.38, 0.20],
            [0.05, 0.10, 0.25, 0.50, 0.75],
        ], dtype=np.float64),
    ),

    # Q5: Systems Thinking
    CalibrationQuestion(
        id="systems_thinking_1",
        category="systems_thinking",
        question_text="When facing a recurring problem, how do you usually approach it?",
        question_type="mc",
        options=[
            "Fix the immediate issue and move on",
            "Think about it but struggle to see the bigger picture",
            "Try to identify root causes sometimes",
            "Naturally look for patterns and underlying structures",
        ],
        a_matrix=np.array([
            [0.55, 0.30, 0.10, 0.03, 0.01],
            [0.28, 0.38, 0.25, 0.10, 0.04],
            [0.12, 0.22, 0.40, 0.37, 0.20],
            [0.05, 0.10, 0.25, 0.50, 0.75],
        ], dtype=np.float64),
    ),

    # Q6: Self-Trust
    CalibrationQuestion(
        id="self_trust_1",
        category="self_trust",
        question_text="When you make a decision, how often do you second-guess yourself afterward?",
        question_type="mc",
        options=[
            "Almost always — I rarely feel confident in my choices",
            "Often — more than I'd like",
            "Sometimes, but I usually trust my judgment",
            "Rarely — I commit and move forward",
        ],
        a_matrix=np.array([
            [0.60, 0.32, 0.10, 0.03, 0.01],
            [0.25, 0.38, 0.28, 0.10, 0.04],
            [0.10, 0.20, 0.38, 0.40, 0.22],
            [0.05, 0.10, 0.24, 0.47, 0.73],
        ], dtype=np.float64),
    ),

    # Q7: Task Clarity
    CalibrationQuestion(
        id="task_clarity_1",
        category="task_clarity",
        question_text="Before starting a piece of work, how clear are you on exactly what 'done' looks like?",
        question_type="mc",
        options=[
            "Usually vague — I just start and see what happens",
            "I have a rough idea but not specifics",
            "I usually have a clear enough picture",
            "I define it precisely before I start",
        ],
        a_matrix=np.array([
            [0.58, 0.30, 0.08, 0.03, 0.01],
            [0.27, 0.40, 0.28, 0.10, 0.04],
            [0.10, 0.20, 0.40, 0.40, 0.22],
            [0.05, 0.10, 0.24, 0.47, 0.73],
        ], dtype=np.float64),
    ),

    # Q8: Consistency
    CalibrationQuestion(
        id="consistency_1",
        category="consistency",
        question_text="Think of a routine you tried to build in the last year. How many consecutive days did you maintain it?",
        question_type="mc",
        options=[
            "Less than a week",
            "1-2 weeks",
            "About a month",
            "More than a month",
        ],
        a_matrix=np.array([
            [0.60, 0.30, 0.08, 0.02, 0.01],
            [0.25, 0.40, 0.30, 0.10, 0.04],
            [0.10, 0.20, 0.38, 0.38, 0.22],
            [0.05, 0.10, 0.24, 0.50, 0.73],
        ], dtype=np.float64),
    ),

    # Q9: Free text — friction + multi-category
    CalibrationQuestion(
        id="friction_freetext",
        category="friction",
        question_text="What is the thing you most want to change about how you work or live, and what has stopped you so far?",
        question_type="free_text",
        information_gain_weight=1.5,  # High value: covers multiple dimensions
    ),

    # Q10: Free text — blindspot + multi-category
    CalibrationQuestion(
        id="blindspot_freetext",
        category="blindspot",
        question_text="Describe a recent moment where you surprised yourself — either positively or negatively.",
        question_type="free_text",
        information_gain_weight=1.2,
    ),
]


def get_question_by_id(question_id: str) -> Optional[CalibrationQuestion]:
    """Look up a question by ID."""
    for q in QUESTION_BANK:
        if q.id == question_id:
            return q
    return None


def get_adaptive_question_order(
    beliefs: Dict[str, np.ndarray],
    asked_ids: List[str],
) -> List[CalibrationQuestion]:
    """
    Order remaining questions by expected information gain.

    Questions targeting factors with high uncertainty (flat beliefs)
    are prioritized.
    """
    from ..core.inference import compute_information_gain
    from ..core.model import SphereModel

    model = SphereModel()
    remaining = [q for q in QUESTION_BANK if q.id not in asked_ids]

    def _score(q: CalibrationQuestion) -> float:
        # MC questions get a base bonus to ensure they come first
        # (they have direct A-matrix updates; free-text is supplementary)
        mc_bonus = 10.0 if q.question_type == "mc" else 0.0

        if q.question_type == "free_text":
            return q.information_gain_weight

        if q.category in beliefs and q.category in model.A:
            ig = compute_information_gain(beliefs[q.category], model.A[q.category])
            return mc_bonus + ig * q.information_gain_weight
        return mc_bonus + q.information_gain_weight

    remaining.sort(key=_score, reverse=True)
    return remaining
