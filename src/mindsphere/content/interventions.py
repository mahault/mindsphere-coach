"""
Micro-intervention templates for MindSphere Coach.

Each intervention is a small, actionable step designed to be low-friction.
Interventions are tagged with properties used by the empathy planner to
predict user response.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Intervention:
    """A coaching micro-intervention."""
    id: str
    target_skill: str
    description: str
    difficulty: float           # 0-1 (0 = trivial, 1 = very hard)
    duration_minutes: float     # Expected time to complete
    evaluative: float           # 0-1 (how much it feels like being tested)
    structured: float           # 0-1 (how structured/prescriptive)
    playful: float              # 0-1 (fun/game-like quality)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "target_skill": self.target_skill,
            "description": self.description,
            "difficulty": self.difficulty,
            "duration_minutes": self.duration_minutes,
            "evaluative": self.evaluative,
            "structured": self.structured,
            "playful": self.playful,
        }


# Organized by target skill, with gentle/push pairs
INTERVENTION_BANK: Dict[str, List[Intervention]] = {
    "focus": [
        Intervention(
            id="focus_gentle_1",
            target_skill="focus",
            description="Set a timer for 2 minutes. Do one thing. When it rings, you're done.",
            difficulty=0.1, duration_minutes=2, evaluative=0.0,
            structured=0.8, playful=0.3, tags=["gentle", "micro"],
        ),
        Intervention(
            id="focus_push_1",
            target_skill="focus",
            description="Block all notifications and do a 25-minute deep work session on your most important task.",
            difficulty=0.6, duration_minutes=25, evaluative=0.2,
            structured=0.7, playful=0.1, tags=["push", "deep_work"],
        ),
    ],
    "follow_through": [
        Intervention(
            id="ft_gentle_1",
            target_skill="follow_through",
            description="Pick one small thing you started this week but didn't finish. Spend 5 minutes on just that.",
            difficulty=0.2, duration_minutes=5, evaluative=0.1,
            structured=0.5, playful=0.2, tags=["gentle"],
        ),
        Intervention(
            id="ft_push_1",
            target_skill="follow_through",
            description="Write down the three steps to finish that project, then do the first one right now.",
            difficulty=0.5, duration_minutes=15, evaluative=0.3,
            structured=0.8, playful=0.1, tags=["push"],
        ),
    ],
    "social_courage": [
        Intervention(
            id="sc_gentle_1",
            target_skill="social_courage",
            description="Think of one opinion you held back recently. Write it down — just for yourself.",
            difficulty=0.1, duration_minutes=2, evaluative=0.1,
            structured=0.3, playful=0.2, tags=["gentle", "reflective"],
        ),
        Intervention(
            id="sc_push_1",
            target_skill="social_courage",
            description="In your next conversation today, share one honest thought you'd normally keep to yourself.",
            difficulty=0.6, duration_minutes=5, evaluative=0.5,
            structured=0.3, playful=0.1, tags=["push", "social"],
        ),
    ],
    "emotional_reg": [
        Intervention(
            id="er_gentle_1",
            target_skill="emotional_reg",
            description="Next time something annoys you, just notice the feeling and name it silently. That's it.",
            difficulty=0.1, duration_minutes=1, evaluative=0.0,
            structured=0.2, playful=0.2, tags=["gentle", "mindful"],
        ),
        Intervention(
            id="er_push_1",
            target_skill="emotional_reg",
            description="Keep a 'triggers log' for one day: note what triggered a strong reaction and what you did next.",
            difficulty=0.4, duration_minutes=10, evaluative=0.2,
            structured=0.7, playful=0.1, tags=["push", "tracking"],
        ),
    ],
    "systems_thinking": [
        Intervention(
            id="st_gentle_1",
            target_skill="systems_thinking",
            description="Pick one recurring frustration. Ask 'why?' three times in a row. See where it leads.",
            difficulty=0.2, duration_minutes=3, evaluative=0.0,
            structured=0.4, playful=0.4, tags=["gentle", "inquiry"],
        ),
        Intervention(
            id="st_push_1",
            target_skill="systems_thinking",
            description="Draw a simple diagram of how your daily habits connect. Which ones cascade into others?",
            difficulty=0.5, duration_minutes=15, evaluative=0.2,
            structured=0.7, playful=0.3, tags=["push", "mapping"],
        ),
    ],
    "self_trust": [
        Intervention(
            id="str_gentle_1",
            target_skill="self_trust",
            description="Think of one decision you made recently that turned out well. Acknowledge it to yourself.",
            difficulty=0.1, duration_minutes=1, evaluative=0.0,
            structured=0.2, playful=0.1, tags=["gentle", "reflective"],
        ),
        Intervention(
            id="str_push_1",
            target_skill="self_trust",
            description="Make one decision today without consulting anyone. Commit to it for 24 hours.",
            difficulty=0.5, duration_minutes=5, evaluative=0.4,
            structured=0.4, playful=0.1, tags=["push", "practice"],
        ),
    ],
    "task_clarity": [
        Intervention(
            id="tc_gentle_1",
            target_skill="task_clarity",
            description="Pick your most important task for tomorrow. Write exactly what 'done' looks like in one sentence.",
            difficulty=0.2, duration_minutes=2, evaluative=0.1,
            structured=0.7, playful=0.1, tags=["gentle", "planning"],
        ),
        Intervention(
            id="tc_push_1",
            target_skill="task_clarity",
            description="For your current project, write a concrete checklist of every sub-task with clear completion criteria.",
            difficulty=0.5, duration_minutes=20, evaluative=0.3,
            structured=0.9, playful=0.0, tags=["push", "planning"],
        ),
    ],
    "consistency": [
        Intervention(
            id="con_gentle_1",
            target_skill="consistency",
            description="Choose one tiny thing (30 seconds) to do every morning for the next 3 days. Just 3 days.",
            difficulty=0.1, duration_minutes=0.5, evaluative=0.0,
            structured=0.6, playful=0.3, tags=["gentle", "habit"],
        ),
        Intervention(
            id="con_push_1",
            target_skill="consistency",
            description="Commit to a 15-minute daily practice for one week. Track it with a visible streak counter.",
            difficulty=0.5, duration_minutes=15, evaluative=0.3,
            structured=0.8, playful=0.2, tags=["push", "habit"],
        ),
    ],
}


def get_interventions_for_skill(
    skill: str, variant: str = "gentle"
) -> List[Intervention]:
    """Get interventions for a skill, filtered by variant tag."""
    interventions = INTERVENTION_BANK.get(skill, [])
    return [i for i in interventions if variant in i.tags]


def get_gentle_push_pair(skill: str) -> tuple[Intervention, Intervention]:
    """Get a (gentle, push) pair for counterfactual display."""
    gentle = get_interventions_for_skill(skill, "gentle")
    push = get_interventions_for_skill(skill, "push")
    return (
        gentle[0] if gentle else INTERVENTION_BANK[skill][0],
        push[0] if push else INTERVENTION_BANK[skill][-1],
    )
