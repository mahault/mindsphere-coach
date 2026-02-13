"""
LLM prompt templates for various coaching contexts.
"""

from __future__ import annotations

WELCOME_MESSAGE = (
    "Welcome to MindSphere Coach. I'm going to ask you a few quick questions "
    "to understand how you think, work, and where you might be getting stuck. "
    "There are no right or wrong answers — just be honest.\n\n"
    "Ready? Let's start."
)

SPHERE_INTRO = (
    "Here's your MindSphere — a snapshot of your current patterns across "
    "eight key areas. The rounder the shape, the more balanced things are. "
    "Dents show where there's room to grow, and I'll show you how some "
    "areas affect each other."
)

PLAN_INTRO = (
    "Based on your sphere, I've identified the most impactful area to "
    "work on first. I'm going to suggest one small step — designed to be "
    "something you'll actually do, not just something that sounds good."
)

SAFETY_CHECK_MESSAGE = (
    "Quick pulse check — I'm optimizing for your genuine wellbeing, not just "
    "compliance. If anything I've suggested feels coercive or off-base, "
    "that means I'm miscalibrated. Please tell me and I'll adjust."
)

SESSION_WRAP = (
    "Great session. Here's the key insight: {insight}\n\n"
    "Your one step: {step}\n\n"
    "I predict {confidence}% likelihood you'll follow through on this. "
    "Let's check in next time."
)

DEPENDENCY_EXPLANATION = (
    "I noticed that your {blocker} score is holding back your "
    "{blocked} potential. {explanation} "
    "That's why I'm suggesting we start there."
)

# =============================================================================
# COACHING PROMPTS — probing questions organized by skill
# =============================================================================

COACHING_PROBES = {
    "focus": [
        "When you lose focus, what's usually the thing that pulls you away? Is it external (notifications, people) or internal (thoughts, anxiety)?",
        "Think about the last time you were completely absorbed in something. What was it, and what was different about that situation?",
        "Do you notice a pattern in *when* you focus well? Time of day, environment, mood?",
        "What's the cost of your focus challenges? What does it actually take from you?",
    ],
    "follow_through": [
        "When you drop something, is it usually because you lose interest, hit a wall, or something new comes along?",
        "What's the longest streak you've maintained on anything? What made that one different?",
        "Do you feel the drop-off is about motivation, or about the system you set up?",
        "If you could magically finish one thing you've started, what would it be and why does it matter to you?",
    ],
    "social_courage": [
        "When you hold back in a social situation, what's the feeling underneath? Fear of conflict, rejection, being wrong?",
        "Think of someone you admire who speaks their mind. What do they do differently from you?",
        "What's one thing you wish you had said recently but didn't? What stopped you?",
        "How would your life change if you were 20% more direct with people?",
    ],
    "emotional_reg": [
        "When a strong emotion hits you, what's your first instinct? Do you push it away, sit with it, or get swept up?",
        "What does stress show up as in your body? Where do you feel it first?",
        "You mentioned feeling stressed. Can you walk me through what a typical stressful day looks like for you?",
        "What's your current recovery strategy when things go wrong? Does it work?",
    ],
    "systems_thinking": [
        "When the same problem keeps coming back, what usually happens? Do you address the symptoms or dig into why?",
        "Can you think of a time when fixing one thing accidentally improved something else? What was the connection?",
        "If you could map out how your habits connect, what would you notice?",
        "What's one pattern in your life that you know is a pattern but haven't figured out how to change?",
    ],
    "self_trust": [
        "When you second-guess a decision, what's the voice in your head saying?",
        "Think of a decision you trusted yourself on that turned out well. How did that feel?",
        "Do you tend to over-research before making decisions, or do you avoid deciding altogether?",
        "If you trusted yourself 50% more, what would you do differently starting tomorrow?",
    ],
    "task_clarity": [
        "When you start something without a clear picture of 'done', what happens?",
        "Do you find it easier to define the outcome or the process? Which one do you usually skip?",
        "What's your current project or goal right now? Can you describe exactly what success looks like?",
        "How much time do you think you lose to ambiguity — not knowing what to work on next?",
    ],
    "consistency": [
        "What's the main thing that breaks your routines? Is it boredom, life events, or lack of accountability?",
        "Do you work better with external structure (deadlines, accountability partners) or internal motivation?",
        "If you could only be consistent at one thing for the next month, what would you choose?",
        "What does your ideal morning look like vs what actually happens?",
    ],
}

COACHING_EXERCISES = {
    "focus": [
        "Try the 'one tab' rule tomorrow: only one browser tab open at a time. Notice how it feels.",
        "Before your next work session, write down the single thing you'll work on. Put it on a sticky note in front of you.",
        "Do a distraction audit: for one hour, note every time your attention shifts and what caused it.",
    ],
    "follow_through": [
        "Pick the smallest possible next step on something you've abandoned. I mean truly small — something you can do in under 2 minutes.",
        "Create a 'finish something' day: pick one incomplete thing and spend 30 minutes completing it, even if imperfectly.",
        "Write down three things you started this month. Circle the one that matters most. Let the other two go guilt-free.",
    ],
    "social_courage": [
        "Tomorrow, share one genuine opinion that you'd normally keep to yourself. It doesn't have to be confrontational.",
        "Write a message to someone you've been meaning to reach out to. You don't have to send it today — just write it.",
        "Practice the 'response delay': next time someone asks your opinion, count to 3 before answering. Use that time to say what you actually think.",
    ],
    "emotional_reg": [
        "Try the 90-second rule: when a strong emotion hits, set a timer for 90 seconds and just observe the feeling. Most emotional floods peak and subside in that window.",
        "Start a one-line mood journal: each evening, write one sentence about how you felt today and why.",
        "Pick a stress trigger you know is coming this week. Pre-decide how you want to respond to it.",
    ],
    "systems_thinking": [
        "Draw a simple cause-and-effect map of one recurring problem. What feeds into it? What does it feed into?",
        "Ask 'why' five times about a frustration. Go deeper each time. Write down where it leads.",
        "List your three biggest time sinks. For each, ask: is this a symptom or a root cause?",
    ],
    "self_trust": [
        "Make one small decision today without asking anyone else's opinion. Notice what happens.",
        "Write a list of five decisions you made in the past year that turned out well. Keep it somewhere visible.",
        "Next time you catch yourself second-guessing, ask: 'What would I do if I trusted my first instinct?'",
    ],
    "task_clarity": [
        "Before starting your next task, write exactly one sentence describing what 'done' looks like. Be specific.",
        "Take your current biggest project and break it into steps that are each under 30 minutes.",
        "At the end of today, review: did you know what to work on at every point? Where did ambiguity slow you down?",
    ],
    "consistency": [
        "Choose one micro-habit (under 2 minutes) and attach it to something you already do daily. Do it for 3 days.",
        "Set a daily reminder at the same time for one thing. Make it so small it would be embarrassing to skip.",
        "Track just one thing for 7 days: sleep, steps, water, or anything. The act of tracking is the exercise.",
    ],
}
