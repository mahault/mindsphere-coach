"""
REST API routes for MindSphere Coach.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException

from .schemas import (
    StartSessionRequest,
    StartSessionResponse,
    StepRequest,
    StepResponse,
    EmpathyDialRequest,
    SphereData,
    QuestionData,
)
from .session import SessionManager

router = APIRouter(prefix="/api")

# Global session manager (initialized in app.py lifespan)
session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    global session_manager
    if session_manager is None:
        session_manager = SessionManager()
    return session_manager


@router.post("/session/start", response_model=StartSessionResponse)
async def start_session(request: StartSessionRequest = StartSessionRequest()):
    """Start a new coaching session."""
    sm = get_session_manager()
    session_id = sm.create_session(
        lambda_empathy=request.lambda_empathy,
        n_particles=request.n_particles,
    )
    agent = sm.get_agent(session_id)
    if agent is None:
        raise HTTPException(500, "Failed to create session")

    result = agent.start_session()

    # Store welcome message in history
    sm.add_to_history(session_id, "assistant", result["message"])

    question_data = None
    if result.get("question"):
        q = result["question"]
        question_data = QuestionData(**q)

    return StartSessionResponse(
        session_id=session_id,
        phase=result["phase"],
        message=result["message"],
        question=question_data,
        is_complete=result.get("is_complete", False),
    )


@router.post("/session/{session_id}/step", response_model=StepResponse)
async def step(session_id: str, request: StepRequest):
    """Process one step of the coaching session."""
    sm = get_session_manager()
    if not sm.session_exists(session_id):
        raise HTTPException(404, f"Session {session_id} not found")

    agent = sm.get_agent(session_id)
    if agent is None:
        raise HTTPException(500, "Agent not found for session")

    # Build user input dict for the agent
    user_input = {
        "answer": request.user_message,
        "message_type": request.message_type,
    }
    if request.answer_index is not None:
        user_input["answer_index"] = request.answer_index
    if request.choice is not None:
        user_input["choice"] = request.choice

    # Store user message in history
    sm.add_to_history(session_id, "user", request.user_message)

    # Process step
    result = agent.step(user_input)

    # Store assistant response in history
    if result.get("message"):
        sm.add_to_history(session_id, "assistant", result["message"])

    # Build response
    question_data = None
    if result.get("question"):
        question_data = QuestionData(**result["question"])

    intervention_data = None
    if result.get("intervention"):
        intervention_data = result["intervention"]

    return StepResponse(
        phase=result.get("phase", "unknown"),
        message=result.get("message", ""),
        question=question_data,
        intervention=intervention_data,
        counterfactual=result.get("counterfactual"),
        sphere_data=result.get("sphere_data"),
        belief_summary=result.get("belief_summary") or (agent.get_belief_summary() if result.get("sphere_data") else None),
        tom_stats=result.get("tom_stats"),
        user_type_summary=result.get("user_type_summary"),
        dependency_explanation=result.get("dependency_explanation"),
        progress=result.get("progress"),
        is_complete=result.get("is_complete", False),
    )


@router.get("/session/{session_id}/sphere")
async def get_sphere(session_id: str):
    """Get current sphere data for a session."""
    sm = get_session_manager()
    if not sm.session_exists(session_id):
        raise HTTPException(404, f"Session {session_id} not found")

    agent = sm.get_agent(session_id)
    if agent is None:
        raise HTTPException(500, "Agent not found")

    return agent.get_sphere_data()


@router.get("/session/{session_id}/beliefs")
async def get_beliefs(session_id: str):
    """Get current belief summary for a session."""
    sm = get_session_manager()
    if not sm.session_exists(session_id):
        raise HTTPException(404, f"Session {session_id} not found")

    agent = sm.get_agent(session_id)
    if agent is None:
        raise HTTPException(500, "Agent not found")

    return agent.get_belief_summary()


@router.get("/session/{session_id}/user-state")
async def get_user_state(session_id: str):
    """Get the inferred cognitive/emotional state of the user."""
    sm = get_session_manager()
    if not sm.session_exists(session_id):
        raise HTTPException(404, f"Session {session_id} not found")

    agent = sm.get_agent(session_id)
    if agent is None:
        raise HTTPException(500, "Agent not found")

    return agent.get_inferred_user_state()


@router.post("/session/{session_id}/empathy-dial")
async def set_empathy_dial(session_id: str, request: EmpathyDialRequest):
    """Adjust the empathy dial for a session."""
    sm = get_session_manager()
    if not sm.session_exists(session_id):
        raise HTTPException(404, f"Session {session_id} not found")

    agent = sm.get_agent(session_id)
    if agent is None:
        raise HTTPException(500, "Agent not found")

    agent.set_empathy_dial(request.lambda_value)
    return {"lambda_empathy": request.lambda_value}
