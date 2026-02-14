"""
REST API routes for MindSphere Coach.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from starlette.responses import StreamingResponse

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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Global session manager (initialized in app.py lifespan)
session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    global session_manager
    if session_manager is None:
        session_manager = SessionManager()
    return session_manager


@router.get("/status")
async def status():
    """Check system status including LLM availability."""
    try:
        from ..llm.client import MistralClient
        client = MistralClient()
        llm_available = client.is_available
    except Exception:
        llm_available = False
    return {"llm_available": llm_available}


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


@router.post("/session/{session_id}/step-stream")
async def step_stream(session_id: str, request: StepRequest):
    """SSE streaming version of step(). Streams LLM tokens as they arrive."""
    sm = get_session_manager()
    if not sm.session_exists(session_id):
        raise HTTPException(404, f"Session {session_id} not found")

    agent = sm.get_agent(session_id)
    if agent is None:
        raise HTTPException(500, "Agent not found for session")

    user_input = {
        "answer": request.user_message,
        "message_type": request.message_type,
    }
    if request.answer_index is not None:
        user_input["answer_index"] = request.answer_index
    if request.choice is not None:
        user_input["choice"] = request.choice

    sm.add_to_history(session_id, "user", request.user_message)

    def event_generator():
        full_message = []
        for event in agent.step_stream(user_input):
            event_type = event.get("event", "token")
            data = event.get("data", {})
            payload = json.dumps(data, default=str)
            yield f"event: {event_type}\ndata: {payload}\n\n"
            if event_type == "token":
                full_message.append(data.get("text", ""))
        # Store the full assembled message in session history
        msg = "".join(full_message)
        if msg:
            sm.add_to_history(session_id, "assistant", msg)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
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


@router.get("/session/{session_id}/profile-data")
async def get_profile_data(session_id: str):
    """Get detailed profile data for visualization panel."""
    sm = get_session_manager()
    if not sm.session_exists(session_id):
        raise HTTPException(404, f"Session {session_id} not found")

    agent = sm.get_agent(session_id)
    if agent is None:
        raise HTTPException(500, "Agent not found")

    return agent.get_profile_data()


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
