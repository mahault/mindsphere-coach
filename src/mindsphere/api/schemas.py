"""
Pydantic request/response models for the MindSphere Coach API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# REQUEST MODELS
# =============================================================================

class StartSessionRequest(BaseModel):
    """Request to start a new coaching session."""
    lambda_empathy: float = Field(0.5, ge=0.0, le=1.0, description="Empathy dial (0=challenging, 1=gentle)")
    n_particles: int = Field(50, ge=10, le=200, description="Number of ToM particles")


class StepRequest(BaseModel):
    """Request to process one step of the session."""
    user_message: str = Field(..., description="User's text response")
    message_type: str = Field("text", description="Type: text, mc_choice, user_choice")
    answer_index: Optional[int] = Field(None, description="MC answer index (0-3)")
    choice: Optional[str] = Field(None, description="User choice: accept, too_hard, not_relevant")


class EmpathyDialRequest(BaseModel):
    """Request to adjust the empathy dial."""
    lambda_value: float = Field(..., ge=0.0, le=1.0)


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class QuestionData(BaseModel):
    """Data for a calibration question."""
    id: str
    category: str
    question_text: str
    question_type: str
    options: Optional[List[str]] = None


class InterventionData(BaseModel):
    """Data for a proposed intervention."""
    description: str
    target_skill: str
    duration_minutes: float
    difficulty: float


class CounterfactualOption(BaseModel):
    """One option in a counterfactual comparison."""
    description: str
    duration_minutes: float
    p_completion: float
    p_dropout: float
    felt_cost: float


class CounterfactualData(BaseModel):
    """Counterfactual comparison between two approaches."""
    gentle: CounterfactualOption
    push: CounterfactualOption
    recommendation: str
    confidence: float


class BottleneckData(BaseModel):
    """A skill bottleneck."""
    blocker: str
    blocked: List[str]
    score: float
    impact: float


class SphereData(BaseModel):
    """Full sphere visualization data."""
    categories: Dict[str, float]
    bottlenecks: List[Dict[str, Any]]
    dependency_edges: List[Dict[str, Any]]


class StartSessionResponse(BaseModel):
    """Response from starting a new session."""
    session_id: str
    phase: str
    message: str
    question: Optional[QuestionData] = None
    is_complete: bool = False


class StepResponse(BaseModel):
    """Response from processing one step."""
    phase: str
    message: str
    question: Optional[QuestionData] = None
    intervention: Optional[InterventionData] = None
    counterfactual: Optional[Dict[str, Any]] = None
    sphere_data: Optional[Dict[str, Any]] = None
    belief_summary: Optional[Dict[str, Any]] = None
    tom_stats: Optional[Dict[str, Any]] = None
    user_type_summary: Optional[Dict[str, float]] = None
    dependency_explanation: Optional[str] = None
    progress: Optional[float] = None
    is_complete: bool = False
