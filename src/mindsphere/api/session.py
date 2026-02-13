"""
In-memory session manager for MindSphere Coach.

Stores active CoachingAgent instances keyed by session_id.
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from ..core.agent import CoachingAgent


class SessionManager:
    """
    Manages active coaching sessions in memory.

    Each session has its own CoachingAgent instance and conversation history.
    """

    def __init__(self):
        self._sessions: Dict[str, Dict] = {}

    def create_session(
        self,
        lambda_empathy: float = 0.5,
        n_particles: int = 50,
    ) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())[:8]
        agent = CoachingAgent(
            lambda_empathy=lambda_empathy,
            n_particles=n_particles,
        )
        self._sessions[session_id] = {
            "agent": agent,
            "conversation_history": [],
        }
        return session_id

    def get_agent(self, session_id: str) -> Optional[CoachingAgent]:
        """Get the agent for a session."""
        session = self._sessions.get(session_id)
        return session["agent"] if session else None

    def get_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session."""
        session = self._sessions.get(session_id)
        return session["conversation_history"] if session else []

    def add_to_history(
        self, session_id: str, role: str, content: str
    ) -> None:
        """Add a message to the conversation history."""
        session = self._sessions.get(session_id)
        if session:
            session["conversation_history"].append({
                "role": role,
                "content": content,
            })

    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        return session_id in self._sessions

    def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        self._sessions.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self._sessions.keys())
