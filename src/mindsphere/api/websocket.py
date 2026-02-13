"""
WebSocket handler for real-time MindSphere Coach updates.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import WebSocket, WebSocketDisconnect

from .session import SessionManager


async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    session_manager: SessionManager,
):
    """
    WebSocket handler for a coaching session.

    Protocol:
        Client -> Server:
            {"type": "user_message", "content": "...", "message_type": "text|mc|choice"}
            {"type": "user_message", "content": "...", "answer_index": 0}
            {"type": "user_message", "content": "...", "choice": "accept"}
            {"type": "continue"}  (for phase transitions)
            {"type": "set_empathy", "value": 0.7}

        Server -> Client:
            {"type": "assistant_message", "content": "...", "phase": "...", "action": "..."}
            {"type": "question", "data": {...}}
            {"type": "sphere_update", "data": {...}}
            {"type": "counterfactual", "data": {...}}
            {"type": "intervention", "data": {...}}
            {"type": "phase_change", "phase": "..."}
            {"type": "belief_update", "data": {...}}
            {"type": "error", "message": "..."}
    """
    await websocket.accept()

    if not session_manager.session_exists(session_id):
        await websocket.send_json({"type": "error", "message": f"Session {session_id} not found"})
        await websocket.close()
        return

    agent = session_manager.get_agent(session_id)
    if agent is None:
        await websocket.send_json({"type": "error", "message": "Agent not found"})
        await websocket.close()
        return

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "user_message":
                content = data.get("content", "")
                message_type = data.get("message_type", "text")

                # Build user input
                user_input: Dict[str, Any] = {"answer": content}
                if "answer_index" in data:
                    user_input["answer_index"] = data["answer_index"]
                if "choice" in data:
                    user_input["choice"] = data["choice"]

                # Store in history
                session_manager.add_to_history(session_id, "user", content)

                # Process step
                result = agent.step(user_input)

                # Send assistant message
                if result.get("message"):
                    session_manager.add_to_history(session_id, "assistant", result["message"])
                    await websocket.send_json({
                        "type": "assistant_message",
                        "content": result["message"],
                        "phase": result.get("phase", ""),
                    })

                # Send question if present
                if result.get("question"):
                    await websocket.send_json({
                        "type": "question",
                        "data": result["question"],
                    })

                # Send sphere update if present
                if result.get("sphere_data"):
                    await websocket.send_json({
                        "type": "sphere_update",
                        "data": result["sphere_data"],
                    })

                # Send counterfactual if present
                if result.get("counterfactual"):
                    await websocket.send_json({
                        "type": "counterfactual",
                        "data": result["counterfactual"],
                    })

                # Send intervention if present
                if result.get("intervention"):
                    await websocket.send_json({
                        "type": "intervention",
                        "data": result["intervention"],
                    })

                # Send phase change
                await websocket.send_json({
                    "type": "phase_change",
                    "phase": result.get("phase", ""),
                    "progress": result.get("progress"),
                    "is_complete": result.get("is_complete", False),
                })

                # Send belief update
                if result.get("tom_stats") or result.get("user_type_summary"):
                    await websocket.send_json({
                        "type": "belief_update",
                        "data": {
                            "tom_stats": result.get("tom_stats"),
                            "user_type": result.get("user_type_summary"),
                            "beliefs": agent.get_belief_summary(),
                        },
                    })

            elif msg_type == "continue":
                # Trigger phase transition
                result = agent.step({})
                await websocket.send_json({
                    "type": "phase_change",
                    "phase": result.get("phase", ""),
                })
                if result.get("message"):
                    await websocket.send_json({
                        "type": "assistant_message",
                        "content": result["message"],
                        "phase": result.get("phase", ""),
                    })
                if result.get("sphere_data"):
                    await websocket.send_json({
                        "type": "sphere_update",
                        "data": result["sphere_data"],
                    })
                if result.get("intervention"):
                    await websocket.send_json({
                        "type": "intervention",
                        "data": result["intervention"],
                    })
                if result.get("counterfactual"):
                    await websocket.send_json({
                        "type": "counterfactual",
                        "data": result["counterfactual"],
                    })

            elif msg_type == "set_empathy":
                value = float(data.get("value", 0.5))
                agent.set_empathy_dial(value)
                await websocket.send_json({
                    "type": "empathy_updated",
                    "value": value,
                })

    except WebSocketDisconnect:
        pass
