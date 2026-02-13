"""
FastAPI application for MindSphere Coach.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket

# Configure logging to show INFO from mindsphere modules
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logging.getLogger("mindsphere").setLevel(logging.INFO)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .routes import router, get_session_manager
from .websocket import websocket_endpoint

app = FastAPI(
    title="MindSphere Coach",
    description="Interactive ToM-powered coaching agent with Active Inference",
    version="0.1.0",
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router)

# Static frontend files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the main frontend page."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "MindSphere Coach API", "docs": "/docs"}


@app.websocket("/ws/{session_id}")
async def ws_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time coaching."""
    sm = get_session_manager()
    await websocket_endpoint(websocket, session_id, sm)
