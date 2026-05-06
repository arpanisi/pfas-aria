"""
WebSocket handler.
Streams real-time pipeline status updates to the React frontend.
Clients connect to /ws/{run_id} and receive JSON status messages.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["WebSocket"])

# Active WebSocket connections per run_id
_connections: dict[str, list[WebSocket]] = {}


@router.websocket("/ws/{run_id}")
async def pipeline_websocket(websocket: WebSocket, run_id: str) -> None:
    """
    WebSocket endpoint for real-time pipeline updates.
    Client connects here and receives status messages as the pipeline runs.
    """
    await websocket.accept()

    if run_id not in _connections:
        _connections[run_id] = []
    _connections[run_id].append(websocket)

    logger.info(f"WebSocket connected for run {run_id}")

    try:
        # Keep connection alive and handle pings
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if data == "ping":
                    await websocket.send_text("pong")
            except TimeoutError:
                # Send keepalive
                await websocket.send_json({"type": "keepalive"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for run {run_id}")
    finally:
        if run_id in _connections:
            _connections[run_id] = [
                ws for ws in _connections[run_id] if ws != websocket
            ]


async def broadcast_status(run_id: str, message: dict[str, Any]) -> None:
    """
    Broadcast a status message to all connected clients for a run.
    Called by the pipeline at each stage transition.
    """
    if run_id not in _connections:
        return

    dead: list[WebSocket] = []
    for ws in _connections[run_id]:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)

    # Clean up dead connections
    for ws in dead:
        _connections[run_id].remove(ws)


def build_status_message(
    run_id: str,
    stage: str,
    round_number: int = 0,
    match_score: float = 0.0,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standardized status message for the frontend."""
    return {
        "type": "status",
        "run_id": run_id,
        "stage": stage,
        "round": round_number,
        "match_score": match_score,
        "details": details or {},
    }
