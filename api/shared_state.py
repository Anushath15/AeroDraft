"""
Shared state and command queue for AeroDraft REST API.

This module is the only communication bridge between the main OpenCV loop
(which runs in the main thread) and the FastAPI server (which runs in a
background daemon thread).

Thread safety:
    SharedState uses a threading.Lock for all reads and writes.
    command_queue is a stdlib queue.Queue — inherently thread-safe.

Design rule:
    The API thread NEVER touches OpenCV, MediaPipe, or any gesture state
    directly. It only reads from SharedState and pushes to command_queue.
    The main loop reads commands and executes them at the top of each frame.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class _State:
    """Internal snapshot — written by main loop, read by API."""
    tracking: bool = False
    gesture: str = "NONE"
    box_state: str = "IDLE"
    object_type: str = "cube"
    category: str = "Demo"
    fps: float = 0.0
    depth: Optional[float] = None
    hand_position: Optional[Tuple[int, int]] = None
    demo_mode: bool = False
    notification: Optional[str] = None
    frame_count: int = 0
    uptime_seconds: float = 0.0


class SharedState:
    """
    Thread-safe live telemetry store.

    The main loop calls update() once per frame.
    FastAPI route handlers call snapshot() to get a JSON-safe dict.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data = _State()

    def update(self, **kwargs) -> None:
        """
        Update one or more fields atomically.
        Unknown keys are silently ignored (forward-compatible).
        frame_count is auto-incremented on every update call.
        """
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._data, key):
                    setattr(self._data, key, value)
            self._data.frame_count += 1

    def snapshot(self) -> dict:
        """Returns a JSON-serialisable copy of the current state."""
        with self._lock:
            d = self._data
            return {
                "tracking": d.tracking,
                "gesture": d.gesture,
                "box_state": d.box_state,
                "object_type": d.object_type,
                "category": d.category,
                "fps": round(d.fps, 1),
                "depth": round(d.depth, 3) if d.depth is not None else None,
                "hand_position": list(d.hand_position) if d.hand_position else None,
                "demo_mode": d.demo_mode,
                "notification": d.notification,
                "frame_count": d.frame_count,
                "uptime_seconds": round(d.uptime_seconds, 1),
            }


# ── Singletons used by both main.py and server.py ──────────────────────────

shared_state: SharedState = SharedState()
"""Live telemetry — main loop writes, API reads."""

command_queue: queue.Queue = queue.Queue()
"""
Control commands from API to main loop.

Each item is a dict with at least a "type" key:
    {"type": "switch_product", "value": "switchboard"}
    {"type": "toggle_demo"}
    {"type": "screenshot"}
    {"type": "reset"}
"""