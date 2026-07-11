import os

api = '''import time
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import uvicorn

from api.shared_state import shared_state, command_queue
from engine.catalog import ProductCatalog

VALID_PRODUCTS = {"cube", "switchboard", "socket", "ceiling_light", "junction_box", "conduit_box", "distribution_board"}
_start_time = time.time()

def _get(attr, default=None):
    if hasattr(shared_state, '_data'):
        return shared_state._data.get(attr, default)
    if hasattr(shared_state, '_state'):
        return shared_state._state.get(attr, default)
    return getattr(shared_state, attr, default)

@asynccontextmanager
async def lifespan(app: FastAPI):
    shared_state.is_running = True
    yield
    shared_state.is_running = False

app = FastAPI(title="AeroDraft API", version="1.0.0", lifespan=lifespan)

@app.get("/")
async def root():
    return {"status": "running", "version": "1.0.0", "uptime_seconds": int(time.time() - _start_time)}

@app.get("/status")
async def get_status():
    return {
        "tracking": _get("tracking", False), "gesture": _get("gesture", "NONE"),
        "box_state": _get("box_state", "IDLE"), "object_type": _get("object_type", "cube"),
        "category": _get("category", "Demo"), "fps": _get("fps", 0.0),
        "depth": _get("depth", 0.0), "hand_position": _get("hand_position", None),
        "demo_mode": _get("demo_mode", False), "notification": _get("notification", None),
        "frame_count": _get("frame_count", 0),
    }

@app.get("/products")
async def get_products():
    products = []
    for key in VALID_PRODUCTS:
        info = ProductCatalog.get(key)
        if info:
            products.append({"key": key, "display_name": getattr(info, 'display_name', getattr(info, 'name', key)), "category": getattr(info, 'category', 'Unknown'), "dimensions": getattr(info, 'dimensions', (0,0,0))})
    return {"total": len(products), "products": products}

@app.post("/product")
async def switch_product(data: dict):
    product = data.get("product")
    if not product or product not in VALID_PRODUCTS:
        raise HTTPException(status_code=422, detail="Invalid or missing product")
    command_queue.put({"type": "switch_product", "value": product})
    return {"success": True, "product": product, "message": f"Switched to {product}"}

@app.post("/demo")
async def toggle_demo():
    command_queue.put({"type": "toggle_demo"})
    return {"success": True}

@app.post("/screenshot")
async def take_screenshot():
    command_queue.put({"type": "screenshot"})
    return {"success": True}

@app.post("/reset")
async def reset_state():
    command_queue.put({"type": "reset"})
    return {"success": True}

@app.get("/benchmark")
async def get_benchmark():
    return {"benchmark_enabled": _get("benchmark_enabled", False), "modules": []}

@app.get("/gestures")
async def get_gestures():
    return {"gestures": [{"name": "PINCH", "description": "Place object"}, {"name": "FIST", "description": "Lock object"}, {"name": "OPEN_PALM", "description": "Reset"}], "keyboard_shortcuts": {"1-7": "Switch products", "D": "Toggle demo", "C": "Clear sketch"}}

def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    def run(): uvicorn.run(app, host=host, port=port, log_level="warning")
    threading.Thread(target=run, daemon=True).start()
'''

ht = '''"""
Hand tracking inference module.
Uses the modern MediaPipe Tasks API (>= 0.10).
Isolates all AI logic from camera I/O and rendering.
"""
from __future__ import annotations
from types import TracebackType
from typing import Any, Optional, Type
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
from loguru import logger

from config import TrackerConfig


class HandTracker:
    MODEL_PATH = "hand_landmarker.task"

    def __init__(self, config: TrackerConfig) -> None:
        self._config = config
        self._detector: Optional[Any] = None

    def __enter__(self) -> HandTracker:
        base_options = mp_python.BaseOptions(model_asset_path=self.MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self._config.max_num_hands,
            min_hand_detection_confidence=self._config.min_detection_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)
        logger.info("MediaPipe HandLandmarker initialized (VIDEO mode).")
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None
            logger.info("MediaPipe HandLandmarker closed.")

    def process_frame(self, bgr_frame: np.ndarray, timestamp_ms: Optional[int] = None) -> Any:
        if self._detector is None:
            raise RuntimeError("HandTracker not initialized. Use a with block.")
        if timestamp_ms is None:
            timestamp_ms = int(time.perf_counter() * 1000)
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self._detector.detect(mp_image, timestamp_ms)

    def draw_landmarks(self, bgr_image: np.ndarray, results: Any) -> np.ndarray:
        if not results or not results.hand_landmarks:
            return bgr_image
        h, w = bgr_image.shape[:2]
        for hand in results.hand_landmarks:
            points = []
            for lm in hand:
                px, py = int(lm.x * w), int(lm.y * h)
                points.append((px, py))
                cv2.circle(bgr_image, (px, py), 4, (0, 255, 0), -1)
            for start_idx, end_idx in HAND_CONNECTIONS:
                if start_idx < len(points) and end_idx < len(points):
                    cv2.line(bgr_image, points[start_idx], points[end_idx], (255, 255, 255), 2)
        return bgr_image

HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
'''

with open("api/server.py", "w") as f: f.write(api)
with open("hand_tracker.py", "w") as f: f.write(ht)

print("Final fixes applied!")