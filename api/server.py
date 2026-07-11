"""
AeroDraft REST API server.

Runs in a background daemon thread via uvicorn.
All routes are read-only on SharedState or push to command_queue.
The main OpenCV loop is never touched directly.

Start the server:
    from api.server import start_api_server
    start_api_server(host="0.0.0.0", port=8000)

Base URL (default): http://localhost:8000
Docs (auto-generated): http://localhost:8000/docs
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from api.shared_state import command_queue, shared_state

# ── Valid product keys (mirrors engine/catalog.py) ──────────────────────────
VALID_PRODUCTS = {
    "cube",
    "switchboard",
    "socket",
    "ceiling_light",
    "junction_box",
    "conduit_box",
    "distribution_board",
}

# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AeroDraft API",
    description=(
        "REST control interface for AeroDraft — "
        "AI-powered spatial product visualization for MSMEs. "
        "Use this API to read live telemetry and send control commands "
        "from Postman, a mobile device, or any HTTP client."
    ),
    version="1.0.0",
    contact={"name": "Anushath S", "url": "https://github.com/Anushath15/AeroDraft"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_server_start_time = time.time()


# ── Request / response models ────────────────────────────────────────────────

class SwitchProductRequest(BaseModel):
    product: str

    @field_validator("product")
    @classmethod
    def must_be_valid(cls, v: str) -> str:
        if v not in VALID_PRODUCTS:
            raise ValueError(
                f"Unknown product '{v}'. Valid options: {sorted(VALID_PRODUCTS)}"
            )
        return v


class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get(
    "/",
    summary="Health check",
    tags=["System"],
)
def root() -> dict:
    """
    Quick health-check. Returns API version and uptime.
    Use this in Postman to verify AeroDraft is running.
    """
    return {
        "name": "AeroDraft API",
        "version": "1.0.0",
        "status": "running",
        "uptime_seconds": round(time.time() - _server_start_time, 1),
        "docs": "/docs",
    }


@app.get(
    "/status",
    summary="Live telemetry",
    tags=["Telemetry"],
    response_model=None,
)
def get_status() -> dict:
    """
    Returns the current live state of the AeroDraft pipeline:
    - tracking: whether a hand is detected
    - gesture: PINCH / FIST / OPEN_PALM / NONE
    - box_state: IDLE / DRAWING / PLACED / LOCKED
    - object_type: which product is selected
    - fps: current frame rate
    - depth: estimated hand depth (larger = closer to camera)
    - hand_position: [x, y] pixel coordinates of wrist
    - frame_count: total frames processed since startup

    Poll this every second to build a live dashboard in Postman.
    """
    state = shared_state.snapshot()
    state["api_uptime_seconds"] = round(time.time() - _server_start_time, 1)
    return state


@app.get(
    "/products",
    summary="Product catalog",
    tags=["Catalog"],
)
def get_products() -> dict:
    """
    Returns all 7 products available for visualization with their
    business metadata: display name, category, and real-world dimensions.

    Use the 'key' field as the value when calling POST /product.
    """
    return {
        "products": [
            {
                "key": "cube",
                "display_name": "Wireframe Cube",
                "category": "Demo",
                "dimensions_cm": {"width": 20, "height": 20, "depth": 20},
                "description": "Basic demo object for visualization",
            },
            {
                "key": "switchboard",
                "display_name": "Electrical Switchboard",
                "category": "Electrical",
                "dimensions_cm": {"width": 30, "height": 30, "depth": 10},
                "description": "Main electrical distribution panel for residential use",
            },
            {
                "key": "socket",
                "display_name": "Wall Socket",
                "category": "Electrical",
                "dimensions_cm": {"width": 8, "height": 8, "depth": 5},
                "description": "Standard 16A wall power outlet",
            },
            {
                "key": "ceiling_light",
                "display_name": "LED Ceiling Light",
                "category": "Lighting",
                "dimensions_cm": {"width": 25, "height": 25, "depth": 5},
                "description": "Surface-mounted LED panel for indoor lighting",
            },
            {
                "key": "junction_box",
                "display_name": "Junction Box",
                "category": "Electrical",
                "dimensions_cm": {"width": 10, "height": 10, "depth": 5},
                "description": "Electrical wiring junction enclosure",
            },
            {
                "key": "conduit_box",
                "display_name": "PVC Conduit Box",
                "category": "Conduit",
                "dimensions_cm": {"width": 15, "height": 15, "depth": 10},
                "description": "PVC conduit junction box for cable management",
            },
            {
                "key": "distribution_board",
                "display_name": "Distribution Board",
                "category": "Electrical",
                "dimensions_cm": {"width": 40, "height": 50, "depth": 15},
                "description": "Main power distribution unit for commercial buildings",
            },
        ],
        "total": 7,
    }


@app.post(
    "/product",
    summary="Switch displayed product",
    tags=["Control"],
    response_model=ApiResponse,
)
def switch_product(body: SwitchProductRequest) -> ApiResponse:
    """
    Switch the visualized product instantly.

    Send a JSON body:
    ```json
    { "product": "switchboard" }
    ```

    Valid product keys: cube, switchboard, socket, ceiling_light,
    junction_box, conduit_box, distribution_board.

    The change takes effect on the very next video frame (~33ms).
    """
    command_queue.put({"type": "switch_product", "value": body.product})
    return ApiResponse(
        success=True,
        message=f"Switching to '{body.product}' on next frame.",
        data={"product": body.product},
    )


@app.post(
    "/demo",
    summary="Toggle demo panel",
    tags=["Control"],
    response_model=ApiResponse,
)
def toggle_demo() -> ApiResponse:
    """
    Toggles the MSME demo help panel on or off.
    The panel shows control instructions for judges and visitors.
    No request body needed — just POST to this endpoint.
    """
    command_queue.put({"type": "toggle_demo"})
    return ApiResponse(
        success=True,
        message="Demo panel toggle queued. Effect on next frame.",
    )


@app.post(
    "/screenshot",
    summary="Save screenshot",
    tags=["Control"],
    response_model=ApiResponse,
)
def take_screenshot() -> ApiResponse:
    """
    Saves the current annotated video frame as a PNG file in the
    project root directory. Filename format: aerodraft_{timestamp}.png

    Use this from Postman during a demo to capture placement moments
    without touching the keyboard.
    """
    command_queue.put({"type": "screenshot"})
    return ApiResponse(
        success=True,
        message="Screenshot command queued. File saved on next frame.",
    )


@app.post(
    "/reset",
    summary="Reset interaction state",
    tags=["Control"],
    response_model=ApiResponse,
)
def reset_state() -> ApiResponse:
    """
    Resets the gesture state machine to IDLE, clearing the current
    wireframe box placement. Equivalent to the user showing an open palm
    gesture and losing tracking.

    Useful during a demo to restart a product placement cleanly.
    """
    command_queue.put({"type": "reset"})
    return ApiResponse(
        success=True,
        message="State machine reset queued. Effect on next frame.",
    )


@app.get(
    "/benchmark",
    summary="Performance report",
    tags=["Telemetry"],
)
def get_benchmark() -> dict:
    """
    Returns per-module timing statistics collected since startup.
    Only populated when AeroDraft is started with:
        AERODRAFT_BENCHMARK=1 python -m main

    Fields per module: count, mean_ms, min_ms, max_ms, total_ms.
    Modules: camera_read, hand_tracking, gesture_classification,
             depth_estimation, coordinate_smoothing, state_machine,
             projection_rendering, hud_render.
    """
    try:
        from core.benchmark import get_report
        report = get_report()
        return {
            "benchmark_enabled": bool(report),
            "modules": report,
            "tip": (
                "Run with AERODRAFT_BENCHMARK=1 python -m main "
                "to enable timing collection."
            ) if not report else None,
        }
    except Exception as e:
        return {"benchmark_enabled": False, "error": str(e), "modules": {}}


@app.get(
    "/gestures",
    summary="Gesture reference",
    tags=["Reference"],
)
def get_gestures() -> dict:
    """
    Returns the complete gesture reference guide — what each hand gesture
    does at each stage of the interaction, with tips for reliable detection.
    """
    return {
        "gestures": [
            {
                "name": "PINCH",
                "hand_shape": "Thumb tip and index finger tip touching",
                "actions": {
                    "IDLE": "Spawns the product wireframe at hand position → DRAWING",
                    "DRAWING": "Holds and moves the wireframe — box follows your hand",
                    "PLACED": "Re-enters DRAWING mode",
                    "LOCKED": "No effect — use reset to restart",
                },
                "tip": "Touch thumb and index firmly. The gap must be < 15% of hand size.",
            },
            {
                "name": "OPEN_PALM",
                "hand_shape": "All five fingers fully extended",
                "actions": {
                    "IDLE": "No effect",
                    "DRAWING": "Releases pinch → PLACED (box freezes in space)",
                    "PLACED": "Cancels any active fist-hold timer",
                    "LOCKED": "No effect",
                },
                "tip": "Spread fingers wide and hold flat toward the camera.",
            },
            {
                "name": "FIST",
                "hand_shape": "All fingers curled into palm",
                "actions": {
                    "IDLE": "No effect",
                    "DRAWING": "No effect",
                    "PLACED": "Hold for 1 second → LOCKED (confirms placement)",
                    "LOCKED": "No effect",
                },
                "tip": "Hold the fist steady for a full second. Any other gesture resets the timer.",
            },
        ],
        "keyboard_shortcuts": {
            "1": "cube",
            "2": "switchboard",
            "3": "socket",
            "4": "ceiling_light",
            "5": "junction_box",
            "6": "conduit_box",
            "7": "distribution_board",
            "S": "screenshot",
            "D": "toggle demo panel",
            "Q / ESC": "exit",
        },
    }


# ── Server launcher ───────────────────────────────────────────────────────────

def start_api_server(host: str = "0.0.0.0", port: int = 8000) -> threading.Thread:
    """
    Starts the FastAPI server in a background daemon thread.

    Args:
        host: Network interface to bind (0.0.0.0 = all interfaces).
        port: Port number (default 8000).

    Returns:
        The daemon thread (already started). The main process does not
        need to join it — daemon threads exit automatically when the
        main thread exits.

    Usage:
        thread = start_api_server()
        # main OpenCV loop continues here
    """
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, name="AeroDraft-API", daemon=True)
    thread.start()
    return thread