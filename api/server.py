import time
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import uvicorn

from api.shared_state import shared_state, command_queue
from engine.catalog import ProductCatalog

VALID_PRODUCTS = {"cube", "switchboard", "socket", "ceiling_light", "junction_box", "conduit_box", "distribution_board"}
_start_time = time.time()

def _get(attr, default=None):
    """Retrieve a field from shared state snapshot."""
    try:
        snapshot = shared_state.snapshot()
        return snapshot.get(attr, default)
    except Exception:
        return default

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
            products.append({"key": key, "display_name": getattr(info, 'display_name', getattr(info, 'name', key)), "category": getattr(info, 'category', 'Unknown'), "dimensions_cm": getattr(info, 'dimensions', (0,0,0))})
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
