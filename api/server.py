import time
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from api.shared_state import shared_state, command_queue
from engine.catalog import ProductCatalog

VALID_PRODUCTS = ["cube", "switchboard", "socket", "ceiling_light", "junction_box", "conduit_box", "distribution_board"]
_start_time = time.time()

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
        "tracking": shared_state.tracking,
        "gesture": shared_state.gesture,
        "box_state": shared_state.box_state,
        "object_type": shared_state.object_type,
        "category": shared_state.category,
        "fps": shared_state.fps,
        "depth": shared_state.depth,
        "hand_position": shared_state.hand_position,
        "demo_mode": shared_state.demo_mode,
        "notification": shared_state.notification,
        "frame_count": shared_state.frame_count,
    }

@app.get("/products")
async def get_products():
    products = []
    for key in VALID_PRODUCTS:
        info = ProductCatalog.get(key)
        if info:
            products.append({"key": key, "name": info.name, "category": info.category, "dimensions": info.dimensions})
    return {"total": len(products), "products": products}

@app.post("/product")
async def switch_product(data: dict):
    product = data.get("product")
    if not product or product not in VALID_PRODUCTS:
        raise HTTPException(status_code=422, detail="Invalid or missing product")
    command_queue.put({"type": "switch_product", "value": product})
    return {"success": True, "product": product}

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
    return {"benchmark_enabled": shared_state.benchmark_enabled, "modules": []}

@app.get("/gestures")
async def get_gestures():
    return {
        "gestures": [
            {"name": "Pinch", "description": "Place object"},
            {"name": "Fist", "description": "Lock object"},
            {"name": "Open Palm", "description": "Reset"}
        ],
        "keyboard_shortcuts": {"1-7": "Switch products", "D": "Toggle demo", "C": "Clear sketch"}
    }

def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    def run():
        uvicorn.run(app, host=host, port=port, log_level="warning")
    thread = threading.Thread(target=run, daemon=True)
    thread.start()