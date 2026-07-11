import os
import json
from pathlib import Path
from dataclasses import asdict
from typing import Optional
from loguru import logger

CONFIG_FILENAMES = ["aerodraft.json", "aerodraft.config.json", ".aerodraft.json"]

def load_config_file() -> Optional[dict]:
    for filename in CONFIG_FILENAMES:
        path = Path(filename)
        if path.exists():
            try:
                with open(path) as f: return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
    return None

def load_from_environment() -> dict:
    env_config = {}
    if v := os.environ.get("AERODRAFT_CAMERA_WIDTH"):
        env_config.setdefault("camera", {})["width"] = int(v)
    if v := os.environ.get("AERODRAFT_CAMERA_HEIGHT"):
        env_config.setdefault("camera", {})["height"] = int(v)
    if os.environ.get("AERODRAFT_DEMO_MODE", "").lower() in ("1", "true", "yes"):
        env_config["demo"] = {"enabled": True}
    return env_config

def merge_config(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result

def build_settings() -> "AppConfig":
    from config import AppConfig, CameraConfig, TrackerConfig, RenderConfig, DemoConfig
    defaults = asdict(AppConfig())
    merged = merge_config(defaults, load_config_file() or {})
    merged = merge_config(merged, load_from_environment())
    
    return AppConfig(
        camera=CameraConfig(**merged.get("camera", {})),
        tracker=TrackerConfig(**merged.get("tracker", {})),
        render=RenderConfig(**merged.get("render", {})),
        demo=DemoConfig(**merged.get("demo", {})),
    )
