"""
Heads-Up Display renderer for AeroDraft.
Provides real-time telemetry, status indicators, and demo overlays.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from config import HUDConfig
from gestures.gesture_classifier import GestureType
from gestures.state_machine import BoxState


@dataclass
class HUDData:
    tracking: bool = False
    gesture: Optional[GestureType] = None
    state: Optional[BoxState] = None
    object_type: Optional[str] = None
    fps: Optional[float] = None
    depth: Optional[float] = None
    hand_position: Optional[Tuple[int, int]] = None
    # Phase 12 (all optional defaults for backward compatibility)
    category: Optional[str] = None
    notification: Optional[str] = None
    notification_color: Optional[Tuple[int, int, int]] = None
    demo_mode: bool = False


class HUDRenderer:
    def __init__(self, config: HUDConfig) -> None:
        self.config = config

    def render(self, frame: np.ndarray, data: HUDData) -> np.ndarray:
        if frame.size == 0:
            return frame
        self._draw_info_panel(frame, data)
        self._draw_notification_banner(frame, data)
        self._draw_demo_panel(frame, data)
        self._draw_controls_legend(frame)
        return frame

    def _draw_info_panel(self, frame: np.ndarray, data: HUDData) -> None:
        cfg = self.config
        x = cfg.left_margin
        y = cfg.top_margin

        # Title
        cv2.putText(frame, "AeroDraft", (x, y), cfg.font_face, cfg.font_scale, cfg.info_color, cfg.font_thickness + 1)
        y += cfg.line_spacing

        # Tracking indicator (exact position preserved for tests)
        tracking_text = "TRACKING : YES" if data.tracking else "TRACKING : NO"
        tracking_color = cfg.text_color if data.tracking else cfg.alert_color
        indicator_cx = cfg.left_margin + 8
        indicator_cy = cfg.top_margin + cfg.line_spacing - 4
        cv2.circle(frame, (indicator_cx, indicator_cy), cfg.indicator_radius, tracking_color, -1)
        cv2.putText(frame, tracking_text, (x + 20, y), cfg.font_face, cfg.font_scale, tracking_color, cfg.font_thickness)
        y += cfg.line_spacing

        # Object & Category
        if data.object_type:
            obj_name = self._format_object_type(data.object_type)
            cv2.putText(frame, f"OBJECT : {obj_name}", (x, y), cfg.font_face, cfg.font_scale, cfg.text_color, cfg.font_thickness)
            y += cfg.line_spacing
            if data.category:
                cv2.putText(frame, f"CATEGORY : {data.category.upper()}", (x, y), cfg.font_face, cfg.font_scale, cfg.text_color, cfg.font_thickness)
                y += cfg.line_spacing

        # State with color
        if data.state is not None:
            state_name = self._format_state(data.state)
            state_color = self._resolve_state_color(data.state)
            cv2.putText(frame, f"STATE : {state_name}", (x, y), cfg.font_face, cfg.font_scale, state_color, cfg.font_thickness)
            y += cfg.line_spacing

        # Depth
        depth_str = f"DEPTH : {data.depth:.2f}" if data.depth is not None else "DEPTH : --"
        cv2.putText(frame, depth_str, (x, y), cfg.font_face, cfg.font_scale, cfg.text_color, cfg.font_thickness)
        y += cfg.line_spacing

        # FPS
        fps_str = f"FPS : {data.fps:.1f}" if data.fps is not None else "FPS : --"
        cv2.putText(frame, fps_str, (x, y), cfg.font_face, cfg.font_scale, cfg.text_color, cfg.font_thickness)
        y += cfg.line_spacing

        # Hand position
        hand_str = f"HAND : {self._format_position(data.hand_position)}" if data.hand_position else "HAND : --"
        cv2.putText(frame, hand_str, (x, y), cfg.font_face, cfg.font_scale, cfg.text_color, cfg.font_thickness)

    def _draw_notification_banner(self, frame: np.ndarray, data: HUDData) -> None:
        if not data.notification:
            return
        h, w = frame.shape[:2]
        banner_h = 42
        color = data.notification_color or self.config.info_color
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        text_size = cv2.getTextSize(data.notification, self.config.font_face, self.config.font_scale * 1.1, self.config.font_thickness + 1)[0]
        text_x = (w - text_size[0]) // 2
        text_y = banner_h - 12
        cv2.putText(frame, data.notification, (text_x, text_y), self.config.font_face, self.config.font_scale * 1.1, color, self.config.font_thickness + 1)

    def _draw_demo_panel(self, frame: np.ndarray, data: HUDData) -> None:
        if not data.demo_mode:
            return
        h, w = frame.shape[:2]
        panel_w = self.config.demo_panel_width
        panel_x = w - panel_w - 10
        panel_y = 10
        panel_h = 200
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (25, 25, 25), -1)
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (80, 80, 80), 1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        x = panel_x + 15
        y = panel_y + 30
        cfg = self.config
        cv2.putText(frame, "Demo Scenario", (x, y), cfg.font_face, cfg.font_scale * 1.1, cfg.info_color, cfg.font_thickness + 1)
        y += 30
        lines = [
            "Customer wants to preview",
            "electrical products before",
            "installation.",
            "",
            "Controls:",
            "Pinch  -> Place Object",
            "Move   -> Position",
            "Fist   -> Lock / Confirm",
        ]
        for line in lines:
            cv2.putText(frame, line, (x, y), cfg.font_face, cfg.font_scale, cfg.text_color, cfg.font_thickness)
            y += cfg.line_spacing

    def _draw_controls_legend(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        legend = "Q/ESC: Exit | Pinch: Place | Fist: Lock"
        text_size = cv2.getTextSize(legend, self.config.font_face, self.config.font_scale, self.config.font_thickness)[0]
        x = (w - text_size[0]) // 2
        y = h - 15
        cv2.putText(frame, legend, (x, y), self.config.font_face, self.config.font_scale, (180, 180, 180), self.config.font_thickness)

    def _resolve_state_color(self, state: BoxState) -> Tuple[int, int, int]:
        from config import settings
        return settings.render.state_colors.get(state.name, self.config.text_color)

    @staticmethod
    def _format_gesture(gesture: Optional[GestureType]) -> str:
        if gesture is None:
            return "NONE"
        return gesture.name.replace("_", " ")

    @staticmethod
    def _format_state(state: Optional[BoxState]) -> str:
        if state is None:
            return "NONE"
        return state.name

    @staticmethod
    def _format_position(pos: Optional[Tuple[int, int]]) -> str:
        if pos is None:
            return "--"
        return f"({pos[0]}, {pos[1]})"

    @staticmethod
    def _format_object_type(obj_type: Optional[str]) -> str:
        if obj_type is None:
            return "NONE"
        return obj_type.replace("_", " ").upper()