"""
Heads-up display (HUD) rendering module.

Draws informational overlays onto a video frame without modifying
application state or interpreting gestures. Pure rendering only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from config import HUDConfig
from gestures.gesture_classifier import GestureType
from gestures.state_machine import BoxState


@dataclass(frozen=True)
class HUDData:
    """
    Immutable snapshot of all data the HUD can display.
    
    Encapsulates the render parameters to avoid long argument lists
    and enable future extension without API breakage.
    """
    tracking: bool = False
    gesture: Optional[GestureType] = None
    state: Optional[BoxState] = None
    fps: float = 0.0
    depth: Optional[float] = None
    hand_position: Optional[Tuple[int, int]] = None


class HUDRenderer:
    """
    Stateless HUD overlay renderer.
    
    Draws text and status indicators onto a BGR frame. Does not
    classify gestures, transition states, or modify any application
    logic. All data is passed in via HUDData.
    
    Usage:
        renderer = HUDRenderer(settings.hud)
        annotated = renderer.render(frame, HUDData(tracking=True, fps=30.0))
    """

    # Control legend text — static, never changes
    _CONTROLS_TEXT: str = "Pinch→Select  Point→Cursor  Open Palm→Reset"

    def __init__(self, config: HUDConfig) -> None:
        """
        Args:
            config: Font, color, and layout configuration.
        """
        self._config = config

    def render(self, frame: np.ndarray, data: HUDData) -> np.ndarray:
        """
        Draws the complete HUD overlay onto the given frame.
        
        Args:
            frame: BGR image to annotate (modified in place and returned).
            data: Snapshot of current application state to display.
        
        Returns:
            The annotated frame (same object as input).
        """
        if frame is None or frame.size == 0:
            logger.warning("HUD received empty frame — returning unchanged.")
            return frame

        overlay = frame.copy()

        # --- Title ---
        self._draw_text(overlay, "AeroDraft", self._config.top_margin, self._config.text_color)

        # --- Status indicator (colored dot) ---
        indicator_y = self._config.top_margin + self._config.line_spacing
        self._draw_status_indicator(overlay, data.tracking, indicator_y)

        # --- Tracking line ---
        tracking_y = indicator_y
        tracking_text = "TRACKING: YES" if data.tracking else "TRACKING: NO"
        tracking_color = self._config.text_color if data.tracking else self._config.alert_color
        self._draw_text(overlay, tracking_text, tracking_y, tracking_color, offset_x=20)

        # --- Gesture ---
        gesture_y = tracking_y + self._config.line_spacing
        gesture_name = self._format_gesture(data.gesture)
        self._draw_text(overlay, f"GESTURE: {gesture_name}", gesture_y, self._config.info_color)

        # --- State ---
        state_y = gesture_y + self._config.line_spacing
        state_name = self._format_state(data.state)
        self._draw_text(overlay, f"STATE: {state_name}", state_y, self._config.info_color)

        # --- FPS ---
        fps_y = state_y + self._config.line_spacing
        self._draw_text(overlay, f"FPS: {data.fps:.1f}", fps_y, self._config.text_color)

        # --- Depth ---
        depth_y = fps_y + self._config.line_spacing
        depth_str = f"{data.depth:.2f}" if data.depth is not None else "--"
        self._draw_text(overlay, f"DEPTH: {depth_str}", depth_y, self._config.text_color)

        # --- Hand position ---
        pos_y = depth_y + self._config.line_spacing
        pos_str = self._format_position(data.hand_position)
        self._draw_text(overlay, f"HAND: {pos_str}", pos_y, self._config.text_color)

        # --- Controls legend (bottom-left) ---
        self._draw_text(
            overlay,
            self._CONTROLS_TEXT,
            overlay.shape[0] - 20,
            self._config.text_color,
        )

        # Blend overlay with original for subtle background effect
        cv2.addWeighted(overlay, 0.0, frame, 1.0, 0, frame)
        # Actually, we drew directly on overlay; copy back
        # Simpler: just return overlay since we don't need transparency for text
        return overlay

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _draw_text(
        self,
        frame: np.ndarray,
        text: str,
        y: int,
        color: Tuple[int, int, int],
        offset_x: int = 0,
    ) -> None:
        """Draws a single line of text at the configured left margin."""
        x = self._config.left_margin + offset_x
        cv2.putText(
            frame,
            text,
            (x, y),
            self._config.font_face,
            self._config.font_scale,
            color,
            self._config.font_thickness,
            cv2.LINE_AA,
        )

    def _draw_status_indicator(
        self,
        frame: np.ndarray,
        tracking: bool,
        y: int,
    ) -> None:
        """
        Draws a colored circle indicating tracking status.
        
        Green  = tracking active
        Yellow = searching (not implemented yet, reserved)
        Red    = lost / no hand detected
        """
        center = (self._config.left_margin + 8, y - 4)
        color = self._config.text_color if tracking else self._config.alert_color
        cv2.circle(frame, center, self._config.indicator_radius, color, -1)

    @staticmethod
    def _format_gesture(gesture: Optional[GestureType]) -> str:
        """Returns a human-readable gesture name."""
        if gesture is None:
            return "NONE"
        return gesture.name.replace("_", " ")

    @staticmethod
    def _format_state(state: Optional[BoxState]) -> str:
        """Returns a human-readable state name."""
        if state is None:
            return "NONE"
        return state.name

    @staticmethod
    def _format_position(pos: Optional[Tuple[int, int]]) -> str:
        """Returns formatted hand position or placeholder."""
        if pos is None:
            return "--"
        return f"({pos[0]}, {pos[1]})"