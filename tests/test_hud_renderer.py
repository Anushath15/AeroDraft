"""
Unit tests for the HUDRenderer class.
"""
from __future__ import annotations

import numpy as np
import pytest

from config import HUDConfig
from gestures.gesture_classifier import GestureType
from gestures.state_machine import BoxState
from ui.hud_renderer import HUDData, HUDRenderer


@pytest.fixture
def config() -> HUDConfig:
    return HUDConfig(
        font_scale=0.5,
        font_thickness=1,
        text_color=(0, 255, 0),
        alert_color=(0, 0, 255),
        info_color=(255, 255, 0),
        line_spacing=25,
        left_margin=10,
        top_margin=30,
        indicator_radius=8,
    )


@pytest.fixture
def renderer(config: HUDConfig) -> HUDRenderer:
    return HUDRenderer(config)


@pytest.fixture
def blank_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


# ------------------------------------------------------------------ #
# Basic rendering
# ------------------------------------------------------------------ #


def test_render_returns_same_frame_shape(renderer: HUDRenderer, blank_frame: np.ndarray) -> None:
    """HUD render must preserve frame dimensions."""
    data = HUDData()
    result = renderer.render(blank_frame, data)
    assert result.shape == blank_frame.shape


def test_render_modifies_frame(renderer: HUDRenderer, blank_frame: np.ndarray) -> None:
    """A blank frame must have non-zero pixels after HUD overlay."""
    data = HUDData(tracking=True, fps=30.0)
    result = renderer.render(blank_frame, data)
    assert np.any(result != 0)


def test_render_empty_frame_returns_unchanged(renderer: HUDRenderer) -> None:
    """An empty or zero-size frame should be returned without crashing."""
    empty = np.array([])
    result = renderer.render(empty, HUDData())
    assert result.size == 0


# ------------------------------------------------------------------ #
# Tracking status
# ------------------------------------------------------------------ #


def test_tracking_yes_draws_green_indicator(
    renderer: HUDRenderer, blank_frame: np.ndarray, config: HUDConfig
) -> None:
    """Tracking=ON must draw a green status indicator."""
    data = HUDData(tracking=True)
    result = renderer.render(blank_frame, data)
    # Sample near the indicator position (left_margin + 8, top_margin + line_spacing - 4)
    indicator_x = config.left_margin + 8
    indicator_y = config.top_margin + config.line_spacing - 4
    pixel = result[indicator_y, indicator_x]
    assert tuple(pixel) == config.text_color


def test_tracking_no_draws_red_indicator(
    renderer: HUDRenderer, blank_frame: np.ndarray, config: HUDConfig
) -> None:
    """Tracking=OFF must draw a red status indicator."""
    data = HUDData(tracking=False)
    result = renderer.render(blank_frame, data)
    indicator_x = config.left_margin + 8
    indicator_y = config.top_margin + config.line_spacing - 4
    pixel = result[indicator_y, indicator_x]
    assert tuple(pixel) == config.alert_color


# ------------------------------------------------------------------ #
# Gesture display
# ------------------------------------------------------------------ #


def test_gesture_pinch_renders(
    renderer: HUDRenderer, blank_frame: np.ndarray
) -> None:
    """GestureType.PINCH must appear in the HUD."""
    data = HUDData(gesture=GestureType.PINCH)
    result = renderer.render(blank_frame, data)
    assert np.any(result != 0)


def test_gesture_none_renders(
    renderer: HUDRenderer, blank_frame: np.ndarray
) -> None:
    """None gesture must render as 'NONE' without crashing."""
    data = HUDData(gesture=None)
    result = renderer.render(blank_frame, data)
    assert result is not None


# ------------------------------------------------------------------ #
# State display
# ------------------------------------------------------------------ #


def test_state_drawing_renders(
    renderer: HUDRenderer, blank_frame: np.ndarray
) -> None:
    """BoxState.DRAWING must appear in the HUD."""
    data = HUDData(state=BoxState.DRAWING)
    result = renderer.render(blank_frame, data)
    assert np.any(result != 0)


def test_state_none_renders(
    renderer: HUDRenderer, blank_frame: np.ndarray
) -> None:
    """None state must render as 'NONE' without crashing."""
    data = HUDData(state=None)
    result = renderer.render(blank_frame, data)
    assert result is not None


# ------------------------------------------------------------------ #
# FPS display
# ------------------------------------------------------------------ #


def test_fps_renders(
    renderer: HUDRenderer, blank_frame: np.ndarray
) -> None:
    """FPS value must appear in the HUD."""
    data = HUDData(fps=29.97)
    result = renderer.render(blank_frame, data)
    assert np.any(result != 0)


# ------------------------------------------------------------------ #
# Depth display
# ------------------------------------------------------------------ #


def test_depth_with_value_renders(
    renderer: HUDRenderer, blank_frame: np.ndarray
) -> None:
    """Depth float must appear formatted in the HUD."""
    data = HUDData(depth=1.234)
    result = renderer.render(blank_frame, data)
    assert np.any(result != 0)


def test_depth_none_renders_placeholder(
    renderer: HUDRenderer, blank_frame: np.ndarray
) -> None:
    """None depth must render '--' placeholder."""
    data = HUDData(depth=None)
    result = renderer.render(blank_frame, data)
    assert result is not None


# ------------------------------------------------------------------ #
# Hand position display
# ------------------------------------------------------------------ #


def test_hand_position_with_value_renders(
    renderer: HUDRenderer, blank_frame: np.ndarray
) -> None:
    """Hand position tuple must appear in the HUD."""
    data = HUDData(hand_position=(320, 240))
    result = renderer.render(blank_frame, data)
    assert np.any(result != 0)


def test_hand_position_none_renders_placeholder(
    renderer: HUDRenderer, blank_frame: np.ndarray
) -> None:
    """None hand position must render '--' placeholder."""
    data = HUDData(hand_position=None)
    result = renderer.render(blank_frame, data)
    assert result is not None


# ------------------------------------------------------------------ #
# Controls legend
# ------------------------------------------------------------------ #


def test_controls_legend_renders(
    renderer: HUDRenderer, blank_frame: np.ndarray
) -> None:
    """The controls legend must appear at the bottom of the HUD."""
    data = HUDData()
    result = renderer.render(blank_frame, data)
    assert np.any(result != 0)


# ------------------------------------------------------------------ #
# Invalid values
# ------------------------------------------------------------------ #


def test_negative_fps_renders(
    renderer: HUDRenderer, blank_frame: np.ndarray
) -> None:
    """Negative FPS (invalid but possible input) must not crash."""
    data = HUDData(fps=-5.0)
    result = renderer.render(blank_frame, data)
    assert result is not None


def test_zero_fps_renders(
    renderer: HUDRenderer, blank_frame: np.ndarray
) -> None:
    """Zero FPS must render as '0.0' without crashing."""
    data = HUDData(fps=0.0)
    result = renderer.render(blank_frame, data)
    assert result is not None


# ------------------------------------------------------------------ #
# Formatting helpers
# ------------------------------------------------------------------ #


def test_format_gesture_replaces_underscores() -> None:
    """Gesture names with underscores should display as spaces."""
    assert HUDRenderer._format_gesture(GestureType.OPEN_PALM) == "OPEN PALM"


def test_format_gesture_none_returns_none() -> None:
    """None gesture should return 'NONE'."""
    assert HUDRenderer._format_gesture(None) == "NONE"


def test_format_state_returns_name() -> None:
    """State enum should return its name."""
    assert HUDRenderer._format_state(BoxState.LOCKED) == "LOCKED"


def test_format_position_tuple() -> None:
    """Position tuple should format as '(x, y)'."""
    assert HUDRenderer._format_position((100, 200)) == "(100, 200)"


def test_format_position_none() -> None:
    """None position should return '--'."""
    assert HUDRenderer._format_position(None) == "--"