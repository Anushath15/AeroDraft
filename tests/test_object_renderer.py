"""
Unit tests for the ObjectRenderer and individual object renderers.
"""
from __future__ import annotations

import numpy as np
import pytest

from config import RenderConfig
from engine.object_renderer import (
    ObjectRenderer,
    CubeRenderer,
    SwitchboardRenderer,
    CeilingLightRenderer,
    SocketRenderer,
    JunctionBoxRenderer,
    ConduitBoxRenderer,
)


@pytest.fixture
def sample_box_points() -> np.ndarray:
    """A plausible 8-point box roughly centered in a 640x480 frame."""
    return np.array([
        [270, 190], [370, 190], [370, 290], [270, 290],
        [280, 200], [360, 200], [360, 280], [280, 280],
    ], dtype=np.float64)


@pytest.fixture
def render_config() -> RenderConfig:
    return RenderConfig(
        box_color_bgr=(0, 255, 255),
        line_thickness=2,
        default_object="cube",
    )


# ------------------------------------------------------------------ #
# ObjectRenderer factory
# ------------------------------------------------------------------ #


def test_object_renderer_selects_cube_by_default(render_config: RenderConfig) -> None:
    """Default configuration should select cube renderer."""
    renderer = ObjectRenderer(render_config)
    assert renderer.object_type == "cube"


def test_object_renderer_selects_switchboard(render_config: RenderConfig) -> None:
    """Switchboard configuration should select switchboard renderer."""
    cfg = RenderConfig(
        box_color_bgr=(0, 255, 255),
        line_thickness=2,
        default_object="switchboard",
    )
    renderer = ObjectRenderer(cfg)
    assert renderer.object_type == "switchboard"


def test_object_renderer_selects_ceiling_light(render_config: RenderConfig) -> None:
    """Ceiling light configuration should select ceiling light renderer."""
    cfg = RenderConfig(
        box_color_bgr=(0, 255, 255),
        line_thickness=2,
        default_object="ceiling_light",
    )
    renderer = ObjectRenderer(cfg)
    assert renderer.object_type == "ceiling_light"


def test_object_renderer_selects_socket(render_config: RenderConfig) -> None:
    """Socket configuration should select socket renderer."""
    cfg = RenderConfig(
        box_color_bgr=(0, 255, 255),
        line_thickness=2,
        default_object="socket",
    )
    renderer = ObjectRenderer(cfg)
    assert renderer.object_type == "socket"


def test_object_renderer_selects_junction_box(render_config: RenderConfig) -> None:
    """Junction box configuration should select junction box renderer."""
    cfg = RenderConfig(
        box_color_bgr=(0, 255, 255),
        line_thickness=2,
        default_object="junction_box",
    )
    renderer = ObjectRenderer(cfg)
    assert renderer.object_type == "junction_box"


def test_object_renderer_selects_conduit_box(render_config: RenderConfig) -> None:
    """Conduit box configuration should select conduit box renderer."""
    cfg = RenderConfig(
        box_color_bgr=(0, 255, 255),
        line_thickness=2,
        default_object="conduit_box",
    )
    renderer = ObjectRenderer(cfg)
    assert renderer.object_type == "conduit_box"


def test_object_renderer_fallback_on_unknown_type(render_config: RenderConfig) -> None:
    """Unknown object type should fall back to cube with a warning."""
    cfg = RenderConfig(
        box_color_bgr=(0, 255, 255),
        line_thickness=2,
        default_object="nonexistent_object",
    )
    renderer = ObjectRenderer(cfg)
    assert renderer.object_type == "cube"


# ------------------------------------------------------------------ #
# Drawing interface
# ------------------------------------------------------------------ #


def test_cube_renderer_draws_on_frame(sample_box_points: np.ndarray) -> None:
    """CubeRenderer should modify the frame."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    renderer = CubeRenderer(color=(0, 255, 255), thickness=2)
    result = renderer.draw(frame, sample_box_points)
    assert np.any(result != 0)


def test_switchboard_renderer_draws_on_frame(sample_box_points: np.ndarray) -> None:
    """SwitchboardRenderer should modify the frame."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    renderer = SwitchboardRenderer(color=(0, 255, 255), thickness=2)
    result = renderer.draw(frame, sample_box_points)
    assert np.any(result != 0)


def test_ceiling_light_renderer_draws_on_frame(sample_box_points: np.ndarray) -> None:
    """CeilingLightRenderer should modify the frame."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    renderer = CeilingLightRenderer(color=(0, 255, 255), thickness=2)
    result = renderer.draw(frame, sample_box_points)
    assert np.any(result != 0)


def test_socket_renderer_draws_on_frame(sample_box_points: np.ndarray) -> None:
    """SocketRenderer should modify the frame."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    renderer = SocketRenderer(color=(0, 255, 255), thickness=2)
    result = renderer.draw(frame, sample_box_points)
    assert np.any(result != 0)


def test_junction_box_renderer_draws_on_frame(sample_box_points: np.ndarray) -> None:
    """JunctionBoxRenderer should modify the frame."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    renderer = JunctionBoxRenderer(color=(0, 255, 255), thickness=2)
    result = renderer.draw(frame, sample_box_points)
    assert np.any(result != 0)


def test_conduit_box_renderer_draws_on_frame(sample_box_points: np.ndarray) -> None:
    """ConduitBoxRenderer should modify the frame."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    renderer = ConduitBoxRenderer(color=(0, 255, 255), thickness=2)
    result = renderer.draw(frame, sample_box_points)
    assert np.any(result != 0)


def test_object_renderer_draws_selected_object(sample_box_points: np.ndarray, render_config: RenderConfig) -> None:
    """ObjectRenderer should delegate drawing to the selected renderer."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    renderer = ObjectRenderer(render_config)
    result = renderer.draw(frame, sample_box_points)
    assert np.any(result != 0)


# ------------------------------------------------------------------ #
# Edge cases
# ------------------------------------------------------------------ #


def test_invalid_thickness_raises() -> None:
    """Thickness must be strictly positive."""
    with pytest.raises(ValueError):
        CubeRenderer(color=(255, 255, 255), thickness=0)


def test_invalid_vertices_shape_raises() -> None:
    """Wrong vertex count must raise ValueError."""
    renderer = CubeRenderer(color=(0, 255, 255), thickness=2)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bad_points = np.array([[100, 100], [200, 200]], dtype=np.float64)
    with pytest.raises(ValueError):
        renderer.draw(frame, bad_points)


def test_all_renderers_return_same_frame_object(sample_box_points: np.ndarray) -> None:
    """All renderers should modify in place and return the same array."""
    renderers = [
        CubeRenderer((0, 255, 255), 2),
        SwitchboardRenderer((0, 255, 255), 2),
        CeilingLightRenderer((0, 255, 255), 2),
        SocketRenderer((0, 255, 255), 2),
        JunctionBoxRenderer((0, 255, 255), 2),
        ConduitBoxRenderer((0, 255, 255), 2),
    ]
    for r in renderers:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = r.draw(frame, sample_box_points)
        assert result is frame


def test_small_face_skips_detail(sample_box_points: np.ndarray) -> None:
    """Renderers should not crash when face is too small for detail."""
    # Very small projected points
    tiny_points = np.array([
        [10, 10], [12, 10], [12, 12], [10, 12],
        [11, 11], [13, 11], [13, 13], [11, 13],
    ], dtype=np.float64)
    
    renderers = [
        SwitchboardRenderer((0, 255, 255), 2),
        CeilingLightRenderer((0, 255, 255), 2),
        SocketRenderer((0, 255, 255), 2),
        JunctionBoxRenderer((0, 255, 255), 2),
        ConduitBoxRenderer((0, 255, 255), 2),
    ]
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for r in renderers:
        result = r.draw(frame.copy(), tiny_points)
        assert result is not None  # Should not crash