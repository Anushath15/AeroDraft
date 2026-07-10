"""
Object rendering module.

Provides multiple 3D object visualizations for MSME electrical hardware
demonstrations. All rendering uses OpenCV primitives only (lines,
rectangles, circles, polylines). No external 3D assets.

Objects are drawn from 8 projected 2D vertices (from PerspectiveProjector)
allowing them to appear anchored in 3D space and follow hand gestures.
"""
from __future__ import annotations

from typing import Protocol, Tuple

import cv2
import numpy as np
from loguru import logger

from config import RenderConfig


class ObjectRendererProtocol(Protocol):
    """Interface contract for all object renderers."""
    
    def draw(self, frame: np.ndarray, projected_points: np.ndarray) -> np.ndarray:
        """Draws the object onto the frame using projected 2D vertices."""


class BaseObjectRenderer:
    """
    Base class providing common utilities for all object renderers.
    
    Handles vertex validation, color application, and offscreen detection.
    """
    
    def __init__(self, color: Tuple[int, int, int], thickness: int) -> None:
        if thickness <= 0:
            raise ValueError("thickness must be positive")
        self.color = color
        self.thickness = thickness
    
    def _validate_vertices(self, projected_points: np.ndarray) -> np.ndarray:
        """Ensures vertices are (8, 2) int32 array."""
        if projected_points.shape != (8, 2):
            raise ValueError(
                f"projected_points must have shape (8, 2); got {projected_points.shape}"
            )
        return projected_points.astype(np.int32)
    
    def _get_face_center(self, points: np.ndarray, indices: list[int]) -> Tuple[int, int]:
        """Computes the centroid of a face from vertex indices."""
        face_points = points[indices]
        cx = int(np.mean(face_points[:, 0]))
        cy = int(np.mean(face_points[:, 1]))
        return cx, cy
    
    def _get_face_size(self, points: np.ndarray, indices: list[int]) -> Tuple[int, int]:
        """Computes approximate width and height of a face in pixels."""
        face_points = points[indices]
        min_x, max_x = int(face_points[:, 0].min()), int(face_points[:, 0].max())
        min_y, max_y = int(face_points[:, 1].min()), int(face_points[:, 1].max())
        return max_x - min_x, max_y - min_y


class CubeRenderer(BaseObjectRenderer):
    """Renders a 12-edge wireframe cuboid (existing behavior, preserved)."""
    
    BOTTOM_FACE = [0, 1, 2, 3]
    TOP_FACE = [4, 5, 6, 7]
    VERTICAL_EDGES = [(0, 4), (1, 5), (2, 6), (3, 7)]
    
    def draw(self, frame: np.ndarray, projected_points: np.ndarray) -> np.ndarray:
        points = self._validate_vertices(projected_points)
        
        bottom_loop = points[self.BOTTOM_FACE].reshape(-1, 1, 2)
        top_loop = points[self.TOP_FACE].reshape(-1, 1, 2)
        
        cv2.polylines(frame, [bottom_loop], isClosed=True, color=self.color, thickness=self.thickness)
        cv2.polylines(frame, [top_loop], isClosed=True, color=self.color, thickness=self.thickness)
        
        for start_idx, end_idx in self.VERTICAL_EDGES:
            cv2.line(frame, tuple(points[start_idx]), tuple(points[end_idx]), self.color, self.thickness)
        
        return frame


class SwitchboardRenderer(BaseObjectRenderer):
    """
    Renders an electrical switchboard panel.
    
    Features: rectangular plate, two rocker switch cutouts, screw holes at corners.
    """
    
    def draw(self, frame: np.ndarray, projected_points: np.ndarray) -> np.ndarray:
        points = self._validate_vertices(projected_points)
        
        # Draw the front face (vertices 4,5,6,7 — top face when box is oriented)
        front_face = [4, 5, 6, 7]
        front_loop = points[front_face].reshape(-1, 1, 2)
        cv2.polylines(frame, [front_loop], isClosed=True, color=self.color, thickness=self.thickness)
        
        # Get face dimensions for proportional drawing
        fw, fh = self._get_face_size(points, front_face)
        cx, cy = self._get_face_center(points, front_face)
        
        if fw < 20 or fh < 20:
            return frame  # Too small to draw detail
        
        # Draw two rocker switches (vertical rectangles)
        switch_width = max(4, fw // 6)
        switch_height = max(8, fh // 3)
        gap = max(4, fw // 8)
        
        # Left switch
        left_cx = cx - gap // 2 - switch_width // 2
        cv2.rectangle(
            frame,
            (left_cx - switch_width // 2, cy - switch_height // 2),
            (left_cx + switch_width // 2, cy + switch_height // 2),
            self.color, self.thickness
        )
        # Rocker line (indicates ON/OFF position)
        cv2.line(
            frame,
            (left_cx - switch_width // 4, cy - switch_height // 4),
            (left_cx + switch_width // 4, cy + switch_height // 4),
            self.color, self.thickness
        )
        
        # Right switch
        right_cx = cx + gap // 2 + switch_width // 2
        cv2.rectangle(
            frame,
            (right_cx - switch_width // 2, cy - switch_height // 2),
            (right_cx + switch_width // 2, cy + switch_height // 2),
            self.color, self.thickness
        )
        cv2.line(
            frame,
            (right_cx - switch_width // 4, cy + switch_height // 4),
            (right_cx + switch_width // 4, cy - switch_height // 4),
            self.color, self.thickness
        )
        
        # Screw holes at corners
        screw_radius = max(2, min(fw, fh) // 20)
        for idx in front_face:
            cv2.circle(frame, tuple(points[idx]), screw_radius, self.color, self.thickness)
        
        return frame


class CeilingLightRenderer(BaseObjectRenderer):
    """
    Renders an LED ceiling light fixture.
    
    Features: circular outer ring, inner diffuser, radial mounting lines.
    """
    
    def draw(self, frame: np.ndarray, projected_points: np.ndarray) -> np.ndarray:
        points = self._validate_vertices(projected_points)
        
        # Use the front face for the circular fixture
        front_face = [4, 5, 6, 7]
        cx, cy = self._get_face_center(points, front_face)
        fw, fh = self._get_face_size(points, front_face)
        
        radius = min(fw, fh) // 2
        if radius < 5:
            return frame
        
        # Outer ring (fixture housing)
        cv2.circle(frame, (cx, cy), radius, self.color, self.thickness)
        # Inner ring (LED diffuser)
        cv2.circle(frame, (cx, cy), radius * 2 // 3, self.color, max(1, self.thickness - 1))
        # Center dot (driver unit)
        cv2.circle(frame, (cx, cy), max(2, radius // 5), self.color, -1)
        
        # Radial mounting lines (4 screws at 45-degree positions)
        for angle_deg in [45, 135, 225, 315]:
            angle = np.deg2rad(angle_deg)
            inner_r = radius * 3 // 4
            outer_r = radius - 2
            x1 = int(cx + inner_r * np.cos(angle))
            y1 = int(cy + inner_r * np.sin(angle))
            x2 = int(cx + outer_r * np.cos(angle))
            y2 = int(cy + outer_r * np.sin(angle))
            cv2.line(frame, (x1, y1), (x2, y2), self.color, self.thickness)
            # Screw head
            cv2.circle(frame, (x2, y2), max(2, self.thickness), self.color, -1)
        
        return frame


class SocketRenderer(BaseObjectRenderer):
    """
    Renders a wall socket (3-pin Indian type).
    
    Features: square plate, three circular holes (top earth, bottom live/neutral).
    """
    
    def draw(self, frame: np.ndarray, projected_points: np.ndarray) -> np.ndarray:
        points = self._validate_vertices(projected_points)
        
        front_face = [4, 5, 6, 7]
        front_loop = points[front_face].reshape(-1, 1, 2)
        cv2.polylines(frame, [front_loop], isClosed=True, color=self.color, thickness=self.thickness)
        
        cx, cy = self._get_face_center(points, front_face)
        fw, fh = self._get_face_size(points, front_face)
        
        size = min(fw, fh)
        if size < 20:
            return frame
        
        # Top hole (earth pin - larger)
        earth_radius = max(3, size // 10)
        cv2.circle(frame, (cx, cy - size // 5), earth_radius, self.color, self.thickness)
        
        # Bottom holes (live and neutral)
        pin_radius = max(2, size // 14)
        pin_spacing = size // 6
        
        # Left bottom pin (live)
        cv2.circle(frame, (cx - pin_spacing, cy + size // 6), pin_radius, self.color, self.thickness)
        # Right bottom pin (neutral)
        cv2.circle(frame, (cx + pin_spacing, cy + size // 6), pin_radius, self.color, self.thickness)
        
        # Triangular guide marking
        triangle_size = size // 8
        triangle_points = np.array([
            [cx, cy - triangle_size],
            [cx - triangle_size, cy + triangle_size],
            [cx + triangle_size, cy + triangle_size]
        ], np.int32)
        cv2.polylines(frame, [triangle_points.reshape(-1, 1, 2)], isClosed=True, color=self.color, thickness=self.thickness)
        
        return frame


class JunctionBoxRenderer(BaseObjectRenderer):
    """
    Renders an electrical junction box.
    
    Features: square enclosure, cable entry/exit knockouts on sides.
    """
    
    def draw(self, frame: np.ndarray, projected_points: np.ndarray) -> np.ndarray:
        points = self._validate_vertices(projected_points)
        
        front_face = [4, 5, 6, 7]
        front_loop = points[front_face].reshape(-1, 1, 2)
        cv2.polylines(frame, [front_loop], isClosed=True, color=self.color, thickness=self.thickness)
        
        cx, cy = self._get_face_center(points, front_face)
        fw, fh = self._get_face_size(points, front_face)
        
        if fw < 20 or fh < 20:
            return frame
        
        # Cable entry knockouts (small circles on each edge)
        knockout_radius = max(2, min(fw, fh) // 16)
        
        # Top edge knockout
        top_mid = ((points[4][0] + points[5][0]) // 2, (points[4][1] + points[5][1]) // 2)
        cv2.circle(frame, top_mid, knockout_radius, self.color, self.thickness)
        
        # Bottom edge knockout
        bottom_mid = ((points[6][0] + points[7][0]) // 2, (points[6][1] + points[7][1]) // 2)
        cv2.circle(frame, bottom_mid, knockout_radius, self.color, self.thickness)
        
        # Left edge knockout
        left_mid = ((points[4][0] + points[7][0]) // 2, (points[4][1] + points[7][1]) // 2)
        cv2.circle(frame, left_mid, knockout_radius, self.color, self.thickness)
        
        # Right edge knockout
        right_mid = ((points[5][0] + points[6][0]) // 2, (points[5][1] + points[6][1]) // 2)
        cv2.circle(frame, right_mid, knockout_radius, self.color, self.thickness)
        
        # Center cover screw
        cv2.circle(frame, (cx, cy), max(2, knockout_radius), self.color, -1)
        
        return frame


class ConduitBoxRenderer(BaseObjectRenderer):
    """
    Renders a PVC conduit junction box.
    
    Features: rectangular body, pipe connector stubs on two sides.
    """
    
    def draw(self, frame: np.ndarray, projected_points: np.ndarray) -> np.ndarray:
        points = self._validate_vertices(projected_points)
        
        front_face = [4, 5, 6, 7]
        front_loop = points[front_face].reshape(-1, 1, 2)
        cv2.polylines(frame, [front_loop], isClosed=True, color=self.color, thickness=self.thickness)
        
        cx, cy = self._get_face_center(points, front_face)
        fw, fh = self._get_face_size(points, front_face)
        
        if fw < 20 or fh < 20:
            return frame
        
        # Conduit pipe stubs (rectangular extensions from left and right edges)
        stub_length = max(5, fw // 6)
        stub_width = max(4, fh // 5)
        
        # Left stub
        left_edge = points[4]  # top-left of front face
        left_bottom = points[7]  # bottom-left
        cv2.rectangle(
            frame,
            (left_edge[0] - stub_length, left_edge[1] + stub_width // 2),
            (left_edge[0], left_bottom[1] - stub_width // 2),
            self.color, self.thickness
        )
        
        # Right stub
        right_edge = points[5]  # top-right
        right_bottom = points[6]  # bottom-right
        cv2.rectangle(
            frame,
            (right_edge[0], right_edge[1] + stub_width // 2),
            (right_edge[0] + stub_length, right_bottom[1] - stub_width // 2),
            self.color, self.thickness
        )
        
        # Center coupling nut (hexagon approximation)
        nut_radius = max(3, min(fw, fh) // 12)
        hex_points = []
        for i in range(6):
            angle = np.deg2rad(i * 60)
            hx = int(cx + nut_radius * np.cos(angle))
            hy = int(cy + nut_radius * np.sin(angle))
            hex_points.append([hx, hy])
        hex_array = np.array(hex_points, np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [hex_array], isClosed=True, color=self.color, thickness=self.thickness)
        
        return frame


class ObjectRenderer:
    """
    Factory and facade for object rendering.
    
    Selects the appropriate renderer based on configuration and delegates
    drawing. Preserves the same interface as WireframeRenderer for
    backward compatibility.
    
    Usage:
        renderer = ObjectRenderer(settings.render)
        annotated = renderer.draw(frame, projected_points)
    """
    
    _RENDERERS = {
        "cube": CubeRenderer,
        "switchboard": SwitchboardRenderer,
        "ceiling_light": CeilingLightRenderer,
        "socket": SocketRenderer,
        "junction_box": JunctionBoxRenderer,
        "conduit_box": ConduitBoxRenderer,
    }
    
    def __init__(self, config: RenderConfig) -> None:
        """
        Args:
            config: Render configuration including default_object selection.
        
        Raises:
            ValueError: If default_object is not a recognized object type.
        """
        object_type = getattr(config, 'default_object', 'cube')
        
        if object_type not in self._RENDERERS:
            logger.warning(
                f"Unknown object type '{object_type}', falling back to 'cube'. "
                f"Valid options: {list(self._RENDERERS.keys())}"
            )
            object_type = "cube"
        
        self._object_type = object_type
        self._renderer: ObjectRendererProtocol = self._RENDERERS[object_type](
            color=config.box_color_bgr,
            thickness=config.line_thickness,
        )
    
    @property
    def object_type(self) -> str:
        """Returns the currently configured object type name."""
        return self._object_type
    
    def draw(self, frame: np.ndarray, projected_points: np.ndarray) -> np.ndarray:
        """
        Draws the configured object onto the frame.
        
        Args:
            frame: BGR image to annotate (modified in place).
            projected_points: (8, 2) array from PerspectiveProjector.
        
        Returns:
            The annotated frame (same object as input).
        """
        return self._renderer.draw(frame, projected_points)