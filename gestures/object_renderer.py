"""
Object rendering engine for AeroDraft.
Routes object types to specialized wireframe renderers.
Supports state-based colors and selection highlighting.
"""
from typing import Optional, Tuple

import cv2
import numpy as np

from config import settings


class CubeRenderer:
    def draw(self, frame: np.ndarray, projected_points: np.ndarray, thickness: int = 2, color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        if projected_points is None or projected_points.shape[0] < 8:
            raise ValueError("projected_points must have shape (8, 2)")
        color = color or settings.render.box_color_bgr
        pts = projected_points.astype(np.int32)
        cv2.line(frame, tuple(pts[0]), tuple(pts[1]), color, thickness)
        cv2.line(frame, tuple(pts[1]), tuple(pts[2]), color, thickness)
        cv2.line(frame, tuple(pts[2]), tuple(pts[3]), color, thickness)
        cv2.line(frame, tuple(pts[3]), tuple(pts[0]), color, thickness)
        cv2.line(frame, tuple(pts[4]), tuple(pts[5]), color, thickness)
        cv2.line(frame, tuple(pts[5]), tuple(pts[6]), color, thickness)
        cv2.line(frame, tuple(pts[6]), tuple(pts[7]), color, thickness)
        cv2.line(frame, tuple(pts[7]), tuple(pts[4]), color, thickness)
        cv2.line(frame, tuple(pts[0]), tuple(pts[4]), color, thickness)
        cv2.line(frame, tuple(pts[1]), tuple(pts[5]), color, thickness)
        cv2.line(frame, tuple(pts[2]), tuple(pts[6]), color, thickness)
        cv2.line(frame, tuple(pts[3]), tuple(pts[7]), color, thickness)
        return frame


class SwitchboardRenderer:
    def draw(self, frame: np.ndarray, projected_points: np.ndarray, thickness: int = 2, color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        if projected_points is None or projected_points.shape[0] < 8:
            raise ValueError("projected_points must have shape (8, 2)")
        color = color or settings.render.box_color_bgr
        pts = projected_points.astype(np.int32)
        CubeRenderer().draw(frame, projected_points, thickness, color)
        door_pts = np.array([
            pts[4] + (pts[5] - pts[4]) // 4,
            pts[5] - (pts[5] - pts[4]) // 4,
            pts[6] - (pts[6] - pts[5]) // 4,
            pts[7] + (pts[7] - pts[6]) // 4,
        ], dtype=np.int32)
        for i in range(4):
            cv2.line(frame, tuple(door_pts[i]), tuple(door_pts[(i + 1) % 4]), color, max(1, thickness - 1))
        return frame


class CeilingLightRenderer:
    def draw(self, frame: np.ndarray, projected_points: np.ndarray, thickness: int = 2, color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        if projected_points is None or projected_points.shape[0] < 8:
            raise ValueError("projected_points must have shape (8, 2)")
        color = color or settings.render.box_color_bgr
        pts = projected_points.astype(np.int32)
        cv2.line(frame, tuple(pts[0]), tuple(pts[1]), color, thickness)
        cv2.line(frame, tuple(pts[1]), tuple(pts[2]), color, thickness)
        cv2.line(frame, tuple(pts[2]), tuple(pts[3]), color, thickness)
        cv2.line(frame, tuple(pts[3]), tuple(pts[0]), color, thickness)
        cv2.line(frame, tuple(pts[4]), tuple(pts[5]), color, thickness)
        cv2.line(frame, tuple(pts[5]), tuple(pts[6]), color, thickness)
        cv2.line(frame, tuple(pts[6]), tuple(pts[7]), color, thickness)
        cv2.line(frame, tuple(pts[7]), tuple(pts[4]), color, thickness)
        for i in range(4):
            cv2.line(frame, tuple(pts[i]), tuple(pts[i + 4]), color, max(1, thickness - 1))
        cv2.line(frame, tuple(pts[4]), tuple(pts[6]), color, max(1, thickness - 1))
        cv2.line(frame, tuple(pts[5]), tuple(pts[7]), color, max(1, thickness - 1))
        return frame


class SocketRenderer:
    def draw(self, frame: np.ndarray, projected_points: np.ndarray, thickness: int = 2, color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        if projected_points is None or projected_points.shape[0] < 8:
            raise ValueError("projected_points must have shape (8, 2)")
        color = color or settings.render.box_color_bgr
        pts = projected_points.astype(np.int32)
        CubeRenderer().draw(frame, projected_points, thickness, color)
        mid_top = (pts[4] + pts[5]) // 2
        mid_bottom = (pts[7] + pts[6]) // 2
        cv2.line(frame, tuple(mid_top), tuple(mid_bottom), color, max(1, thickness - 1))
        return frame


class JunctionBoxRenderer:
    def draw(self, frame: np.ndarray, projected_points: np.ndarray, thickness: int = 2, color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        if projected_points is None or projected_points.shape[0] < 8:
            raise ValueError("projected_points must have shape (8, 2)")
        color = color or settings.render.box_color_bgr
        pts = projected_points.astype(np.int32)
        CubeRenderer().draw(frame, projected_points, thickness, color)
        seam_y = (pts[4][1] + pts[7][1]) // 2
        seam_x1 = (pts[4][0] + pts[7][0]) // 2
        seam_x2 = (pts[5][0] + pts[6][0]) // 2
        cv2.line(frame, (seam_x1, seam_y), (seam_x2, seam_y), color, max(1, thickness - 1))
        return frame


class ConduitBoxRenderer:
    def draw(self, frame: np.ndarray, projected_points: np.ndarray, thickness: int = 2, color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        if projected_points is None or projected_points.shape[0] < 8:
            raise ValueError("projected_points must have shape (8, 2)")
        color = color or settings.render.box_color_bgr
        pts = projected_points.astype(np.int32)
        CubeRenderer().draw(frame, projected_points, thickness, color)
        for a, b in [(0, 4), (1, 5), (2, 6), (3, 7)]:
            mid = (pts[a] + pts[b]) // 2
            cv2.circle(frame, tuple(mid), 2, color, -1)
        return frame


class DistributionBoardRenderer:
    def draw(self, frame: np.ndarray, projected_points: np.ndarray, thickness: int = 2, color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        if projected_points is None or projected_points.shape[0] < 8:
            raise ValueError("projected_points must have shape (8, 2)")
        color = color or settings.render.box_color_bgr
        pts = projected_points.astype(np.int32)
        CubeRenderer().draw(frame, projected_points, thickness, color)
        front_tl, front_tr = pts[4], pts[5]
        front_bl, front_br = pts[7], pts[6]
        for i in range(1, 4):
            t = i / 4
            y = int(front_tl[1] * (1 - t) + front_bl[1] * t)
            x1 = int(front_tl[0] * (1 - t) + front_bl[0] * t)
            x2 = int(front_tr[0] * (1 - t) + front_br[0] * t)
            cv2.line(frame, (x1, y), (x2, y), color, max(1, thickness - 1))
        return frame


class ObjectRenderer:
    def __init__(self, default_object: str = "cube") -> None:
        self._renderers = {
            "cube": CubeRenderer(),
            "switchboard": SwitchboardRenderer(),
            "ceiling_light": CeilingLightRenderer(),
            "socket": SocketRenderer(),
            "junction_box": JunctionBoxRenderer(),
            "conduit_box": ConduitBoxRenderer(),
            "distribution_board": DistributionBoardRenderer(),
        }
        self.default_object = default_object

    def render(self, frame: np.ndarray, object_type: Optional[str], projected_points: np.ndarray, thickness: Optional[int] = None, color: Optional[Tuple[int, int, int]] = None, highlight: bool = False) -> np.ndarray:
        if thickness is not None and thickness < 1:
            raise ValueError("thickness must be >= 1")
        if projected_points is None or len(projected_points.shape) != 2 or projected_points.shape[1] != 2:
            raise ValueError("projected_points must be a 2D array with shape (N, 2)")
        thickness = thickness or settings.render.line_thickness
        color = color or settings.render.box_color_bgr
        renderer = self._renderers.get(object_type, self._renderers[self.default_object])
        renderer.draw(frame, projected_points, thickness, color)
        if highlight:
            frame = self._draw_highlight(frame, projected_points, color)
        return frame

    @staticmethod
    def _draw_highlight(frame: np.ndarray, projected_points: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
        pts = projected_points.astype(np.int32)
        xs = pts[:, 0]
        ys = pts[:, 1]
        x1, y1 = int(xs.min()) - 5, int(ys.min()) - 5
        x2, y2 = int(xs.max()) + 5, int(ys.max()) + 5
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        highlight_color = settings.render.highlight_color
        cv2.rectangle(frame, (x1, y1), (x2, y2), highlight_color, settings.render.selection_thickness)
        return frame