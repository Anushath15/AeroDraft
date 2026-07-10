"""
Video I/O module.
Handles all hardware camera interaction via OpenCV.
Uses the context manager protocol to guarantee safe resource release.
"""
import cv2
import numpy as np
from loguru import logger
from typing import Optional, Tuple


class VideoStream:
    """
    Manages the OpenCV VideoCapture stream.

    Usage:
        with VideoStream(0, 640, 480) as stream:
            success, frame = stream.read_frame()
    """

    def __init__(self, device_index: int, width: int, height: int) -> None:
        self.device_index = device_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None

    def __enter__(self) -> "VideoStream":
        """Opens and configures the camera hardware."""
        logger.info(f"Initializing camera at index {self.device_index}...")
        self.cap = cv2.VideoCapture(self.device_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {self.device_index}.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera ready — resolution: {actual_w}x{actual_h}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Releases camera hardware unconditionally on exit."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info("Camera released.")

    def read_frame(self) -> Tuple[bool, np.ndarray]:
        """
        Reads one frame from the camera.

        Returns:
            Tuple[bool, np.ndarray]: (success flag, BGR frame array)

        Raises:
            RuntimeError: If called before the context manager is entered.
        """
        if self.cap is None:
            raise RuntimeError("Camera not initialized. Use a 'with' block.")

        success, frame = self.cap.read()

        if not success or frame is None:
            return False, np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Mirror horizontally for natural interaction
        frame = cv2.flip(frame, 1)
        return True, frame