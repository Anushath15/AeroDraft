"""Unit tests for aerodraft.core.camera.CameraCapture.

All tests mock ``cv2.VideoCapture`` so they run deterministically without
requiring a physical webcam.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from aerodraft.config.settings import CameraSettings
from aerodraft.core.camera import CameraCapture
from aerodraft.core.exceptions import CameraInitializationError, CameraReadError


@pytest.fixture
def camera_settings() -> CameraSettings:
    """Returns a deterministic CameraSettings instance for tests."""
    return CameraSettings(
        device_index=0, frame_width=640, frame_height=480, capture_fps=30
    )


@patch("aerodraft.core.camera.cv2.VideoCapture")
def test_open_success(
    mock_video_capture: MagicMock, camera_settings: CameraSettings
) -> None:
    """open() should succeed when the device reports itself as opened."""
    mock_capture_instance = MagicMock()
    mock_capture_instance.isOpened.return_value = True
    mock_video_capture.return_value = mock_capture_instance

    camera = CameraCapture(camera_settings)
    camera.open()

    assert camera.is_open() is True
    mock_capture_instance.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 640)
    mock_capture_instance.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 480)


@patch("aerodraft.core.camera.cv2.VideoCapture")
def test_open_failure_raises(
    mock_video_capture: MagicMock, camera_settings: CameraSettings
) -> None:
    """open() should raise CameraInitializationError when isOpened() is False."""
    mock_capture_instance = MagicMock()
    mock_capture_instance.isOpened.return_value = False
    mock_video_capture.return_value = mock_capture_instance

    camera = CameraCapture(camera_settings)

    with pytest.raises(CameraInitializationError):
        camera.open()


def test_read_frame_without_open_raises(camera_settings: CameraSettings) -> None:
    """read_frame() should raise CameraReadError if open() was never called."""
    camera = CameraCapture(camera_settings)

    with pytest.raises(CameraReadError):
        camera.read_frame()


@patch("aerodraft.core.camera.cv2.VideoCapture")
def test_read_frame_success(
    mock_video_capture: MagicMock, camera_settings: CameraSettings
) -> None:
    """read_frame() should return the frame provided by the underlying device."""
    expected_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_capture_instance = MagicMock()
    mock_capture_instance.isOpened.return_value = True
    mock_capture_instance.read.return_value = (True, expected_frame)
    mock_video_capture.return_value = mock_capture_instance

    camera = CameraCapture(camera_settings)
    camera.open()
    frame = camera.read_frame()

    assert frame is expected_frame


@patch("aerodraft.core.camera.cv2.VideoCapture")
def test_read_frame_failure_raises(
    mock_video_capture: MagicMock, camera_settings: CameraSettings
) -> None:
    """read_frame() should raise CameraReadError if the device fails to read."""
    mock_capture_instance = MagicMock()
    mock_capture_instance.isOpened.return_value = True
    mock_capture_instance.read.return_value = (False, None)
    mock_video_capture.return_value = mock_capture_instance

    camera = CameraCapture(camera_settings)
    camera.open()

    with pytest.raises(CameraReadError):
        camera.read_frame()


@patch("aerodraft.core.camera.cv2.VideoCapture")
def test_context_manager_releases_on_exit(
    mock_video_capture: MagicMock, camera_settings: CameraSettings
) -> None:
    """Using CameraCapture as a context manager should release the device."""
    mock_capture_instance = MagicMock()
    mock_capture_instance.isOpened.return_value = True
    mock_video_capture.return_value = mock_capture_instance

    with CameraCapture(camera_settings) as camera:
        assert camera.is_open() is True

    mock_capture_instance.release.assert_called_once()


@patch("aerodraft.core.camera.cv2.VideoCapture")
def test_release_is_safe_when_never_opened(
    mock_video_capture: MagicMock, camera_settings: CameraSettings
) -> None:
    """release() should be a no-op (not raise) if the camera was never opened."""
    camera = CameraCapture(camera_settings)
    camera.release()  # Should not raise.
    assert camera.is_open() is False