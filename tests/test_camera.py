"""
Unit tests for the VideoStream module.
All tests mock cv2.VideoCapture — no physical camera required.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from camera import VideoStream


@patch('camera.cv2.VideoCapture')
def test_open_success(mock_cap_class: MagicMock) -> None:
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = True
    mock_instance.get.return_value = 640
    mock_cap_class.return_value = mock_instance

    with VideoStream(0, 640, 480) as stream:
        assert stream.cap is not None


@patch('camera.cv2.VideoCapture')
def test_open_failure_raises(mock_cap_class: MagicMock) -> None:
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = False
    mock_cap_class.return_value = mock_instance

    with pytest.raises(RuntimeError):
        with VideoStream(0, 640, 480) as stream:
            pass


@patch('camera.cv2.VideoCapture')
def test_read_frame_success(mock_cap_class: MagicMock) -> None:
    expected_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = True
    mock_instance.get.return_value = 640
    mock_instance.read.return_value = (True, expected_frame)
    mock_cap_class.return_value = mock_instance

    with VideoStream(0, 640, 480) as stream:
        success, frame = stream.read_frame()
        assert success is True
        assert frame is not None


@patch('camera.cv2.VideoCapture')
def test_read_frame_failure_returns_false(mock_cap_class: MagicMock) -> None:
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = True
    mock_instance.get.return_value = 640
    mock_instance.read.return_value = (False, None)
    mock_cap_class.return_value = mock_instance

    with VideoStream(0, 640, 480) as stream:
        success, frame = stream.read_frame()
        assert success is False


def test_read_frame_raises_without_context() -> None:
    stream = VideoStream(0, 640, 480)
    with pytest.raises(RuntimeError):
        stream.read_frame()
