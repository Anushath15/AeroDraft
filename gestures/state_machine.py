"""
Gesture-driven state machine module.

Owns the wireframe box's lifecycle state (position, size, stage) and
decides transitions based on the incoming gesture + hand position +
depth stream over time. This is the only module in the gesture layer
that carries memory across frames.
"""
from __future__ import annotations
from enum import Enum, auto
from typing import Optional
import numpy as np
from loguru import logger
from config import StateMachineConfig
from gestures.gesture_classifier import GestureType

class InteractionState(Enum):
    MOVING = auto()
    ANCHORED = auto()

class StateMachine:
    def __init__(self):
        self.state = InteractionState.MOVING
        self.anchor_pos = None

    def update(self, is_pinched: bool, current_pos: tuple):
        if is_pinched:
            if self.state == InteractionState.MOVING:
                self.state = InteractionState.ANCHORED
                self.anchor_pos = current_pos
            else:
                self.state = InteractionState.MOVING


class BoxState(Enum):
    """Enumerates the lifecycle stages of the wireframe box."""
    IDLE = auto()
    DRAWING = auto()
    PLACED = auto()
    LOCKED = auto()


class GestureStateMachine:
    """
    Drives box state transitions from a stream of gesture/position/depth updates.

    State transition rules:
        IDLE     --(PINCH)-->              DRAWING
        DRAWING  --(gesture != PINCH)-->    PLACED
        PLACED   --(FIST held >= hold_duration)--> LOCKED
        LOCKED   --(no automatic exit; call reset() to restart)

    Usage:
        machine = GestureStateMachine(settings.state_machine)
        state = machine.update(gesture, hand_position, depth, timestamp)
    """

    def __init__(self, config: StateMachineConfig) -> None:
        """
        Args:
            config: State machine timing and default sizing configuration.
        """
        self._config = config
        self._state = BoxState.IDLE
        self._box_center: Optional[np.ndarray] = None
        self._box_half_extents: Optional[np.ndarray] = None
        self._fist_hold_start: Optional[float] = None

    def update(
        self,
        gesture: GestureType,
        hand_position: np.ndarray,
        depth: float,
        timestamp: float,
    ) -> BoxState:
        """
        Advances the state machine by one frame's worth of gesture data.

        Args:
            gesture: The classified gesture for this frame.
            hand_position: 2D array [x, y] — hand position in virtual
                projection space (not screen pixels).
            depth: Pseudo-depth value from DepthEstimator for this frame.
            timestamp: Monotonic timestamp in seconds.

        Returns:
            The resulting BoxState after processing this update.
        """
        if self._state == BoxState.IDLE:
            self._handle_idle(gesture, hand_position, depth)

        elif self._state == BoxState.DRAWING:
            self._handle_drawing(gesture, hand_position, depth)

        elif self._state == BoxState.PLACED:
            self._handle_placed(gesture, timestamp)

        elif self._state == BoxState.LOCKED:
            pass  # Terminal state — no automatic transitions out.

        return self._state

    def _handle_idle(
        self, gesture: GestureType, hand_position: np.ndarray, depth: float
    ) -> None:
        """IDLE -> DRAWING on pinch detection; spawns the box at hand position."""
        if gesture == GestureType.PINCH:
            self._box_center = np.array([hand_position[0], hand_position[1], depth])
            self._box_half_extents = np.array(self._config.default_box_half_extents)
            self._state = BoxState.DRAWING
            logger.debug("State transition: IDLE -> DRAWING")

    def _handle_drawing(
        self, gesture: GestureType, hand_position: np.ndarray, depth: float
    ) -> None:
        """
        While pinch is held, box follows hand position/depth.
        On pinch release (any other gesture), box freezes and moves to PLACED.
        """
        if gesture == GestureType.PINCH:
            self._box_center = np.array([hand_position[0], hand_position[1], depth])
        else:
            self._state = BoxState.PLACED
            logger.debug("State transition: DRAWING -> PLACED")

    def _handle_placed(self, gesture: GestureType, timestamp: float) -> None:
        """
        PLACED -> LOCKED after a fist gesture is held continuously for
        at least lock_hold_duration_s seconds. A non-fist gesture resets
        the hold timer without leaving PLACED.
        """
        if gesture == GestureType.FIST:
            if self._fist_hold_start is None:
                self._fist_hold_start = timestamp
            elif (timestamp - self._fist_hold_start) >= self._config.lock_hold_duration_s:
                self._state = BoxState.LOCKED
                logger.debug("State transition: PLACED -> LOCKED")
        elif gesture != GestureType.NONE:
            # A gesture other than FIST or NONE (e.g. re-pinching, open palm)
            # cancels an in-progress hold. GestureType.NONE is treated as a
            # momentary detection gap and does not reset the hold timer.
            self._fist_hold_start = None

    @property
    def current_state(self) -> BoxState:
        """The current lifecycle state of the box."""
        return self._state

    @property
    def current_box_center(self) -> Optional[np.ndarray]:
        """The box's current 3D center, or None if no box exists yet (IDLE)."""
        return self._box_center

    @property
    def current_box_half_extents(self) -> Optional[np.ndarray]:
        """The box's current half-extents, or None if no box exists yet (IDLE)."""
        return self._box_half_extents

    def reset(self) -> None:
        """Resets the state machine to IDLE with no box, clearing all history."""
        self._state = BoxState.IDLE
        self._box_center = None
        self._box_half_extents = None
        self._fist_hold_start = None
        logger.debug("State machine reset to IDLE.")