"""
Unit tests for the GestureStateMachine class.
"""
import numpy as np
import pytest
from config import StateMachineConfig
from gestures.gesture_classifier import GestureType
from gestures.state_machine import GestureStateMachine, BoxState


@pytest.fixture
def config() -> StateMachineConfig:
    return StateMachineConfig(
        lock_hold_duration_s=1.0,
        default_box_half_extents=(0.1, 0.1, 0.1),
    )


@pytest.fixture
def machine(config: StateMachineConfig) -> GestureStateMachine:
    return GestureStateMachine(config)


def test_initial_state_is_idle(machine: GestureStateMachine) -> None:
    """A freshly constructed machine must start in IDLE with no box."""
    assert machine.current_state == BoxState.IDLE
    assert machine.current_box_center is None
    assert machine.current_box_half_extents is None


def test_idle_ignores_non_pinch_gestures(machine: GestureStateMachine) -> None:
    """Gestures other than PINCH must not move the machine out of IDLE."""
    state = machine.update(GestureType.OPEN_PALM, np.array([0.0, 0.0]), depth=1.0, timestamp=0.0)
    assert state == BoxState.IDLE


def test_pinch_transitions_idle_to_drawing(machine: GestureStateMachine) -> None:
    """A PINCH gesture in IDLE must spawn the box and enter DRAWING."""
    state = machine.update(GestureType.PINCH, np.array([1.0, 2.0]), depth=5.0, timestamp=0.0)
    assert state == BoxState.DRAWING
    np.testing.assert_allclose(machine.current_box_center, [1.0, 2.0, 5.0])


def test_drawing_follows_hand_while_pinch_held(machine: GestureStateMachine) -> None:
    """While PINCH is held in DRAWING, box center must update to follow the hand."""
    machine.update(GestureType.PINCH, np.array([1.0, 2.0]), depth=5.0, timestamp=0.0)
    machine.update(GestureType.PINCH, np.array([3.0, 4.0]), depth=6.0, timestamp=0.1)
    np.testing.assert_allclose(machine.current_box_center, [3.0, 4.0, 6.0])


def test_pinch_release_transitions_drawing_to_placed(machine: GestureStateMachine) -> None:
    """Releasing PINCH (any other gesture) in DRAWING must transition to PLACED."""
    machine.update(GestureType.PINCH, np.array([1.0, 2.0]), depth=5.0, timestamp=0.0)
    state = machine.update(GestureType.OPEN_PALM, np.array([1.0, 2.0]), depth=5.0, timestamp=0.1)
    assert state == BoxState.PLACED


def test_full_happy_path_to_locked(machine: GestureStateMachine) -> None:
    """
    Full sequence: IDLE -> pinch -> DRAWING -> release -> PLACED
    -> fist held >= 1.0s -> LOCKED.
    """
    machine.update(GestureType.PINCH, np.array([1.0, 2.0]), depth=5.0, timestamp=0.0)
    machine.update(GestureType.OPEN_PALM, np.array([1.0, 2.0]), depth=5.0, timestamp=0.1)
    assert machine.current_state == BoxState.PLACED

    machine.update(GestureType.FIST, np.array([1.0, 2.0]), depth=5.0, timestamp=0.2)
    assert machine.current_state == BoxState.PLACED  # hold just started

    state = machine.update(GestureType.FIST, np.array([1.0, 2.0]), depth=5.0, timestamp=1.3)
    assert state == BoxState.LOCKED


def test_fist_held_less_than_duration_does_not_lock(machine: GestureStateMachine) -> None:
    """A fist held for less than lock_hold_duration_s must not transition to LOCKED."""
    machine.update(GestureType.PINCH, np.array([1.0, 2.0]), depth=5.0, timestamp=0.0)
    machine.update(GestureType.OPEN_PALM, np.array([1.0, 2.0]), depth=5.0, timestamp=0.1)
    machine.update(GestureType.FIST, np.array([1.0, 2.0]), depth=5.0, timestamp=0.2)
    state = machine.update(GestureType.FIST, np.array([1.0, 2.0]), depth=5.0, timestamp=0.5)
    assert state == BoxState.PLACED


def test_non_fist_gesture_resets_hold_timer(machine: GestureStateMachine) -> None:
    """An OPEN_PALM (not FIST, not NONE) during a fist hold must reset the timer."""
    machine.update(GestureType.PINCH, np.array([1.0, 2.0]), depth=5.0, timestamp=0.0)
    machine.update(GestureType.OPEN_PALM, np.array([1.0, 2.0]), depth=5.0, timestamp=0.1)
    machine.update(GestureType.FIST, np.array([1.0, 2.0]), depth=5.0, timestamp=0.2)
    machine.update(GestureType.OPEN_PALM, np.array([1.0, 2.0]), depth=5.0, timestamp=0.5)
    # Re-start fist hold at t=0.6; only 0.7s elapsed by t=1.3, must NOT lock yet
    machine.update(GestureType.FIST, np.array([1.0, 2.0]), depth=5.0, timestamp=0.6)
    state = machine.update(GestureType.FIST, np.array([1.0, 2.0]), depth=5.0, timestamp=1.3)
    assert state == BoxState.PLACED


def test_gesture_none_does_not_reset_hold_timer(machine: GestureStateMachine) -> None:
    """A momentary NONE gesture during a fist hold must NOT reset the timer (detection gap tolerance)."""
    machine.update(GestureType.PINCH, np.array([1.0, 2.0]), depth=5.0, timestamp=0.0)
    machine.update(GestureType.OPEN_PALM, np.array([1.0, 2.0]), depth=5.0, timestamp=0.1)
    machine.update(GestureType.FIST, np.array([1.0, 2.0]), depth=5.0, timestamp=0.2)
    machine.update(GestureType.NONE, np.array([1.0, 2.0]), depth=5.0, timestamp=0.5)  # momentary gap
    state = machine.update(GestureType.FIST, np.array([1.0, 2.0]), depth=5.0, timestamp=1.3)
    assert state == BoxState.LOCKED


def test_locked_state_is_terminal(machine: GestureStateMachine) -> None:
    """Once LOCKED, further gesture updates must not change the state."""
    machine.update(GestureType.PINCH, np.array([1.0, 2.0]), depth=5.0, timestamp=0.0)
    machine.update(GestureType.OPEN_PALM, np.array([1.0, 2.0]), depth=5.0, timestamp=0.1)
    machine.update(GestureType.FIST, np.array([1.0, 2.0]), depth=5.0, timestamp=0.2)
    machine.update(GestureType.FIST, np.array([1.0, 2.0]), depth=5.0, timestamp=1.3)
    assert machine.current_state == BoxState.LOCKED

    state = machine.update(GestureType.PINCH, np.array([9.0, 9.0]), depth=9.0, timestamp=2.0)
    assert state == BoxState.LOCKED


def test_reset_returns_to_idle_and_clears_box(machine: GestureStateMachine) -> None:
    """reset() must clear all state and box data, returning to IDLE."""
    machine.update(GestureType.PINCH, np.array([1.0, 2.0]), depth=5.0, timestamp=0.0)
    machine.reset()

    assert machine.current_state == BoxState.IDLE
    assert machine.current_box_center is None
    assert machine.current_box_half_extents is None