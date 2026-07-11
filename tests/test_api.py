"""
Unit tests for the AeroDraft REST API.

Uses FastAPI's TestClient — no real HTTP server or webcam needed.
These tests run alongside the existing 132 tests with 'pytest'.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.server import app, VALID_PRODUCTS
from api.shared_state import SharedState, command_queue


@pytest.fixture(autouse=True)
def clear_command_queue():
    """Drain the command queue before each test for isolation."""
    while not command_queue.empty():
        command_queue.get_nowait()
    yield
    while not command_queue.empty():
        command_queue.get_nowait()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ─── Health check ─────────────────────────────────────────────────────────────

class TestRoot:
    def test_returns_200(self, client: TestClient) -> None:
        r = client.get("/")
        assert r.status_code == 200

    def test_returns_running_status(self, client: TestClient) -> None:
        data = r = client.get("/")
        assert r.json()["status"] == "running"

    def test_returns_version(self, client: TestClient) -> None:
        assert client.get("/").json()["version"] == "1.0.0"

    def test_returns_uptime(self, client: TestClient) -> None:
        assert "uptime_seconds" in client.get("/").json()


# ─── Status endpoint ──────────────────────────────────────────────────────────

class TestStatus:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/status").status_code == 200

    def test_has_all_required_fields(self, client: TestClient) -> None:
        data = client.get("/status").json()
        required = {
            "tracking", "gesture", "box_state", "object_type",
            "category", "fps", "depth", "hand_position",
            "demo_mode", "notification", "frame_count",
        }
        assert required.issubset(data.keys())

    def test_default_box_state_is_idle(self, client: TestClient) -> None:
        assert client.get("/status").json()["box_state"] == "IDLE"

    def test_reflects_shared_state_update(self, client: TestClient) -> None:
        from api.shared_state import shared_state
        shared_state.update(tracking=True, fps=29.5, object_type="switchboard")
        data = client.get("/status").json()
        assert data["tracking"] is True
        assert data["object_type"] == "switchboard"

    def test_fps_is_rounded(self, client: TestClient) -> None:
        from api.shared_state import shared_state
        shared_state.update(fps=29.999)
        fps = client.get("/status").json()["fps"]
        assert fps == round(fps, 1)


# ─── Products endpoint ────────────────────────────────────────────────────────

class TestProducts:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/products").status_code == 200

    def test_returns_seven_products(self, client: TestClient) -> None:
        data = client.get("/products").json()
        assert data["total"] == 7
        assert len(data["products"]) == 7

    def test_all_products_have_required_fields(self, client: TestClient) -> None:
        products = client.get("/products").json()["products"]
        for p in products:
            assert "key" in p
            assert "display_name" in p
            assert "category" in p
            assert "dimensions_cm" in p

    def test_distribution_board_present(self, client: TestClient) -> None:
        keys = [p["key"] for p in client.get("/products").json()["products"]]
        assert "distribution_board" in keys

    def test_all_keys_match_valid_products_set(self, client: TestClient) -> None:
        keys = {p["key"] for p in client.get("/products").json()["products"]}
        assert keys == VALID_PRODUCTS


# ─── POST /product ────────────────────────────────────────────────────────────

class TestSwitchProduct:
    def test_valid_product_returns_200(self, client: TestClient) -> None:
        r = client.post("/product", json={"product": "switchboard"})
        assert r.status_code == 200

    def test_valid_product_queues_command(self, client: TestClient) -> None:
        client.post("/product", json={"product": "socket"})
        cmd = command_queue.get_nowait()
        assert cmd["type"] == "switch_product"
        assert cmd["value"] == "socket"

    def test_response_includes_product_name(self, client: TestClient) -> None:
        r = client.post("/product", json={"product": "ceiling_light"})
        assert r.json()["success"] is True
        assert "ceiling_light" in r.json()["message"]

    def test_unknown_product_returns_422(self, client: TestClient) -> None:
        r = client.post("/product", json={"product": "toaster"})
        assert r.status_code == 422

    def test_missing_body_returns_422(self, client: TestClient) -> None:
        r = client.post("/product", json={})
        assert r.status_code == 422

    def test_all_valid_products_accepted(self, client: TestClient) -> None:
        for key in VALID_PRODUCTS:
            r = client.post("/product", json={"product": key})
            assert r.status_code == 200, f"Failed for product: {key}"

    def test_command_queue_has_correct_structure(self, client: TestClient) -> None:
        client.post("/product", json={"product": "distribution_board"})
        cmd = command_queue.get_nowait()
        assert set(cmd.keys()) == {"type", "value"}


# ─── POST /demo ───────────────────────────────────────────────────────────────

class TestToggleDemo:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.post("/demo").status_code == 200

    def test_queues_toggle_demo_command(self, client: TestClient) -> None:
        client.post("/demo")
        cmd = command_queue.get_nowait()
        assert cmd["type"] == "toggle_demo"

    def test_success_true_in_response(self, client: TestClient) -> None:
        assert client.post("/demo").json()["success"] is True

    def test_double_toggle_queues_two_commands(self, client: TestClient) -> None:
        client.post("/demo")
        client.post("/demo")
        cmds = [command_queue.get_nowait(), command_queue.get_nowait()]
        assert all(c["type"] == "toggle_demo" for c in cmds)


# ─── POST /screenshot ─────────────────────────────────────────────────────────

class TestScreenshot:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.post("/screenshot").status_code == 200

    def test_queues_screenshot_command(self, client: TestClient) -> None:
        client.post("/screenshot")
        cmd = command_queue.get_nowait()
        assert cmd["type"] == "screenshot"

    def test_success_true_in_response(self, client: TestClient) -> None:
        assert client.post("/screenshot").json()["success"] is True


# ─── POST /reset ──────────────────────────────────────────────────────────────

class TestReset:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.post("/reset").status_code == 200

    def test_queues_reset_command(self, client: TestClient) -> None:
        client.post("/reset")
        cmd = command_queue.get_nowait()
        assert cmd["type"] == "reset"

    def test_success_true_in_response(self, client: TestClient) -> None:
        assert client.post("/reset").json()["success"] is True


# ─── GET /benchmark ───────────────────────────────────────────────────────────

class TestBenchmark:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/benchmark").status_code == 200

    def test_has_benchmark_enabled_field(self, client: TestClient) -> None:
        assert "benchmark_enabled" in client.get("/benchmark").json()

    def test_has_modules_field(self, client: TestClient) -> None:
        assert "modules" in client.get("/benchmark").json()


# ─── GET /gestures ────────────────────────────────────────────────────────────

class TestGestures:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/gestures").status_code == 200

    def test_returns_three_gestures(self, client: TestClient) -> None:
        data = client.get("/gestures").json()
        assert len(data["gestures"]) == 3

    def test_pinch_gesture_present(self, client: TestClient) -> None:
        names = [g["name"] for g in client.get("/gestures").json()["gestures"]]
        assert "PINCH" in names

    def test_keyboard_shortcuts_present(self, client: TestClient) -> None:
        assert "keyboard_shortcuts" in client.get("/gestures").json()


# ─── SharedState unit tests ───────────────────────────────────────────────────

class TestSharedState:
    def test_initial_snapshot_has_defaults(self) -> None:
        s = SharedState()
        snap = s.snapshot()
        assert snap["tracking"] is False
        assert snap["box_state"] == "IDLE"
        assert snap["fps"] == 0.0

    def test_update_changes_fields(self) -> None:
        s = SharedState()
        s.update(tracking=True, fps=28.5, object_type="socket")
        snap = s.snapshot()
        assert snap["tracking"] is True
        assert snap["fps"] == 28.5
        assert snap["object_type"] == "socket"

    def test_frame_count_increments_on_update(self) -> None:
        s = SharedState()
        s.update(fps=30.0)
        s.update(fps=30.0)
        assert s.snapshot()["frame_count"] == 2

    def test_unknown_key_is_ignored_safely(self) -> None:
        s = SharedState()
        s.update(nonexistent_field="should_not_crash")
        assert s.snapshot()["tracking"] is False

    def test_depth_none_stays_none_in_snapshot(self) -> None:
        s = SharedState()
        snap = s.snapshot()
        assert snap["depth"] is None

    def test_depth_float_is_rounded_in_snapshot(self) -> None:
        s = SharedState()
        s.update(depth=1.23456789)
        assert s.snapshot()["depth"] == 1.235

    def test_hand_position_none_stays_none(self) -> None:
        s = SharedState()
        assert s.snapshot()["hand_position"] is None

    def test_hand_position_tuple_becomes_list(self) -> None:
        s = SharedState()
        s.update(hand_position=(320, 240))
        assert s.snapshot()["hand_position"] == [320, 240]