"""
Unit tests for the benchmark module.
"""
from __future__ import annotations

import os
import time

import pytest

from core.benchmark import (
    BenchmarkCollector,
    benchmark,
    benchmark_func,
    get_report,
    print_report,
    reset_benchmarks,
)


@pytest.fixture(autouse=True)
def clear_benchmarks():
    """Ensures clean state before each test."""
    reset_benchmarks()
    yield
    reset_benchmarks()


def test_collector_records_samples() -> None:
    """Collector should accumulate timing samples."""
    collector = BenchmarkCollector()
    collector.record("test", 5.0)
    collector.record("test", 15.0)
    
    stats = collector.report()
    assert stats["test"]["count"] == 2
    assert stats["test"]["mean_ms"] == 10.0
    assert stats["test"]["min_ms"] == 5.0
    assert stats["test"]["max_ms"] == 15.0


def test_collector_empty_report() -> None:
    """Empty collector should return empty dict."""
    collector = BenchmarkCollector()
    assert collector.report() == {}


def test_benchmark_disabled_by_default() -> None:
    """Benchmark context manager should not crash when disabled."""
    with benchmark("noop"):
        pass  # Should execute without error even when disabled


def test_benchmark_func_disabled() -> None:
    """Decorated function should execute normally when benchmarking disabled."""
    @benchmark_func()
    def add(a: int, b: int) -> int:
        return a + b
    
    assert add(2, 3) == 5


def test_get_report_empty() -> None:
    """get_report should return empty dict with no data."""
    assert get_report() == {}


def test_reset_clears_data() -> None:
    """reset_benchmarks should clear all collected data."""
    collector = BenchmarkCollector()
    collector.record("x", 1.0)
    collector.reset()
    assert collector.report() == {}


def test_benchmark_enabled_records() -> None:
    """When enabled, benchmark should record timing data."""
    # Temporarily enable benchmarking
    os.environ["AERODRAFT_BENCHMARK"] = "1"
    
    # Must reimport to pick up the env var
    from core import benchmark as benchmark_module
    benchmark_module._BENCHMARK_ENABLED = True
    
    try:
        with benchmark("enabled_test"):
            time.sleep(0.001)
        
        report = get_report()
        assert "enabled_test" in report
        assert report["enabled_test"]["count"] == 1
        assert report["enabled_test"]["mean_ms"] > 0
    finally:
        benchmark_module._BENCHMARK_ENABLED = False
        os.environ.pop("AERODRAFT_BENCHMARK", None)
        reset_benchmarks()