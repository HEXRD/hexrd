import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'benchmark_fit_grains',
    Path(__file__).resolve().parents[1] / 'scripts/benchmark_fit_grains.py',
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_exact_benchmark_comparison_accepts_matching_missing_data():
    result = ([(0, 1.0, np.nan, np.array([1.0, np.nan]))], {'panel': [True, np.nan]})
    benchmark.assert_identical(result, result)


@pytest.mark.parametrize(
    'left,right',
    [
        (1.0, np.nextafter(1.0, 2.0)),
        (np.array([1.0]), np.array([np.nextafter(1.0, 2.0)])),
        (np.array([np.nan]), np.array([0.0])),
        (np.array([1.0]), np.array([[1.0]])),
        (np.array([1.0], dtype=np.float32), np.array([1.0], dtype=np.float64)),
        ({'a': 1, 'b': 2}, {'b': 2, 'a': 1}),
        ([1], (1,)),
        ([1], [1, 2]),
    ],
)
def test_exact_benchmark_comparison_rejects_differences(left, right):
    with pytest.raises(AssertionError):
        benchmark.assert_identical(left, right)


def test_memory_monitor_stops_and_reports_parent():
    monitor = benchmark.MemoryMonitor()
    monitor.start()
    result = monitor.stop()
    assert not monitor.thread.is_alive()
    assert result['samples'] >= 2
    assert result['peak_processes'] >= 1
    assert result['peak_bytes']['rss'] > 0
