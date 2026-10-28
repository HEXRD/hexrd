import importlib.util
from pathlib import Path
import subprocess
import sys

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


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux allocation guard')
def test_allocation_guard_rejects_allocation_before_memory_is_used():
    script = f'''
import importlib.util
import resource
spec = importlib.util.spec_from_file_location('benchmark', {str(spec.origin)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.limit_address_space(128)
assert resource.getrlimit(resource.RLIMIT_AS) == (128 * 1024**2,) * 2
assert module.limit_address_space(256) == 128
try:
    bytearray(256 * 1024**2)
except MemoryError:
    print('allocation rejected')
else:
    raise AssertionError('allocation ceiling was not enforced')
'''
    result = subprocess.run(
        [sys.executable, '-c', script], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == 'allocation rejected'


def test_allocation_guard_rejects_multiple_workers(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(spec.origin),
            'run',
            '--source',
            str(tmp_path),
            '--config',
            str(tmp_path),
            '--grains',
            str(tmp_path),
            '--output',
            str(tmp_path),
            '--workers',
            '2',
            '--memory-limit-mb',
            '128',
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert 'requires --workers 1' in result.stderr


def test_direct_io_requires_explicit_lazy_mode(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(spec.origin),
            'run',
            '--source',
            str(tmp_path),
            '--config',
            str(tmp_path),
            '--grains',
            str(tmp_path),
            '--output',
            str(tmp_path),
            '--direct-io',
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert 'requires --image-mode lazy' in result.stderr
