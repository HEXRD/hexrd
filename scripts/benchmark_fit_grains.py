#!/usr/bin/env python3
"""Run and compare fit-grains across source trees, including legacy master.

Run with --help for usage. Result pickles are local benchmark artifacts;
only compare files produced by this script from trusted source trees.
"""

import argparse
import cProfile
import hashlib
import json
import logging
from numbers import Number
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading
import time


def limit_address_space(mebibytes):
    """Hard allocation ceiling, installed before importing numerical libraries.

    Unlike a sampled RSS watchdog, RLIMIT_AS rejects an allocation before it
    can exhaust RAM. This guard is for serial benchmarks only: the limit is
    per process, so it must not be advertised as an aggregate worker limit.
    """
    if mebibytes is None:
        return
    if sys.platform != 'linux':
        raise RuntimeError('The benchmark allocation guard requires Linux RLIMIT_AS')
    import resource

    requested = int(mebibytes * 1024**2)
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        requested = min(requested, hard)
    resource.setrlimit(resource.RLIMIT_AS, (requested, requested))
    # Thread stacks and BLAS workspaces count against the allocation ceiling.
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    return requested // 1024**2


class MemoryMonitor:
    """Sample physical memory across the parent and its worker processes.

    Summed RSS double-counts fork-shared images. PSS apportions shared pages;
    USS counts private pages. Record all three so comparisons are interpretable.
    """

    def __init__(self, interval=0.1):
        import psutil

        self.process = psutil.Process()
        self.interval = interval
        self.stop_event = threading.Event()
        self.peaks = dict(rss=0, pss=0, uss=0, processes=0)
        self.samples = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def sample(self):
        import psutil

        totals = dict(rss=0, pss=0, uss=0, processes=0)
        for process in [self.process, *self.process.children(recursive=True)]:
            try:
                info = process.memory_full_info()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            totals['processes'] += 1
            for key in ('rss', 'pss', 'uss'):
                totals[key] += getattr(info, key, 0)
        for key, value in totals.items():
            self.peaks[key] = max(self.peaks[key], value)
        self.samples += 1

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def start(self):
        self.sample()
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.sample()
        return dict(
            peak_bytes={k: self.peaks[k] for k in ('rss', 'pss', 'uss')},
            peak_processes=self.peaks['processes'],
            samples=self.samples,
            sample_interval_seconds=self.interval,
        )


def assert_identical(expected, actual, path='result'):
    """Compare nested fits and spots exactly, treating missing data NaNs alike."""
    import numpy as np

    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray), f'{path}: array type differs'
        assert actual.dtype == expected.dtype, f'{path}: array dtype differs'
        if not np.array_equal(actual, expected, equal_nan=True):
            np.testing.assert_array_equal(actual, expected, err_msg=path, strict=True)
    elif isinstance(expected, dict):
        assert list(expected) == list(actual), f'{path}: dictionary keys differ'
        for key in expected:
            assert_identical(expected[key], actual[key], f'{path}[{key!r}]')
    elif isinstance(expected, (list, tuple)):
        assert type(expected) is type(actual), f'{path}: sequence types differ'
        assert len(expected) == len(actual), f'{path}: sequence lengths differ'
        for i, (left, right) in enumerate(zip(expected, actual)):
            assert_identical(left, right, f'{path}[{i}]')
    else:
        equal = actual == expected
        if not equal and isinstance(expected, Number) and isinstance(actual, Number):
            equal = np.isnan(expected) and np.isnan(actual)
        assert equal, f'{path}: {actual!r} != {expected!r}'


def _merge(base, overlay):
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value


def run(args):
    allocation_limit = limit_address_space(args.memory_limit_mb)
    if args.direct_io:
        os.environ['HEXRD_FIT_GRAINS_DIRECT_IO'] = '1'
    # Select the source tree before importing any part of HEXRD. Master needs
    # its own compiled extensions; sharing the branch's would invalidate the
    # comparison of the native transforms.
    source = args.source.resolve()
    if args.direct_io and not (source / 'hexrd/hedm/_image_cache.py').is_file():
        raise ValueError('--direct-io requires a source tree with lazy FCH5 readers')
    sys.path.insert(0, str(source))
    import hexrd
    import numpy as np
    import scipy
    import yaml

    from hexrd.hedm.fitgrains import fit_grains

    assert Path(hexrd.__file__).resolve().parent == source / 'hexrd'
    config = args.config.resolve()
    grains_path = args.grains.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)

    docs = list(yaml.safe_load_all(config.read_text()))
    if not 0 <= args.study < len(docs):
        raise ValueError(
            f'Config has {len(docs)} documents; invalid study {args.study}'
        )
    settings = docs[0]
    # Multi-document configs inherit all earlier documents.
    for overlay in docs[1 : args.study + 1]:
        _merge(settings, overlay)
    os.chdir(config.parent)
    settings['working_dir'] = str(
        Path(settings.get('working_dir', config.parent)).resolve()
    )
    settings['multiprocessing'] = args.workers
    if getattr(args, 'image_mode', None) is not None:
        settings.setdefault('fit_grains', {}).setdefault('image_cache', {})[
            'mode'
        ] = args.image_mode
    if getattr(args, 'image_cache_mb', None) is not None:
        settings.setdefault('fit_grains', {}).setdefault('image_cache', {})[
            'max_size_mb'
        ] = args.image_cache_mb
    # Keep spot files and other analysis output away from the input dataset.
    settings['analysis_name'] = str(output) + '-analysis'
    temporary_config = output.with_suffix('.yml')
    if temporary_config == config:
        raise ValueError('Choose an output prefix different from the input config')
    temporary_config.write_text(yaml.safe_dump(settings))
    original_grains = np.loadtxt(grains_path, ndmin=2)[: args.limit].copy()

    memory = MemoryMonitor() if args.memory else None
    if memory:
        memory.start()
    started = time.perf_counter()
    try:
        from hexrd.hedm.experiment import HedmExperiment
    except ModuleNotFoundError as error:
        if error.name != 'hexrd.hedm.experiment':
            raise
        from hexrd.hedm import config as legacy_config

        experiment = legacy_config.open(str(temporary_config))[0]
    else:
        experiment = HedmExperiment(str(temporary_config))
    setup_seconds = time.perf_counter() - started

    timings = []
    reference = None
    for repetition in range(args.repeat):
        profiler = cProfile.Profile() if args.profile else None
        if profiler:
            profiler.enable()
        started = time.perf_counter()
        result = fit_grains(
            experiment,
            original_grains.copy(),
            write_spots_files=args.write_spots,
            return_pull_spots_data=not args.no_return_spots,
        )
        timings.append(time.perf_counter() - started)
        if profiler:
            profiler.disable()
            profiler.dump_stats(str(output) + f'-{repetition}.prof')
        if reference is None:
            reference = result
        else:
            assert_identical(reference, result)
        print(f'Run {repetition + 1}: {timings[-1]:.6f} s', flush=True)

    memory_summary = memory.stop() if memory else None
    with output.with_suffix('.pkl').open('wb') as stream:
        pickle.dump(reference, stream, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {
        'source': str(source),
        'config': str(config),
        'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
        'grains': str(grains_path),
        'grains_sha256': hashlib.sha256(grains_path.read_bytes()).hexdigest(),
        'grain_count': len(original_grains),
        'study': args.study,
        'workers': args.workers,
        'address_space_limit_mb': allocation_limit,
        'direct_fch5_reads': os.environ.get('HEXRD_FIT_GRAINS_DIRECT_IO') == '1',
        'image_cache': settings.get('fit_grains', {}).get('image_cache', {}),
        'setup_seconds': setup_seconds,
        'seconds': timings,
        'median_seconds': float(np.median(timings)),
        'profiled': args.profile,
        'write_spots': args.write_spots,
        'return_spot_data': not args.no_return_spots,
        'memory': memory_summary,
        'python': sys.version,
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'thread_environment': {
            key: os.environ.get(key)
            for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
        },
    }
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


def compare(args):
    with args.reference.open('rb') as stream:
        expected = pickle.load(stream)
    with args.actual.open('rb') as stream:
        actual = pickle.load(stream)
    assert_identical(expected, actual)
    print('Identical fit results and extracted spot data (no numerical tolerance).')


def examples(args):
    root = args.examples.resolve()
    output = args.output_dir.resolve()
    ruby = root / 'NIST_ruby/single_GE'
    multi = root / 'NIST_ruby/multiruby_dexelas'
    zirconium = root / 'DAC/alpha_zirconium_axial_multi'
    cases = [
        (
            f'ruby-{i}',
            ruby / 'include/ruby_config.yml',
            ruby / f'results/ruby-b035e/scan-{i}/grains.out',
            i,
        )
        for i in range(6)
    ]
    for name, suffix in [('dexelas', ''), ('dexelas-roi', '_roi')]:
        cases.append(
            (
                name,
                multi / f'include/mruby_config_composite{suffix}.yml',
                multi / f'results/composite{suffix}_fit_grains_grains.out',
                0,
            )
        )
    cases.append(
        (
            'zirconium',
            zirconium / 'include/alpha_zr_state_0000_config.yml',
            zirconium / 'results/results_alpha_Zr_state_0000/grains.out',
            0,
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, config, grains, study in cases:
        results = []
        timings = []
        for label, source in [
            ('reference', args.reference_source),
            ('branch', args.source),
        ]:
            prefix = output / f'{label}-{name}'
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                'run',
                '--source',
                str(source.resolve()),
                '--config',
                str(config),
                '--grains',
                str(grains),
                '--output',
                str(prefix),
                '--study',
                str(study),
                '--workers',
                str(args.workers),
            ]
            with prefix.with_suffix('.log').open('w') as log:
                subprocess.run(
                    command, check=True, stdout=log, stderr=subprocess.STDOUT
                )
            with prefix.with_suffix('.pkl').open('rb') as stream:
                results.append(pickle.load(stream))
            timings.append(
                json.loads(prefix.with_suffix('.json').read_text())['median_seconds']
            )
        assert_identical(*results)
        summary.append(
            {
                'case': name,
                'identical': True,
                'reference_seconds': timings[0],
                'branch_seconds': timings[1],
            }
        )
        (output / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(f'{name}: identical fits and spots', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    runner = commands.add_parser(
        'run', help='Fit the same initial grains in a selected source tree'
    )
    runner.add_argument('--source', type=Path, required=True)
    runner.add_argument('--config', type=Path, required=True)
    runner.add_argument('--grains', type=Path, required=True)
    runner.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output prefix for .yml, .json, .pkl, and optional profiles',
    )
    runner.add_argument('--workers', type=int, default=1)
    runner.add_argument(
        '--memory-limit-mb',
        type=int,
        help='Hard Linux address-space limit in MiB; requires --workers 1',
    )
    runner.add_argument('--image-mode', choices=('auto', 'eager', 'lazy'))
    runner.add_argument(
        '--direct-io',
        action='store_true',
        help='Bypass filesystem cache for lazy FCH5 reads using Linux O_DIRECT',
    )
    runner.add_argument(
        '--image-cache-mb', type=float, help='Total image cache MiB across workers'
    )
    runner.add_argument('--limit', type=int)
    runner.add_argument(
        '--study',
        type=int,
        default=0,
        help='Config document index (0 is the base analysis)',
    )
    runner.add_argument('--repeat', type=int, default=1)
    runner.add_argument(
        '--profile',
        action='store_true',
        help='Use one worker to profile the numerical work',
    )
    runner.add_argument('--write-spots', action='store_true')
    runner.add_argument(
        '--no-return-spots',
        action='store_true',
        help='Measure production fitting without retaining all spot data',
    )
    runner.add_argument(
        '--memory',
        action='store_true',
        help='Sample aggregate RSS/PSS/USS for parent and workers',
    )
    runner.set_defaults(func=run)
    comparison = commands.add_parser(
        'compare', help='Require exact agreement of trusted local result files'
    )
    comparison.add_argument('reference', type=Path)
    comparison.add_argument('actual', type=Path)
    comparison.set_defaults(func=compare)
    suite = commands.add_parser(
        'examples', help='Require exact agreement on all nine rotation-series examples'
    )
    suite.add_argument('--examples', type=Path, required=True)
    suite.add_argument('--reference-source', type=Path, required=True)
    suite.add_argument('--source', type=Path, required=True)
    suite.add_argument('--output-dir', type=Path, required=True)
    suite.add_argument('--workers', type=int, default=1)
    suite.set_defaults(func=examples)
    args = parser.parse_args()
    if args.command in ('run', 'examples') and args.workers < 1:
        parser.error('--workers must be positive')
    if args.command == 'run' and (
        args.repeat < 1 or (args.limit is not None and args.limit < 1)
    ):
        parser.error('--repeat and --limit must be positive')
    if args.command == 'run' and args.memory_limit_mb is not None:
        if args.memory_limit_mb < 1 or args.workers != 1:
            parser.error('--memory-limit-mb must be positive and requires --workers 1')
    if args.command == 'run' and args.direct_io and args.image_mode != 'lazy':
        parser.error('--direct-io requires --image-mode lazy')
    args.func(args)


if __name__ == '__main__':
    main()
