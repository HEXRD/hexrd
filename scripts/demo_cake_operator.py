"""Build, reload, and verify a saved cake operator on the Eiger ceria fixture.

Run from this checkout with HEXRD_EXAMPLE_REPO_PATH pointing at HEXRD/examples.
This demonstrates the core API; workflow CLI integration belongs to Alessandro.
"""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import h5py
import numba
import numpy as np

from hexrd.core.cake import CakeOperator, CakeOperatorSpec, export_cake_operator
from hexrd.core.instrument import HEDMInstrument
from hexrd.core.projections.polar import PolarView


def median_ms(function, repeats=15):
    function()  # Compile and warm caches before timing.
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        samples.append((perf_counter() - start) * 1000)
    return float(np.median(samples))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--examples', type=Path, default=os.environ.get('HEXRD_EXAMPLE_REPO_PATH')
    )
    parser.add_argument('--output', type=Path, default=Path('/tmp/hexrd-cake-demo'))
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.examples is None:
        parser.error('provide --examples or HEXRD_EXAMPLE_REPO_PATH')
    numba.set_num_threads(args.threads)
    fixture = args.examples / 'eiger/first_ceria'
    with h5py.File(fixture / 'eiger_ceria_uncalibrated_composite.hexrd') as stream:
        instrument = HEDMInstrument(stream)
    start = perf_counter()
    with h5py.File(fixture / 'ff_000_data_000001.h5') as stream:
        frame = stream['entry/data/data'][0]
    read_ms = (perf_counter() - start) * 1000
    images = {
        name: frame[
            panel.roi[0][0] : panel.roi[0][1], panel.roi[1][0] : panel.roi[1][1]
        ]
        for name, panel in instrument.detectors.items()
    }
    spec = CakeOperatorSpec((0, 14), (-180, 180), (0.01, 5))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'eiger_ceria.hexrdcake'
    start = perf_counter()
    summary = export_cake_operator(
        path, instrument, spec, eta_chunk_rows=12, overwrite=True
    )
    build_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    operator = CakeOperator.load(path, verify_digest=True)
    verified_load_ms = (perf_counter() - start) * 1000
    reference = PolarView(
        spec.tth_range,
        instrument,
        eta_min=-180,
        eta_max=180,
        pixel_size=spec.pixel_size,
        cache_coordinate_map=True,
    )
    expected = reference.warp_image(images)
    cake = operator.apply_composite(frame)
    np.testing.assert_array_equal(cake.mask, expected.mask)
    np.testing.assert_array_equal(cake.compressed(), expected.compressed())
    np.testing.assert_array_equal(operator.apply(images), cake)
    active = operator.gather_active_composite_pixels(frame)
    batch = np.column_stack([active] * 8)
    batch_result = operator.apply_active_pixels(batch)
    for i in range(8):
        np.testing.assert_array_equal(batch_result[..., i], cake)
    regional = CakeOperator.load(path, eta_slice=slice(35, 37), compact_input=True)
    np.testing.assert_array_equal(regional.apply_composite(frame), cake[35:37])
    operator.save(args.output / 'roundtrip.hexrdcake', overwrite=True)
    roundtrip = CakeOperator.load(
        args.output / 'roundtrip.hexrdcake', verify_digest=True
    )
    np.testing.assert_array_equal(roundtrip.apply_composite(frame), cake)
    np.savez_compressed(
        args.output / 'results.npz',
        cake=cake.filled(np.nan),
        tth=operator.grid.tth_centers,
        eta=operator.grid.eta_centers,
        full_eta_mean=cake.mean(axis=0).filled(np.nan),
    )
    report = {
        **operator.summary(),
        'bitwise_equal_to_polar_view': True,
        'batch_and_region_equal': True,
        'threads': numba.get_num_threads(),
        'file_bytes': summary.file_size,
        'build_ms': build_ms,
        'read_frame_ms_single_sample': read_ms,
        'verified_load_ms': verified_load_ms,
        'cached_polar_ms': median_ms(lambda: reference.warp_image(images)),
        'composite_gather_ms': median_ms(
            lambda: operator.gather_active_composite_pixels(frame)
        ),
        'active_apply_ms': median_ms(lambda: operator.apply_active_pixels(active)),
        'composite_gather_and_apply_ms': median_ms(
            lambda: operator.apply_composite(frame)
        ),
        'batch_8_apply_ms': median_ms(lambda: operator.apply_active_pixels(batch)),
        'regional_active_pixels': regional.active_pixel_count,
        'note': 'Coarse demonstration grid; timings exclude JIT. Batch timing excludes gathering.',
    }
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax, profile) = plt.subplots(2, 1, figsize=(10, 7), layout='constrained')
    image = ax.pcolormesh(
        operator.grid.tth_edges,
        operator.grid.eta_edges,
        np.log1p(cake),
        shading='flat',
        cmap='viridis',
    )
    ax.set(
        xlabel='Two-theta (degrees)',
        ylabel='Eta (degrees)',
        title='Saved HEXRD cake operator · Eiger ceria',
    )
    fig.colorbar(image, ax=ax, label='log(1 + intensity)')
    profile.plot(operator.grid.tth_centers, cake.mean(axis=0), linewidth=1)
    profile.set(
        xlabel='Two-theta (degrees)',
        ylabel='Mean intensity',
        title='Full-eta masked mean · matches cached PolarView bit for bit',
    )
    fig.savefig(args.output / 'cake_demo.png', dpi=160)
    print(json.dumps(report, indent=2))
    print(f'Artifacts: {args.output.resolve()}')


if __name__ == '__main__':
    main()
