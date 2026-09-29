import argparse
import json
from dataclasses import replace
from pathlib import Path
import warnings

import h5py
import numpy as np
import pytest

from hexrd.powder.cake import CakeMatrix, export_cake_matrix
from hexrd.core.instrument import HEDMInstrument, PlanarDetector
from hexrd.core.projections.polar import PolarView
from hexrd.powder.cake_data import CakeDataConfig, _profiles, cake_data, config_as_json
from hexrd.powder.cli.cake_data import configure_parser


@pytest.fixture
def small_scan(tmp_path: Path) -> tuple[CakeDataConfig, HEDMInstrument, np.ndarray]:
    instrument = HEDMInstrument()
    instrument.detectors.clear()
    instrument.detectors['panel'] = PlanarDetector(
        rows=32,
        cols=32,
        pixel_size=(0.2, 0.2),
        tvec=np.array([0.0, 0.0, -100.0]),
        saturation_level=65535,
    )
    matrix_path = tmp_path / 'geometry.hexrdcake'
    export_cake_matrix(
        matrix_path,
        instrument,
        tth_range=(0, 3),
        eta_range=(-180, 180),
        pixel_size=(0.2, 10),
    )
    raw = np.broadcast_to(
        np.array([99, 3, 6, 9, 14], dtype=np.uint32)[:, None, None], (5, 32, 32)
    ).copy()
    # A real bad pixel must not turn every eta sample in its lineout into NaN.
    used = CakeMatrix.load(matrix_path).matrix.indices[100]
    raw[1:, used // 32, used % 32] = np.iinfo(np.uint32).max
    with h5py.File(tmp_path / 'raw.h5', 'w') as f:
        f['entry/data/data'] = raw
    par = np.zeros((1, 25))
    par[0, [3, 6, 7, 8, 9, 10, 11, 24]] = [7, 12, 1, 2, 3, 1, 0, 4]
    np.savetxt(tmp_path / 'input.par', par)
    # Deliberately out of order: the numeric keys, not dictionary order, matter.
    (tmp_path / 'input.json').write_text(
        json.dumps({'11': 'fly_axis0_falling', '10': 'fly_axis0_rising'})
    )
    config = CakeDataConfig(
        data_dir=str(tmp_path),
        output_dir=str(tmp_path / 'out'),
        sample_name='sample',
        matrix=str(matrix_path),
        par_file=str(tmp_path / 'input.par'),
        image_pattern='raw.h5',
        image_format='hdf5',
        image_options={'path': 'entry/data', 'dataname': 'data'},
        ome_start=0,
        ome_end=2,
        dome=1,
        skip_frames=1,
        cake_width=90,
    )
    return config, instrument, raw


def test_saved_matrix_command_matches_polar_lineouts(
    tmp_path: Path, small_scan: tuple[CakeDataConfig, HEDMInstrument, np.ndarray]
) -> None:
    config, instrument, raw = small_scan
    config_path = tmp_path / 'cake.json'
    config_path.write_text(config_as_json(config))
    parser = argparse.ArgumentParser()
    configure_parser(parser.add_subparsers())
    args = parser.parse_args(['cake-data', '--config', str(config_path), '-c', '90'])
    args.func(args, parser)

    view = PolarView(
        (0, 3), instrument, eta_min=-180, eta_max=180, pixel_size=(0.2, 10)
    )
    # Average and floor two frames, then reverse omega as the par metadata asks.
    images = raw[1:].astype(float)
    images[images >= np.iinfo(np.uint32).max] = np.nan
    integrated = np.floor(images.reshape(2, 2, 32, 32).mean(axis=1))[::-1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        cakes = np.array(
            [
                view.warp_image({'panel': image}, pad_with_nans=True)
                for image in integrated
            ]
        )
        expected_wedges = np.nanmean(cakes.reshape(2, 4, 9, -1), axis=2)
        expected_full = np.nanmean(expected_wedges, axis=1)
    with h5py.File(tmp_path / 'out/sample.h5') as f:
        g = f['Cake/0']
        np.testing.assert_allclose(g['IvTTHvW'][:], expected_full, equal_nan=True)
        np.testing.assert_allclose(
            g['IvTTH_Full'][:], expected_full.mean(axis=0), equal_nan=True
        )
        np.testing.assert_allclose(
            g['Ome/0/IvTTH'][:], expected_wedges[0], equal_nan=True
        )
        np.testing.assert_allclose(g['ome_key'][1], [0.5, 1.5])
        np.testing.assert_allclose(g['tth'][:], np.degrees(view.angular_grid[1][0]))
        assert g['nFrames'][()] == 2
        assert np.isfinite(g['IvTTH_Full'][:]).sum() > 5
    assert np.loadtxt(tmp_path / 'out/sample.par').shape == (12,)
    # Limiting input frames truncates the acquisition; it must not rescale omega.
    cake_data(replace(config, max_input_frames=2))
    with h5py.File(tmp_path / 'out/sample.h5') as f:
        np.testing.assert_allclose(f['Cake/0/ome_key'][1], [1.5])
        np.testing.assert_allclose(
            f['Cake/0/IvTTHvW'][:], expected_full[1:], equal_nan=True
        )


def test_invalid_scan_keeps_previous_outputs(
    tmp_path: Path, small_scan: tuple[CakeDataConfig, HEDMInstrument, np.ndarray]
) -> None:
    config, _, _ = small_scan
    cake_data(config)
    output = tmp_path / 'out/sample.h5'
    original = output.read_bytes()
    # A typo in output settings must never replace the raw data or par input.
    for overrides in (
        dict(output_dir=str(tmp_path), sample_name='raw'),
        dict(output_dir=str(tmp_path), output_par='input.par'),
        dict(output_par='sample.h5'),
    ):
        with pytest.raises(ValueError, match='output paths must differ'):
            cake_data(replace(config, **overrides))
    with h5py.File(tmp_path / 'raw.h5', 'w') as f:
        f['entry/data/data'] = np.ones((5, 16, 64))
    with pytest.raises(ValueError, match='input image shape'):
        cake_data(config)
    assert output.read_bytes() == original
    assert not list((tmp_path / 'out').glob('.cake-data-*'))


def test_combined_lineout_weights_wedges_equally() -> None:
    cake = np.ma.array(
        [[10, 0, np.nan], [10, np.nan, np.nan], [20, 4, np.nan], [999, 999, 999]],
        mask=[[False] * 3] * 3 + [[True] * 3],
    )
    # Two wedges have unequal coverage; the third has no samples at all.
    full, wedges = _profiles([cake[:2], cake[2:], cake[4:]])
    np.testing.assert_allclose(
        wedges, [[10, 0, np.nan], [20, 4, np.nan], [np.nan] * 3], equal_nan=True
    )
    np.testing.assert_allclose(full, [15, 2, np.nan], equal_nan=True)
