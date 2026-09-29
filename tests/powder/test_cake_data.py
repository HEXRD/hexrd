import argparse
import json
from dataclasses import replace
import warnings

import h5py

import numpy as np
import pytest

from hexrd.powder.cake import CakeMatrix, CakeOperatorSpec, export_cake_matrix
from hexrd.core.instrument import HEDMInstrument, PlanarDetector
from hexrd.core.projections.polar import PolarView
from hexrd.powder.cake_data import CakeDataConfig, cake_data, config_as_json
from hexrd.powder.cli.cake_data import configure_parser


def test_config_file_and_command_line_update(tmp_path):
    path = tmp_path / "cake.json"
    path.write_text(
        json.dumps(
            {
                "data_dir": "raw/{sample_name}/{image_number}",
                "output_dir": "output",
                "sample_name": "ceria",
                "instrument": "instrument.hexrd",
                "par_file": "input.par",
                "cake_width": 10,
            }
        ),
        encoding="utf-8",
    )
    config = CakeDataConfig.from_json(path)
    config.update({"tth_start": 4.0, "cake_width": None})

    assert config.tth_start == 4.0
    assert config.cake_width == 10
    config.validate()


def test_config_rejects_unknown_parameters(tmp_path):
    path = tmp_path / "cake.json"
    path.write_text('{"not_a_parameter": 1}', encoding="utf-8")

    with pytest.raises(ValueError, match="unknown cake-data parameters"):
        CakeDataConfig.from_json(path)


def test_cli_exposes_short_and_long_parameter_forms():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    configure_parser(subparsers)

    short = parser.parse_args(["cake-data", "-b", "4", "-p", "input.par"])
    long = parser.parse_args(
        ["cake-data", "--tth-start", "4", "--par-file", "input.par"]
    )

    assert short.tth_start == long.tth_start == 4
    assert short.par_file == long.par_file == "input.par"


@pytest.fixture
def small_scan(tmp_path):
    instrument = HEDMInstrument()
    instrument.detectors.clear()
    instrument.detectors['panel'] = PlanarDetector(
        rows=32,
        cols=32,
        pixel_size=(0.2, 0.2),
        tvec=np.array([0.0, 0.0, -100.0]),
        saturation_level=65535,
    )
    spec = CakeOperatorSpec((0, 3), (-180, 180), (0.2, 10))
    matrix_path = tmp_path / 'geometry.hexrdcake'
    export_cake_matrix(matrix_path, instrument, spec)
    raw = np.broadcast_to(
        np.array([99, 3, 5, 9, 13], dtype=np.uint32)[:, None, None], (5, 32, 32)
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
    return config, instrument, spec, raw


def test_saved_matrix_command_matches_polar_lineouts(tmp_path, small_scan):
    config, instrument, spec, raw = small_scan
    config_path = tmp_path / 'cake.json'
    config_path.write_text(config_as_json(config))
    parser = argparse.ArgumentParser()
    configure_parser(parser.add_subparsers())
    args = parser.parse_args(['cake-data', '--config', str(config_path)])
    args.func(args, parser)

    view = PolarView(
        spec.tth_range,
        instrument,
        eta_min=-180,
        eta_max=180,
        pixel_size=spec.pixel_size,
    )
    # Average and floor two frames, then reverse omega as the par metadata asks.
    images = raw[1:].astype(float)
    images[images >= np.iinfo(np.uint32).max] = np.nan
    integrated = np.floor(images.reshape(2, 2, 32, 32).mean(axis=1))[::-1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        cakes = [
            view.warp_image({'panel': image}, pad_with_nans=True)
            for image in integrated
        ]
        expected_full = np.array([np.nanmean(cake, axis=0) for cake in cakes])
        expected_wedges = np.array(
            [
                [
                    np.nanmean(cake[start : start + 9], axis=0)
                    for start in range(0, 36, 9)
                ]
                for cake in cakes
            ]
        )
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


def test_invalid_scan_keeps_previous_outputs(tmp_path, small_scan):
    config, _, _, _ = small_scan
    cake_data(config)
    output = tmp_path / 'out/sample.h5'
    original = output.read_bytes()
    with pytest.raises(ValueError, match='matrix fixes the geometry'):
        cake_data(replace(config, refinement=1))
    with pytest.raises(ValueError, match='omega range must be divisible'):
        cake_data(replace(config, dome=0.7))
    with pytest.raises(ValueError, match='complete omega interval'):
        cake_data(replace(config, max_input_frames=3))
    with h5py.File(tmp_path / 'raw.h5', 'w') as f:
        f['entry/data/data'] = np.ones((5, 16, 64))
    with pytest.raises(ValueError, match='input image shape'):
        cake_data(config)
    assert output.read_bytes() == original
    assert not list((tmp_path / 'out').glob('.cake-data-*'))
