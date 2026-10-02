import sys
import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from hexrd.powder.cake import CakeMatrix, export_cake_matrix
from hexrd.core.instrument import HEDMInstrument, PlanarDetector
from hexrd.powder.cake_data import CakeDataConfig, _profiles, cake_data, config_as_yaml
from hexrd.cli.main import main


@pytest.fixture
def small_scan(tmp_path: Path) -> CakeDataConfig:
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
    return CakeDataConfig(
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
        cake_width=45,
    )


@pytest.mark.parametrize(
    'frame_limit, intensities, omegas',
    [(None, [11, 4], [0.5, 1.5]), (2, [4], [1.5])],
    ids=['complete-scan', 'truncated-scan'],
)
def test_command_integrates_frames(
    tmp_path: Path,
    monkeypatch,
    small_scan: CakeDataConfig,
    frame_limit: int | None,
    intensities: list[int],
    omegas: list[float],
) -> None:
    config = replace(
        small_scan, max_input_frames=frame_limit, matrix=Path(small_scan.matrix)
    )
    config_path = tmp_path / 'cake.yml'
    config_path.write_text(config_as_yaml(config))
    monkeypatch.setattr(
        sys, 'argv', ['hexrd', 'cake-data', '--config', str(config_path), '-c', '90']
    )
    main()

    # Skip 99; floor the pair means (3, 6) -> 4 and (9, 14) -> 11.
    # Reversal changes bin order; truncation keeps the original omega labels.
    with h5py.File(tmp_path / 'out/sample.h5') as output:
        layer = output['Cake/0']
        np.testing.assert_allclose(layer['ome_key'][1], omegas)
        assert layer['nFrames'][()] == len(intensities)
        for index, intensity in enumerate(intensities):
            wedges = layer[f'Ome/{index}/IvTTH'][:]
            assert wedges.shape == (4, 15)  # CLI overrides 45-degree wedges.
            np.testing.assert_allclose(wedges[np.isfinite(wedges)], intensity)
        full = layer['IvTTHvW'][:]
        covered = np.isfinite(full).all(axis=0)
        assert covered.sum() > 5  # One bad raw pixel must not spoil the lineout.
        for profile, intensity in zip(full, intensities):
            np.testing.assert_allclose(profile[covered], intensity)
        np.testing.assert_allclose(
            layer['IvTTH_Full'][:][covered], np.mean(intensities)
        )
        np.testing.assert_allclose(layer['tth'][:], np.arange(15) * 0.2 + 0.1)
    assert np.loadtxt(tmp_path / 'out/sample.par').shape == (12,)


@pytest.mark.parametrize('options', [None, [], 'reader options', 1])
def test_reader_options_must_be_an_object(
    small_scan: CakeDataConfig, options: object
) -> None:
    with pytest.raises(ValueError, match='image_options must be a mapping'):
        replace(small_scan, image_options=options).validate()


@pytest.mark.parametrize('count', [0, 3.9])
def test_invalid_frame_count(small_scan: CakeDataConfig, count: float) -> None:
    metadata = np.loadtxt(small_scan.par_file, ndmin=2)
    metadata[0, 24] = count
    np.savetxt(small_scan.par_file, metadata)
    with pytest.raises(ValueError, match='frame counts'):
        cake_data(small_scan)


@pytest.mark.parametrize('filename', ['raw.h5', 'input.par', 'sample.h5', 'directory'])
def test_invalid_output_path(
    tmp_path: Path, small_scan: CakeDataConfig, filename: str
) -> None:
    (tmp_path / 'directory').mkdir()
    config = replace(small_scan, output_dir=str(tmp_path), output_par=filename)
    with pytest.raises(ValueError, match='output paths must'):
        cake_data(config)


def test_failed_processing_preserves_outputs(
    tmp_path: Path, small_scan: CakeDataConfig
) -> None:
    outputs = cake_data(small_scan)
    originals = [path.read_bytes() for path in outputs]
    with h5py.File(tmp_path / 'raw.h5', 'w') as stream:
        stream['entry/data/data'] = np.ones((5, 16, 64))  # Wrong detector shape.
    with pytest.raises(ValueError, match='input image shape'):
        cake_data(small_scan)
    assert [path.read_bytes() for path in outputs] == originals
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


@pytest.mark.parametrize('contents', ['', '[]', 'cake_width: [', 'unknown: 1\n2: bad'])
def test_invalid_yaml_config(tmp_path, monkeypatch, capsys, contents):
    path = tmp_path / 'invalid.yml'
    path.write_text(contents)
    monkeypatch.setattr(sys, 'argv', ['hexrd', 'cake-data', '-y', str(path)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert 'error:' in capsys.readouterr().err


@pytest.mark.parametrize(
    'name, value',
    [
        ('skip_frames', None),
        ('skip_frames', True),
        ('start_layer', None),
        ('dome', '1e-3'),
        ('sample_name', 123),
        ('image_format', []),
        ('image_options', {1: 'path'}),
    ],
)
def test_invalid_parameter_type(small_scan, name, value):
    with pytest.raises(ValueError, match=name):
        replace(small_scan, **{name: value}).validate()


@pytest.mark.parametrize('column', [-1, 25, 10])
def test_invalid_fly_axis_metadata(small_scan, column):
    path = Path(small_scan.par_file)
    path.with_suffix('.json').write_text(
        json.dumps(
            {
                str(column): 'fly_axis0_rising',
                '11': 'fly_axis0_falling',
            }
        )
    )
    if column == 10:
        data = np.loadtxt(path, ndmin=2)
        data[0, column] = np.nan
        np.savetxt(path, data)
    with pytest.raises(ValueError, match='fly-axis'):
        cake_data(small_scan)
