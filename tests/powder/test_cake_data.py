import sys
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from hexrd.powder.cake import CakeMatrix, export_cake_matrix
from hexrd.core.instrument import HEDMInstrument, PlanarDetector
from hexrd.core import imageseries
from hexrd.powder.cake_data import (
    CakeDataConfig,
    _profiles,
    _wedge_edges,
    cake_data,
    config_as_yaml,
)
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
        np.array([99] + [1] * 10 + [3, 4] * 5, dtype=np.uint32)[:, None, None],
        (21, 32, 32),
    ).copy()
    # A real bad pixel must not turn every eta sample in its lineout into NaN.
    used = CakeMatrix.load(matrix_path).matrix.indices[100]
    raw[1:, used // 32, used % 32] = np.iinfo(np.uint32).max
    with h5py.File(tmp_path / 'raw.h5', 'w') as f:
        f['entry/data/data'] = raw
    par = np.zeros((1, 25))
    par[0, [3, 6, 7, 8, 9, 10, 11, 24]] = [7, 12, 1, 2, 3, 1, 0, 20]
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
        image_options={
            'path': 'entry/data', 'dataname': 'data', 'close_when_finished': True
        },
        ome_start=0,
        ome_end=2,
        dome=1,
        skip_frames=1,
        cake_width=45,
    )


@pytest.mark.parametrize(
    'frame_limit, template, intensities, omegas',
    [
        (None, '%s', [3, 1], [0.5, 1.5]),
        (10, '{sample_name}', [1], [1.5]),
    ],
    ids=['complete-scan', 'truncated-scan'],
)
def test_command_integrates_frames(
    tmp_path: Path,
    monkeypatch,
    small_scan: CakeDataConfig,
    frame_limit: int | None,
    template: str,
    intensities: list[int],
    omegas: list[float],
) -> None:
    sample_dir = tmp_path / 'sample'
    sample_dir.mkdir()
    (tmp_path / 'raw.h5').rename(sample_dir / 'raw.h5')
    config = replace(
        small_scan,
        data_dir=str(tmp_path / template),
        max_input_frames=frame_limit,
        matrix=Path(small_scan.matrix),
    )
    config_path = tmp_path / 'cake.yml'
    config_path.write_text(config_as_yaml(config))
    monkeypatch.setattr(
        sys, 'argv', ['hexrd', 'cake-data', '--config', str(config_path), '-c', '90']
    )
    main()

    # Skip 99; ten ones must average to 1, and alternating 3/4 must floor to 3.
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


def test_failed_processing_preserves_outputs(
    tmp_path: Path, small_scan: CakeDataConfig
) -> None:
    outputs = cake_data(small_scan)
    originals = [path.read_bytes() for path in outputs]
    with pytest.raises(ValueError, match='cake_width must be at least'):
        cake_data(replace(small_scan, cake_width=5))
    assert [path.read_bytes() for path in outputs] == originals

    with h5py.File(tmp_path / 'raw.h5', 'w') as stream:
        stream['entry/data/data'] = np.ones((21, 16, 64))  # Wrong detector shape.
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


def test_invalid_yaml_config(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'invalid.yml'
    path.write_text('cake_width: [')
    monkeypatch.setattr(sys, 'argv', ['hexrd', 'cake-data', '-y', str(path)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert 'error:' in capsys.readouterr().err


@pytest.mark.parametrize(
    'image_format, options, error',
    [
        ('hdf5', {'path': 'entry/data', 'bogus': 1}, 'unknown image_options'),
        ('eiger-stream-v2', {'threshold': 'threshold_2'}, 'unknown image_options'),
        ('eiger-stream-v2', {'threshold_setting': 'man_diff'}, 'does not support man_diff'),
    ],
)
def test_reader_options(small_scan, monkeypatch, image_format, options, error):
    # Validate configuration even when optional Dectris support is not installed.
    monkeypatch.setitem(imageseries.load.Registry.adapter_registry, image_format, object)
    config = replace(small_scan, image_format=image_format, image_options=options)
    with pytest.raises(ValueError, match=error):
        config.validate()


@pytest.mark.parametrize(
    'step, count, width, wedges, last_width',
    [
        (10, 36, 110, 4, 30),
        (1.3, 277, 5, 72, 5.1),  # Rounded endpoint must not add an empty wedge.
        (2, 8, 5, 4, 1),  # The final center lies exactly on a wedge boundary.
    ],
)
def test_wedges_preserve_partial_and_rounded_endpoints(
    step, count, width, wedges, last_width
):
    centers = -180 + (np.arange(count) + 0.5) * step
    operator = SimpleNamespace(eta=(-180.0, step, count), eta_centers=centers)
    edges = _wedge_edges(CakeDataConfig(cake_width=width), operator)
    assert len(edges) == wedges + 1
    assert edges[-1] - edges[-2] == pytest.approx(last_width)
    assert np.all(np.diff(np.searchsorted(centers, edges)) > 0)


def test_final_wedge_keeps_its_weight() -> None:
    centers = np.arange(1, 16, 2)
    operator = SimpleNamespace(eta=(0.0, 2.0, 8), eta_centers=centers)
    edges = _wedge_edges(CakeDataConfig(cake_width=5), operator)
    bounds = np.searchsorted(centers, edges)
    cake = np.ma.array([[1.], [1.], [2.], [2.], [2.], [3.], [3.], [40.]])
    full, profiles = _profiles(
        cake[first:last] for first, last in zip(bounds[:-1], bounds[1:])
    )
    np.testing.assert_allclose(profiles[:, 0], [1, 2, 3, 40])
    np.testing.assert_allclose(full, [11.5])


def test_invalid_destinations_do_not_create_directory(tmp_path, small_scan):
    output_dir = tmp_path / 'new/nested'
    config = replace(small_scan, output_dir=output_dir, output_par='sample.h5')
    with pytest.raises(ValueError, match='output paths must differ'):
        cake_data(config)
    assert not (tmp_path / 'new').exists()
