"""Per-frame energies stay aligned with images and correct prediction and fitting."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from hexrd.core import constants as ct, imageseries
from hexrd.core.experiment import ImageSeries as SparseImageSeries
from hexrd.core.imageseries.omega import OmegaImageSeries
from hexrd.core.imageseries.process import ProcessedImageSeries
from hexrd.core.instrument import HEDMInstrument
from hexrd.core.instrument.beam_energy import (
    load_beam_energy_file,
    omegas_to_frames,
    validate_beam_energy_series,
)
from hexrd.core.material.crystallography import PlaneData
from hexrd.core.transforms import xfcapi
from hexrd.hedm import fitgrains
from hexrd.hedm.experiment import FitGrains


def make_series(
    energies: np.ndarray, omega: np.ndarray | None = None
) -> OmegaImageSeries:
    if omega is None:
        edges = np.linspace(-180, 180, len(energies) + 1)
        omega = np.column_stack([edges[:-1], edges[1:]])
    data = np.zeros((len(energies), 5, 5))
    data[:, 0, 0] = 1
    return OmegaImageSeries(
        imageseries.open(
            None,
            'array',
            data=data,
            meta={'omega': omega, 'beam_energy': energies},
        )
    )


def make_grain() -> tuple[HEDMInstrument, PlaneData, np.ndarray]:
    instrument = HEDMInstrument()
    instrument.beam_energy = 65.0
    next(iter(instrument.detectors.values())).saturation_level = 65535
    hkls = np.array([[1, 1, 1], [2, 0, 0], [2, 2, 0], [3, 1, 1]]).T
    plane_data = PlaneData(hkls, [3.6], 'Oh', 65.0, 0.01)
    grain = np.r_[0.2, -0.1, 0.3, 0.1, 0.2, -0.1, 1, 1, 1, 0, 0, 0]
    return instrument, plane_data, grain


def test_energy_file_and_frame_selection(tmp_path: Path) -> None:
    series = make_series(np.array([64.5, 65.5]))
    filename = tmp_path / 'energies.txt'
    filename.write_text('# keV\n64.5\n\n6.55e1\n')
    np.testing.assert_array_equal(load_beam_energy_file(filename, series), [64.5, 65.5])
    for bad in ('65\n', '64 65\n', 'nan\n65\n', '0\n65\n'):
        filename.write_text(bad)
        with pytest.raises(ValueError):
            load_beam_energy_file(filename, series)

    subset = ProcessedImageSeries(series, [], frame_list=[1])
    np.testing.assert_array_equal(subset.metadata['beam_energy'], [65.5])
    np.testing.assert_array_equal(subset.metadata['omega'], series.omega[[1]])

    # File overrides must retain sparse pixel indexing and leave the source alone.
    cache = tmp_path / 'frames.npz'
    imageseries.write(series, cache, 'frame-cache', threshold=0, max_workers=1)
    sparse = OmegaImageSeries(SparseImageSeries('detector', str(cache)))
    override = fitgrains._FrameEnergySeries(sparse, np.array([60.0, 61.0]))
    pixels = override[0, np.array([[0]]), np.array([[0]])]
    assert isinstance(pixels, np.ndarray)
    np.testing.assert_array_equal(pixels, [[1.0]])
    assert 'beam_energy' not in sparse.metadata
    series.metadata['beam_energy'] = np.ones(3)
    with pytest.raises(ValueError, match='one value per image frame'):
        ProcessedImageSeries(series, [], frame_list=[1])


def test_frame_lookup_and_detector_fallback() -> None:
    series = make_series(
        np.array([60.0, 61.0]), np.array([[350.0, 360.0], [0.0, 10.0]])
    )
    validate_beam_energy_series(series.metadata['beam_energy'], series)
    np.testing.assert_array_equal(
        omegas_to_frames([-5.0, 5.0, 20.0, np.nan], series), [0, 1, -1, -1]
    )
    for omega in (np.array([[0.0, 720.0]]), np.array([[0.0, 2.0], [1.0, 3.0]])):
        ambiguous = make_series(np.ones(len(omega)), omega)
        with pytest.raises(ValueError, match='omega wedges'):
            validate_beam_energy_series(ambiguous.metadata['beam_energy'], ambiguous)

    plain = make_series(np.ones(2))
    del plain.metadata['beam_energy']
    spots = {
        key: [[None] * 6 + [np.radians([0.0, 0.0, 5.0])]] for key in ('plain', 'energy')
    }
    images = {'plain': plain, 'energy': series}
    wavelengths = fitgrains._per_refl_wavelengths(spots, images, 0.2)
    assert wavelengths['plain'] == 0.2
    np.testing.assert_allclose(wavelengths['energy'], [ct.keVToAngstrom(61.0)])
    spots['energy'][0][6][-1] = np.radians(20.0)
    with pytest.raises(ValueError, match='measured omega has no frame energy'):
        fitgrains._per_refl_wavelengths(spots, images, 0.2)


def test_prediction_iterations() -> None:
    instrument, plane_data, grain = make_grain()
    detector = next(iter(instrument.detectors))
    reference = []
    for energy in (65.0, 65.2, 65.4):
        result = instrument.simulate_rotation_series(
            PlaneData(None, plane_data),
            [grain],
            wavelength=ct.keVToAngstrom(energy),
        )[detector]
        reference.append(np.r_[result[2][0][0], result[3][0][0]])

    # The first correction crosses this boundary; the next must use the new frame.
    initial, corrected = np.degrees([reference[0][2], reference[1][2]])
    boundary = (initial + corrected) / 2
    energies = [65.2, 65.4] if initial < boundary else [65.4, 65.2]
    series = make_series(
        np.array(energies), np.array([[-180, boundary], [boundary, 180]])
    )
    for count in (0, 1, 2, 3):
        result = instrument.simulate_rotation_series(
            plane_data,
            [grain],
            imgser_dict={detector: series},
            beam_energy_iterations=count,
        )[detector]
        np.testing.assert_allclose(
            np.r_[result[2][0][0], result[3][0][0]],
            reference[min(count, 2)],
            atol=1e-12,
        )
    for count in (-1, 1.5, True):
        with pytest.raises(ValueError, match='beam_energy_iterations'):
            instrument.simulate_rotation_series(
                plane_data, [grain], beam_energy_iterations=count
            )
        with pytest.raises(ValueError, match='beam_energy_iterations'):
            FitGrains.from_dict({'beam_energy_iterations': count}, str)


def test_prediction_and_fit_from_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instrument, plane_data, grain = make_grain()
    instrument.energy_correction = {'axis': 'y', 'slope': 100.0, 'intercept': 0.0}
    detector, panel = next(iter(instrument.detectors.items()))
    panel.group, panel.roi = 'group', np.array([0, 0])
    energies = np.repeat([64.9, 65.1], 36)
    series = make_series(energies)

    # Independent scalar-energy predictions supply ideal measured centroids.
    spots = []
    for energy in np.unique(energies):
        result = instrument.simulate_rotation_series(
            PlaneData(None, plane_data),
            [grain],
            wavelength=ct.keVToAngstrom(energy),
        )[detector]
        _, hkls, angles, xy, _ = [values[0] for values in result]
        for hkl, angle, point in zip(hkls, angles, xy):
            frame = series.omega_to_frame(np.degrees(angle[2]))[0]
            if energies[frame] == energy:
                spots.append([len(spots), 0, hkl, 1.0, 1.0, angle, angle, point])
    expected = sorted(np.r_[spot[2], spot[6], spot[7]].tolist() for spot in spots)
    for key in ('group', ct.shared_ims_key, f'{detector}-cache'):
        result = instrument.simulate_rotation_series(
            plane_data, [grain], imgser_dict={key: series}
        )[detector]
        actual = np.column_stack([result[1][0], result[2][0], result[3][0]])
        np.testing.assert_allclose(sorted(actual.tolist()), expected, atol=1e-12)

    # Mock only spot extraction and the subset selected for refitting.
    pull_spots = Mock(return_value=([True] * len(spots), {detector: spots}))
    monkeypatch.setattr(instrument, 'pull_spots', pull_spots)
    monkeypatch.setattr(
        fitgrains,
        'refine_valid_reflections',
        Mock(return_value=({detector: spots[::2]}, len(spots[::2]))),
    )
    np.savetxt(tmp_path / 'energies.txt', energies)
    experiment = SimpleNamespace(
        instrument=instrument,
        image_series={'group': series},
        max_workers=1,
        analysis_dir=tmp_path,
        eta_ranges=np.radians([[-180.0, 180.0]]),
    )
    seed = grain.copy()
    seed[:3] += 1e-4
    seed[6:9] += 1e-4

    def resolve(name: str) -> str:
        return str(tmp_path / name)

    for filename in (None, 'energies.txt'):
        experiment.fit_grains = FitGrains.from_dict(
            {
                'beam_energy_file': filename,
                'beam_energy_iterations': 2,
                'threshold': 0,
                'refit': [2, 2],
                'tolerance': {'tth': [0.25], 'eta': [1.0], 'omega': [1.0]},
            },
            resolve,
        )
        # The file must override stale metadata without changing the source.
        original = energies if filename is None else np.full(72, 65.0)
        series.metadata['beam_energy'] = original.copy()
        result = fitgrains.fit_grains(
            experiment,
            np.r_[0.0, 1.0, 0.0, seed],
            material=SimpleNamespace(_material=SimpleNamespace(planeData=plane_data)),
            write_spots_files=False,
        )[0]
        np.testing.assert_allclose(result[3], grain, atol=1e-6)
        assert result[2] < 1e-9
        assert pull_spots.call_args.kwargs['beam_energy_iterations'] == 2
        assert next(iter(pull_spots.call_args.args[2])) == 'group'
        np.testing.assert_array_equal(series.metadata['beam_energy'], original)

    assert fitgrains.refine_valid_reflections.call_count == 2

    # A file must match every detector's frame count, not just the first one.
    experiment.image_series[detector] = make_series(energies[:-1])
    with pytest.raises(ValueError, match=r'expected \(71,\), got \(72,\)'):
        fitgrains.fit_grains(experiment, np.r_[0.0, 1.0, 0.0, seed])


def test_pull_spots_preserves_first_series(monkeypatch: pytest.MonkeyPatch) -> None:
    instrument, plane_data, grain = make_grain()
    detector, panel = next(iter(instrument.detectors.items()))
    instrument.detectors['other'] = panel
    # Input order differs from detector order, with different omega ranges/steps.
    images = {
        'other': make_series(np.full(2, 65.0), np.array([[10.0, 11.0], [11.0, 12.0]])),
        detector: make_series(np.full(2, 65.0), np.array([[20.0, 22.0], [22.0, 24.0]])),
    }
    predict = Mock(side_effect=RuntimeError('stop before extraction'))
    monkeypatch.setattr(instrument, 'simulate_rotation_series', predict)
    for check_only in (False, True):
        with pytest.raises(RuntimeError, match='stop before extraction'):
            instrument.pull_spots(plane_data, grain, images, check_only=check_only)
        np.testing.assert_allclose(
            predict.call_args.kwargs['ome_ranges'], np.radians([[10.0, 12.0]])
        )
        np.testing.assert_allclose(
            predict.call_args.kwargs['ome_period'], np.radians([10.0, 370.0])
        )


def test_array_wavelengths_match_scalar_solver() -> None:
    hkls = np.array([[1.0, 1.0, 1.0], [2.0, 0.0, 0.0], [3.0, 1.0, 1.0]])
    rmat = xfcapi.make_rmat_of_expmap(np.array([0.2, -0.1, 0.3]))
    bmat = np.eye(3) / 3.6
    wavelengths = np.array([0.19, 0.20, 0.21])[::-1]  # Non-contiguous input.
    expected = np.concatenate(
        [
            xfcapi.oscill_angles_of_hkls(hkls[i : i + 1], 0.0, rmat, bmat, wave)
            for i, wave in enumerate(wavelengths)
        ],
        axis=1,
    )
    actual = xfcapi.oscill_angles_of_hkls(hkls, 0.0, rmat, bmat, wavelengths)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    for bad in (wavelengths[:1], wavelengths[:, None]):
        with pytest.raises(ValueError, match='wavelength'):
            xfcapi.oscill_angles_of_hkls(hkls, 0.0, rmat, bmat, bad)
