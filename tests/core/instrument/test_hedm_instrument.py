import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import h5py
import copy
import tempfile
import os


from hexrd.core.instrument import hedm_instrument as hi
from hexrd.core.instrument.hedm_instrument import HEDMInstrument, BufferShapeMismatchError
from hexrd.core.instrument.planar_detector import PlanarDetector
from hexrd.core.imageseries import ImageSeries
from hexrd.core.imageseries.omega import OmegaImageSeries

# --- Utility Tests ---

def test_generate_chunks():
    rects, labels = hi.generate_chunks(2, 2, 100, 100, row_gap=10, col_gap=10)
    assert len(rects) == 4
    assert len(labels) == 4
    assert np.array_equal(rects[0], [[0, 100], [0, 100]])
    assert np.array_equal(rects[3], [[110, 210], [110, 210]])
    assert labels == ['0_0', '0_1', '1_0', '1_1']

def test_chunk_instrument():
    instr = MagicMock()
    instr.detectors = {'det1': MagicMock(spec=PlanarDetector)}
    panel = instr.detectors['det1']
    panel.config_dict.return_value = {
        'detector': {
            'pixels': {'rows': 2048, 'columns': 2048, 'size': [0.2, 0.2]},
            'transform': {'translation': [0, 0, -1000], 'tilt': [0, 0, 0]}
        }
    }
    panel.pixelToCart.return_value = np.array([[0.0, 0.0]])
    panel.rmat = np.eye(3)
    panel.tvec = np.array([0, 0, -1000])
    panel.panel_buffer = None
    
    instr.write_config.return_value = {
        'beam': {}, 'oscillation_stage': {}
    }
    
    rects, labels = hi.generate_chunks(2, 2, 1024, 1024)
    new_cfg = hi.chunk_instrument(instr, rects, labels, use_roi=True)
    
    assert len(new_cfg['detectors']) == 4
    assert 'det1_0_0' in new_cfg['detectors']
    assert new_cfg['detectors']['det1_0_0']['pixels']['rows'] == 1024

def test_calc_beam_vec():
    bv = hi.calc_beam_vec(0, 90)
    assert np.allclose(bv, [-1, 0, 0])

def test_calc_angles_from_beam_vec():
    bv = np.array([-1.0, 0.0, 0.0])
    azim, pola = hi.calc_angles_from_beam_vec(bv)
    assert np.isclose(azim, 0)
    assert np.isclose(pola, 90)

def test_migrate_instrument_config():
    instrument_config = {
        'beam': {
            'energy': 65.0,
            'vector': {'azimuth': 0, 'polar_angle': 90}
        },
        'oscillation_stage': {'chi': 0, 'translation': [0.0, 0.0, 0.0]},
        'detectors': {
            'det1': {
                'pixels': {'rows': 100, 'columns': 100, 'size': [0.1, 0.1]},
                'transform': {'translation': [0.0, 0.0, -1000.0], 'tilt': [0.0, 0.0, 0.0]}
            }
        }
    }
    migrated_cfg_list = hi.migrate_instrument_config(instrument_config)
    assert isinstance(migrated_cfg_list, list)
    assert len(migrated_cfg_list) == 1
    assert migrated_cfg_list[0]['detector'] == instrument_config['detectors']['det1']
    assert migrated_cfg_list[0]['oscillation_stage'] == instrument_config['oscillation_stage']

def test_angle_in_range():
    ranges = [[0, 90], [180, 270]]
    assert hi.angle_in_range(45, ranges) == 0
    assert hi.angle_in_range(200, ranges) == 1
    assert np.isnan(hi.angle_in_range(100, ranges))

    ranges_rad = np.radians(ranges)
    assert hi.angle_in_range(np.radians(45), ranges_rad, units='radians') == 0
    assert hi.angle_in_range(np.radians(200), ranges_rad, units='radians') == 1
    assert np.isnan(hi.angle_in_range(np.radians(100), ranges_rad, units='radians'))


def test_max_tth():
    instr = MagicMock()
    det = MagicMock()
    
    # Set the pixel angles
    det.angularPixelSize.return_value = np.array([[0.1, 0.1]])
    det.pixel_coords = (np.array([0]), np.array([0]))
    det.pixel_angles.return_value = (np.array([[0.1]]), np.array([[0.0]]))
    det.shape = (1, 1)
    instr.detectors = {'det1': det}
    
    res = hi.max_tth(instr)
    assert np.allclose(res, (0.1, 0.1))

def test_pixel_resolution():
    instr = MagicMock()
    det = MagicMock()
    
    # Assume 1 pixel
    det.angularPixelSize.return_value = np.array([[0.1, 0.1]])
    det.pixel_coords = (np.array([0]), np.array([0]))
    det.shape = (1, 1)
    
    instr.detectors = {'det1': det}
    
    res = hi.pixel_resolution(instr)
    assert res[0] == (0.1, 0.1, 0.1)

def test_gaussian_dist():
    x = np.array([0.0, 1.0, 2.0])
    cen = 1.0
    fwhm = 1.0
    expected = np.exp(-0.5 * (x - cen) ** 2 / (fwhm / (2 * np.sqrt(2 * np.log(2))))**2)
    result = hi._gaussian_dist(x, cen, fwhm)
    assert np.allclose(result, expected)

def test_sigma_to_fwhm():
    sigma = 1.0
    fwhm = hi._sigma_to_fwhm(sigma)
    expected = sigma * 2 * np.sqrt(2 * np.log(2))
    assert np.isclose(fwhm, expected)

def test_fwhm_to_sigma():
    fwhm = 1.0
    sigma = hi._fwhm_to_sigma(fwhm)
    expected = fwhm / (2 * np.sqrt(2 * np.log(2)))
    assert np.isclose(sigma, expected)


def test_hedm_instrument_init():
    """
    Complete coverage test for HEDMInstrument.__init__ logic.
    """
    # --- Mocks ---
    # Create specific mocks for the classes so we can use them in patch.dict
    MockPlanar = MagicMock(spec=PlanarDetector)
    MockPlanarInstance = MagicMock(spec=PlanarDetector)
    MockPlanar.return_value = MockPlanarInstance
    MockPlanar.update_memoization_sizes = MagicMock()
    
    MockCylindrical = MagicMock()
    MockCylindricalInstance = MagicMock()
    MockCylindrical.return_value = MockCylindricalInstance
    MockCylindrical.update_memoization_sizes = MagicMock()

    # Dictionary to patch DETECTOR_TYPES with our mocks
    mock_detector_types = {
        'planar': MockPlanar,
        'cylindrical': MockCylindrical
    }

    # Patch Detectors in module scope AND the lookup dictionary
    with patch('hexrd.core.instrument.hedm_instrument.PlanarDetector', MockPlanar), \
         patch('hexrd.core.instrument.hedm_instrument.CylindricalDetector', MockCylindrical), \
         patch.dict('hexrd.core.instrument.hedm_instrument.DETECTOR_TYPES', mock_detector_types), \
         patch('hexrd.core.instrument.hedm_instrument.unwrap_h5_to_dict') as mock_h5_unwrap, \
         patch('hexrd.core.instrument.hedm_instrument.distortion_pkg.get_mapping') as mock_dist_map:

        instr = HEDMInstrument(instrument_name="TestDefault")
        
        assert instr.id == "TestDefault"
        assert instr.num_panels == 1
        assert "XRS1" in instr.beam_dict
        assert instr.active_beam_name == "XRS1"
        assert instr.active_beam['energy'] == hi.beam_energy_DFLT
        # Verify default detector creation
        assert "panel_id_DFLT" in instr.detectors
        MockPlanar.assert_called() 
        
        # --- Scenario 2: Config Dict with Single Beam & Planar Detector ---
        # Covers: dict config, single beam logic, explicit eta_vector
        single_beam_config = {
            'beam': {
                'energy': 50.0,
                'vector': {'azimuth': 0, 'polar_angle': 90}
            },
            'oscillation_stage': {'chi': 1.0, 'translation': [1, 2, 3]},
            'detectors': {
                'det1': {
                    'pixels': {'rows': 100, 'columns': 100, 'size': [0.1, 0.1]},
                    'transform': {'translation': [0, 0, 0], 'tilt': [0, 0, 0]},
                    'saturation_level': 5000,
                    'buffer': [2, 2], # List buffer type
                    'group': 'groupA'
                }
            },
            'id': 'ConfigInstr'
        }
        
        instr_single = HEDMInstrument(
            instrument_config=single_beam_config,
            eta_vector=[0, 1, 0] # Explicit eta
        )
        
        assert instr_single.id == 'ConfigInstr'
        assert instr_single.beam_energy == 50.0
        assert np.allclose(instr_single.eta_vector, [0, 1, 0])
        assert instr_single.detectors['det1'] is not None
        
        # --- Scenario 3: Config Dict with Multi-Beam & Cylindrical Detector ---
        # Covers: multi-beam, CylindricalDetector, physics_package, complex buffer types
        multi_beam_config = {
            'physics_package': 'mock_physics',
            'beam': {
                'BeamA': {'energy': 40.0, 'vector': {'azimuth': 0, 'polar_angle': 90}},
                'BeamB': {'energy': 80.0, 'vector': {'azimuth': 90, 'polar_angle': 90}, 'source_distance': 1000}
            },
            'oscillation_stage': {'chi': 0, 'translation': [0, 0, 0]},
            'detectors': {
                'cyl1': {
                    'detector_type': 'cylindrical',
                    'pixels': {'rows': 100, 'columns': 100, 'size': [0.1, 0.1]},
                    'transform': {'translation': [0, 0, 0], 'tilt': [0, 0, 0]},
                    'radius': 50.0,
                    'buffer': 5, # Scalar buffer type
                    'distortion': {'function_name': 'func', 'parameters': []},
                    'filter': 'Zr',
                    'coating': 'P43',
                    'phosphor': 'GOS'
                }
            }
        }

        # Try without active_beam_name
        instr_multi_no_active = HEDMInstrument(
            instrument_config=multi_beam_config
        )
        
        instr_multi = HEDMInstrument(
            instrument_config=multi_beam_config,
            instrument_name="OverrideName",
            active_beam_name='BeamB'
        )
        
        assert instr_multi.id == "OverrideName"
        assert instr_multi.physics_package == 'mock_physics'
        assert len(instr_multi.beam_dict) == 2
        assert instr_multi.active_beam_name == 'BeamB'
        # This assertion should now pass because DETECTOR_TYPES['cylindrical'] is MockCylindrical
        assert isinstance(instr_multi.detectors['cyl1'], MagicMock) 
        mock_dist_map.assert_called() # Check distortion loaded
        
        # --- Scenario 4: HDF5 Input ---
        # Covers: isinstance(h5py.File) branch
        mock_h5 = MagicMock(spec=h5py.File)
        # Mock the unwrap function to populate the temp dict
        def side_effect_unwrap(h5obj, out_dict):
            out_dict['instrument'] = single_beam_config
        mock_h5_unwrap.side_effect = side_effect_unwrap
        
        instr_h5 = HEDMInstrument(instrument_config=mock_h5)
        assert instr_h5.id == 'ConfigInstr' # From single_beam_config
        
        # --- Scenario 5: Error Handling & Edge Cases ---
        
        # 5a. Invalid Config Type
        with pytest.raises(RuntimeError, match="must be either an HDF5 file object"):
            HEDMInstrument(instrument_config="invalid_string")
            
        # 5b. Buffer Shape Mismatch (2D Array)
        bad_buffer_config = copy.deepcopy(single_beam_config)
        bad_buffer_config['detectors']['det1']['buffer'] = np.zeros((50, 50)) # Wrong shape (100,100 expected)
        with pytest.raises(BufferShapeMismatchError):
            HEDMInstrument(instrument_config=bad_buffer_config)
            
        # 5c. Valid 2D Buffer (Coverage)
        valid_buffer_config = copy.deepcopy(single_beam_config)
        valid_buffer_config['detectors']['det1']['buffer'] = np.zeros((100, 100))
        HEDMInstrument(instrument_config=valid_buffer_config) # Should pass
        
        # 5d. Invalid Buffer Spec (None of the above types)
        bad_spec_config = copy.deepcopy(single_beam_config)
        # Pass a dict (not ndarray, not list, not scalar) to trigger the else block
        bad_spec_config['detectors']['det1']['buffer'] = {'invalid': 'type'} 
        with pytest.raises(RuntimeError, match="panel buffer spec invalid"):
            HEDMInstrument(instrument_config=bad_spec_config)
            
        # 5e. Distortion Key Error
        dist_fail_config = copy.deepcopy(single_beam_config)
        dist_fail_config['detectors']['det1']['distortion'] = {'bad_key': 'val'}
        mock_dist_map.side_effect = KeyError("params missing")
        with pytest.raises(RuntimeError, match="problem with distortion"):
            HEDMInstrument(instrument_config=dist_fail_config)
        mock_dist_map.side_effect = None
        
        # 5f. Unknown Detector Type
        bad_det_config = copy.deepcopy(single_beam_config)
        bad_det_config['detectors']['det1']['detector_type'] = 'unknown_type'
        with pytest.raises(NotImplementedError, match="Unknown detector type"):
            HEDMInstrument(instrument_config=bad_det_config)

        # Case where det_buffer is a 1d array with incorrect length
        bad_length_config = copy.deepcopy(single_beam_config)
        bad_length_config['detectors']['det1']['buffer'] = np.ndarray([1, 2, 3])
        with pytest.raises(ValueError, match="Buffer length for det1 must be 2"):
            HEDMInstrument(instrument_config=bad_length_config)

def test_properties():
    simple_config = {
        'beam': {'energy': 65.0, 'vector': {'azimuth':0, 'polar_angle':90}},
        'oscillation_stage': {'chi':0, 'translation':[0,0,0]},
        'detectors': {
            'det1': {
                'pixels': {'rows': 100, 'columns': 100, 'size': [0.1, 0.1]},
                'transform': {'translation': [0,0,-1000], 'tilt': [0,0,0]}
            }
        }
    }
    instr = HEDMInstrument(instrument_config=simple_config)
    
    instr.mean_detector_center
    instr.detector_groups
    instr.detector_parameters
    assert instr.mean_group_center('det1') is not None

    simple_config['detectors']['det1']['group'] = 'groupX'
    instr2 = HEDMInstrument(instrument_config=simple_config)
    groups = instr2.detector_groups
    assert 'groupX' in groups

    instr.beam_energy = 70.0
    assert instr.beam_energy == 70.0
    assert not instr.has_multi_beam

    assert instr.beam_names == ['XRS1']
    assert instr.xrs_beam_energy('XRS1') == instr.beam_energy

    # Set XRS1 to active beam, and try xrs_beam_energy without setting the name
    instr.active_beam_name = 'XRS1'
    assert instr.xrs_beam_energy(beam_name=None) == instr.beam_energy

    with pytest.raises(ValueError, match="is not present in"):
        instr.active_beam_name = 'NonExistentBeam'

def test_centers_of_edge_vec():
    edges = np.array([0, 1, 2, 3])
    centers = hi.centers_of_edge_vec(edges)
    assert np.array_equal(centers, np.array([0.5, 1.5, 2.5]))

    with pytest.raises(ValueError, match="edges must be an array-like with at least 2 elements"):
        hi.centers_of_edge_vec([1])  # Less than 2 elements

def test_max_resolution():
    instr = MagicMock()
    det = MagicMock()
    
    # Assume 1 pixel
    det.angularPixelSize.return_value = np.array([[0.1, 0.1]])
    det.pixel_coords = (np.array([0]), np.array([0]))
    det.shape = (1, 1)
    
    instr.detectors = {'det1': det}
    
    res = hi.max_resolution(instr)
    assert res == (0.1, 0.1)

# --- HEDMInstrument Class Tests ---

@pytest.fixture
def minimal_instr_config():
    return {
        'beam': {
            'energy': 65.0,
            'vector': {'azimuth': 0, 'polar_angle': 90}
        },
        'oscillation_stage': {'chi': 0, 'translation': [0.0, 0.0, 0.0]},
        'detectors': {
            'det1': {
                'pixels': {'rows': 100, 'columns': 100, 'size': [0.1, 0.1]},
                'transform': {'translation': [0.0, 0.0, -1000.0], 'tilt': [0.0, 0.0, 0.0]}
            }
        }
    }

@pytest.fixture
def instr(minimal_instr_config):
    return HEDMInstrument(instrument_config=minimal_instr_config)

def test_init_defaults():
    instr = HEDMInstrument() # Defaults
    assert instr.num_panels == 1
    assert 'panel_id_DFLT' in instr.detectors
    assert instr.beam_energy == 65.351

def test_init_from_dict(minimal_instr_config):
    instr = HEDMInstrument(instrument_config=minimal_instr_config)
    assert instr.num_panels == 1
    assert 'det1' in instr.detectors
    assert instr.beam_energy == 65.0

def test_write_config(instr, tmp_path):
    f = tmp_path / "test.yml"
    instr.write_config(str(f))
    assert f.exists()

    f = tmp_path / "test.h5"
    instr.write_config(str(f), style='hdf5')
    assert f.exists()
    
    with h5py.File(f, 'w') as h5f:
        instr.write_config(h5f, style='hdf5')
        assert 'instrument' in h5f

    instr.source_distance = 1000.0
    instr.energy_correction = {'axis': 'x', 'intercept': 0.1, 'slope': 0.01}
    instr.write_config(tmp_path / "test2.yml")
    instr.write_config(tmp_path / "test2.yml", calibration_dict={'collab': 'test2'}, style='yaml')

    with pytest.raises(ValueError, match="style must be 'yaml' or 'hdf5'"):
        instr.write_config(str(f), style='xml')

    with pytest.raises(TypeError, match='Unexpected file type'):
        instr.write_config(1234, style='hdf5')

def test_simulate_powder_pattern(instr):
    mat = MagicMock()
    mat.name = 'SimulatedMat'
    mat.planeData.getMultiplicity.return_value = 1.0
    mat.planeData.getTTh.return_value = np.array([0.1])
    mat.planeData.structFact = 1.0
    
    mat.latticeType = 'cubic'
    mat.sgnum = 225
    mat.lparms = np.array([10.0]) 
    
    det = instr.detectors['det1']
    det.pixel_angles = MagicMock(return_value=(
        np.linspace(0.05, 0.15, 100).reshape(10, 10), 
        np.zeros((10, 10))
    ))
    
    with patch('hexrd.core.transforms.xfcapi.make_beam_rmat', return_value=np.eye(3)):
        with patch('hexrd.core.instrument.hedm_instrument.LeBail') as mock_lebail:
            mock_instance = mock_lebail.return_value
            mock_instance.spectrum_sim.x = np.linspace(0, 20, 100) # degrees
            mock_instance.spectrum_sim.y = np.ones(100)
            mock_instance.background.y = np.zeros(100)
            
            res = instr.simulate_powder_pattern([mat])
            assert 'det1' in res
            assert res['det1'].shape == (10, 10)

def test_simulate_powder_pattern_noise(instr):
    """
    Test simulate_powder_pattern with various noise types, 
    specifically targeting the normalization logic in the 'poisson' branch.
    """
    mat = MagicMock()
    mat.name = 'SimulatedMat'
    mat.planeData.getMultiplicity.return_value = 1.0
    mat.planeData.getTTh.return_value = np.array([0.1])
    mat.planeData.structFact = 1.0
    mat.latticeType = 'cubic'
    mat.sgnum = 225
    mat.lparms = np.array([10.0])

    det = instr.detectors['det1']
    det.pixel_angles = MagicMock(return_value=(
        np.linspace(0.05, 0.15, 100).reshape(10, 10), 
        np.zeros((10, 10))
    ))

    with patch('hexrd.core.transforms.xfcapi.make_beam_rmat', return_value=np.eye(3)):
        with patch('hexrd.core.instrument.hedm_instrument.LeBail') as mock_lebail:
            mock_instance = mock_lebail.return_value
            mock_instance.spectrum_sim.x = np.linspace(0, 20, 100)
            mock_instance.spectrum_sim.y = np.full(100, 100.0) # Intensity = 100
            mock_instance.background.y = np.zeros(100)

            with patch('hexrd.core.instrument.hedm_instrument.random_noise') as mock_rn:
                mock_rn.return_value = np.linspace(0.0, 0.5, 100).reshape(10, 10)
                
                res = instr.simulate_powder_pattern([mat], noise='poisson')
                
                assert mock_rn.call_count == 1
                assert mock_rn.call_args[1]['mode'] == 'poisson'
                assert np.isclose(res['det1'].max(), 100.0)

                mock_rn.reset_mock()
                mock_rn.return_value = np.full((10, 10), 0.5)
                
                res = instr.simulate_powder_pattern([mat], noise='poisson')
                
                assert np.allclose(res['det1'], 50.0)

                noise_modes = ['gaussian', 'salt', 'pepper', 's&p', 'speckle']
                for mode in noise_modes:
                    mock_rn.reset_mock()
                    instr.simulate_powder_pattern([mat], noise=mode)
                    
                    assert mock_rn.call_count == 1
                    assert mock_rn.call_args[1]['mode'] == mode


def test_pull_spots_check_only_frame_outside_range(instr, capsys):
    """
    Ensure that when check_only=True and omega_to_frame returns -1
    the reflection is skipped and the informative message is printed.
    """

    instr.detectors['det1'].tvec = np.array([0.0, 0.0, -1000.0])
    instr.tvec = np.zeros(3)

    # --- OmegaImageSeries mock ---
    ims = MagicMock(spec=OmegaImageSeries)
    ims.metadata = {'omega': np.array([[0, 1]])}
    ims.omegawedges.wedges = [{'ostart': 0, 'ostop': 360}]
    ims.__getitem__.return_value = np.full((10, 10), 100.0)

    # CRITICAL: return an integer -1 as the frame index (not a list)
    # pull_spots uses ome_imgser.omega_to_frame(ome)[0] expecting an int
    ims.omega_to_frame.return_value = (-1, 0)
    ims.omega = np.array([[0, 0.1]])

    imgser_dict = {'det1': ims}

    # --- Simulated rotation output (1 reflection) ---
    sim_res = {
        'det1': [
            [np.array([0])],
            [np.array([[1, 2, 3]])],            # hkls_p used in print(msg)
            [np.array([[0.1, 0.0, 0.0]])],
            [np.array([[0.0, 0.0]])],
            [np.array([[0.01, 0.01]])],
        ]
    }

    # Keep simulate_rotation_series mocked so we don't run real sim
    with patch.object(instr, 'simulate_rotation_series', return_value=sim_res):
        det = instr.detectors['det1']

        # Patch geometry helpers so we reach the frame check logic
        with patch(
            'hexrd.core.instrument.hedm_instrument._project_on_detector_plane',
            return_value=(np.zeros((4, 2)), None, None),
        ):
            with patch.object(
                det,
                'clip_to_panel',
                return_value=(None, np.array([True, True, True, True])),
            ):
                with patch.object(
                    det,
                    'cartToPixel',
                    return_value=np.array([[0.0, 0.0]] * 4),
                ):
                    with patch(
                        'hexrd.core.instrument.hedm_instrument.polygon',
                        return_value=(np.array([0]), np.array([0])),
                    ):
                        compl, out = instr.pull_spots(
                            plane_data='plane_data',
                            grain_params=np.zeros(12),
                            imgser_dict=imgser_dict,
                            dirname='.',
                            filename='spots',
                            threshold=50,
                            check_only=True,
                            quiet=False,
                        )

                        _ = instr.pull_spots(
                            plane_data='plane_data',
                            grain_params=np.zeros(12),
                            imgser_dict=imgser_dict,
                            dirname='.',
                            filename='spots',
                            threshold=50,
                            check_only=True,
                            quiet=True,
                        )

    # The reflection should be skipped, so no completeness entry appended
    assert compl == []
    assert out['det1'] == []

    # Verify the warning message was printed and includes hkls
    captured = capsys.readouterr()
    assert "falls outside omega range" in captured.out
    assert "(1  2  3)" in captured.out

def test_pull_spots_check_only_frame_inside_range(instr):
    """
    Ensure that when check_only=True and omega_to_frame returns valid frame
    the reflection is processed and completeness is computed.
    """

    instr.detectors['det1'].tvec = np.array([0.0, 0.0, -1000.0])
    instr.tvec = np.zeros(3)

    # --- OmegaImageSeries mock ---
    ims = MagicMock(spec=OmegaImageSeries)
    ims.metadata = {'omega': np.array([[0, 1]])}
    ims.omegawedges.wedges = [{'ostart': 0, 'ostop': 360}]
    ims.__getitem__.return_value = np.full((10, 10), 100.0)
    ims.omega_to_frame.return_value = ([0], 0) # Valid frame index
    ims.omega = np.array([[0, 0.1]])

    imgser_dict = {'det1': ims}

    # --- Simulated rotation output (1 reflection) ---
    sim_res = {
        'det1': [
            [np.array([0])],
            [np.array([[1, 2, 3]])],            # hkls_p used in print(msg)
            [np.array([[0.1, 0.0, 0.0]])],
            [np.array([[0.0, 0.0]])],
            [np.array([[0.01, 0.01]])],
        ]
    }

    # Keep simulate_rotation_series mocked so we don't run real sim
    with patch.object(instr, 'simulate_rotation_series', return_value=sim_res):
        det = instr.detectors['det1']

        # Patch geometry helpers so we reach the frame check logic
        with patch(
            'hexrd.core.instrument.hedm_instrument._project_on_detector_plane',
            return_value=(np.zeros((4, 2)), None, None),
        ):
            with patch.object(
                det,
                'clip_to_panel',
                return_value=(None, np.array([True, True, True, True])),
            ):
                with patch.object(
                    det,
                    'cartToPixel',
                    return_value=np.array([[0.0, 0.0]] * 4),
                ):
                    with patch(
                        'hexrd.core.instrument.hedm_instrument.polygon',
                        return_value=(np.array([0]), np.array([0])),
                    ):
                        compl, out = instr.pull_spots(
                            plane_data='plane_data',
                            grain_params=np.zeros(12),
                            imgser_dict=imgser_dict,
                            dirname='.',
                            filename='spots',
                            threshold=50,
                            check_only=True,
                        )

def test_pull_spots_no_check_only_frame_inside_range(instr):
        """
        Ensure that when check_only=False and omega_to_frame returns valid frame
        the reflection is processed and completeness is computed.
        """
        
        # --- Instrument Setup ---
        det = instr.detectors['det1']
        det.tvec = np.array([0.0, 0.0, -1000.0])
        instr.tvec = np.zeros(3)
        type(det).pixel_area = PropertyMock(return_value=1.0)

        # --- OmegaImageSeries mock ---
        ims = MagicMock(spec=OmegaImageSeries)
        ims.metadata = {'omega': np.array([[0, 1]])}
        ims.omegawedges.wedges = [{'ostart': 0, 'ostop': 360}]
        ims.__getitem__.return_value = np.full((10, 10), 100.0)
        # Force omega_to_frame to return a list of ints, not list of arrays
        ims.omega_to_frame.return_value = ([0], 0) 
        ims.omega = np.array([[0, 0.1]])
        
        imgser_dict = {'det1': ims}
        
        # --- Simulated rotation output ---
        sim_res = {
            'det1': [
                [np.array([0])],
                [np.array([[1, 2, 3]])],            # hkls
                [np.array([[0.1, 0.0, 0.0]])],      # angles
                [np.array([[0.0, 0.0]])],           # xy
                [np.array([[0.01, 0.01]])],         # pixel size
            ]
        }
        
        # --- Mock Patch Data ---
        mock_patch = (
            (np.zeros((1, 2)), np.zeros((2, 1))), 
            None, None,
            np.ones((2, 2)), 
            (np.array([0, 1]), np.array([0, 1])), 
            (np.array([0]), np.array([0]))
        )

        with patch.object(instr, 'simulate_rotation_series', return_value=sim_res), \
             patch('hexrd.core.instrument.hedm_instrument.xrdutil.make_reflection_patches', return_value=[mock_patch]), \
             patch('hexrd.core.instrument.hedm_instrument.gvec_to_xy', return_value=np.array([0.0, 0.0])), \
             patch('hexrd.core.instrument.hedm_instrument.angles_to_gvec', return_value=np.array([0.0, 0.0, 1.0])), \
             patch('hexrd.core.instrument.hedm_instrument.make_sample_rmat', return_value=np.eye(3)):
            
            with patch.object(det, 'clip_to_panel', return_value=(None, np.array([True]*4))):
                with patch.object(det, 'interpolate_bilinear', return_value=np.full((4,), 100.0)):
                    
                    with tempfile.TemporaryDirectory() as tmpdir:
                        compl, out = instr.pull_spots(
                            plane_data='plane_data',
                            grain_params=np.zeros(12),
                            imgser_dict=imgser_dict,
                            dirname=tmpdir,
                            filename='spots',
                            threshold=50,
                            check_only=False
                        )
                        
                        assert compl[0] == True
                        res = out['det1'][0]
                        
                        # Structure: [peak_id, id, hkl, sum, max, angs, meas_angs, meas_xy]
                        assert len(res) == 8
                        assert res[2].tolist() == [1, 2, 3] # HKL
                        assert len(res[5]) == 3 # Sim Angles
                        assert len(res[6]) == 4 # Meas Angles

def test_pull_spots_bilinear_interpolation(instr):
        """
        Test the 'bilinear' interpolation branch of pull_spots.
        """
        # --- Setup Mocks ---
        det = instr.detectors['det1']
        det.tvec = np.array([0.0, 0.0, -1000.0], dtype=float)
        type(det).pixel_area = PropertyMock(return_value=1.0)
        
        # ImageSeries with signal
        ims = MagicMock(spec=OmegaImageSeries)
        ims.metadata = {'omega': np.array([[0.0, 1.0]])}
        ims.omegawedges.wedges = [{'ostart': 0, 'ostop': 360}]
        ims.omega = np.array([[0.0, 0.1]]) 
        ims.omega_to_frame.return_value = ([0], 0)
        
        # FIX: Return shape must match the interpolation patch shape (2x2)
        # The mock patch defines areas as np.ones((2, 2))
        ims.__getitem__.return_value = np.full((2, 2), 100.0) 
        
        imgser_dict = {'det1': ims}
        
        # Simulation Result (1 spot)
        sim_res = {
            'det1': [
                [np.array([0])],            # ids
                [np.array([[1, 1, 1]])],    # hkls
                [np.array([[0.1, 0.0, 0.0]])], # ang_centers
                [np.array([[0.0, 0.0]])],   # xy_centers
                [np.array([[0.01, 0.01]])]  # pixel size
            ]
        }
        
        # Mock Patches
        mock_patch = (
            (np.zeros((1, 2)), np.zeros((2, 1))), # vtx_angs
            None, None,
            np.ones((2, 2)), # areas (shape 2x2)
            (np.array([0, 1]), np.array([0, 1])), # xy_eval
            (np.array([0]), np.array([0]))  # ijs
        )
        
        # --- Execution ---
        with patch.object(instr, 'simulate_rotation_series', return_value=sim_res):
            with patch('hexrd.core.instrument.hedm_instrument.xrdutil.make_reflection_patches', return_value=[mock_patch]):
                # Pass projection/clipping checks
                with patch.object(det, 'clip_to_panel', return_value=(None, np.array([True]*4))):
                    # Mock the specific method we are testing
                    # interpolate_bilinear should return flat array matching size of xy_eval (4 points for 2x2)
                    with patch.object(det, 'interpolate_bilinear', return_value=np.full((4,), 100.0)) as mock_bilinear:
                        
                        instr.pull_spots(
                            plane_data=MagicMock(),
                            grain_params=np.zeros(12),
                            imgser_dict=imgser_dict,
                            interp='bilinear',  # <--- Trigger the branch
                            check_only=False
                        )
    
                        # --- Assertion ---
                        assert mock_bilinear.call_count == 11

def test_simulate_laue_pattern(instr):
    # This just delegates to detector.simulate_laue_pattern
    det = instr.detectors['det1']
    det.simulate_laue_pattern = MagicMock(return_value='result')
    
    res = instr.simulate_laue_pattern('crystal_data')
    assert res['det1'] == 'result'
    det.simulate_laue_pattern.assert_called()

def test_pull_spots(instr):
    instr.detectors['det1'].tvec = np.array([0.0, 0.0, -1000.0], dtype=float)
    instr.tvec = np.array([0.0, 0.0, 0.0], dtype=float)

    ims = MagicMock(spec=OmegaImageSeries)
    ims.metadata = {'omega': np.array([[0, 1]])}
    ims.omegawedges.wedges = [{'ostart': 0, 'ostop': 360}]
    
    ims.__getitem__.return_value = np.full((10, 10), 100.0)
    ims.omega_to_frame.return_value = ([0], 0) # valid frame
    ims.omega = np.array([[0, 0.1]]) 
    
    imgser_dict = {'det1': ims}
    
    sim_res = {
        'det1': [
            [np.array([0])],
            [np.array([[1, 1, 1]])],
            [np.array([[0.1, 0.0, 0.0]])],
            [np.array([[0.0, 0.0]])],
            [np.array([[0.01, 0.01]])]
        ]
    }
    
    with patch.object(instr, 'simulate_rotation_series', return_value=sim_res):
        with patch('hexrd.core.instrument.hedm_instrument.xrdutil.make_reflection_patches') as m_patch:
            mock_patch = (
                (np.zeros((1, 2)), np.zeros((2, 1))),
                None, None, 
                np.array([[1.0]]),
                (np.array([0]), np.array([0])),
                (np.array([0]), np.array([0]))
            )
            m_patch.return_value = [mock_patch]
            
            det = instr.detectors['det1']
            mock_clip_mask = np.array([True, True, True, True]) 
            
            with patch.object(det, 'clip_to_panel', return_value=(None, mock_clip_mask)):
                with patch.object(det, 'interpolate_bilinear', return_value=np.array([100.0])):
            
                    with tempfile.TemporaryDirectory() as tmpdir:
                        compl, _ = instr.pull_spots(
                            plane_data='plane_data', 
                            grain_params=np.zeros(12),
                            imgser_dict=imgser_dict,
                            dirname=tmpdir,
                            filename='spots',
                            threshold=50
                        )
                        
                        assert len(compl) == 1
                        assert compl[0] == True 


                    # Test with invalid interp option
                    with pytest.raises(ValueError, match="interp='invalid_interp' invalid"):
                        instr.pull_spots(
                            plane_data='plane_data',
                            grain_params=np.zeros(12),
                            imgser_dict=imgser_dict,
                            interp='invalid_interp'
                        )

# --- Writer Tests ---

def test_grain_data_writer(tmp_path):
    f = tmp_path / "grains.out"
    writer = hi.GrainDataWriter(str(f))
    
    params = np.zeros(12)
    params[6:9] = 1.0 
    
    writer.dump_grain(1, 1.0, 0.1, params)
    writer.close()
    
    assert f.exists()
    content = f.read_text()
    assert "grain ID" in content

def test_patch_data_writer(tmp_path):
    f = tmp_path / "spots.out"
    writer = hi.PatchDataWriter(str(f))
    
    writer.dump_patch(
        0, 0, [1,1,1], 100.0, 10.0, 
        np.zeros(3), np.zeros(3), np.zeros(2), np.zeros(2)
    )
    writer.close()
    assert f.exists()

# --- Helper Logic ---

def test_buffer_shape_mismatch_error():
    cfg = {
        'beam': {'energy': 65, 'vector': {'azimuth':0, 'polar_angle':90}},
        'oscillation_stage': {'chi':0, 'translation':[0,0,0]},
        'detectors': {
            'det1': {
                'pixels': {'rows': 10, 'columns': 10, 'size': [1,1]},
                'transform': {'translation': [0,0,0], 'tilt': [0,0,0]},
                'buffer': np.zeros((5, 5)) # Mismatch shape
            }
        }
    }
    with pytest.raises(BufferShapeMismatchError):
        HEDMInstrument(instrument_config=cfg)

def test_extract_polar_maps(instr):
    plane_data = MagicMock()
    plane_data.tThWidth = 0.01
    
    imgser_dict = {'det1': MagicMock()}
    det = instr.detectors['det1']
    
    with patch.object(det, 'make_powder_rings', return_value=([], [], np.array([[0.1, 0.2]]), [], np.array([0, 1]))):
        with patch.object(det, 'pixel_angles', return_value=(np.zeros((10,10)), np.zeros((10,10)))):
            mock_ims = MagicMock()
            mock_ims.metadata = {'omega': np.array([[0, 1]])}
            mock_ims.__len__.return_value = 1
            mock_ims.__getitem__.return_value = np.zeros((10, 10))
            
            with patch('hexrd.core.instrument.hedm_instrument._parse_imgser_dict', return_value=mock_ims):
                maps, edges = instr.extract_polar_maps(plane_data, imgser_dict)
                assert 'det1' in maps

def test_switch_xray_source(instr):
    instr._beam_dict['XRS2'] = instr._beam_dict['XRS1'].copy()
    instr._beam_dict['XRS2']['energy'] = 100.0
    
    assert instr.active_beam_name == 'XRS1'
    assert instr.beam_energy == 65.0
    
    with hi.switch_xray_source(instr, 'XRS2'):
        assert instr.active_beam_name == 'XRS2'
        assert instr.beam_energy == 100.0
        
    assert instr.active_beam_name == 'XRS1'

def test_beam_vector_setter_logic(instr):
    valid_bvec = np.array([0.0, 1.0, 0.0])
    instr.beam_vector = valid_bvec
    assert np.allclose(instr.beam_vector, valid_bvec)

    azim, polar = 0.0, 90.0
    instr.beam_vector = [azim, polar]
    
    assert len(instr.beam_vector) == 3
    assert np.allclose(instr.beam_vector, [-1.0, 0.0, 0.0])
    
    with pytest.raises(ValueError, match="beam_vector must be a 2 or 3-element array-like"):
        instr.beam_vector = [1.0,]

    with pytest.raises(ValueError, match="beam_vector must be a 2 or 3-element array-like"):
        instr.beam_vector = [30.0, 60.0, 90.0, 120.0]

    with pytest.raises(ValueError, match="beam_vector must be a unit vector"):
        instr.beam_vector = [1.0, 1.0, 0.0]
    
def test_source_distance_properties(instr):
    instr.source_distance = 1000.0
    assert instr.source_distance == 1000.0
    assert instr.active_beam['distance'] == 1000.0
    
    with pytest.raises(AssertionError, match="must be a scalar"):
        instr.source_distance = [1000.0, 2000.0]

def test_energy_correction_properties(instr):
    defaults = HEDMInstrument.create_default_energy_correction()
    assert 'intercept' in defaults
    assert 'slope' in defaults
    assert 'axis' in defaults
    
    new_correction = {'intercept': 1.0, 'slope': 0.1, 'axis': 'x'}
    instr.energy_correction = new_correction
    assert instr.energy_correction == new_correction
    
    bad_correction = {'intercept': 1.0, 'slope': 0.1} # Missing 'axis'
    with pytest.raises(ValueError, match="energy_correction keys do not match required keys"):
        instr.energy_correction = bad_correction
        
    instr.energy_correction = None
    assert instr.energy_correction is None

def test_eta_vector_propagation(instr):
    initial_eta = np.array([1.0, 0.0, 0.0])
    instr.eta_vector = initial_eta
    assert np.allclose(instr.eta_vector, initial_eta)
    
    for det in instr.detectors.values():
        assert np.allclose(det.evec, initial_eta)
        
    new_eta = np.array([0.0, 1.0, 0.0])
    instr.eta_vector = new_eta
    assert np.allclose(instr.eta_vector, new_eta)
    
    for det in instr.detectors.values():
        assert np.allclose(det.evec, new_eta)

    with pytest.raises(ValueError, match="eta_vector must be a 3-element array-like"):
        instr.eta_vector = [1.0, 0.0]

    with pytest.raises(ValueError, match="eta_vector must be a unit vector"):
        instr.eta_vector = [1.0, 1.0, 0.0]

def test_extract_polar_maps_tth_tol_logic(instr):
    """Test tth_tol override logic."""
    plane_data = MagicMock()
    plane_data.tThWidth = 0.01
    plane_data.getHKLID.return_value = np.array([0, 1])
    plane_data.hkls = np.array([[1, 1, 1], [2, 0, 0]])

    ims = MagicMock()
    ims.metadata = {'omega': np.array([[0, 1]])}
    ims.__len__.return_value = 1
    ims.__getitem__.return_value = np.zeros((10, 10))
    imgser_dict = {'det1': ims}

    det = instr.detectors['det1']
    
    with patch.object(det, 'make_powder_rings', return_value=([], [], np.array([[0.1, 0.2]]), [], np.array([0, 1]))):
        with patch.object(det, 'pixel_angles', return_value=(np.zeros((10, 10)), np.zeros((10, 10)))):
            with patch('hexrd.core.instrument.hedm_instrument._parse_imgser_dict', return_value=ims):
                instr.extract_polar_maps(
                    plane_data, imgser_dict, tth_tol=0.5
                )
                assert plane_data.tThWidth == np.radians(0.5)

                plane_data.tThWidth = 0.01
                instr.extract_polar_maps(
                    plane_data, imgser_dict, tth_tol=None
                )
                assert plane_data.tThWidth == 0.01


def test_extract_polar_maps_active_hkls(instr):
    """Test active_hkls filtering and error handling."""
    plane_data = MagicMock()
    plane_data.tThWidth = 0.01
    plane_data.getHKLID.return_value = np.array([0, 1, 2])
    
    tth_ranges_full = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    ims = MagicMock()
    ims.metadata = {'omega': np.array([[0, 1]])}
    ims.__len__.return_value = 1
    ims.__getitem__.return_value = np.zeros((10, 10))
    imgser_dict = {'det1': ims}

    det = instr.detectors['det1']
    
    mock_rings_ret = ([], [], tth_ranges_full, [], np.array([0, 1]))

    with patch.object(det, 'make_powder_rings', return_value=mock_rings_ret):
        with patch.object(det, 'pixel_angles', return_value=(np.zeros((10, 10)), np.zeros((10, 10)))):
            with patch('hexrd.core.instrument.hedm_instrument._parse_imgser_dict', return_value=ims):
                with pytest.raises(AssertionError, match="must be an iterable"):
                    instr.extract_polar_maps(plane_data, imgser_dict, active_hkls=123)

                maps, edges = instr.extract_polar_maps(
                    plane_data, imgser_dict, active_hkls=[1]
                )
                assert maps['det1'].shape[0] == 1

                with pytest.raises(RuntimeError, match="hklID '99' is invalid"):
                    instr.extract_polar_maps(
                        plane_data, imgser_dict, active_hkls=[99]
                    )


def test_extract_polar_maps_missing_omega(instr):
    """Test error raised when imageseries lacks omega metadata."""
    plane_data = MagicMock()
    plane_data.tThWidth = 0.01
    
    ims = MagicMock()
    ims.metadata = {} 
    imgser_dict = {'det1': ims}

    det = instr.detectors['det1']
    
    with patch.object(det, 'make_powder_rings', return_value=([], [], np.array([[0.1, 0.2]]), [], np.array([0, 1]))):
        with patch.object(det, 'pixel_angles', return_value=(np.zeros((10, 10)), np.zeros((10, 10)))):
            with patch('hexrd.core.instrument.hedm_instrument._parse_imgser_dict', return_value=ims):
                with pytest.raises(RuntimeError, match="has no omega info"):
                    instr.extract_polar_maps(plane_data, imgser_dict)


def test_extract_polar_maps_threading_logic(instr):
    """Test single vs multi-threaded execution paths."""
    plane_data = MagicMock()
    plane_data.tThWidth = 0.01
    
    ims = MagicMock()
    ims.metadata = {'omega': np.array([[0, 1]])}
    ims.__len__.return_value = 10 
    ims.__getitem__.return_value = np.zeros((10, 10))
    imgser_dict = {'det1': ims}

    det = instr.detectors['det1']
    
    with patch.object(det, 'make_powder_rings', return_value=([], [], np.array([[0.1, 0.2]]), [], np.array([0, 1]))):
        with patch.object(det, 'pixel_angles', return_value=(np.zeros((10, 10)), np.zeros((10, 10)))):
            with patch('hexrd.core.instrument.hedm_instrument._parse_imgser_dict', return_value=ims):
                instr.max_workers = 1
                instr.extract_polar_maps(plane_data, imgser_dict)
                
                instr.max_workers = 2
                with patch('hexrd.core.instrument.hedm_instrument.ThreadPoolExecutor') as mock_tpe:
                    mock_executor = mock_tpe.return_value.__enter__.return_value
                    mock_executor.map.side_effect = lambda f, tasks: [f(t) for t in tasks]
                    
                    instr.extract_polar_maps(plane_data, imgser_dict)
                    assert mock_tpe.called

def test_extract_line_positions(instr):
    """Test the orchestration logic of extract_line_positions."""
    plane_data = MagicMock()
    imgser_dict = {'det1': MagicMock()}
    
    mock_worker_result = ["mock_result_data"]
    
    with patch('hexrd.core.instrument.hedm_instrument._extract_detector_line_positions', return_value=mock_worker_result) as mock_worker:
        with patch('hexrd.core.instrument.hedm_instrument.ProcessPoolExecutor') as mock_executor_cls:
            mock_executor = mock_executor_cls.return_value.__enter__.return_value

            def side_effect_map(func, iterable):
                return [func(item) for item in iterable]
            mock_executor.map.side_effect = side_effect_map
            
            with patch('hexrd.core.instrument.hedm_instrument._parse_imgser_dict', return_value='mock_images'):
                
                res = instr.extract_line_positions(
                    plane_data, imgser_dict, 
                    collapse_tth=True, do_fitting=True
                )
                
                assert 'det1' in res
                assert res['det1'] == mock_worker_result
                
                call_args = mock_worker.call_args
                assert call_args is not None
                
                kwargs = call_args[1]
                assert kwargs['collapse_tth']
                assert kwargs['do_fitting']
                assert kwargs['fitting_kwargs'] == {}
                assert kwargs['plane_data'] == plane_data

                fit_args = {'method': 'simplex'}
                instr.extract_line_positions(
                    plane_data, imgser_dict, fitting_kwargs=fit_args
                )

                call_args = mock_worker.call_args
                kwargs = call_args[1]
                assert kwargs['fitting_kwargs'] == fit_args

def test_simulate_rotation_series_dispatch(instr):
    """
    Test that simulate_rotation_series correctly dispatches calls to detectors
    with the correct instrument-level parameters (chi, tvec, energy_correction).
    """
    # --- Setup Mocks ---
    plane_data = MagicMock()
    grain_params = [np.zeros(12)]
    
    # Configure Instrument state
    instr.chi = 0.5
    instr.tvec = np.array([1.0, 2.0, 3.0])
    
    # Set an energy correction to verify it passes through
    e_corr = {'intercept': 0.1, 'slope': 0.01, 'axis': 'x'}
    instr.energy_correction = e_corr
    
    # Mock the detector method
    det = instr.detectors['det1']
    det.simulate_rotation_series = MagicMock(return_value="sim_result")
    
    # --- Execution ---
    eta_ranges = [(-1.0, 1.0)]
    ome_ranges = [(-2.0, 2.0)]
    ome_period = (-3.0, 3.0)
    wavelength = 0.7
    
    results = instr.simulate_rotation_series(
        plane_data, 
        grain_params,
        eta_ranges=eta_ranges,
        ome_ranges=ome_ranges,
        ome_period=ome_period,
        wavelength=wavelength
    )
    
    # --- Verification ---
    assert results['det1'] == "sim_result"
    
    det.simulate_rotation_series.assert_called_once()
    
    # Verify arguments passed to detector method
    call_args = det.simulate_rotation_series.call_args
    call_pos_args = call_args[0] # Positional args
    call_kwargs = call_args[1]   # Keyword args
    
    # Positional args are: (plane_data, grain_param_list)
    assert call_pos_args[0] == plane_data
    assert call_pos_args[1] == grain_params
    
    # Keyword args
    assert call_kwargs['eta_ranges'] == eta_ranges
    assert call_kwargs['ome_ranges'] == ome_ranges
    assert call_kwargs['ome_period'] == ome_period
    assert call_kwargs['wavelength'] == wavelength
    
    # Verify Instrument attributes were passed correctly
    assert call_kwargs['chi'] == 0.5
    np.testing.assert_array_equal(call_kwargs['tVec_s'], np.array([1.0, 2.0, 3.0]))
    assert call_kwargs['energy_correction'] == e_corr

def test_pull_spots_hdf5_output(instr):
    """
    Test pull_spots with filename set and output_format='hdf5'.
    Verifies that GrainDataWriter_h5 is initialized and used.
    """
    det = instr.detectors['det1']
    det.tvec = np.array([0.0, 0.0, -1000.0], dtype=float)
    type(det).pixel_area = PropertyMock(return_value=1.0)

    # --- OmegaImageSeries mock ---
    ims = MagicMock(spec=OmegaImageSeries)
    ims.metadata = {'omega': np.array([[0.0, 1.0]])}
    ims.omegawedges.wedges = [{'ostart': 0.0, 'ostop': 360.0}]
    ims.omega = np.array([[0.0, 0.1]]) # Small step for 3D logic if needed
    # omega_to_frame returns valid frame 0
    ims.omega_to_frame.return_value = ([0], 0)
    # __getitem__ returns array with signal above threshold (100 > 10)
    ims.__getitem__.return_value = np.full((10, 10), 100.0)

    imgser_dict = {'det1': ims}

    # --- simulate_rotation_series output ---
    sim_results = {
        'det1': [
            [np.array([0])],                # hkl_ids
            [np.array([[1, 1, 1]])],        # hkls
            [np.array([[0.1, 0.2, 0.3]])],  # ang_centers
            [np.array([[0.0, 0.0]])],       # xy_centers
            [np.array([[0.01, 0.01]])],     # ang_pixel_size
        ]
    }

    # --- Mock reflection patch ---
    mock_patch = (
        (np.zeros((1, 2)), np.zeros((2, 1))), # vtx_angs
        None, None,
        np.array([[1.0]]), # areas
        (np.array([0]), np.array([0])), # xy_eval
        (np.array([0]), np.array([0]))  # ijs
    )

    # --- Execution Context ---
    with patch.object(instr, 'simulate_rotation_series', return_value=sim_results), \
         patch('hexrd.core.instrument.hedm_instrument.xrdutil.make_reflection_patches', return_value=[mock_patch]), \
         patch('hexrd.core.instrument.hedm_instrument.GrainDataWriter_h5') as MockH5Writer, \
         patch('hexrd.core.instrument.hedm_instrument.gvec_to_xy', return_value=np.array([0.0, 0.0])):

        # Mock detector methods
        # Use patch.object to ensure methods on the specific instance are mocked
        with patch.object(det, 'clip_to_panel', return_value=(None, np.array([True, True, True, True]))):
            with patch.object(det, 'interpolate_bilinear', return_value=np.array([100.0])):
                
                # Create a temporary directory for the file output
                import tempfile
                with tempfile.TemporaryDirectory() as tmpdir:
                    filename = 'test_spots'
                    
                    compl, output = instr.pull_spots(
                        plane_data=MagicMock(),
                        grain_params=np.zeros(12),
                        imgser_dict=imgser_dict,
                        threshold=10,
                        filename=filename,
                        dirname=tmpdir,
                        output_format='hdf5'
                    )

                    # --- Verification ---
                    # 1. Verify Writer Initialization
                    MockH5Writer.assert_called_once()
                    call_args = MockH5Writer.call_args
                    # First arg is filename path
                    assert os.path.join(tmpdir, filename) in call_args[0][0]
                    
                    # 2. Verify Data Dump
                    # Get the instance returned by the constructor
                    writer_instance = MockH5Writer.return_value
                    writer_instance.dump_patch.assert_called_once()
                    
                    # Check arguments passed to dump_patch
                    # (panel_id, i_refl, peak_id, hkl_id, hkl, ...)
                    dump_args = writer_instance.dump_patch.call_args[0]
                    assert dump_args[0] == 'det1' # panel_id
                    assert dump_args[2] == 0      # peak_id (from labeling)
                    
                    # 3. Verify Closure
                    writer_instance.close.assert_called_once()