"""Regression coverage for the find-orientations launch-config workflow."""
import os

import numba
import numpy as np
import pytest
import yaml

from hexrd.hedm.experiment import HedmExperiment
from hexrd.hedm.find_orientations import find_orientations


@pytest.mark.skipif(
    'HEXRD_EXAMPLE_REPO_PATH' not in os.environ,
    reason='requires the NIST ruby example data',
)
def test_ruby_pipeline(example_repo_path, test_data_dir, tmp_path, monkeypatch):
    include = example_repo_path / 'NIST_ruby/single_GE/include'
    with open(include / 'cstudy.yml') as stream:
        config = next(yaml.safe_load_all(stream))
    config['analysis_name'] = str(tmp_path / 'results')
    config['multiprocessing'] = 1
    filename = tmp_path / 'config.yml'
    filename.write_text(yaml.safe_dump(config))

    reference = test_data_dir / 'find_orientations_pipeline'
    expected_scores = np.load(reference / 'ruby_completeness.npy')
    expected_qbar = np.load(reference / 'ruby_qbar.npy')
    previous_threads = numba.get_num_threads()
    try:
        for clean in (True, False):
            experiment = HedmExperiment(str(filename), working_dir=str(include))
            load_material = experiment.get_active_material

            def check_workers():
                assert numba.get_num_threads() == 1
                return load_material()

            monkeypatch.setattr(experiment, 'get_active_material', check_workers)
            numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))
            result = find_orientations(experiment, clean=clean)
            np.testing.assert_array_equal(result.completeness, expected_scores)
            np.testing.assert_allclose(result.grain_orientations, expected_qbar,
                                       rtol=0, atol=1e-12)
            if not clean:
                assert experiment.image_series_list[0]._images is None
    finally:
        numba.set_num_threads(previous_threads)
