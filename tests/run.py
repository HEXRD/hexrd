import os

import numpy as np

from hexrd import config

import find_orientations_testing

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# =============================================================================
# %% REFERENCE DATA SPEC
# =============================================================================

# load reference data
working_dir = os.getcwd()
data_dirname = 'include'
results_dirname = 'results'

# config file for test
config_fname = os.path.join(data_dirname, 'test_config.yml')

ref_maps_fname = os.path.join(
    data_dirname, results_dirname,
    'results_mruby_composite_hexrd06_py27_ruby_eta-ome_maps.npz'
)

ref_quats_fname = os.path.join(
    data_dirname, results_dirname,
    'accepted_orientations_results_mruby_composite_hexrd06_py27_ruby.dat'
)


# =============================================================================
# %% EXECUTE INDEXING
# =============================================================================
cfg = config.open(config_fname)[0]

# ??? You guys have changed the CLI callback to be indteractive, yes?

# =============================================================================
# %% EXECUTE COMPARISONS
# =============================================================================
ref_maps = find_orientations_testing.load(ref_maps_fname)
# FIXME: get this from interactive indexing execution
test_maps = find_orientations_testing.load(
    os.path.join(
        data_dirname,
        'results_mruby_composite_hexrd07_py38_ruby_eta-ome_maps.npz'
    )
)

# not positive how you want to catch the comparison
map_comparison = find_orientations_testing.Comparison(ref_maps, test_maps)
print("Compare test with ref: ", map_comparison.compare())

ref_quats = np.loadtxt(ref_quats_fname, ndmin=2)
# FIXME: get this from interactive indexing execution
test_quats = np.loadtxt(
     os.path.join(
         data_dirname,
         'accepted_orientations_results_mruby_composite_hexrd07_py38_ruby.dat'
    ), ndmin=2
)

# test accepted orientations
# !!! currently just a func that raises a RuntimeError if test fails.
find_orientations_testing.compare_quaternion_lists(test_quats, ref_quats)
