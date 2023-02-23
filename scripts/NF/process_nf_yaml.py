import numpy as np

import multiprocessing as mp
import yaml

import os
import logging
import psutil
import argparse

from hexrd import rotations as rot
from hexrd.grainmap import nfutil_yaml
from hexrd.material import Material
from hexrd.valunits import valWUnit
from hexrd import nf_config

try:
    import matplotlib.pyplot as plt
    matplot = True
except(ImportError):
    logging.warning(f'no matplotlib, debug plotting disabled')
    matplot = False


#%%
# fname = '/Users/lim37/Documents/AFRL/NF_config_v1.yml'
parser = argparse.ArgumentParser(description='Preprocess NF image stack')

parser.add_argument('input_file', type=str, help='Input File for NF reconstruction')

args = parser.parse_args()
fname = args.input_file

#%%

cfg = nf_config.open(fname)[0]

output_stem = cfg.analysis_name
main_dir = cfg.main_dir
output_dir = cfg.output_dir

output_plot_check = cfg.output_plot_check


grain_out_file = cfg.input_files.grain_out_file
det_file = cfg.input_files.det_file
mat_file = cfg.input_files.mat_file
mat_name = cfg.experiment.mat_name
max_tth = cfg.experiment.max_tth
comp_thresh = cfg.experiment.comp_thresh
chi2_thresh = cfg.experiment.chi2_thresh

nframes = cfg.images.nframes

# if cfg.experiment.misorientation:
misorientation_bnd = cfg.experiment.misorientation['misorientation_bnd']
misorientation_spacing = cfg.experiment.misorientation['misorientation_spacing']

ome_range_deg = cfg.experiment.ome_range

beam_stop_parms = np.array(
    [cfg.reconstruction.beam_stop_y_cen, cfg.reconstruction.beam_stop_width])

use_mask = cfg.reconstruction.tomography.use_mask

max_RAM = cfg.multiprocessing.max_RAM

ncpus = cfg.multiprocessing.ncpus
chunk_size = cfg.multiprocessing.chunk_size
check = cfg.multiprocessing.check
generate = cfg.multiprocessing.generate
limit = cfg.multiprocessing.limit


#==============================================================================
# %% LOAD GRAIN AND EXPERIMENT DATA
#==============================================================================

experiment, nf_to_ff_id_map = nfutil.gen_trial_exp_data(grain_out_file, det_file,
                                                        mat_file, mat_name,
                                                        max_tth, ome_range_deg,
                                                        nframes, beam_stop_parms,
                                                        comp_thresh, chi2_thresh,
                                                        misorientation_bnd,
                                                        misorientation_spacing)


#==============================================================================
# %% LOAD / GENERATE TEST DATA
#==============================================================================

use_mask = cfg.reconstruction.tomography['use_mask']

if use_mask:
    mask_data = np.load(cfg.reconstruction.tomography['mask_data_file'])

    mask_full = mask_data['mask']
    Xs_mask = mask_data['Xs']
    Ys_mask = mask_data['Ys'] - \
        (cfg.reconstruction.tomography['mask_vert_offset'])
    Zs_mask = mask_data['Zs']
    voxel_spacing = mask_data['voxel_spacing']

    # need to think about how to handle a single layer in this context
    tomo_layer_centers = np.squeeze(Ys_mask[:, 0, 0])
    above = np.where(tomo_layer_centers >= cfg.reconstruction.v_bnds[0])
    below = np.where(tomo_layer_centers < cfg.reconstruction.v_bnds[1])

    in_bnds = np.intersect1d(above, below)

    mask = mask_full[in_bnds]
    Xs = Xs_mask[in_bnds]
    Ys = Ys_mask[in_bnds]
    Zs = Zs_mask[in_bnds]

    test_crds_full = np.vstack([Xs.flatten(), Ys.flatten(), Zs.flatten()]).T
    n_crds = len(test_crds_full)

    to_use = np.where(mask.flatten())[0]


else:
    test_crds_full, n_crds, Xs, Ys, Zs = nfutil.gen_nf_test_grid(
        cfg.reconstruction.cross_sectional_dim, cfg.reconstruction.v_bnds,
        cfg.reconstruction.voxel_spacing)
    to_use = np.arange(len(test_crds_full))

test_crds = test_crds_full[to_use, :]

#load cleaned image stack
image_stack = np.load(os.path.join(
    output_dir, (output_stem+'.npz')))['image_stack']
#==============================================================================
# %% NEAR FIELD - splitting
#==============================================================================


RAM = max_RAM * 1e9  # turn into number of bytes

RAM_to_use = 0.75 * RAM

n_oris = len(nf_to_ff_id_map)
n_voxels = len(test_crds)

bits_for_arrays = 64*n_oris*n_voxels + 192 * \
    n_voxels  # bits raw conf + bits voxel positions
bytes_for_array = bits_for_arrays/8.

n_groups = np.floor(bytes_for_array/RAM_to_use)  # number of full groups
leftover_voxels = np.mod(n_voxels, n_groups)

print('Splitting data into %d groups with %d leftover voxels' %
      (int(n_groups), int(leftover_voxels)))


grouped_voxels = n_voxels - leftover_voxels

if n_groups != 0:
    voxels_per_group = grouped_voxels/n_groups
#==============================================================================
# %% BUILD MP CONTROLLER
#==============================================================================

# assume that if os has fork, it will be used by multiprocessing.
# note that on python > 3.4 we could use multiprocessing get_start_method and
# set_start_method for a cleaner implementation of this functionality.
multiprocessing_start_method = 'fork' if hasattr(os, 'fork') else 'spawn'


controller = nfutil.build_controller(
    ncpus=ncpus, chunk_size=chunk_size, check=check, generate=generate, limit=limit)


#==============================================================================
# %% TEST ORIENTATIONS
#==============================================================================

## Would be nice to pack the images, quant_and_clip will need to be edited if this is added, dcp 6.2.2021
#packed_image_stack = nfutil.dilate_image_stack(image_stack, experiment,controller)

print('Testing Orientations...')
#% Test orientations in groups

if n_groups == 0:
    raw_confidence = nfutil.test_orientations(
        image_stack, experiment, test_crds, controller, multiprocessing_start_method)

    del controller

    raw_confidence_full = np.zeros(
        [len(experiment.exp_maps), len(test_crds_full)])

    for ii in np.arange(raw_confidence_full.shape[0]):
        raw_confidence_full[ii, to_use] = raw_confidence[ii, :]

else:
    grain_map_list = np.zeros(n_voxels)
    confidence_map_list = np.zeros(n_voxels)

    # test voxels in groups
    for group in range(int(n_groups)):
        voxels_to_test = test_crds[int(
            group) * int(voxels_per_group):int(group + 1) * int(voxels_per_group), :]
        print('Calculating group %d' % group)
        raw_confidence = nfutil.test_orientations(
            image_stack, experiment, voxels_to_test, controller, multiprocessing_start_method)
        print('Calculated raw confidence group %d' % group)
        grain_map_group_list, confidence_map_group_list = nfutil.process_raw_confidence(
            raw_confidence, id_remap=nf_to_ff_id_map, min_thresh=0.0)

        grain_map_list[int(
            group) * int(voxels_per_group):int(group + 1) * int(voxels_per_group)] = grain_map_group_list

        confidence_map_list[int(
            group) * int(voxels_per_group):int(group + 1) * int(voxels_per_group)] = confidence_map_group_list
        del raw_confidence

    if leftover_voxels > 0:
        #now for the leftover voxels
        voxels_to_test = test_crds[int(
            n_groups) * int(voxels_per_group):, :]
        raw_confidence = nfutil.test_orientations(
            image_stack, experiment, voxels_to_test,
            controller, multiprocessing_start_method)
        grain_map_group_list, confidence_map_group_list = nfutil.process_raw_confidence(
            raw_confidence, id_remap=nf_to_ff_id_map, min_thresh=0.0)

        grain_map_list[int(
            n_groups) * int(voxels_per_group):] = grain_map_group_list

        confidence_map_list[int(
            n_groups) * int(voxels_per_group):] = confidence_map_group_list

    #fix so that chunking will work with tomography
    grain_map_list_full = np.zeros(Xs.shape[0]*Xs.shape[1]*Xs.shape[2])
    confidence_map_list_full = np.zeros(Xs.shape[0]*Xs.shape[1]*Xs.shape[2])

    for jj in np.arange(len(to_use)):
        grain_map_list_full[to_use[jj]] = grain_map_list[jj]
        confidence_map_list_full[to_use[jj]] = confidence_map_list[jj]

    #reshape them
    grain_map = grain_map_list_full.reshape(Xs.shape)
    confidence_map = confidence_map_list_full.reshape(Xs.shape)

del controller

#==============================================================================
# %% POST PROCESS W WHEN TOMOGRAPHY HAS BEEN USED
#==============================================================================

#note all masking is already handled by not evaluating specific points
grain_map, confidence_map = nfutil.process_raw_confidence(
    raw_confidence_full, Xs.shape, id_remap=nf_to_ff_id_map, min_thresh=0.0)

#==============================================================================
# %% SAVE PROCESSED GRAIN MAP DATA
#==============================================================================

nfutil.save_nf_data(output_dir, output_stem, grain_map, confidence_map,
                    Xs, Ys, Zs, experiment.exp_maps, id_remap=nf_to_ff_id_map)

#==============================================================================
# %% CHECKING OUTPUT
#==============================================================================


if matplot:
    if output_plot_check:
        beam_energy = cfg.experiment.beam_energy
        mat = Material(name=mat_name, material_file=mat_file, dmin=valWUnit(
            'lp', 'length',  0.05, 'nm'), kev=valWUnit('kev', 'energy', beam_energy, 'keV'))
        nfutil.plot_ori_map(grain_map, confidence_map,
                            experiment.exp_maps, 0, mat, id_remap=nf_to_ff_id_map)
