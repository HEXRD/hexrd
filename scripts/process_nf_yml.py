#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import multiprocessing as mp

import os
import logging
import psutil

from hexrd import nf_config
from hexrd import rotations as rot
from hexrd.grainmap import nfutil
from hexrd.material import Material
from hexrd.valunits import valWUnit

try:
    import matplotlib.pyplot as plt
    matplot = True
except (ImportError):
    logging.warning(f'no matplotlib, debug plotting disabled')
    matplot = False

# ==============================================================================
# %% Sets parameters from input yaml
# ==============================================================================

config_fname = '/Users/lim37/Library/CloudStorage/OneDrive-LLNL/ti7al_cyclic_HEDM/ti7-1/NF/NF_config_test.yml'

cfg = nf_config.open(config_fname)[0]

analysis_name = cfg.analysis_name
main_dir = cfg.main_dir
output_dir = cfg.output_dir

output_plot_check = cfg.output_plot_check

beam_energy = cfg.experiment.beam_energy
grain_out_file = cfg.input_files.grain_out_file
det_file = cfg.input_files.det_file
mat_file = cfg.input_files.mat_file
mat_name = cfg.experiment.mat_name
max_tth = cfg.experiment.max_tth
comp_thresh = cfg.experiment.comp_thresh
chi2_thresh = cfg.experiment.chi2_thresh

nframes = cfg.images.nframes

if cfg.experiment.misorientation:
    misorientation_bnd = np.array(
        cfg.experiment.misorientation['misorientation_bnd'])
    misorientation_spacing = \
        cfg.experiment.misorientation['misorientation_spacing']

ome_range_deg = cfg.experiment.ome_range

beam_stop_parms = np.array(
    [cfg.reconstruction.beam_stop_y_cen, cfg.reconstruction.beam_stop_width])


# reconstruction size parameters
cross_sectional_dim = cfg.reconstruction.cross_sectional_dim  # 2.0
voxel_spacing = cfg.reconstruction.voxel_spacing
v_bnds = cfg.reconstruction.v_bnds


# project_single_layer = cfg.reconstruction.tomorgrproject_single_layer
tomo_mask = cfg.reconstruction.tomography
if tomo_mask:
    mask_data_file = tomo_mask['mask_data_file']
    mask_vert_offset = tomo_mask['mask_vert_offset']
    project_single_layer = tomo_mask['project_single_layer']


ncpus = cfg.multiprocessing.num_cpus
chunk_size = cfg.multiprocessing.chunk_size
check = cfg.multiprocessing.check
limit = cfg.multiprocessing.limit
generate = cfg.multiprocessing.generate

max_RAM = cfg.multiprocessing.max_RAM  # this is in bytes


# ==============================================================================
# %% LOAD GRAIN AND EXPERIMENT DATA
# ==============================================================================

experiment, nf_to_ff_id_map = \
    nfutil.gen_trial_exp_data(grain_out_file, det_file, mat_file, mat_name,
                              max_tth, comp_thresh, chi2_thresh,
                              misorientation_bnd, misorientation_spacing,
                              ome_range_deg, nframes, beam_stop_parms)


# ==============================================================================
# %% LOAD / GENERATE TEST DATA
# ==============================================================================

test_crds_full, n_crds, Xs, Ys, Zs = nfutil.gen_nf_test_grid(
    cross_sectional_dim, v_bnds, voxel_spacing)


if tomo_mask:
    mask_data = np.load(mask_data_file)
    mask_full = mask_data['mask'][0:1, :, :]
    to_use = mask_full

    Xs = Xs[to_use]
    Ys = Ys[to_use]
    Zs = Zs[to_use]

    test_crds_full = np.vstack([Xs.flatten(), Ys.flatten(), Zs.flatten()]).T
    n_crds = len(test_crds_full)

    to_use = np.where(mask_full.flatten())[0]

else:
    to_use = np.arange(len(test_crds_full))

test_crds = test_crds_full

image_data = np.load(os.path.join(output_dir, (analysis_name+'.npz')))
image_stack = image_data['image_stack']

# ==============================================================================
# %% NEAR FIELD - splitting
# ==============================================================================


if max_RAM < 1000:  # a terabyte of RAM
    RAM = max_RAM * 1e9  # turn into number of bytes
else:
    RAM = psutil.virtual_memory().available  # in GB

RAM_to_use = 0.75 * RAM

n_oris = len(nf_to_ff_id_map)
n_voxels = len(test_crds)

bits_for_arrays = 64*n_oris*n_voxels + 192 * \
    n_voxels  # bits raw conf + bits voxel positions
bytes_for_array = bits_for_arrays/8.

n_groups = np.floor(bytes_for_array/RAM_to_use)  # number of full groups
if n_groups == 0:
    leftover_voxels = n_voxels
else:
    leftover_voxels = np.mod(n_voxels, n_groups)

print('Splitting data into %d groups with %d leftover voxels' %
      (int(n_groups), int(leftover_voxels)))


grouped_voxels = n_voxels - leftover_voxels

if n_groups != 0:
    voxels_per_group = grouped_voxels/n_groups
# ==============================================================================
# %% BUILD MP CONTROLLER
# ==============================================================================

# assume that if os has fork, it will be used by multiprocessing.
# note that on python > 3.4 we could use multiprocessing get_start_method and
# set_start_method for a cleaner implementation of this functionality.
multiprocessing_start_method = 'fork' if hasattr(os, 'fork') else 'spawn'

controller = nfutil.build_controller(
    ncpus=ncpus, chunk_size=chunk_size, check=check,
    generate=generate, limit=limit)

# ==============================================================================
# %% TEST ORIENTATIONS
# ==============================================================================

print('Testing Orientations...')
# % Test orientations in groups

if n_groups == 0:
    raw_confidence = nfutil.test_orientations(
        image_stack, experiment, test_crds, controller,
        multiprocessing_start_method)

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
            image_stack, experiment, voxels_to_test, controller,
            multiprocessing_start_method)
        print('Calculated raw confidence group %d' % group)
        grain_map_group_list, confidence_map_group_list = \
            nfutil.process_raw_confidence(
                raw_confidence, id_remap=nf_to_ff_id_map, min_thresh=0.0)

        grain_map_list[int(
            group) * int(voxels_per_group):int(group + 1) *
            int(voxels_per_group)] = grain_map_group_list

        confidence_map_list[int(
            group) * int(voxels_per_group):int(group + 1) *
            int(voxels_per_group)] = confidence_map_group_list
        del raw_confidence

    if leftover_voxels > 0:
        # now for the leftover voxels
        voxels_to_test = test_crds[int(
            n_groups) * int(voxels_per_group):, :]
        raw_confidence = nfutil.test_orientations(
            image_stack, experiment, voxels_to_test, controller,
            multiprocessing_start_method)
        grain_map_group_list, confidence_map_group_list = \
            nfutil.process_raw_confidence(
                raw_confidence, id_remap=nf_to_ff_id_map, min_thresh=0.0)

        grain_map_list[int(
            n_groups) * int(voxels_per_group):] = grain_map_group_list

        confidence_map_list[int(
            n_groups) * int(voxels_per_group):] = confidence_map_group_list

    # fix so that chunking will work with tomography
    grain_map_list_full = np.zeros(Xs.shape[0]*Xs.shape[1]*Xs.shape[2])
    confidence_map_list_full = np.zeros(Xs.shape[0]*Xs.shape[1]*Xs.shape[2])

    for jj in np.arange(len(to_use)):
        grain_map_list_full[to_use[jj]] = grain_map_list[jj]
        confidence_map_list_full[to_use[jj]] = confidence_map_list[jj]

    # reshape them
    grain_map = grain_map_list_full.reshape(Xs.shape)
    confidence_map = confidence_map_list_full.reshape(Xs.shape)

# del controller

# ==============================================================================
# %% POST PROCESS W WHEN TOMOGRAPHY HAS BEEN USED
# ==============================================================================

# note all masking is already handled by not evaluating specific points
grain_map, confidence_map = nfutil.process_raw_confidence(
    raw_confidence_full, Xs.shape, min_thresh=0.0)


# %%
plt.figure()

good = (confidence_map[0, :, :] > 0.5)*1

plt.imshow(grain_map[0, :, :]*good, cmap='jet')

plt.figure()
plt.imshow(confidence_map[0, :, :], vmin=0.5, vmax=1)
# plt.title('%d voxels > 0.5' % np.sum(confidence_map[0, :, :] > 0.5))
plt.colorbar()


# # ==============================================================================
# # %% SAVE PROCESSED GRAIN MAP DATA
# # ==============================================================================

nfutil.save_nf_data(output_dir, analysis_name, grain_map, confidence_map,
                    Xs, Ys, Zs, experiment.exp_maps, id_remap=nf_to_ff_id_map)


# ==============================================================================
# %% CHECKING OUTPUT
# ==============================================================================


mat = Material(name=mat_name, material_file=mat_file, dmin=valWUnit(
    'lp', 'length',  0.05, 'nm'), kev=valWUnit('kev', 'energy', beam_energy, 'keV'))

if matplot:
    if output_plot_check:
        nfutil.plot_ori_map(grain_map, confidence_map,
                            experiment.exp_maps, 0, mat)
