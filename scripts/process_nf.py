#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: dcp5303
"""

#%% Necessary Dependencies


#  PROCESSING NF GRAINS WITH MISORIENTATION
#==============================================================================
import numpy as np

import multiprocessing as mp

import os
import logging
import psutil

from hexrd.grainmap import nfutil

try:
    import matplotlib.pyplot as plt
    matplot = True
except(ImportError):
    logging.warning(f'no matplotlib, debug plotting disabled')
    matplot = False


#==============================================================================
# %% FILES TO LOAD -CAN BE EDITED
#==============================================================================
#These files are attached, retiga.yml is a detector configuration file
#The near field detector was already calibrated

#A materials file, is a cPickle file which contains material information like lattice
#parameters necessary for the reconstruction

main_dir = '/INSERT/WORKDIR/'

det_file = main_dir + 'retiga.yml'
mat_file = main_dir + 'materials.h5'

#==============================================================================
# %% OUTPUT INFO -CAN BE EDITED
#==============================================================================

output_dir = main_dir


#==============================================================================
# %% NEAR FIELD DATA FILES -CAN BE EDITED - ZERO LOAD SCAN
#==============================================================================
#These are the near field data files used for the reconstruction, a grains.out file
#from the far field analaysis is used as orientation guess for the grid that will
grain_out_file = '/LOC/grains.out'

#%%

stem = 'nf_'

#Locations of near field images
data_folder = '/INSERT/nf/'

img_start = 6  # whatever you want the first frame to be
nframes = 1440
img_nums = np.arange(img_start, img_start+nframes, 1)

output_stem = 'NAME_VOL_X'
#==============================================================================
# %% USER OPTIONS -CAN BE EDITED
#==============================================================================

### material for the reconstruction
mat_name = 'INSERT'
max_tth = None  # degrees, if None is input max tth will be set by the geometry


#reconstruction with misorientation included, for many grains, this will quickly
#make the reconstruction size unmanagable
misorientation_bnd = 0.0  # degrees
misorientation_spacing = 0.25  # degrees


#####image processing parameters
num_for_dark = 250  # num images to use for median data

img_threshold = 0
process_type = 'gaussian'
sigma = 2.0
size = 3.0
process_args = np.array([sigma, size])

# process_type='dilations_only'
# num_erosions=2 #num iterations of images erosion, don't mess with unless you know what you're doing
# num_dilations=3 #num iterations of images erosion, don't mess with unless you know what you're doing
# process_args=np.array([num_erosions,num_dilations])

threshold = 1.5
# num iterations of 3d image stack dilations, don't mess with unless you know what you're doing
ome_dilation_iter = 1

#thresholds for grains in reconstructions
# only use orientations from grains with completnesses ABOVE this threshold
comp_thresh = 0.0
chi2_thresh = 1.0  # only use orientations from grains BELOW this chi^2


######reconstruction parameters
ome_range_deg = [(0., 359.75)]  # degrees

use_mask = True
#Mask info, used if use_mask=True
mask_data_file = '/LOC/tomo_mask.npz'
# this is generally the difference in y motor positions between the tomo and nf layer (tomo_motor_z-nf_motor_z), needed for registry
mask_vert_offset = -0.3

#these values will be used if no mask data is provided (use_mask=False) (no tomography)
# cross sectional to reconstruct (should be at least 20%-30% over sample width)
cross_sectional_dim = 1.35
voxel_spacing = 0.005  # in mm, voxel spacing for the near field reconstruction


##vertical (y) reconstruction voxel bounds in mm, ALWAYS USED REGARDLESS OF MASK
# (mm) if bounds are equal, a single layer is produced
v_bnds = [-0.085, 0.085]

beam_stop_y_cen = 0.0  # mm, measured from the origin of the detector paramters
beam_stop_width = 0.6  # mm, width of the beam stop verticallu
beam_stop_parms = np.array([beam_stop_y_cen, beam_stop_width])

##### multiprocessing controller parameters
check = None
limit = None
generate = None
ncpus = mp.cpu_count()
# chunksize for multiprocessing, don't mess with unless you know what you're doing
chunk_size = 500

RAM_set = False  # if True, manually set max amount of ram
max_RAM = 256  # only used if RAM_set is true. in GB

#### debug plotting
output_plot_check = True


#==============================================================================
# %% LOAD GRAIN AND EXPERIMENT DATA
#==============================================================================

experiment, nf_to_ff_id_map = nfutil.gen_trial_exp_data(grain_out_file, det_file,
                                                        mat_file, mat_name, max_tth,
                                                        comp_thresh, chi2_thresh, misorientation_bnd,
                                                        misorientation_spacing, ome_range_deg,
                                                        nframes, beam_stop_parms)


#==============================================================================
# %% LOAD / GENERATE TEST DATA
#==============================================================================


if use_mask:
    mask_data = np.load(mask_data_file)

    mask_full = mask_data['mask']
    Xs_mask = mask_data['Xs']
    Ys_mask = mask_data['Ys']-(mask_vert_offset)
    Zs_mask = mask_data['Zs']
    voxel_spacing = mask_data['voxel_spacing']

    # need to think about how to handle a single layer in this context
    tomo_layer_centers = np.squeeze(Ys_mask[:, 0, 0])
    above = np.where(tomo_layer_centers >= v_bnds[0])
    below = np.where(tomo_layer_centers < v_bnds[1])

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
        cross_sectional_dim, v_bnds, voxel_spacing)
    to_use = np.arange(len(test_crds_full))

test_crds = test_crds_full[to_use, :]


#==============================================================================
# %% NEAR FIELD - MAKE MEDIAN DARK
#==============================================================================

dark = nfutil.gen_nf_dark(data_folder, img_nums, num_for_dark, experiment.nrows,
                          experiment.ncols, dark_type='median', num_digits=6, stem=stem)

#==============================================================================
# %% NEAR FIELD - LOAD IMAGE DATA AND PROCESS
#==============================================================================

image_stack = nfutil.gen_nf_cleaned_image_stack(data_folder, img_nums, dark, experiment.nrows, experiment.ncols, process_type=process_type,
                                                process_args=process_args, threshold=img_threshold, ome_dilation_iter=ome_dilation_iter, num_digits=6, stem=stem)


#==============================================================================
# %% NEAR FIELD - splitting
#==============================================================================

if RAM_set is True:
    RAM = max_RAM * 1e9
else:
    RAM = psutil.virtual_memory().available  # in GB
#  # turn into number of bytes

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

#%% Test orientations in groups


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
            abcd) * int(voxels_per_group):int(abcd + 1) * int(voxels_per_group)] = confidence_map_group_list
        del raw_confidence

    if leftover_voxels > 0:
        #now for the leftover voxels
        voxels_to_test = test_crds[int(
            n_groups) * int(voxels_per_group):, :]
        raw_confidence = nfutil.test_orientations(
            image_stack, experiment, voxels_to_test, controller, multiprocessing_start_method)
        grain_map_group_list, confidence_map_group_list = nfutil.process_raw_confidence(
            raw_confidence, id_remap=nf_to_ff_id_map, min_thresh=0.0)

        grain_map_list[int(
            n_groups) * int(voxels_per_group):] = grain_map_group_list

        confidence_map_list[int(
            n_groups) * int(voxels_per_group):] = confidence_map_group_list
    
    #fix so that chunking will work with tomography
    grain_map_list_full=np.zeros(Xs.shape[0]*Xs.shape[1]*Xs.shape[2])
    confidence_map_list_full=np.zeros(Xs.shape[0]*Xs.shape[1]*Xs.shape[2])
    
    for jj in np.arange(len(to_use)):
        grain_map_list_full[to_use[jj]]=grain_map_list[jj]
        confidence_map_list_full[to_use[jj]]=confidence_map_list[jj]
    
    
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

        plt.figure(1)
        plt.imshow(confidence_map[0])
        plt.title('Bottom Layer Confidence')

        plt.figure(2)
        plt.imshow(confidence_map[-1])
        plt.title('Bottom Layer Confidence')

        plt.figure(3)
        nfutil.plot_ori_map(grain_map, confidence_map,
                            experiment.exp_maps, 0, id_remap=nf_to_ff_id_map)
        plt.title('Top Layer Grain Map')

        plt.figure(4)
        nfutil.plot_ori_map(grain_map, confidence_map,
                            experiment.exp_maps, -1, id_remap=nf_to_ff_id_map)
        plt.title('Top Layer Grain Map')

        plt.show()
