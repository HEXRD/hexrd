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

import nfutil

#==============================================================================
# %% FILES TO LOAD -CAN BE EDITED
#==============================================================================
#These files are attached, retiga.yml is a detector configuration file
#The near field detector was already calibrated

#A materials file, is a cPickle file which contains material information like lattice
#parameters necessary for the reconstruction

main_dir = '/INSERT/WORKDIR/'

det_file = main_dir + 'retiga.yml'
mat_file= main_dir + 'materials.h5' 

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

#Locations of near field images
data_folder='/INSERT/nf/' 

img_start=6 #whatever you want the first frame to be 
nframes=1440
img_nums=np.arange(img_start,img_start+nframes,1)

output_stem='NAME_VOL_X'
#==============================================================================
# %% USER OPTIONS -CAN BE EDITED
#==============================================================================

### material for the reconstruction
mat_name='INSERT'
max_tth=None #degrees, if None is input max tth will be set by the geometry


#reconstruction with misorientation included, for many grains, this will quickly
#make the reconstruction size unmanagable
misorientation_bnd=0.0 #degrees 
misorientation_spacing=0.25 #degrees


#####image processing parameters
num_for_dark=250#num images to use for median data

process_type='gaussian'
sigma=4.5
size=5
process_args=np.array([sigma,size])

# process_type='dilations_only'
# num_erosions=2 #num iterations of images erosion, don't mess with unless you know what you're doing
# num_dilations=3 #num iterations of images erosion, don't mess with unless you know what you're doing
# process_args=np.array([num_erosions,num_dilations])

threshold=1.5
ome_dilation_iter=1 #num iterations of 3d image stack dilations, don't mess with unless you know what you're doing

#thresholds for grains in reconstructions
comp_thresh=0.0 #only use orientations from grains with completnesses ABOVE this threshold
chi2_thresh=1.0 #only use orientations from grains BELOW this chi^2


######reconstruction parameters
ome_range_deg=[(0.,359.75)] #degrees 

use_mask=True
#Mask info, used if use_mask=True
mask_data_file='/LOC/tomo_mask.npz'
mask_vert_offset=-0.3 #this is generally the difference in y motor positions between the tomo and nf layer (tomo_motor_z-nf_motor_z), needed for registry

#these values will be used if no mask data is provided (use_mask=False) (no tomography)
cross_sectional_dim=1.35 #cross sectional to reconstruct (should be at least 20%-30% over sample width)
voxel_spacing = 0.005 #in mm, voxel spacing for the near field reconstruction


##vertical (y) reconstruction voxel bounds in mm, ALWAYS USED REGARDLESS OF MASK
v_bnds=[-0.085,0.085] #(mm) if bounds are equal, a single layer is produced

beam_stop_y_cen=0.0 #mm, measured from the origin of the detector paramters
beam_stop_width=0.6 #mm, width of the beam stop verticallu
beam_stop_parms=np.array([beam_stop_y_cen,beam_stop_width])


##### multiprocessing controller parameters
check=None
limit=None
generate=None
ncpus=mp.cpu_count()
chunk_size=500#chunksize for multiprocessing, don't mess with unless you know what you're doing





#==============================================================================
# %% LOAD GRAIN AND EXPERIMENT DATA
#==============================================================================

experiment, nf_to_ff_id_map  = nfutil.gen_trial_exp_data(grain_out_file,det_file,mat_file, mat_name, max_tth, comp_thresh, chi2_thresh, misorientation_bnd, \
                       misorientation_spacing,ome_range_deg, nframes, beam_stop_parms)
    
    
#==============================================================================
# %% LOAD / GENERATE TEST DATA
#==============================================================================



if use_mask:
    mask_data=np.load(mask_data_file)
    
    
    mask_full=mask_data['mask']
    Xs_mask=mask_data['Xs']
    Ys_mask=mask_data['Ys']-(mask_vert_offset)
    Zs_mask=mask_data['Zs']
    voxel_spacing=mask_data['voxel_spacing']
    
    
    tomo_layer_centers=np.squeeze(Ys_mask[:,0,0])
    above=np.where(tomo_layer_centers>v_bnds[0])
    below=np.where(tomo_layer_centers<v_bnds[1])
    
    
    in_bnds=np.intersect1d(above,below)
    
    mask=mask_full[in_bnds]
    Xs=Xs_mask[in_bnds]
    Ys=Ys_mask[in_bnds]
    Zs=Zs_mask[in_bnds]
    
    test_crds_full = np.vstack([Xs.flatten(), Ys.flatten(), Zs.flatten()]).T
    n_crds = len(test_crds_full)
    
    to_use=np.where(mask.flatten())[0]
    

else:
    test_crds_full, n_crds, Xs, Ys, Zs = nfutil.gen_nf_test_grid(cross_sectional_dim, v_bnds, voxel_spacing)
    to_use=np.arange(len(test_crds_full))


test_crds=test_crds_full[to_use,:]



#==============================================================================
# %% NEAR FIELD - MAKE MEDIAN DARK
#==============================================================================

dark=nfutil.gen_nf_dark(data_folder,img_nums,num_for_dark,experiment.nrows,experiment.ncols,dark_type='median',num_digits=6)

#==============================================================================
# %% NEAR FIELD - LOAD IMAGE DATA AND PROCESS
#==============================================================================

image_stack=nfutil.gen_nf_cleaned_image_stack(data_folder,img_nums,dark,experiment.nrows,experiment.ncols,process_type=process_type,\
                                              process_args=process_args,ome_dilation_iter=ome_dilation_iter,num_digits=6)

    
    
    
#==============================================================================
# %% BUILD MP CONTROLLER
#============================================================================== 

# assume that if os has fork, it will be used by multiprocessing.
# note that on python > 3.4 we could use multiprocessing get_start_method and
# set_start_method for a cleaner implementation of this functionality.
multiprocessing_start_method = 'fork' if hasattr(os, 'fork') else 'spawn'  
    

controller=nfutil.build_controller(ncpus=ncpus,chunk_size=chunk_size,check=check,generate=generate,limit=limit)   
 

#==============================================================================
# %% TEST ORIENTATIONS
#============================================================================== 
   
## Would be nice to pack the images, quant_and_clip will need to be edited if this is added, dcp 6.2.2021
#packed_image_stack = nfutil.dilate_image_stack(image_stack, experiment,controller)

raw_confidence=nfutil.test_orientations(image_stack, experiment,test_crds,controller,multiprocessing_start_method)    
 

del controller 
  


#==============================================================================
# %% PUT IT ALL BACK TOGETHER
#==============================================================================

# note that all masking 
raw_confidence_full=np.zeros([len(experiment.exp_maps),len(test_crds_full)])

for ii in np.arange(raw_confidence_full.shape[0]):
    raw_confidence_full[ii,to_use]=raw_confidence[ii,:]

    
#==============================================================================
# %% POST PROCESS W WHEN TOMOGRAPHY HAS BEEN USED
#==============================================================================


#note all masking is already handled by not evaluating specific points
grain_map, confidence_map = nfutil.process_raw_confidence(raw_confidence_full,Xs.shape,id_remap=nf_to_ff_id_map,min_thresh=0.0)

#==============================================================================
# %% SAVE PROCESSED GRAIN MAP DATA
#==============================================================================

nfutil.save_nf_data(output_dir,output_stem,grain_map,confidence_map,Xs,Ys,Zs,experiment.exp_maps,id_remap=nf_to_ff_id_map)

# #==============================================================================
# # %% CHECKING OUTPUT
# #==============================================================================

# import matplotlib.pyplot as plt

# plt.figure(1)
# plt.imshow(confidence_map[0])

# plt.figure(2)
# plt.imshow(confidence_map[-1])

# plt.figure(3)
# plt.imshow(grain_map[0])

# plt.figure(4)
# plt.imshow(grain_map[-1])



