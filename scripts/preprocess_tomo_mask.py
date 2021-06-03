#%% Necessary Dependencies

import numpy as np

import yaml

import matplotlib.pyplot as plt

import nfutil
import tomoutil

from hexrd import instrument


def load_instrument(yml):
    with open(yml, 'r') as f:
        icfg = yaml.load(f, Loader=yaml.FullLoader)
    return instrument.HEDMInstrument(instrument_config=icfg)

# %% FILES TO LOAD -CAN BE EDITED
#==============================================================================
#These files are attached, retiga.yml is a detector configuration file
#The near field detector was already calibrated

#A materials file, is a cPickle file which contains material information like lattice
#parameters necessary for the reconstruction

main_dir = '/INSERT/WORK/DIR/'

det_file = main_dir + 'retiga.yml'

#==============================================================================
# %% OUTPUT INFO -CAN BE EDITED
#==============================================================================

output_dir = main_dir 
output_stem='tomo_out'

#==============================================================================
# %% TOMOGRAPHY DATA FILES -CAN BE EDITED
#==============================================================================


#Locations of tomography dark field images
tdf_data_folder='/LOC/nf/'

tdf_img_start=52 #for this rate, this is the 6th file in the folder
tdf_num_imgs=10

#Locations of tomography bright field images
tbf_data_folder='/LOC/nf/'

tbf_img_start=68 #for this rate, this is the 6th file in the folder
tbf_num_imgs=10

#Locations of tomography images
tomo_data_folder='/LOC/nf/'

tomo_img_start=84#for this rate, this is the 6th file in the folder
tomo_num_imgs=360

#==============================================================================
# %% USER OPTIONS -CAN BE EDITED
#==============================================================================


ome_range_deg=[(0.,359.75)] #degrees 


#tomography options
recon_thresh=0.0002#usually varies between 0.0001 and 0.0005
#Don't change these unless you know what you are doing, this will close small holes
#and remove noise
noise_obj_size=500
min_hole_size=500
project_single_layer=False #projects the center layers through the volume, faster but not recommended, included for completion / historical purposes


#reconstruction volume options
cross_sectional_dim=1.35 #cross sectional to reconstruct (should be at least 20%-30% over sample width)
voxel_spacing=0.005#in mm
v_bnds=[-0.4,0.4]


#==============================================================================
# %% LOAD INSTRUMENT DATA
#==============================================================================


instr=load_instrument(det_file)

panel = next(iter(instr.detectors.values())) 



nrows=panel.rows
ncols=panel.rows
pixel_size=panel.pixel_size_row


rot_axis_pos=panel.tvec[0] #should match t_vec_d[0] from nf_detector_parameter_file
vert_beam_center=-panel.tvec[1] #this - sign should be checked


# need to do a few calculations because not every row will be reconstructed
# depending on sampling
vert_points=np.arange(v_bnds[0]+voxel_spacing/2.,v_bnds[1],voxel_spacing)

center_layer_row=nrows/2.+vert_beam_center/pixel_size

rows_to_recon=np.round(center_layer_row-vert_points/pixel_size).astype(int)

center_layer_row=int(center_layer_row)

#==============================================================================
# %% TOMO PROCESSING - GENERATE DARK AND BRIGHT FIELD
#==============================================================================
tdf=tomoutil.gen_median_image(tdf_data_folder,tdf_img_start,tdf_num_imgs,nrows,ncols,num_digits=6)
tbf=tomoutil.gen_median_image(tbf_data_folder,tbf_img_start,tbf_num_imgs,nrows,ncols,num_digits=6)

#==============================================================================
# %% TOMO PROCESSING - BUILD RADIOGRAPHS
#==============================================================================

rad_stack=tomoutil.gen_attenuation_rads(tomo_data_folder,tbf,tomo_img_start,tomo_num_imgs,nrows,ncols,num_digits=6,tdf=tdf)
    
#%%


#==============================================================================
# %% TOMO PROCESSING - INVERT SINOGRAM
#==============================================================================
#center = 22.*0.00148
#center = -0.018

test_fbp=tomoutil.tomo_reconstruct_layer(rad_stack,cross_sectional_dim,layer_row=center_layer_row,\
                                                   start_tomo_ang=ome_range_deg[0][0],end_tomo_ang=ome_range_deg[0][1],\
                                                   tomo_num_imgs=tomo_num_imgs, center=rot_axis_pos)

test_binary_recon=tomoutil.threshold_and_clean_tomo_layer(test_fbp,recon_thresh, noise_obj_size,min_hole_size)

tomo_mask_center=tomoutil.crop_and_rebin_tomo_layer(test_binary_recon,recon_thresh,voxel_spacing,pixel_size,cross_sectional_dim)

full_mask=np.zeros([len(rows_to_recon),tomo_mask_center.shape[0],tomo_mask_center.shape[1]])

#==============================================================================
# %% TOMO PROCESSING - VIEW RAW FILTERED BACK PROJECTION
#==============================================================================

plt.figure(1)
plt.imshow(test_fbp,vmin=recon_thresh,vmax=recon_thresh*2)
plt.title('Check Thresholding')
#Use this image to view the raw reconstruction, estimate threshold levels. and
#figure out if the rotation axis position needs to be corrected


plt.figure(2)
plt.imshow(tomo_mask_center,interpolation='none')
plt.title('Check Center Mask')

#==============================================================================
# %% PROCESS REMAINING LAYERS
#==============================================================================

for ii in np.arange(len(rows_to_recon)):
    print('Layer: ' + str(ii) + ' of ' + str(len(rows_to_recon)))
    
    if project_single_layer: #not recommended option
        full_mask[ii]=tomo_mask_center
        
    else:
        reconstruction_fbp=tomoutil.tomo_reconstruct_layer(rad_stack,cross_sectional_dim,layer_row=rows_to_recon[ii],\
                                                       start_tomo_ang=ome_range_deg[0][0],end_tomo_ang=ome_range_deg[0][1],\
                                                       tomo_num_imgs=tomo_num_imgs, center=rot_axis_pos)
        binary_recon=tomoutil.threshold_and_clean_tomo_layer(reconstruction_fbp,recon_thresh, noise_obj_size,min_hole_size)
    
        tomo_mask=tomoutil.crop_and_rebin_tomo_layer(binary_recon,recon_thresh,voxel_spacing,pixel_size,cross_sectional_dim)
        
        full_mask[ii]=tomo_mask

#==============================================================================
# %%  TOMO PROCESSING - VIEW TOMO_MASK FOR SAMPLE BOUNDS
#==============================================================================

plt.figure(3)
plt.imshow(tomo_mask,interpolation='none')
plt.title('Check Center Mask')

#==============================================================================
# %%  TOMO PROCESSING - CONSTRUCT DATA GRID
#==============================================================================

test_crds, n_crds, Xs, Ys, Zs = nfutil.gen_nf_test_grid_tomo(full_mask.shape[2], full_mask.shape[1], v_bnds, voxel_spacing)

#%%

np.savez('tomo_mask.npz',mask=full_mask,Xs=Xs,Ys=Ys,Zs=Zs,voxel_spacing=voxel_spacing)
