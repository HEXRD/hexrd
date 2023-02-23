#%% Necessary Dependencies

import numpy as np
import logging
from hexrd import nf_config
import argparse

import yaml

try:
    import matplotlib.pyplot as plt
    matplot=True
except(ImportError):
    logging.warning(f'no matplotlib, debug plotting disabled')
    matplot=False

from hexrd.grainmap import nfutil_yaml
from hexrd.grainmap import tomoutil

from hexrd import instrument


def load_instrument(yml):
    with open(yml, 'r') as f:
        icfg = yaml.load(f, Loader=yaml.FullLoader)
    return instrument.HEDMInstrument(instrument_config=icfg)

#%%
parser = argparse.ArgumentParser(description='Preprocess NF image stack')

parser.add_argument('input_file', type=str, help='Input File for NF reconstruction')

args = parser.parse_args()
fname = args.input_file
# fname = '/Users/lim37/Documents/AFRL/NF_config_v1.yml'

#%%
cfg = nf_config.open(fname)[0]

output_stem = cfg.analysis_name
main_dir = cfg.main_dir
output_dir = cfg.output_dir

det_file = cfg.input_files.det_file

tomo = cfg.tomo

img_stem = tomo.img_stem
num_digits = tomo.num_digits

#Locations of tomography bright field images
tbf_data_folder = tomo.bright.folder
tbf_img_start = tomo.bright.img_start
tbf_num_imgs = tomo.bright.num_imgs

#Locations of tomography dark field images
tdf_data_folder = tomo.dark.folder
tdf_img_start = tomo.dark.img_start#for this rate, this is the 6th file in the folder
tdf_num_imgs = tomo.dark.num_imgs

#Locations of tomography images
tomo_data_folder = tomo.images.folder
tomo_img_start = tomo.images.img_start#for this rate, this is the 6th file in the folder
tomo_num_imgs = tomo.images.num_imgs

#==============================================================================
# %% USER OPTIONS -CAN BE EDITED
#==============================================================================


ome_range_deg = tomo.ome_range #degrees


#tomography options
recon_thresh = tomo.processing.recon_thresh#usually varies between 0.0001 and 0.0005
noise_obj_size = tomo.processing.noise_obj_size
min_hole_size = tomo.processing.min_hole_size
erosion_iter = tomo.processing.erosion_iter
dilation_iter = tomo.processing.dilation_iter


project_single_layer = tomo.reconstruction.project_single_layer #projects the center layers through the volume
cross_sectional_dim = tomo.reconstruction.cross_sectional_dim #cross sectional to reconstruct (should be at least 20%-30% over sample width)
voxel_spacing = tomo.reconstruction.voxel_spacing#in mm
v_bnds = tomo.reconstruction.v_bnds


#==============================================================================
# %% LOAD INSTRUMENT DATA
#==============================================================================


instr=load_instrument(det_file)

panel = next(iter(instr.detectors.values()))



nrows=panel.rows
ncols=panel.cols
pixel_size=panel.pixel_size_row


rot_axis_pos=panel.tvec[0] #should match t_vec_d[0] from nf_detector_parameter_file
vert_beam_center=panel.tvec[1]


# need to do a few calculations because not every row will be reconstructed
# depending on sampling
vert_points=np.arange(v_bnds[0]+voxel_spacing/2.,v_bnds[1],voxel_spacing)

center_layer_row=nrows/2.+vert_beam_center/pixel_size

rows_to_recon=np.round(center_layer_row-vert_points/pixel_size).astype(int)

center_layer_row=int(center_layer_row)

#==============================================================================
# %% TOMO PROCESSING - GENERATE DARK AND BRIGHT FIELD
#==============================================================================
tdf=tomoutil.gen_median_image(tdf_data_folder,tdf_img_start,tdf_num_imgs,nrows,ncols,stem=img_stem,num_digits=num_digits)
tbf=tomoutil.gen_median_image(tbf_data_folder,tbf_img_start,tbf_num_imgs,nrows,ncols,stem=img_stem,num_digits=num_digits)

#==============================================================================
# %% TOMO PROCESSING - BUILD RADIOGRAPHS
#==============================================================================

rad_stack=tomoutil.gen_attenuation_rads(tomo_data_folder,tbf,tomo_img_start,tomo_num_imgs,nrows,ncols,stem=stem,num_digits=num_digits,tdf=tdf)



#==============================================================================
# %% TOMO PROCESSING - INVERT SINOGRAM
#==============================================================================
# center = 0.0

test_fbp=tomoutil.tomo_reconstruct_layer(rad_stack,cross_sectional_dim,layer_row=center_layer_row,\
                                                   start_tomo_ang=ome_range_deg[0][0],end_tomo_ang=ome_range_deg[0][1],\
                                                   tomo_num_imgs=tomo_num_imgs, center=rot_axis_pos,pixel_size=pixel_size)

test_binary_recon=tomoutil.threshold_and_clean_tomo_layer(test_fbp,recon_thresh, \
                                                          noise_obj_size,min_hole_size, erosion_iter=erosion_iter, \
                                                          dilation_iter=dilation_iter)

tomo_mask_center=tomoutil.crop_and_rebin_tomo_layer(test_binary_recon,recon_thresh,voxel_spacing,pixel_size,cross_sectional_dim)


#==============================================================================
# %% TOMO PROCESSING - VIEW RAW FILTERED BACK PROJECTION
#==============================================================================

if matplot:
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

full_mask=np.zeros([len(rows_to_recon),tomo_mask_center.shape[0],tomo_mask_center.shape[1]])


for ii in np.arange(len(rows_to_recon)):
    print('Layer: ' + str(ii) + ' of ' + str(len(rows_to_recon)))

    if project_single_layer: #not recommended option
        full_mask[ii]=tomo_mask_center

    else:
        reconstruction_fbp=tomoutil.tomo_reconstruct_layer(rad_stack,cross_sectional_dim,layer_row=rows_to_recon[ii],\
                                                       start_tomo_ang=ome_range_deg[0][0],end_tomo_ang=ome_range_deg[0][1],\
                                                       tomo_num_imgs=tomo_num_imgs, center=rot_axis_pos,pixel_size=pixel_size)

        binary_recon=tomoutil.threshold_and_clean_tomo_layer(reconstruction_fbp,recon_thresh, \
                                                             noise_obj_size,min_hole_size,erosion_iter=erosion_iter, \
                                                             dilation_iter=dilation_iter)

        tomo_mask=tomoutil.crop_and_rebin_tomo_layer(binary_recon,recon_thresh,voxel_spacing,pixel_size,cross_sectional_dim)

        full_mask[ii]=tomo_mask

#==============================================================================
# %%  TOMO PROCESSING - VIEW LAST TOMO_MASK FOR SAMPLE BOUNDS
#==============================================================================

if matplot:
    plt.figure(3)
    plt.imshow(tomo_mask,interpolation='none')
    plt.title('Check Center Mask')

#==============================================================================
# %%  TOMO PROCESSING - CONSTRUCT DATA GRID
#==============================================================================

test_crds, n_crds, Xs, Ys, Zs = nfutil.gen_nf_test_grid_tomo(full_mask.shape[2], full_mask.shape[1], v_bnds, voxel_spacing)

#%%

np.savez(os.path.join(data_folder, '%s_tomo_mask.npz' % analysis_name,mask=full_mask,Xs=Xs,Ys=Ys,Zs=Zs,voxel_spacing=voxel_spacing)
