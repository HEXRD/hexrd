import numpy as np

import multiprocessing as mp
import yaml

import os
import logging
import psutil
import argparse

from hexrd import rotations as rot
from hexrd.grainmap import nfutil
from hexrd.material import Material
from hexrd.valunits import valWUnit
from hexrd import nf_config
from hexrd import instrument

try:
    import matplotlib.pyplot as plt
    matplot = True
except(ImportError):
    logging.warning(f'no matplotlib, debug plotting disabled')
    matplot = False


def load_instrument(yml):
    with open(yml, 'r') as f:
        icfg = yaml.safe_load(f)
    return instrument.HEDMInstrument(instrument_config=icfg)


#%%

parser = argparse.ArgumentParser(description='Preprocess NF image stack')

parser.add_argument('input_file', type=str, help='Input File for NF reconstruction')

args = parser.parse_args()
fname = args.input_file

#%%
cfg = nf_config.open(fname)[0]

analysis_name = cfg.analysis_name

img_folder = cfg.images.data_folder
img_stem = cfg.images.stem
num_digits = cfg.images.num_digits
main_dir = cfg.main_dir
output_dir = cfg.output_dir

#full_stem = os.path.join(img_folder, img_stem)

det_fname = cfg.input_files.det_file

img_method = cfg.images.processing.method
img_nums = np.arange(cfg.images.img_start,
                     cfg.images.img_start + cfg.images.nframes, 1)
num_for_dark = cfg.images.processing.num_for_dark

img_threshold = cfg.images.processing.img_threshold
ome_dilation_iter = cfg.images.processing.ome_dilation_iter

det_file = os.path.join(main_dir, det_fname)
instr = load_instrument(det_file)
panel = next(iter(instr.detectors.values()))
nrows, ncols = panel.shape


if 'gaussian' in img_method:
    process_type = 'gaussian'
    process_args = np.array(
        [img_method['gaussian']['sigma'], img_method['gaussian']['size']])
elif 'dilations_only' in img_method:
    process_type = 'dilations_only'
    process_args = np.array([img_method['dilations_only']['num_erosions'],
                            img_method['dilations_only']['num_dilations']])
else:
    print('Invalid image processing method')


#==============================================================================
# %% NEAR FIELD - MAKE MEDIAN DARK
#==============================================================================

dark = nfutil.gen_nf_dark(img_folder, img_nums, num_for_dark, nrows,
                          ncols, dark_type='median',
                          num_digits=num_digits, stem=img_stem)

#==============================================================================
# %% NEAR FIELD - LOAD IMAGE DATA AND PROCESS
#==============================================================================

image_stack = nfutil.gen_nf_cleaned_image_stack(img_folder, img_nums, dark,
                                                nrows, ncols,
                                                process_type=process_type,
                                                process_args=process_args,
                                                threshold=img_threshold,
                                                ome_dilation_iter=ome_dilation_iter,
                                                num_digits=5, stem=img_stem)


np.savez(os.path.join(output_dir, (analysis_name+'.npz')),
         image_stack=image_stack, dark=dark)
