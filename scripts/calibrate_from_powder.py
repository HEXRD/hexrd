#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:04:31 2017

@author: bernier2
"""
import os

import glob

import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt

import numpy as np

from scipy.optimize import leastsq, least_squares
try:
    import dill as cpl
except(ImportError):
    import pickle as cpl

from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.morphology import disk, binary_erosion

import yaml

from hexrd import imageseries
from hexrd import instrument
from hexrd import material
from hexrd.fitting import fitpeak
from hexrd.rotations import \
    angleAxisOfRotMat, angles_from_rmat_xyz, make_rmat_euler, mapAngle
from hexrd.xrdutil import make_reflection_patches


# plane data
def load_pdata(cpkl, key):
    with open(cpkl, "rb") as matf:
        mat_list = cpl.load(matf)
    return dict(list(zip([i.name for i in mat_list], mat_list)))[key].planeData


# images
def load_images(yml):
    return imageseries.open(yml, format="image-files")


# instrument
def load_instrument(yml):
    with open(yml, 'r') as f:
        icfg = yaml.safe_load(f)
    return instrument.HEDMInstrument(instrument_config=icfg)


# build a material on the fly
def make_matl(mat_name, sgnum, lparms, hkl_ssq_max=50):
    matl = material.Material(mat_name)
    matl.sgnum = sgnum
    matl.latticeParameters = lparms
    matl.hklMax = hkl_ssq_max

    nhkls = len(matl.planeData.exclusions)
    matl.planeData.set_exclusions(np.zeros(nhkls, dtype=bool))
    return matl


# %%
# =============================================================================
# USER INPUT
# =============================================================================

# dirs
working_dir = '/Users/Shared/APS/PUP_AFRL_Feb19'
image_dir = os.path.join(working_dir, 'image_data')

samp_name = 'ceria_cal'
scan_number = 0

raw_data_dir_template = os.path.join(
    image_dir,
    'raw_images_%s_%06d-%s.yml'
)

instrument_filename = 'Hydra_Feb19.yml'

#mat_filename = 'materials.hexrd'
#mat_key = 'ruby'
#plane_data = load_pdata(os.path.join(working_dir, mat_filename), mat_key)
#plane_data.lparms = np.r_[2.508753]
matl = make_matl('ceria', 225, [5.411102, ])

# tolerances for patches
tth_tol = 0.15
eta_tol = 5.

tth_max = 9.

# peak fit type
#pktype = 'gaussian'
pktype = 'pvoigt'

# Try it
use_robust_optimization = False

#refinement_type = "translation_only"
refinement_type = "translation_and_tilt"
#refinement_type = "beam_and_translation"


# %%
# =============================================================================
# IMAGESERIES
# =============================================================================

# for applying processing options (flips, dark subtretc...)
PIS = imageseries.process.ProcessedImageSeries

# load instrument
#instr = load_instrument(os.path.join(working_dir, instrument_filename))
instr = load_instrument(instrument_filename)
det_keys = list(instr.detectors.keys())

# hijack panel buffer
edge_buff = 5
for k, v in list(instr.detectors.items()):
    try:
        mask = np.load(mask_template % k)
    except(NameError):
        mask = np.ones((v.rows, v.cols), dtype=bool)
    mask[:edge_buff, :] = False
    mask[:, :edge_buff] = False
    mask[-edge_buff:, :] = False
    mask[:, -edge_buff:] = False
    v.panel_buffer = mask


# grab imageseries filenames
file_names = glob.glob(
    raw_data_dir_template % (samp_name, scan_number, '*')
)
check_files_exist = [os.path.exists(file_name) for file_name in file_names]
if not np.all(check_files_exist):
    raise RuntimeError("files don't exist!")

img_dict = dict.fromkeys(det_keys)
for k, v in img_dict.items():
    file_name = glob.glob(
        raw_data_dir_template % (samp_name, scan_number, k)
    )
    ims = load_images(file_name[0])
    img_dict[k] = ims[0]

# plane data munging
matl.beamEnergy = instr.beam_energy
plane_data = matl.planeData
if tth_tol is not None:
    plane_data.tThWidth = np.radians(tth_tol)

if tth_max is not None:
    plane_data.exclusions = None
    plane_data.tThMax = np.radians(tth_max)

# %%
# =============================================================================
# INSTRUMENT
# =============================================================================


# get powder line profiles
#
# output is as follows:
#
# patch_data = {<detector_key>:[ringset_index][patch_index]}
# patch_data[0] = [two_theta_edges, ref_eta]
# patch_data[1] = [intensities]
#
powder_lines = instr.extract_line_positions(
        plane_data, img_dict,
        tth_tol=np.degrees(plane_data.tThWidth), eta_tol=eta_tol,
        npdiv=2,
        collapse_eta=False, collapse_tth=False,
        do_interpolation=True)


# ideal tth
tth_ideal = plane_data.getTTh()
tth0 = []
for idx in plane_data.getMergedRanges()[0]:
    tth0.append(tth_ideal[idx[0]])

# GRAND LOOP OVER PATCHES
rhs = dict.fromkeys(det_keys)
for det_key in det_keys:
    rhs[det_key] = []
    panel = instr.detectors[det_key]
    for i_ring, ringset in enumerate(powder_lines[det_key]):
        tmp = []
        for angs, intensities in ringset:
            tth_centers = np.average(
                np.vstack([angs[0][:-1], angs[0][1:]]),
                axis=0)
            eta_ref = angs[1]
            int1d = np.sum(np.array(intensities).squeeze(), axis=0)
            """
            DARREN: FIT [tth_centers, intensities[0]] HERE

            RETURN TTH0
            rhs.append([tth0, eta_ref])
            """
            p0 = fitpeak.estimate_pk_parms_1d(
                    tth_centers, int1d, pktype
                 )

            p = fitpeak.fit_pk_parms_1d(
                    p0, tth_centers, int1d, pktype
                )
            # !!! this is where we can kick out bunk fits
            tth_ref = plane_data.getTTh()[i_ring]
            tth_meas = p[1]
            if p[0] < 0.1 or abs(tth_meas - tth_ref) > np.radians(tth_tol/2.):
                continue
            xy_meas = panel.angles_to_cart([[tth_meas, eta_ref], ])
            tmp.append(
                np.hstack([xy_meas.squeeze(), tth_meas, tth0[i_ring], eta_ref])
            )
        rhs[det_key].append(np.vstack(tmp))
    rhs[det_key] = np.array(rhs[det_key])

# %% plot fit poistions
fig, ax = plt.subplots(2, 2)
fig_row, fig_col = np.unravel_index(np.arange(instr.num_panels), (2, 2))

ifig = 0
for det_key, panel in instr.detectors.items():
    all_pts = np.vstack(rhs[det_key])
    '''
    pimg = equalize_adapthist(
            rescale_intensity(img_dict[det_key], out_range=(0, 1)),
            10, clip_limit=0.2)
    '''
    pimg = np.array(img_dict[det_key], dtype=float)
    pimg[~panel.panel_buffer] = np.nan
    ax[fig_row[ifig], fig_col[ifig]].imshow(
        pimg,
        vmin=np.percentile(img_dict[det_key], 5),
        vmax=np.percentile(img_dict[det_key], 90),
        cmap=plt.cm.bone_r
    )
    ideal_angs, ideal_xys = panel.make_powder_rings(plane_data,
                                                    delta_eta=eta_tol)
    rijs = panel.cartToPixel(np.vstack(ideal_xys))
    ax[fig_row[ifig], fig_col[ifig]].plot(rijs[:, 1], rijs[:, 0], 'cx')
    ax[fig_row[ifig], fig_col[ifig]].set_title(det_key)
    rijs = panel.cartToPixel(all_pts[:, :2])
    ax[fig_row[ifig], fig_col[ifig]].plot(rijs[:, 1], rijs[:, 0], 'm+')
    ax[fig_row[ifig], fig_col[ifig]].set_title(det_key)
    ifig += 1

# %%
if refinement_type == "translation_only":
    x0 = []
    for k, v in list(instr.detectors.items()):
        x0.append(v.tvec)
    x0 = np.hstack(x0)

    def multipanel_powder_objfunc(param_list, data_dict, instr):
        """
        """
        npp = 3
        ii = 0
        jj = ii + npp
        resd = []
        for det_key, panel in list(instr.detectors.items()):
            # strip params for this panel
            params = param_list[ii:jj]

            # translation
            panel.tvec = np.asarray(params).flatten()

            # advance counters
            ii += npp
            jj += npp

            pdata = np.vstack(data_dict[det_key])
            if len(pdata) > 0:

                calc_xy = panel.angles_to_cart(pdata[:, -2:])

                resd.append(
                    (pdata[:, :2].flatten() - calc_xy.flatten())
                )
            else:
                continue
        return np.hstack(resd)
elif refinement_type == "translation_and_tilt":
    # build parameter list
    x0 = []
    for k, v in instr.detectors.items():
        tilt = np.degrees(angles_from_rmat_xyz(v.rmat))
        print("XYZ tilt angles for '%s': (%.2f, %.2f, %.2f)"
              % tuple([k, *tilt]))
        x0.append(np.hstack([tilt[:2], v.tvec]))
    x0 = np.hstack(x0)

    def multipanel_powder_objfunc(param_list, data_dict, instr):
        """
        """
        npp = 5
        ii = 0
        jj = ii + npp
        resd = []
        '''
        resd = 0
        '''
        for det_key, panel in instr.detectors.items():
            # strip params for this panel
            params = param_list[ii:jj]

            # tilt first
            tilt = np.degrees(angles_from_rmat_xyz(panel.rmat))
            tilt[:2] = params[:2]
            rmat = make_rmat_euler(
                np.radians(tilt), 'xyz', extrinsic=True
            )
            phi, n = angleAxisOfRotMat(rmat)
            panel.tilt = phi*n.flatten()

            # translation
            panel.tvec = params[2:]

            # advance counters
            ii += npp
            jj += npp

            '''
            for rdata in data_dict[det_key]:
                cangs, cgvecs = panel.cart_to_angles(rdata[:, :2])
                resd += np.sum(np.abs(cangs[:, 0] - np.mean(cangs[:, 0])))
            '''
            pdata = np.vstack(data_dict[det_key])
            if len(pdata) > 0:
                calc_xy = panel.angles_to_cart(pdata[:, -2:])

                resd.append(
                    (pdata[:, :2].flatten() - calc_xy.flatten())
                )
            else:
                continue
        return np.hstack(resd)
elif refinement_type == "beam_and_translation":
    # build parameter list
    x0 = list(instrument.calc_angles_from_beam_vec(instr.beam_vector))
    for k, v in list(instr.detectors.items()):
        x0.append(v.tvec)
    x0 = np.hstack(x0)

    def multipanel_powder_objfunc(param_list, data_dict, instr):
        """
        """
        # update beam vector
        instr.beam_vector = param_list[:2]

        # number of parameters per panel
        npp = 3

        # initialize counters and residual
        ii = 2
        jj = ii + npp
        resd = []
        #resd = 0
        for det_key, panel in list(instr.detectors.items()):
            # strip params for this panel
            params = param_list[ii:jj]

            # update translation
            panel.tvec = params

            # advance counters
            ii += npp
            jj += npp

            #for rdata in data_dict[det_key]:
            #    cangs, cgvecs = panel.cart_to_angles(rdata[:, :2])
            #    resd += np.sum(np.abs(cangs[:, 0] - np.mean(cangs[:, 0])))
            pdata = np.vstack(data_dict[det_key])
            if len(pdata) > 0:
                calc_xy = panel.angles_to_cart(pdata[:, -2:])


                resd.append(
                    (pdata[:, :2].flatten() - calc_xy.flatten())
                )
            else:
                continue
        return np.hstack(resd)    # resd


# %%

det_key = 'ge2'

fig, ax = plt.subplots()

rid = 0
ii = 0
for ring_data in powder_lines[det_key][rid]:
    tth_edges = ring_data[0][0]
    tth_centers = 0.5*np.sum(np.vstack([tth_edges[:-1], tth_edges[1:]]), axis=0)
    eta = np.degrees(mapAngle(ring_data[0][1], (0, 2*np.pi)))
    intensities = np.sum(np.array(ring_data[1]).squeeze(), axis=0)
    if ii < 1:
        iprev = 0
    else:
        iprev = 0.25*np.max(np.sum(powder_lines[det_key][rid][ii - 1][1], axis=0))
    ax.plot(tth_centers, intensities + iprev)
    ii += 1

p0 = fitpeak.estimate_pk_parms_1d(
        tth_centers, intensities, pktype
     )

p = fitpeak.fit_pk_parms_1d(
        p0, tth_centers, intensities, pktype
    )

if pktype == 'pvoigt':
    fp0 = fitpeak.pkfuncs.pvoigt1d(p0, tth_centers)
    fp1 = fitpeak.pkfuncs.pvoigt1d(p, tth_centers)
elif pktype == 'gaussian':
    fp0 = fitpeak.pkfuncs.gaussian1d(p0, tth_centers)
    fp1 = fitpeak.pkfuncs.gaussian1d(p, tth_centers)

# %%

initial_guess = np.array(x0)

if use_robust_optimization:
    oresult = least_squares(
        multipanel_powder_objfunc, x0, args=(rhs, instr),
        method='trf', loss='soft_l1'
    )
    x1 = oresult['x']
else:
    x1, cox_x, infodict, mesg, ierr = leastsq(
        multipanel_powder_objfunc, x0, args=(rhs, instr),
        full_output=True
    )
resd0 = multipanel_powder_objfunc(x0, rhs, instr)
resd1 = multipanel_powder_objfunc(x1, rhs, instr)

delta_r = sum(resd0**2)/float(len(resd0)) - sum(resd1**2)/float(len(resd1))

if delta_r > 0:
    print(('OPTIMIZATION SUCCESSFUL\nfinal ssr: %f' % sum(resd1**2)))
    print(('delta_r: %f' % delta_r))
    instr.write_config(instrument_filename)
    x0 = np.array(x1)
else:
    print('no improvement in residual!!!')

# %%


# =============================================================================
# %% PLOTTING
# =============================================================================

cmap = plt.cm.bone_r

#img = np.log(img_dict[det_key] + (1. - np.min(img_dict[det_key])))
img = equalize_adapthist(img_dict[det_key], 20, clip_limit=0.2)
panel = instr.detectors[det_key]

pcfg = panel.config_dict(instr.chi, instr.tvec)
extent = [-0.5*panel.col_dim, 0.5*panel.col_dim,
          -0.5*panel.row_dim, 0.5*panel.row_dim]
pr = panel.make_powder_rings(plane_data,
                             delta_tth=np.degrees(plane_data.tThWidth),
                             delta_eta=eta_tol)

tth_eta = pr[0][rid]
xy_det = pr[1][rid]
rpatches = make_reflection_patches(
    pcfg, tth_eta, panel.angularPixelSize(xy_det),
    omega=None, tth_tol=np.degrees(plane_data.tThWidth), eta_tol=eta_tol,
    distortion=panel.distortion,
    npdiv=2,
    beamVec=instr.beam_vector)

# Create 2x2 sub plots
widths = [1, 3]
heights = [1, 3]
fig = plt.figure(constrained_layout=True)
gs = gridspec.GridSpec(ncols=2, nrows=2,
                       figure=fig,
                       width_ratios=widths,
                       height_ratios=heights)
ax0 = fig.add_subplot(gs[0, 0])  # row 0, col 0
ax1 = fig.add_subplot(gs[0, 1])  # row 0, col 1
ax2 = fig.add_subplot(gs[1, :])  # row 1, span all columns

#vmin = np.percentile(img, 50)
#vmax = np.percentile(img, 99)
vmin = None
vmax = None
img[~panel.panel_buffer] = np.nan

ax2.imshow(img,
           vmin=vmin,
           vmax=vmax,
           cmap=cmap,
           extent=extent)

for rp in rpatches:
    '''
    pts = np.dot(panel.rmat,
                np.vstack([rp[1][0].flatten(),
                           rp[1][1].flatten(),
                           np.zeros(rp[1][0].size)])
    )
    px = pts[0, :]
    py = pts[1, :]
    '''
    px = rp[1][0].flatten()
    py = rp[1][1].flatten()
    _, on_panel = panel.clip_to_panel(np.vstack([px, py]).T)
    if np.any(~on_panel):
        continue
    else:
        good_patch = [px, py]
    ax2.plot(px, py, 'm.', markersize=0.1)
ax2.plot(good_patch[0], good_patch[1], 'c.', markersize=0.1)
aext = np.degrees(
    [np.min(rp[0][0]),
     np.max(rp[0][0]),
     np.min(rp[0][1]),
     np.max(rp[0][1])]
)
ax0.imshow(np.array(ring_data[1]).squeeze(), cmap=cmap,
           extent=aext, aspect=.05,
           vmin=np.percentile(ring_data[1], 50),
           vmax=np.percentile(ring_data[1], 99))
ax1.plot(np.degrees(tth_centers), intensities, 'k.-')
ax1.plot(np.degrees(tth_centers), fp1, 'r:')

ax2.set_xlabel(r'detector $X$ [mm]')
ax2.set_ylabel(r'detector $Y$ [mm]')

ax0.set_xlabel(r'patch $2\theta$')
ax0.set_ylabel(r'patch $\eta$')
ax0.axis('auto')

ax1.set_xlabel(r'patch $2\theta$')
ax1.set_ylabel(r'summed intensity [arb]')


## =============================================================================
## %% ERROR PLOTTING
## =============================================================================
#
#tths = plane_data.getTTh()
#shkls = plane_data.getHKLs(asStr=True)
#ref_etas = np.radians(np.linspace(0, 360, num=181, endpoint=True))
#
#ii = 1
#for tth, shkl in zip(tths, shkls):
#    #fig, ax = plt.subplots(1, 1, figsize=(12, 6))
#    #fig.suptitle('{%s}' % shkl)
#    fig = plt.figure(ii)
#    ax = fig.gca()
#    ax.set_xlim(-180, 180)
#    ax.set_ylim(-0.001, 0.001)
#    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1e'))
#    ax.set_xlabel(r'azimuth, $\eta$ [deg]', size=18)
#    ax.set_ylabel(r'Relative error, $\frac{2\theta-2\theta_0}{2\theta_0}$', size=18)
#    ax.grid(True)
#    for det_key, rdata in rhs.iteritems():
#        if len(rdata) > 0:
#            idx = rdata[:, 3] == tth
#            if np.any(idx):
#                dtth = (rdata[idx, 2] - tth*np.ones(sum(idx)))/tth
#                ax.scatter(np.degrees(rdata[idx, -1]), dtth, c='b', s=12)
#    ii += 1
#
## %%
#for i, shkl in enumerate(shkls):
#    fig = plt.figure(i+1)
#    ax = fig.gca()
#    plt.savefig("CeO2_tth_errors_%s.png" % '_'.join(shkl.split()), dpi=180)
