
# ============================================================
# Copyright (c) 2012, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by Joel Bernier <bernier2@llnl.gov> and others.
# LLNL-CODE-529294.
# All rights reserved.
#
# This file is part of HEXRD. For details on dowloading the source,
# see the file COPYING.
#
# Please also see the file LICENSE.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License (as published by the
# Free Software Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the terms and conditions of the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this program (see file LICENSE); if not, write to
# the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
# Boston, MA 02111-1307 USA or visit <http://www.gnu.org/licenses/>.
# ============================================================

import numpy as np

from scipy import integrate
from scipy import ndimage as imgproc
from scipy import optimize

from hexrd import constants
from hexrd.fitting import peakfunctions as pkfuncs

import matplotlib.pyplot as plt


# =============================================================================
# Helper Functions and Module Vars
# =============================================================================

ftol = constants.sqrt_epsf
xtol = constants.sqrt_epsf


def snip1d(y, w=4, numiter=2):
    """Return SNIP-estimated baseline-background for given spectrum y."""
    z = np.log(np.log(np.sqrt(y + 1) + 1) + 1)
    b = z
    for i in range(numiter):
        for p in range(w, 0, -1):
            kernel = np.zeros(p*2 + 1)
            kernel[0] = kernel[-1] = 1./2.
            b = np.minimum(
                b,
                imgproc.convolve1d(z, kernel, mode='nearest')
            )
        z = b
    # bfull = np.zeros_like(y)
    # bfull[~zeros_idx] = b
    bkg = (np.exp(np.exp(b) - 1) - 1)**2 - 1
    return bkg


def lin_fit_obj(x, m, b):
    return m*np.asarray(x) + b


def lin_fit_jac(x, m, b):
    return np.vstack([x, np.ones_like(x)]).T


# =============================================================================
# 1-D Peak Fitting
# =============================================================================


def estimate_pk_parms_1d(x, f, pktype='pvoigt'):
    """
    Gives initial guess of parameters for analytic fit of one dimensional peak
    data.

    Required Arguments:
    x -- (n) ndarray of coordinate positions
    f -- (n) ndarray of intensity measurements at coordinate positions x
    pktype -- string, type of analytic function that will be used to fit the
    data, current options are "gaussian", "lorentzian",
    "pvoigt" (psuedo voigt), and "split_pvoigt" (split psuedo voigt)

    Outputs:
    p -- (m) ndarray containing initial guesses for parameters for the input
    peaktype
    (see peak function help for what each parameters corresponds to)
    """
    npts = len(x)
    assert len(f) == npts, "ordinate and data must be same length!"

    # handle background
    # ??? make kernel width a kwarg?
    bkg = snip1d(f, w=int(2*npts/3.))

    # fit linear bg and grab params
    bp, _ = optimize.curve_fit(lin_fit_obj, x, bkg, jac=lin_fit_jac)
    bg0 = bp[-1]
    bg1 = bp[0]

    # set remaining params
    pint = f - lin_fit_obj(x, *bp)
    cen_index = np.argmax(pint)
    A = pint[cen_index]
    x0 = x[cen_index]

    # fix center index
    if cen_index > 0 and cen_index < npts - 1:
        left_hm = np.argmin(abs(pint[:cen_index] - 0.5*A))
        right_hm = np.argmin(abs(pint[cen_index:] - 0.5*A))
    elif cen_index == 0:
        right_hm = np.argmin(abs(pint[cen_index:] - 0.5*A))
        left_hm = right_hm
    elif cen_index == npts - 1:
        left_hm = np.argmin(abs(pint[:cen_index] - 0.5*A))
        right_hm = left_hm

    # FWHM estimation
    try:
        FWHM = x[cen_index + right_hm] - x[left_hm]
    except(IndexError):
        FWHM = 0
    if FWHM <= 0 or FWHM > 0.75*npts:
        # something is weird, so punt...
        FWHM = 0.25*(x[-1] - x[0])

    # set params
    if pktype in ['gaussian', 'lorentzian']:
        p = [A, x0, FWHM, bg0, bg1]
    elif pktype == 'pvoigt':
        p = [A, x0, FWHM, 0.5, bg0, bg1]
    elif pktype == 'split_pvoigt':
        p = [A, x0, FWHM, FWHM, 0.5, 0.5, bg0, bg1]
    else:
        raise RuntimeError("pktype '%s' not understood" % pktype)

    return np.r_[p]


def fit_pk_parms_1d(p0, x, f, pktype='pvoigt'):
    """
    Performs least squares fit to find parameters for 1d analytic functions fit
    to diffraction data

    Required Arguments:
    p0 -- (m) ndarray containing initial guesses for parameters
              for the input peaktype
    x -- (n) ndarray of coordinate positions
    f -- (n) ndarray of intensity measurements at coordinate positions x
    pktype -- string, type of analytic function that will be used to
                      fit the data,
    current options are "gaussian","lorentzian","pvoigt" (psuedo voigt), and
    "split_pvoigt" (split psuedo voigt)


    Outputs:
    p -- (m) ndarray containing fit parameters for the input peaktype
    (see peak function help for what each parameters corresponds to)


    Notes:
    1. Currently no checks are in place to make sure that the guess of
    parameters has a consistent number of parameters with the requested
    peak type
    """

    weight = np.max(f)*10.  # hard coded should be changed
    fitArgs = (x, f, pktype)
    if pktype == 'gaussian':
        p, outflag = optimize.leastsq(
            fit_pk_obj_1d, p0,
            args=fitArgs, Dfun=eval_pk_deriv_1d,
            ftol=ftol, xtol=xtol
        )
    elif pktype == 'lorentzian':
        p, outflag = optimize.leastsq(
            fit_pk_obj_1d, p0,
            args=fitArgs, Dfun=eval_pk_deriv_1d,
            ftol=ftol, xtol=xtol
        )
    elif pktype == 'pvoigt':
        lb = [p0[0]*0.5, np.min(x), 0., 0., 0., None]
        ub = [p0[0]*2.0, np.max(x), 4.*p0[2], 1., 2.*p0[4], None]

        fitArgs = (x, f, pktype, weight, lb, ub)
        p, outflag = optimize.leastsq(
            fit_pk_obj_1d_bnded, p0,
            args=fitArgs,
            ftol=ftol, xtol=xtol
        )
    elif pktype == 'split_pvoigt':
        lb = [p0[0]*0.5, np.min(x), 0., 0., 0., 0., 0., None]
        ub = [p0[0]*2.0, np.max(x), 4.*p0[2], 4.*p0[2], 1., 1., 2.*p0[4], None]
        fitArgs = (x, f, pktype, weight, lb, ub)
        p, outflag = optimize.leastsq(
            fit_pk_obj_1d_bnded, p0,
            args=fitArgs,
            ftol=ftol, xtol=xtol
        )
    elif pktype == 'tanh_stepdown':
        p, outflag = optimize.leastsq(
            fit_pk_obj_1d, p0,
            args=fitArgs,
            ftol=ftol, xtol=xtol)
    else:
        p = p0
        print('non-valid option, returning guess')

    if np.any(np.isnan(p)):
        p = p0
        print('failed fitting, returning guess')

    return p


def fit_mpk_parms_1d(p0,x,f0,pktype,num_pks,bgtype=None,bnds=None):
    """
    Performs least squares fit to find parameters for MULTIPLE 1d analytic functions fit
    to diffraction data


    Required Arguments:
    p0 -- (m x u + v) guess of peak parameters for number of peaks, m is the number of
    parameters per peak ("gaussian" and "lorentzian" - 3, "pvoigt" - 4,  "split_pvoigt"
    - 5), v is the number of parameters for chosen bgtype
    x -- (n) ndarray of coordinate positions
    f -- (n) ndarray of intensity measurements at coordinate positions x
    pktype -- string, type of analytic function that will be used to fit the data,
    current options are "gaussian","lorentzian","pvoigt" (psuedo voigt), and
    "split_pvoigt" (split psuedo voigt)
    num_pks -- integer 'u' indicating the number of pks, must match length of p
    pktype -- string, background functions, available options are "constant",
    "linear", and "quadratic"
    bnds -- tuple containing

    Outputs:
    p -- (m x u) fit peak parameters for number of peaks, m is the number of
    parameters per peak ("gaussian" and "lorentzian" - 3, "pvoigt" - 4,  "split_pvoigt"
    - 5)
    """

    fitArgs=(x,f0,pktype,num_pks,bgtype)

    ftol=1e-6
    xtol=1e-6

    if bnds != None:
        p = optimize.least_squares(fit_mpk_obj_1d, p0,bounds=bnds, args=fitArgs,ftol=ftol,xtol=xtol)
    else:
        p = optimize.least_squares(fit_mpk_obj_1d, p0, args=fitArgs,ftol=ftol,xtol=xtol)

    return p.x


def estimate_mpk_parms_1d(pk_pos_0,x,f,pktype='pvoigt',bgtype='linear',fwhm_guess=0.07,center_bnd=0.02):


    num_pks=len(pk_pos_0)
    min_val=np.min(f)



    if pktype == 'gaussian' or pktype == 'lorentzian':
        p0tmp=np.zeros([num_pks,3])
        p0tmp_lb=np.zeros([num_pks,3])
        p0tmp_ub=np.zeros([num_pks,3])

        #x is just 2theta values
        #make guess for the initital parameters
        for ii in np.arange(num_pks):
            pt=np.argmin(np.abs(x-pk_pos_0[ii]))
            p0tmp[ii,:]=[(f[pt]-min_val),pk_pos_0[ii],fwhm_guess]
            p0tmp_lb[ii,:]=[(f[pt]-min_val)*0.1,pk_pos_0[ii]-center_bnd,fwhm_guess*0.5]
            p0tmp_ub[ii,:]=[(f[pt]-min_val)*10.0,pk_pos_0[ii]+center_bnd,fwhm_guess*2.0]
    elif pktype == 'pvoigt':
        p0tmp=np.zeros([num_pks,4])
        p0tmp_lb=np.zeros([num_pks,4])
        p0tmp_ub=np.zeros([num_pks,4])

        #x is just 2theta values
        #make guess for the initital parameters
        for ii in np.arange(num_pks):
            pt=np.argmin(np.abs(x-pk_pos_0[ii]))
            p0tmp[ii,:]=[(f[pt]-min_val),pk_pos_0[ii],fwhm_guess,0.5]
            p0tmp_lb[ii,:]=[(f[pt]-min_val)*0.1,pk_pos_0[ii]-center_bnd,fwhm_guess*0.5,0.0]
            p0tmp_ub[ii,:]=[(f[pt]-min_val+1.)*10.0,pk_pos_0[ii]+center_bnd,fwhm_guess*2.0,1.0]
    elif pktype == 'split_pvoigt':
        p0tmp=np.zeros([num_pks,6])
        p0tmp_lb=np.zeros([num_pks,6])
        p0tmp_ub=np.zeros([num_pks,6])

        #x is just 2theta values
        #make guess for the initital parameters
        for ii in np.arange(num_pks):
            pt=np.argmin(np.abs(x-pk_pos_0[ii]))
            p0tmp[ii,:]=[(f[pt]-min_val),pk_pos_0[ii],fwhm_guess,fwhm_guess,0.5,0.5]
            p0tmp_lb[ii,:]=[(f[pt]-min_val)*0.1,pk_pos_0[ii]-center_bnd,fwhm_guess*0.5,fwhm_guess*0.5,0.0,0.0]
            p0tmp_ub[ii,:]=[(f[pt]-min_val)*10.0,pk_pos_0[ii]+center_bnd,fwhm_guess*2.0,fwhm_guess*2.0,1.0,1.0]


    if bgtype=='linear':
        num_pk_parms=len(p0tmp.ravel())
        p0=np.zeros(num_pk_parms+2)
        lb=np.zeros(num_pk_parms+2)
        ub=np.zeros(num_pk_parms+2)
        p0[:num_pk_parms]=p0tmp.ravel()
        lb[:num_pk_parms]=p0tmp_lb.ravel()
        ub[:num_pk_parms]=p0tmp_ub.ravel()


        p0[-2]=min_val

        lb[-2]=-float('inf')
        lb[-1]=-float('inf')

        ub[-2]=float('inf')
        ub[-1]=float('inf')

    elif bgtype=='constant':
        num_pk_parms=len(p0tmp.ravel())
        p0=np.zeros(num_pk_parms+1)
        lb=np.zeros(num_pk_parms+1)
        ub=np.zeros(num_pk_parms+1)
        p0[:num_pk_parms]=p0tmp.ravel()
        lb[:num_pk_parms]=p0tmp_lb.ravel()
        ub[:num_pk_parms]=p0tmp_ub.ravel()


        p0[-1]=min_val
        lb[-1]=-float('inf')
        ub[-1]=float('inf')

    elif bgtype=='quadratic':
        num_pk_parms=len(p0tmp.ravel())
        p0=np.zeros(num_pk_parms+3)
        lb=np.zeros(num_pk_parms+3)
        ub=np.zeros(num_pk_parms+3)
        p0[:num_pk_parms]=p0tmp.ravel()
        lb[:num_pk_parms]=p0tmp_lb.ravel()
        ub[:num_pk_parms]=p0tmp_ub.ravel()


        p0[-3]=min_val
        lb[-3]=-float('inf')
        lb[-2]=-float('inf')
        lb[-1]=-float('inf')
        ub[-3]=float('inf')
        ub[-2]=float('inf')
        ub[-1]=float('inf')

    bnds=(lb,ub)




    return p0, bnds

def eval_pk_deriv_1d(p,x,y0,pktype):

    if pktype == 'gaussian':
        d_mat=pkfuncs.gaussian1d_deriv(p,x)
    elif pktype == 'lorentzian':
        d_mat=pkfuncs.lorentzian1d_deriv(p,x)

    return d_mat.T


def fit_pk_obj_1d(p,x,f0,pktype):
    if pktype == 'gaussian':
        f=pkfuncs.gaussian1d(p,x)
    elif pktype == 'lorentzian':
        f=pkfuncs.lorentzian1d(p,x)
    elif pktype == 'pvoigt':
        f=pkfuncs.pvoigt1d(p,x)
    elif pktype == 'split_pvoigt':
        f=pkfuncs.split_pvoigt1d(p,x)
    elif pktype == 'tanh_stepdown':
        f=pkfuncs.tanh_stepdown_nobg(p,x)

    resd = f-f0
    return resd


def fit_pk_obj_1d_bnded(p,x,f0,pktype,weight,lb,ub):
    if pktype == 'gaussian':
        f=pkfuncs.gaussian1d(p,x)
    elif pktype == 'lorentzian':
        f=pkfuncs.lorentzian1d(p,x)
    elif pktype == 'pvoigt':
        f=pkfuncs.pvoigt1d(p,x)
    elif pktype == 'split_pvoigt':
        f=pkfuncs.split_pvoigt1d(p,x)

    num_data=len(f)
    num_parm=len(p)
    resd=np.zeros(num_data+num_parm)
    #tub bnds implementation

    resd[:num_data] = f-f0
    for ii in range(num_parm):
        if lb[ii] is not None:
            resd[num_data+ii]=weight*np.max([-(p[ii]-lb[ii]),0.,(p[ii]-ub[ii])])


    return resd


def fit_mpk_obj_1d(p,x,f0,pktype,num_pks,bgtype):

    f=pkfuncs.mpeak_1d(p,x,pktype,num_pks,bgtype='linear')
    resd = f-f0
    return resd





#### 2-D Peak Fitting

def estimate_pk_parms_2d(x,y,f,pktype):
    """
    Gives initial guess of parameters for analytic fit of two dimensional peak
    data.

    Required Arguments:
    x -- (n x 0) ndarray of coordinate positions for dimension 1 (numpy.meshgrid formatting)
    y -- (n x 0) ndarray of coordinate positions for dimension 2 (numpy.meshgrid formatting)
    f -- (n x 0) ndarray of intensity measurements at coordinate positions x and y
    pktype -- string, type of analytic function that will be used to fit the data,
    current options are "gaussian", "gaussian_rot" (gaussian with arbitrary axes) and
    "split_pvoigt_rot" (split psuedo voigt with arbitrary axes)


    Outputs:
    p -- (m) ndarray containing initial guesses for parameters for the input peaktype
    (see peakfunction help for more information)
    """



    bg0=np.mean([f[0,0],f[-1,0],f[-1,-1],f[0,-1]])
    bg1x=(np.mean([f[-1,-1],f[0,-1]])-np.mean([f[0,0],f[-1,0]]))/(x[0,-1]-x[0,0])
    bg1y=(np.mean([f[-1,-1],f[-1,0]])-np.mean([f[0,0],f[0,-1]]))/(y[-1,0]-y[0,0])

    fnobg=f-(bg0+bg1x*x+bg1y*y)

    labels,numlabels=imgproc.label(fnobg>np.max(fnobg)/2.)

    #looks for the largest peak
    areas=np.zeros(numlabels)
    for ii in np.arange(1,numlabels+1,1):
        areas[ii-1]= np.sum(labels==ii)

    peakIndex=np.argmax(areas)+1


#    #currently looks for peak closest to center
#    dist=np.zeros(numlabels)
#    for ii in np.arange(1,numlabels+1,1):
#        dist[ii-1]= ######
#
#    peakIndex=np.argmin(dist)+1

    FWHMx=np.max(x[labels==peakIndex])-np.min(x[labels==peakIndex])
    FWHMy=np.max(y[labels==peakIndex])-np.min(y[labels==peakIndex])

    coords=imgproc.maximum_position(fnobg, labels=labels, index=peakIndex)
    A=imgproc.maximum(fnobg, labels=labels, index=peakIndex)
    x0=x[coords]
    y0=y[coords]

    if pktype=='gaussian':
        p=[A,x0,y0,FWHMx,FWHMy,bg0,bg1x,bg1y]
    elif pktype=='gaussian_rot':
        p=[A,x0,y0,FWHMx,FWHMy,0.,bg0,bg1x,bg1y]
    elif pktype=='split_pvoigt_rot':
        p=[A,x0,y0,FWHMx,FWHMx,FWHMy,FWHMy,0.5,0.5,0.5,0.5,0.,bg0,bg1x,bg1y]

    p=np.array(p)
    return p


def fit_pk_parms_2d(p0,x,y,f,pktype):
    """
    Performs least squares fit to find parameters for 2d analytic functions fit
    to diffraction data

    Required Arguments:
    p0 -- (m) ndarray containing initial guesses for parameters for the input peaktype
    x -- (n x 0) ndarray of coordinate positions for dimension 1 (numpy.meshgrid formatting)
    y -- (n x 0) ndarray of coordinate positions for dimension 2 (numpy.meshgrid formatting)
    f -- (n x 0) ndarray of intensity measurements at coordinate positions x and y
    pktype -- string, type of analytic function that will be used to fit the data,
    current options are "gaussian", "gaussian_rot" (gaussian with arbitrary axes) and
    "split_pvoigt_rot" (split psuedo voigt with arbitrary axes)


    Outputs:
    p -- (m) ndarray containing fit parameters for the input peaktype (see peak function
    help for what each parameters corresponds to)


    Notes:
    1. Currently no checks are in place to make sure that the guess of parameters
    has a consisten number of parameters with the requested peak type
    """


    fitArgs=(x,y,f,pktype)
    ftol=1e-9
    xtol=1e-9

    if pktype == 'gaussian':
        p, outflag = optimize.leastsq(fit_pk_obj_2d, p0, args=fitArgs,ftol=ftol, xtol=xtol)
    elif pktype == 'gaussian_rot':
        p, outflag = optimize.leastsq(fit_pk_obj_2d, p0, args=fitArgs,ftol=ftol, xtol=xtol)
    elif pktype == 'split_pvoigt_rot':
        p, outflag = optimize.leastsq(fit_pk_obj_2d, p0, args=fitArgs,ftol=ftol, xtol=xtol)


    if np.any(np.isnan(p)):
        p=p0

    return p

def fit_pk_obj_2d(p,x,y,f0,pktype):
    if pktype == 'gaussian':
        f=pkfuncs.gaussian2d(p,x,y)
    elif pktype == 'gaussian_rot':
        f=pkfuncs.gaussian2d_rot(p,x,y)
    elif pktype == 'split_pvoigt_rot':
        f=pkfuncs.split_pvoigt2d_rot(p,x,y)

    resd = f-f0
    return resd.flatten()



#### Extra Utilities

def goodness_of_fit(f,f0):
    """
    Calculates two scalar measures of goodness of fit

    Required Arguments:
    f0 -- (n x 0) ndarray of intensity measurements at coordinate positions
    f -- (n x 0) ndarray of fit intensity values at coordinate positions

    Outputs:
    R -- (1) goodness of fit measure which is sum(error^2)/sum(meas^2)
    Rw -- (1) goodness of fit measure weighted by intensity sum(meas*error^2)/sum(meas^3)
    """


    R=np.sum((f-f0)**2)/np.sum(f0**2)
    Rw=np.sum(np.abs(f0*(f-f0)**2))/np.sum(np.abs(f0**3))

    return R, Rw



def direct_pk_analysis(x,f,remove_bg=True,low_int=1.,edge_pts=3,pts_per_meas=100):
    """
    Performs analysis of a single peak that is not well matched to any analytic functions


    Required Arguments:
    x -- (n) ndarray of coordinate positions
    f -- (n) ndarray of intensity measurements at coordinate positions x

    Optional Arguments:
    remove_bg -- boolean, if selected a linear background will be subtracted from the peak
    low_int -- float, value for area under a peak that defines a lower bound
    on what is recognized as peak
    edge_pts -- int, number of points at the edges of the data to use to calculated background
    pts_per_meas -- how many interpolated points to place between measurement values

    Outputs:
    p -- array of values containing the integrated intensity, center of mass, and
    FWHM of the peak
    """




    plt.plot(x,f)
    #subtract background, assumed linear
    if remove_bg:
        bg_data=np.hstack((f[:(edge_pts+1)],f[-edge_pts:]))
        bg_pts=np.hstack((x[:(edge_pts+1)],x[-edge_pts:]))

        bg_parm=np.polyfit(bg_pts,bg_data,1)

        f=f-(bg_parm[0]*x+bg_parm[1])#pull out high background

        f=f-np.min(f)#set the minimum to 0


    plt.plot(bg_pts,bg_data,'x')
    plt.plot(x,f,'r')

    spacing=np.diff(x)[0]/pts_per_meas
    xfine=np.arange(np.min(x),np.max(x)+spacing,spacing)# make a fine grid of points
    ffine=np.interp(xfine,x,f)

    data_max=np.max(f)#find max intensity values

    total_int=integrate.simps(ffine,xfine)#numerically integrate the peak using the simpson rule

    cen_index=np.argmax(ffine)
    A=data_max

    if(total_int<low_int):#this value is arbitrary, maybe set higher
        com=float('NaN')
        FWHM=float('NaN')
        total_int=total_int
        print('Analysis Failed... Intensity too low')
    else:
        com=np.sum(xfine*ffine)/np.sum(ffine)#center of mass calculation

        a=np.abs(ffine[cen_index+1:]-A/2.)
        b=np.abs(ffine[:cen_index]-A/2.)

        if(a.size==0 or b.size==0): #this is a check to see if the peak is falling out of the bnds
            com=float('NaN')
            FWHM=float('NaN')
            total_int=total_int
            print('Analysis Failed... Peak is not well defined')
        else:
            FWHM=xfine[cen_index+np.argmin(a)]-xfine[np.argmin(b)]
                   #calculate positions on the left and right half of peaks at half maxmum
                   #think about changing to full width @ 10% max?

    p=[total_int,com,FWHM]
    p=np.array(p)
    return p


def calc_pk_integrated_intensities(p,x,pktype,num_pks):
    """
    Calculates the area under the curve (integrated intensities) for fit peaks


    Required Arguments:
    p -- (m x u + v) peak parameters for number of peaks, m is the number of
    parameters per peak ("gaussian" and "lorentzian" - 3, "pvoigt" - 4,  "split_pvoigt"
    - 5), v is the number of parameters for chosen bgtype
    x -- (n) ndarray of coordinate positions
    f -- (n) ndarray of intensity measurements at coordinate positions x
    pktype -- string, type of analytic function that will be used to fit the data,
    current options are "gaussian","lorentzian","pvoigt" (psuedo voigt), and
    "split_pvoigt" (split psuedo voigt)
    num_pks -- integer 'u' indicating the number of pks, must match length of p

    Outputs:
    ints -- (m) integrated intensities for m fit peaks
    """


    ints=np.zeros(num_pks)

    if pktype == 'gaussian' or pktype == 'lorentzian':
        p_fit=np.reshape(p[:3*num_pks],[num_pks,3])
    elif pktype == 'pvoigt':
        p_fit=np.reshape(p[:4*num_pks],[num_pks,4])
    elif pktype == 'split_pvoigt':
        p_fit=np.reshape(p[:6*num_pks],[num_pks,6])

    for ii in np.arange(num_pks):
        if pktype == 'gaussian':
            ints[ii]=integrate.simps(pkfuncs._gaussian1d_no_bg(p_fit[ii],x),x)
        elif pktype == 'lorentzian':
            ints[ii]=integrate.simps(pkfuncs._lorentzian1d_no_bg(p_fit[ii],x),x)
        elif pktype == 'pvoigt':
            ints[ii]=integrate.simps(pkfuncs._pvoigt1d_no_bg(p_fit[ii],x),x)
        elif pktype == 'split_pvoigt':
            ints[ii]=integrate.simps(pkfuncs._split_pvoigt1d_no_bg(p_fit[ii],x),x)


    return ints
