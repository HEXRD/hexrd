# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 11:31:46 2021

@author: berni
"""
import copy
import time
import argparse

import numpy as np

import h5py

from psana import DataSource, Detector, DetNames

from hexrd import constants as cnst
from hexrd import instrument
from hexrd.material import Material
from hexrd.valunits import valWUnit
from hexrd.wppf import wppfsupport, LeBail
from hexrd.xrdutil import PolarView
from scipy.interpolate import interp1d


# =============================================================================
# LOCAL FUNCTIONS AND PARAMETERS
# =============================================================================

def _angstroms(x):
    return valWUnit('lp', 'length',  x, 'angstrom')


def _nm(x):
    return valWUnit('lp', 'length',  x, 'nm')


def _kev(x):
    return valWUnit('kev', 'energy', x, 'keV')


def GetImage(det, exp, run, event=-1):
    """
    Yields the requested detector image.

    Parameters
    ----------
    det : str
        the detector name.
    exp : str
        the experiment name.
    run : int
        the run number.
    event : int, optional
        the event number. The default is -1, which will return the last event.

    Raises
    ------
    RuntimeError
        When either the experiment name or run do not exist.
    IndexError
        When the event number is out of range.

    Returns
    -------
    numpy.ndarray
        The specifed detector image.

    """
    # instantiate psana.DataSource object
    # !!! raises RuntimeError if exp or run don't exist
    ds = DataSource(
        f"exp={exp}:run={run:d}:smd".format(exp=exp, run=run)
    )

    if not any(det in d for d in DetNames()):
        raise RuntimeError("Detector '%s' not in run" % det)
    else:
        detector = Detector(det)
        events = list(ds.events())
        nevents = len(events)
        print("There are %d events in run %d" % (nevents, run))

        if event == -1:
            event = nevents-1

        if event >= nevents:
            raise IndexError(
                "Run %d only has %d events; you requested event %d"
                % (run, nevents, event)
            )
        else:
            return detector.image(events[event])


def write_hdf(run,
             mat,
             polar_view,
             extent,
             azimuthal_integration,
             LeBail_fit,
             difference_curve,
             lp,
             amb_density,
             density,
             temperature,
             delD,
             delT,
             Rwp):
    """
    @AUTHOR Saransh Singh, Lawrence Livermore National Lab
    @DATE 06/24/2021 1.0 original
    @DETAILS this function writes out the HDF5 file given a run number
    and data. the organiation of the file is as follows:

    Group: run number
    attributes: lattice parameter
                density
                temperature
                uncertainty in density
                uncertainty in temperature
    datasets: polar view
              extent of polar view (IN DEGREES)
              azimuthal integration
              LeBail fit
              difference curve

    """
    fname = f"h5data/run_{run}.h5"
    
    if os.path.exists(fname):
        os.remove(fname)
        msg = f"file already exists. overwriting..."
        print(msg)
    
    fid = h5py.File(fname,"w")
    
    msg = f"filename: {fname}"
    print(msg)
    
    gname = f"{run}"
    gid = fid.create_group(gname)
    
    gid.attrs["temperature"] = temperature
    gid.attrs["ambient_density"] = amb_density
    gid.attrs["density"] = density
    gid.attrs["uncertainty_temperature"] = delT
    gid.attrs["uncertainty_density"] = delD
    gid.attrs["lattice_parameter"] = lp
    gid.attrs["percent Rwp"] = Rwp
    gid.attrs["material"] = mat

    msg = f"groupname: {gname}"
    print(msg)
    
    dname = f"azimuthal_integration"
    dset = gid.create_dataset(dname, data=azimuthal_integration)
    
    dname = f"LeBail_fit"
    dset = gid.create_dataset(dname, data=LeBail_fit)
    
    dname = f"difference_curve"
    dset = gid.create_dataset(dname, data=difference_curve)
    
    dname = f"polarview"
    dset = gid.create_dataset(dname, data=polar_view)

    dname = f"extent"
    dset = gid.create_dataset(dname, data=extent)
    
    fid.close()

def append_text(run,
                lp,
                density,
                temperature,
                delD,
                delT,
                Rwp):
    """
    this function appends to a simple text file the 
    results of the run

    @TODO if the run number is already present, instead
    of appending, overwrite that particular line
    """

    fid = open("h5data/Results.txt","a")

    line = (f"{run}\t"
        f"{lp}\t"
        f"{density} +/- {delD}\t"
        f"{temperature} +/- {delT}\t"
        f"{Rwp}\n")

    fid.write(line)

    fid.close()

def _prepare_interp_table():
    """
    the experiment relies on the change in the lattice parameter to
    determine the approximate temperature of the sample. this function
    prepares the interpolation function as a dictionary for all the
    elements in the experiment. 
    """
    func_dict = {}
    elem = ["Au", "Ag", "Cu", "Ta", "Fe"]
    for e in elem:
        data = []
        fname = f"rho_v_T_data/{e}.txt"
        fid = open(fname,"r")
        
        for line in fid:
            data.append([float(x) for x in line.split()])
        data = np.array(data)
        fid.close()
        
        func_dict[e] = interp1d(data[:,1],data[:,0],
            kind="cubic",
            bounds_error=False)
        
    return func_dict

def _get_temperature(func_dict, measured_density, mat):
    """
    get the temperature of the material given a density
    and the interpolation function dictionary derived from
    the last routine
    """
    if mat in func_dict:
        f = func_dict[mat]
        return f(measured_density)
        
    else:
        msg = f"material not one of the ones calculated"
        raise RunTimeError(msg)

# params
niter_lebail = 20
dmin_DFLT = _angstroms(0.75)
kev_DFLT = _kev(10.0)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run automated serial MEC pump-probes analysis."
    )
    parser.add_argument(
        'exp_name',
        help="experiment name (e.g., 'meclv7318')",
        type=str
    )

    parser.add_argument(
        'instr_name',
        help="HEXRD instrument name (.hexrd format)",
        type=str
    )

    parser.add_argument(
        '-s', '--start-run',
        help='start run number to process',
        type=int,
        default=0
    )

    parser.add_argument(
        '-e', '--event-numbers',
        help='the event number(s) to process',
        type=int,
        nargs='+',
        default=-1
    )

    parser.add_argument(
        '-m', '--material-name',
        help='name of material',
        type=str
    )

    args = parser.parse_args()

    exp_name = args.exp_name
    instr_name = args.instr_name
    rn = args.start_run
    en = args.event_numbers
    mat = args.material_name

    if mat not in ["Ag", "Au", "Cu", "Fe", "Ta"]:
        raise ValueError("this material is not part of this MEC run")

    # MEC wavelengths
    # !!! second might be coming from spectrometer fit for each run
    lam1 = cnst.keVToAngstrom(10.0)
    lam2 = cnst.keVToAngstrom(10.08)

    # Load instrument
    instr = instrument.HEDMInstrument(
        h5py.File(instr_name, 'r')
    )

    # Detector keys
    det_keys = ['Epix10kaQuad%d' % i for i in range(4)]
    for det_key in instr.detectors.keys():
        if det_key not in det_keys:
            raise RuntimeWarning(
                "Instrument detector key '%s' does not match filenames"
                % det_key
            )
    img_dict = dict.fromkeys(det_keys)

    # Materials
    mname = f"{mat}_ambient"
    mat_lt = Material(
        name=mname, material_file='material.h5',
        dmin=dmin_DFLT, kev=kev_DFLT
    )
    mat_lt.name = f"{mat}_lt"
    mat_ht = copy.deepcopy(mat_lt)
    mat_ht.latticeParameters = [3.32]
    mat_ht.name = f"{mat}_ht"

    # caking (i.e., "dewarping")
    tth_pixel_size = 0.025  # close to median resolution
    eta_pixel_size = 0.5    # close to median resolution
    tth_min = 14.75  # from inspection
    tth_max = 72.75  # rounded from: np.degrees(instrument.max_tth(instr))

    # caking class
    pv = PolarView(
        (tth_min, tth_max), instr,
        eta_min=-40, eta_max=200,
        pixel_size=(tth_pixel_size, eta_pixel_size)
    )
    tth_bin_centers = np.degrees(pv.angular_grid[1][0, :])

    # PREPARE THE INTERPOLATION FUNCTION FOR TEMPERATURE
    func_dict = _prepare_interp_table()

    # =========================================================================
    # SERIAL LOOP
    # =========================================================================

    while True:
        # run PSANA shit here
        try:
            min_intensity = np.inf
            for det_key, panel in instr.detectors.items():
                img = GetImage(det_key, exp_name, rn, event=en)
                img[~panel.panel_buffer] = np.nan
                img = img - np.nanmin(img)
                img[~panel.panel_buffer] = 0.
                img_dict[det_key] = np.fliplr(img)
        except(RuntimeError, IndexError):
            print("waiting for run # %d..." % rn)
            time.sleep(3)
            continue

        # Do dewarping to polar
        polar_img = pv.warp_image(
            img_dict, pad_with_nans=True, do_interpolation=True
        )

        # lineout using masked array
        lineout = np.ma.average(polar_img, axis=0)  # !!! check fill value
        spec_expt = np.vstack([tth_bin_centers, lineout]).T

        # =====================================================================
        # LEBAIL FITTING
        # =====================================================================

        # setup
        kwargs = {
            'expt_spectrum': spec_expt,
            'phases': [mat_lt, mat_ht],
            'wavelength': {
                mat_lt.name: [_angstroms(lam1), 1],
                mat_ht.name: [_angstroms(lam2), 1]
            },
            'bkgmethod': {'chebyshev': 2},
            'peakshape': 'pvtch'
        }

        # instantiate
        lebail_fitter = LeBail(**kwargs)
        lebail_fitter.params["V"].vary = True
        lebail_fitter.params["W"].vary = True
        lebail_fitter.params["X"].vary = True
        lebail_fitter.params["Y"].vary = True
        mat_lp = f"{mat}_ht_a"
        lebail_fitter.params[mat_lp].vary = True

        # fit
        for i in range(niter_lebail):
            lebail_fitter.RefineCycle()
        mat_ht.latticeParameters = [
            lebail_fitter.params[mat_lp].value*10.0,
        ]
        print(
            f"Material: {mat}"
            )
        print(
            "Fit densities:\n\tLT: %1.3e, HT: %1.3e"
            % (mat_lt.unitcell.density, mat_ht.unitcell.density)
        )

        # =====================================================================
        # OUTPUT
        # =====================================================================

        """
        PREPARE THE DATA FOR I/O
        """
        polar_view = masked
        azimuthal_integration = spec_expt
        amb_density = mat_lt.unitcell.density
        density = mat_ht.unitcell.density
        lp = mat_ht.unitcell.a*10.0
        extent = np.degrees(pv.extent)
        LeBail_fit = np.array(L.spectrum_sim.data).T
        difference_curve = np.copy(LeBail_fit)
        difference_curve[:,1] = azimuthal_integration[:,1] - LeBail_fit[:,1]

        """
        CALCULATE THE UNCERTAINTIES IN DENSITY AND TEMPERATURE
        """
        del_a = L.res.params[mat_lp].stderr
        a = L.res.params[mat_lp].value
        perct_err_a = del_a/a
        perct_err_D = 3*perct_err_a
        delD = perct_err_D*density

        temperature = _get_temperature(func_dict, density, mat)
        temp_lb = _get_temperature(func_dict, density+delD, mat)
        temp_ub = _get_temperature(func_dict, density-delD, mat)
        delT = 0.5*(np.abs(temperature-temp_lb) + np.abs(temperature-temp_ub))
        Rwp = L.Rwp*100

        """
        WRITE OUT THE DATA
        """
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        write_hdf(rn,
                mat,
                polar_view,
                extent,
                azimuthal_integration,
                LeBail_fit,
                difference_curve,
                lp,
                amb_density,
                density,
                temperature,
                delD,
                delT,
                Rwp)
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        append_text(rn,
                    mat,
                    lp,
                    density,
                    temperature,
                    delD,
                    delT,
                    Rwp)

        # incr run number
        rn += 1
