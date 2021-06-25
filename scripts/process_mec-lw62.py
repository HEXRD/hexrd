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
        if event >= nevents:
            raise IndexError(
                "Run %d only has %d events; you requested event %d"
                % (run, nevents, event)
            )
        else:
            return detector.image(events[event])


# params
niter_lebail = 10
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
        default=1
    )

    args = parser.parse_args()

    exp_name = args.exp_name
    instr_name = args.instr_name
    rn = args.start_run
    en = args.event_numbers

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
    Ta_lt = Material(
        name='Ta_ambient', material_file='material.h5',
        dmin=dmin_DFLT, kev=kev_DFLT
    )
    Ta_lt.name = 'Ta_lt'
    Ta_ht = copy.deepcopy(Ta_lt)
    Ta_ht.latticeParameters = [3.32]
    Ta_ht.name = 'Ta_ht'

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
            'phases': [Ta_lt, Ta_ht],
            'wavelength': {
                'Ta_lt': [_angstroms(lam1), 1],
                'Ta_ht': [_angstroms(lam2), 1]
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
        lebail_fitter.params["Ta_ht_a"].vary = True

        # fit
        for i in range(niter_lebail):
            lebail_fitter.RefineCycle()
        Ta_ht.latticeParameters = [
            lebail_fitter.params["Ta_ht_a"].value*10.0,
        ]
        print(
            "Fit densities:\n\tLT: %1.3e, HT: %1.3e"
            % (Ta_lt.unitcell.density, Ta_ht.unitcell.density)
        )

        # =====================================================================
        # OUTPUT
        # =====================================================================

        # incr run number
        rn += 1
