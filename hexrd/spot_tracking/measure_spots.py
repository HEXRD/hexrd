import numpy as np

from hexrd.config.config import Config
from hexrd.instrument import HEDMInstrument
from hexrd.imageseries import ImageSeries
from hexrd.spot_tracking.assign_spots import assign_spots_to_hkls
from hexrd.spot_tracking.track_spots import track_combine_and_chunk_spots


# FIXME: the functions in this file are not designed all that well.
# fit-grains is using `create_measure_spots_params()` to make the
# fit-grains parameters, which is kind of weird. We should refactor
# the calls from fit-grains and the HEDM calibration and clean this
# up before merging.


def measure_spots(
    grain_id: int,
    tols: list[int],
    instr: HEDMInstrument,
    grains_table: np.ndarray,
    ims_dict: dict[str, ImageSeries],
    cfg: Config,
    threshold: float,
    eta_ranges: np.ndarray,
    ome_period: np.ndarray,
    output_spots_filename: str | None = None
) -> dict[str, list]:
    # Run the measure spots operation. Either `pull_spots()` (if
    # no spots data file was specified), or the new spot assignment
    # method if a spots data file was provided.

    # First, create the params
    params = create_measure_spots_params(
        instr,
        grains_table,
        ims_dict,
        cfg,
        threshold,
        eta_ranges,
        ome_period,
        output_spots_filename,
    )

    # Next, run from the params
    return measure_spots_from_params(grain_id, tols, params)


def create_measure_spots_params(
    instr: HEDMInstrument,
    grains_table: np.ndarray,
    ims_dict: dict[str, ImageSeries],
    cfg: Config,
    threshold: float,
    eta_ranges: np.ndarray,
    ome_period: np.ndarray,
    output_spots_filename: str | None = None
) -> dict:
    # Determine how we will measure spots (either via the classic
    # `pull_spots()` method or the new spot data method), and return
    # the parameters that will be passed to that method.

    # This is especially important for fit-grains, where we want to
    # extract all needed parameters from the config rather than pickle
    # the whole thing (done by the multiprocessing automatically),
    # so that we can reduce pickling.
    spots_data_file = cfg.fit_grains.spots_data_file
    run_pull_spots = spots_data_file is None

    # These parameters are used in both, or by fit-grains
    ret = {
        'grains_table': grains_table,
        'plane_data': cfg.material.plane_data,
        'instrument': instr,
        'imgser_dict': ims_dict,
        'tth_tol': cfg.fit_grains.tolerance.tth,
        'eta_tol': cfg.fit_grains.tolerance.eta,
        'ome_tol': cfg.fit_grains.tolerance.omega,
        'refit': cfg.fit_grains.refit,
        'eta_ranges': eta_ranges,
        'ome_period': ome_period,
        'analysis_dirname': cfg.analysis_dir,
    }

    if run_pull_spots:
        # Add parts specific to `pull_spots()`
        return {
            **ret,
            'npdiv': cfg.fit_grains.npdiv,
            'threshold': threshold,
            'spots_filename': output_spots_filename,
        }
    else:
        # Use the spots data file
        return {
            **ret,
            'spots_data_file': spots_data_file,
        }


def measure_spots_from_params(
    grain_id: int,
    tols: list[float],
    params: dict,
) -> dict[str, list]:
    grains_table = params['grains_table']
    grain = grains_table[grain_id]
    grain_params = grain[3:15]

    # Measure the spots either by running pull_spots(), or by obtaining the
    # spots from a data file (if one was specified)
    if params.get('spots_data_file'):
        # There's a spots data file with spots already identified. Use that.
        return _assign_spots_data(grain_params, tols, params)
    else:
        # Run pull_spots() and find the spots.
        return _run_pull_spots(grain_id, grain_params, tols, params)


def _assign_spots_data(
    grain_params: np.ndarray,
    tols: list[float],
    params: dict,
) -> dict[str, list]:
    # This assigns spots data to HKLs using our new spot data setup.
    # It returns a `pull-spots-like` output. Maybe we should modify
    # other parts of the code to make this output nicer...

    spots_filename = params['spots_data_file']
    plane_data = params['plane_data']
    instr = params['instrument']
    ims_dict = params['imgser_dict']
    eta_ranges = params['eta_ranges']
    eta_period = (-np.pi, np.pi)
    omega_period = params['ome_period']

    first_ims = next(iter(ims_dict.values()))
    omegas = first_ims.metadata['omega']

    # Just assume the whole omega range is used
    omega_ranges = [omega_period]

    num_images = len(first_ims)

    # Track, combine, and chunk spots into subpanels
    spot_arrays = track_combine_and_chunk_spots(
        spots_filename,
        instr,
        num_images,
        omegas,
    )

    tolerances = np.asarray(tols)

    # Now assign spots to HKLs
    assigned_spots = assign_spots_to_hkls(
        spot_arrays,
        instr,
        tolerances,
        eta_period,
        plane_data,
        np.atleast_2d(grain_params),
        eta_ranges,
        omega_ranges,
        omega_period,
        num_images,
    )

    """
    To match the current `pull_spots()` output, we reorder
    the output to be more like the following:

    peak_id
    hkl_id
    hkl
    sum_int
    max_int
    pred_angs
    meas_angs
    meas_xy

    FIXME: we can modernize/simplify this sometime? But it might
    take some effort.
    """
    results = {}
    for det_key, spots in assigned_spots.items():
        # We only ran for one grain
        spots = spots[0]

        data_list = []
        for i, hkl in enumerate(spots['hkls']):
            peak_id = i
            hkl_id = i  # FIXME: hkl_id has no meaning in this setup
            sum_int = spots['sum_int'][i]
            max_int = spots['max_int'][i]
            pred_angs = spots['sim_angs'][i]
            meas_angs = spots['meas_angs'][i]
            meas_xy = spots['meas_xys'][i][:2]
            data_list.append([
                peak_id,
                hkl_id,
                hkl,
                sum_int,
                max_int,
                pred_angs,
                meas_angs,
                meas_xy,
            ])

        # Put in some fillers for invalid peaks
        # This is so that completeness can be computed accurately
        for i in range(spots['num_hkls_skipped']):
            data_list.append([
                -1,
                -1,
                [0, 0, 0],
                0,
                0,
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan],
            ])

        results[det_key] = data_list

    return results


def _run_pull_spots(
    grain_id: int,
    grain_params: np.ndarray,
    tols: list[float],
    params: dict,
) -> dict[str, list]:
    plane_data = params['plane_data']
    instrument = params['instrument']
    imgser_dict = params['imgser_dict']
    npdiv = params['npdiv']
    threshold = params['threshold']
    eta_ranges = params['eta_ranges']
    ome_period = params['ome_period']
    analysis_dirname = params['analysis_dirname']
    prefix = params['spots_filename']
    spots_filename = None if prefix is None else prefix % grain_id

    complvec, results = instrument.pull_spots(
        plane_data, grain_params,
        imgser_dict,
        tth_tol=tols[0],
        eta_tol=tols[1],
        ome_tol=tols[2],
        npdiv=npdiv, threshold=threshold,
        eta_ranges=eta_ranges,
        ome_period=ome_period,
        dirname=analysis_dirname, filename=spots_filename,
        return_spot_list=False,
        quiet=True, check_only=False, interp='nearest')

    return results
