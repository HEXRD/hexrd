import copy
import logging
import multiprocessing as mp
import os
import timeit

import numpy as np
# np.seterr(over='ignore', invalid='ignore')

# import tqdm

import scipy.cluster as cluster
from scipy import ndimage

from hexrd import constants as const
from hexrd import matrixutil as mutil
from hexrd import indexer
from hexrd import instrument
from hexrd.imageutil import find_peaks_2d
from hexrd import rotations as rot
from hexrd.transforms import xfcapi
from hexrd.xrdutil import EtaOmeMaps

# just require scikit-learn?
have_sklearn = False
try:
    from sklearn.cluster import dbscan
    from sklearn.metrics.pairwise import pairwise_distances
    have_sklearn = True
except ImportError:
    pass


save_as_ascii = False  # FIXME LATER...
filter_stdev_DFLT = 1.0

logger = logging.getLogger(__name__)


# =============================================================================
# FUNCTIONS
# =============================================================================


def _process_omegas(omegaimageseries_dict):
    """Extract omega period and ranges from an OmegaImageseries dictionary."""
    oims = next(iter(omegaimageseries_dict.values()))
    ome_period = oims.omega[0, 0] + np.r_[0., 360.]
    ome_ranges = [
        ([i['ostart'], i['ostop']])
        for i in oims.omegawedges.wedges
    ]
    return ome_period, ome_ranges


def clean_map(this_map):
    # !!! need to remove NaNs from map in case of eta gaps
    # !!! doing offset and truncation by median value now
    nan_mask = np.isnan(this_map)
    med_val = np.median(this_map[~nan_mask])
    this_map[nan_mask] = med_val
    this_map[this_map <= med_val] = med_val
    this_map -= np.min(this_map)


def generate_orientation_fibers(cfg, eta_ome):
    """
    From ome-eta maps and hklid spec, generate list of
    quaternions from fibers
    """
    # grab the relevant parameters from the root config
    ncpus = cfg.multiprocessing
    chi = cfg.instrument.hedm.chi
    seed_hkl_ids = cfg.find_orientations.seed_search.hkl_seeds
    fiber_ndiv = cfg.find_orientations.seed_search.fiber_ndiv
    method_dict = cfg.find_orientations.seed_search.method

    # strip out method name and kwargs
    # !!! note that the config enforces that method is a dict with length 1
    # TODO: put a consistency check on required kwargs, or otherwise specify
    #       default values for each case?  They must be specified as of now.
    method = next(iter(method_dict.keys()))
    method_kwargs = method_dict[method]
    logger.info('\tusing "%s" method for fiber generation'
                % method)

    # crystallography data from the pd object
    pd = eta_ome.planeData
    tTh = pd.getTTh()
    bMat = pd.latVecOps['B']
    csym = pd.getLaueGroup()

    # !!! changed recently where iHKLList are now master hklIDs
    pd_hkl_ids = eta_ome.iHKLList[seed_hkl_ids]
    pd_hkl_idx = pd.getHKLID(
        pd.getHKLs(*eta_ome.iHKLList).T,
        master=False
    )
    seed_hkls = pd.getHKLs(*pd_hkl_ids)
    seed_tths = tTh[pd_hkl_idx][seed_hkl_ids]
    logger.info('\tusing seed hkls: %s'
                % [str(i) for i in seed_hkls])

    # grab angular grid infor from maps
    del_ome = eta_ome.omegas[1] - eta_ome.omegas[0]
    del_eta = eta_ome.etas[1] - eta_ome.etas[0]

    # =========================================================================
    # Labeling of spots from seed hkls
    # =========================================================================

    numSpots = []
    coms = []
    for i in seed_hkl_ids:
        this_map = copy.deepcopy(eta_ome.dataStore[i])
        clean_map(this_map)  # !!! need to get rid of NaNs for blob detection
        numSpots_t, coms_t = find_peaks_2d(this_map, method, method_kwargs)
        numSpots.append(numSpots_t)
        coms.append(coms_t)

    input_p = []
    for i, (this_hkl, this_tth) in enumerate(zip(seed_hkls, seed_tths)):
        for ispot in range(numSpots[i]):
            if not np.isnan(coms[i][ispot][0]):
                ome_c = eta_ome.omeEdges[0] + (0.5 + coms[i][ispot][0])*del_ome
                eta_c = eta_ome.etaEdges[0] + (0.5 + coms[i][ispot][1])*del_eta
                input_p.append(np.hstack([this_hkl, this_tth, eta_c, ome_c]))

    params = dict(
        bMat=bMat,
        chi=chi,
        csym=csym,
        fiber_ndiv=fiber_ndiv)

    # do the mapping
    start = timeit.default_timer()
    qfib = None
    if ncpus > 1:
        # multiple process version
        # ???: Need a chunksize in map?
        chunksize = max(1, len(input_p)//(10*ncpus))
        pool = mp.Pool(ncpus, discretefiber_init, (params, ))
        qfib = pool.map(
            discretefiber_reduced, input_p,
            chunksize=chunksize
        )
        '''
        # This is an experiment...
        ntotal= 10*ncpus + np.remainder(len(input_p), 10*ncpus) > 0
        for _ in tqdm.tqdm(
                pool.imap_unordered(
                    discretefiber_reduced, input_p, chunksize=chunksize
                ), total=ntotal
            ):
            pass
        print(_.shape)
        '''
        pool.close()
        pool.join()
    else:
        # single process version.
        discretefiber_init(params)  # sets paramMP

        # We convert to a list to ensure the map is full iterated before
        # clean up. Otherwise discretefiber_reduced will be called
        # after cleanup.
        qfib = list(map(discretefiber_reduced, input_p))

        discretefiber_cleanup()
    elapsed = (timeit.default_timer() - start)
    logger.info("\tfiber generation took %.3f seconds", elapsed)
    return np.hstack(qfib)


def discretefiber_init(params):
    global paramMP
    paramMP = params


def discretefiber_cleanup():
    global paramMP
    del paramMP


def discretefiber_reduced(params_in):
    """
    input parameters are [hkl_id, com_ome, com_eta]
    """
    global paramMP
    bMat = paramMP['bMat']
    chi = paramMP['chi']
    csym = paramMP['csym']
    fiber_ndiv = paramMP['fiber_ndiv']

    hkl = params_in[:3].reshape(3, 1)

    gVec_s = xfcapi.angles_to_gvec(
        np.atleast_2d(params_in[3:]),
        chi=chi,
        ).T

    tmp = mutil.uniqueVectors(
        rot.discreteFiber(
            hkl,
            gVec_s,
            B=bMat,
            ndiv=fiber_ndiv,
            invert=False,
            csym=csym
            )[0]
        )
    return tmp


def run_cluster(compl, qfib, qsym, cfg,
                min_samples=None, compl_thresh=None, radius=None):
    """
    """
    algorithm = cfg.find_orientations.clustering.algorithm

    cl_radius = cfg.find_orientations.clustering.radius
    min_compl = cfg.find_orientations.clustering.completeness

    ncpus = cfg.multiprocessing

    # check for override on completeness threshold
    if compl_thresh is not None:
        min_compl = compl_thresh

    # check for override on radius
    if radius is not None:
        cl_radius = radius

    start = timeit.default_timer()  # timeit this

    num_above = sum(np.array(compl) > min_compl)
    if num_above == 0:
        # nothing to cluster
        qbar = cl = np.array([])
    elif num_above == 1:
        # short circuit
        qbar = qfib[:, np.array(compl) > min_compl]
        cl = [1]
    else:
        # use compiled module for distance
        # just to be safe, must order qsym as C-contiguous
        qsym = np.array(qsym.T, order='C').T

        def quat_distance(x, y):
            return xfcapi.quat_distance(
                np.array(x, order='C'), np.array(y, order='C'),
                qsym
            )

        qfib_r = qfib[:, np.array(compl) > min_compl]

        num_ors = qfib_r.shape[1]

        if num_ors > 25000:
            if algorithm == 'sph-dbscan' or algorithm == 'fclusterdata':
                logger.info("falling back to euclidean DBSCAN")
                algorithm = 'ort-dbscan'
            # raise RuntimeError(
            #     "Requested clustering of %d orientations, "
            #     + "which would be too slow!" % qfib_r.shape[1]
            # )

        logger.info(
            "Feeding %d orientations above %.1f%% to clustering",
            num_ors, 100*min_compl
            )

        if algorithm == 'dbscan' and not have_sklearn:
            algorithm = 'fclusterdata'
            logger.warning(
                "sklearn >= 0.14 required for dbscan; using fclusterdata"
                )

        if algorithm in ['dbscan', 'ort-dbscan', 'sph-dbscan']:
            # munge min_samples according to options
            if min_samples is None \
                    or cfg.find_orientations.use_quaternion_grid is not None:
                min_samples = 1

            if algorithm == 'sph-dbscan':
                logger.info("using spherical DBSCAN")
                # compute distance matrix
                pdist = pairwise_distances(
                    qfib_r.T, metric=quat_distance, n_jobs=1
                    )

                # run dbscan
                core_samples, labels = dbscan(
                    pdist,
                    eps=np.radians(cl_radius),
                    min_samples=min_samples,
                    metric='precomputed',
                    n_jobs=ncpus,
                    )
            else:
                if algorithm == 'ort-dbscan':
                    logger.info("using euclidean orthographic DBSCAN")
                    pts = qfib_r[1:, :].T
                    eps = 0.25*np.radians(cl_radius)
                else:
                    logger.info("using euclidean DBSCAN")
                    pts = qfib_r.T
                    eps = 0.5*np.radians(cl_radius)

                # run dbscan
                core_samples, labels = dbscan(
                    pts,
                    eps=eps,
                    min_samples=min_samples,
                    metric='minkowski',
                    p=2,
                    n_jobs=ncpus,
                    )

            # extract cluster labels
            cl = np.array(labels, dtype=int)  # convert to array
            noise_points = cl == -1  # index for marking noise
            cl += 1  # move index to 1-based instead of 0
            cl[noise_points] = -1  # re-mark noise as -1
            logger.info("dbscan found %d noise points", sum(noise_points))
        elif algorithm == 'fclusterdata':
            logger.info("using spherical fclusetrdata")
            cl = cluster.hierarchy.fclusterdata(
                qfib_r.T,
                np.radians(cl_radius),
                criterion='distance',
                metric=quat_distance
                )
        else:
            raise RuntimeError(
                "Clustering algorithm %s not recognized" % algorithm
                )

        # extract number of clusters
        if np.any(cl == -1):
            nblobs = len(np.unique(cl)) - 1
        else:
            nblobs = len(np.unique(cl))

        """ PERFORM AVERAGING TO GET CLUSTER CENTROIDS """
        qbar = np.zeros((4, nblobs))
        for i in range(nblobs):
            npts = sum(cl == i + 1)
            qbar[:, i] = rot.quatAverageCluster(
                qfib_r[:, cl == i + 1], qsym
            ).flatten()

    if algorithm in ('dbscan', 'ort-dbscan') and qbar.size/4 > 1:
        logger.info("\tchecking for duplicate orientations...")
        cl = cluster.hierarchy.fclusterdata(
            qbar.T,
            np.radians(cl_radius),
            criterion='distance',
            metric=quat_distance)
        nblobs_new = len(np.unique(cl))
        if nblobs_new < nblobs:
            logger.info(
                "\tfound %d duplicates within %f degrees",
                nblobs - nblobs_new, cl_radius
            )
            tmp = np.zeros((4, nblobs_new))
            for i in range(nblobs_new):
                npts = sum(cl == i + 1)
                tmp[:, i] = rot.quatAverageCluster(
                    qbar[:, cl == i + 1].reshape(4, npts), qsym
                ).flatten()
            qbar = tmp

    logger.info("clustering took %f seconds", timeit.default_timer() - start)
    logger.info(
        "Found %d orientation clusters with >=%.1f%% completeness"
        " and %2f misorientation",
        qbar.size/4,
        100.*min_compl,
        cl_radius
        )

    return np.atleast_2d(qbar), cl


def load_eta_ome_maps(cfg, pd, image_series, hkls=None, clean=False):
    """
    Load the eta-ome maps specified by the config and CLI flags.

    Parameters
    ----------
    cfg : TYPE
        DESCRIPTION.
    pd : TYPE
        DESCRIPTION.
    image_series : TYPE
        DESCRIPTION.
    hkls : TYPE, optional
        DESCRIPTION. The default is None.
    clean : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # check maps filename
    if cfg.find_orientations.orientation_maps.file is None:
        maps_fname = '_'.join([cfg.analysis_id, "eta-ome_maps.npz"])
    else:
        maps_fname = cfg.find_orientations.orientation_maps.file

    fn = os.path.join(cfg.working_dir, maps_fname)

    # ???: necessary?
    if fn.split('.')[-1] != 'npz':
        fn = fn + '.npz'

    if not clean:
        try:
            res = EtaOmeMaps(fn)
            pd = res.planeData
            logger.info('loaded eta/ome orientation maps from %s', fn)
            shkls = pd.getHKLs(*res.iHKLList, asStr=True)
            logger.info(
                'hkls used to generate orientation maps: %s',
                [f'[{i}]' for i in shkls]
            )
        except (AttributeError, IOError):
            logger.info("specified maps file '%s' not found "
                        + "and clean option specified; "
                        + "recomputing eta/ome orientation maps",
                        fn)
            res = generate_eta_ome_maps(cfg, hkls=hkls)
    else:
        logger.info('clean option specified; '
                    + 'recomputing eta/ome orientation maps')
        res = generate_eta_ome_maps(cfg, hkls=hkls)
    filter_maps_if_requested(res, cfg)
    return res


def filter_maps_if_requested(eta_ome, cfg):
    # filter if requested
    filter_maps = cfg.find_orientations.orientation_maps.filter_maps
    # !!! current logic:
    #  if False/None don't do anything
    #  if True, only do median subtraction
    #  if scalar, do median + LoG filter with that many pixels std dev
    if filter_maps:
        if not isinstance(filter_maps, bool):
            sigm = const.fwhm_to_sigma * filter_maps
            logger.info("filtering eta/ome maps incl LoG with %.2f std dev",
                        sigm)
            _filter_eta_ome_maps(
                eta_ome,
                filter_stdev=sigm
            )
        else:
            logger.info("filtering eta/ome maps")
            _filter_eta_ome_maps(eta_ome)


def generate_eta_ome_maps(cfg, hkls=None, save=True):
    """
    Generates the eta-omega maps specified in the input config.

    Parameters
    ----------
    cfg : hexrd.config.root.RootConfig
        A hexrd far-field HEDM config instance.
    hkls : array_like, optional
        If not None, an override for the hkls used to generate maps. This can
        be either a list of unique hklIDs, or a list of [h, k, l] vectors.
        The default is None.
    save : bool, optional
        If True, write map archive to npz format according to path spec in cfg.
        The default is True.

    Raises
    ------
    RuntimeError
        DESCRIPTION.

    Returns
    -------
    eta_ome : TYPE
        DESCRIPTION.

    """
    # extract PlaneData from config and set active hkls
    plane_data = cfg.material.plane_data

    # all active hkl ids masked by exclusions
    active_hklIDs = plane_data.getHKLID(plane_data.hkls, master=True)

    # need the below
    use_all = False

    # handle optional override
    if hkls:
        # overriding hkls are specified
        hkls = np.asarray(hkls)

        # if input is 2-d list of hkls, convert to hklIDs
        if hkls.ndim == 2:
            hkls = plane_data.getHKLID(hkls.tolist(), master=True)
    else:
        # handle logic for active hkl spec in config
        # !!!: default to all hkls defined for material,
        #      override with hkls from config, if specified;
        temp = np.asarray(cfg.find_orientations.orientation_maps.active_hkls)
        if temp.ndim == 0:
            # !!! this is only possible if active_hkls is None
            use_all = True
        elif temp.ndim == 1:
            # we have hklIDs
            hkls = temp
        elif temp.ndim == 2:
            # we have actual hkls
            hkls = plane_data.getHKLID(temp.tolist(), master=True)
        else:
            raise RuntimeError('active_hkls spec must be 1-d or 2-d, not %d-d'
                               % temp.ndim)

    # apply some checks to active_hkls specificaton
    if not use_all:
        # !!! hkls --> list of hklIDs now
        # catch duplicates
        assert len(np.unique(hkls)) == len(hkls), "duplicate hkls specified!"

        # catch excluded hkls
        excluded = np.zeros_like(hkls, dtype=bool)
        for i, hkl in enumerate(hkls):
            if hkl not in active_hklIDs:
                excluded[i] = True
        if np.any(excluded):
            raise RuntimeError(
                "The following requested hkls are marked as excluded: "
                + f"{hkls[excluded]}"
            )

        # ok, now re-assign active_hklIDs
        active_hklIDs = hkls

    # logging output
    shkls = plane_data.getHKLs(*active_hklIDs, asStr=True)
    logger.info(
        "building eta_ome maps using hkls: %s",
        [f'[{i}]' for i in shkls]
    )

    # grad imageseries dict from cfg
    imsd = cfg.image_series

    # handle omega period
    ome_period, _ = _process_omegas(imsd)

    start = timeit.default_timer()

    # make eta_ome maps
    eta_ome = instrument.GenerateEtaOmeMaps(
        imsd, cfg.instrument.hedm, plane_data,
        active_hkls=active_hklIDs,
        eta_step=cfg.find_orientations.orientation_maps.eta_step,
        threshold=cfg.find_orientations.orientation_maps.threshold,
        ome_period=ome_period)

    logger.info("\t\t...took %f seconds", timeit.default_timer() - start)

    if save:
        # save maps
        # ???: should perhaps set default maps name at module level
        map_fname = cfg.find_orientations.orientation_maps.file \
            or '_'.join([cfg.analysis_id, "eta-ome_maps.npz"])

        if not os.path.exists(cfg.working_dir):
            os.mkdir(cfg.working_dir)

        fn = os.path.join(
            cfg.working_dir,
            map_fname
        )

        eta_ome.save(fn)

        logger.info('saved eta/ome orientation maps to "%s"', fn)

    return eta_ome


def _filter_eta_ome_maps(eta_ome, filter_stdev=False):
    """
    Apply median and gauss-laplace filtering to remove streak artifacts.

    Parameters
    ----------
    eta_ome : TYPE
        DESCRIPTION.
    filter_stdev : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    eta_ome : TYPE
        DESCRIPTION.

    """
    gl_filter = ndimage.filters.gaussian_laplace
    for i, pf in enumerate(eta_ome.dataStore):
        # first compoute row-wise median over omega channel
        ome_median = np.tile(np.nanmedian(pf, axis=0), (len(pf), 1))

        # subtract
        # !!! this changes the reference obj, but fitlering does not
        pf -= ome_median

        # filter
        # !!! False/None: only row median subtraction
        #     True: use default
        #     scalar: stdev for filter in pixels
        # ??? simplify this behavior
        if filter_stdev:
            if isinstance(filter_stdev, bool):
                filter_stdev = filter_stdev_DFLT
            pf[:] = -gl_filter(pf, filter_stdev)


def create_clustering_parameters(cfg, eta_ome):
    """
    Compute min samples and mean reflections per grain for clustering

    Parameters
    ----------
    cfg : TYPE
        DESCRIPTION.
    eta_ome : TYPE
        DESCRIPTION.

    Returns
    -------
    Tuple of (min_samples, mean_rpg)

    """

    # grab objects from config
    plane_data = cfg.material.plane_data
    imsd = cfg.image_series
    instr = cfg.instrument.hedm
    eta_ranges = cfg.find_orientations.eta.range
    compl_thresh = cfg.find_orientations.clustering.completeness

    # handle omega period
    ome_period, ome_ranges = _process_omegas(imsd)

    # grab the active hkl ids
    # !!! these are master hklIDs
    active_hkls = eta_ome.iHKLList

    # !!! These are indices into the active hkls
    fiber_seeds = cfg.find_orientations.seed_search.hkl_seeds

    # Simulate N random grains to get neighborhood size
    seed_hkl_ids = active_hkls[fiber_seeds]

    # !!! default to use 100 grains
    ngrains = 100
    rand_q = mutil.unitVector(np.random.randn(4, ngrains))
    rand_e = np.tile(2.*np.arccos(rand_q[0, :]), (3, 1)) \
        * mutil.unitVector(rand_q[1:, :])
    grain_param_list = np.vstack(
            [rand_e,
             np.zeros((3, ngrains)),
             np.tile(const.identity_6x1, (ngrains, 1)).T]
        ).T
    sim_results = instr.simulate_rotation_series(
            plane_data, grain_param_list,
            eta_ranges=np.radians(eta_ranges),
            ome_ranges=np.radians(ome_ranges),
            ome_period=np.radians(ome_period)
    )

    refl_per_grain = np.zeros(ngrains)
    seed_refl_per_grain = np.zeros(ngrains)
    for sim_result in sim_results.values():
        for i, refl_ids in enumerate(sim_result[0]):
            refl_per_grain[i] += len(refl_ids)
            seed_refl_per_grain[i] += np.sum(
                    [sum(refl_ids == hkl_id) for hkl_id in seed_hkl_ids]
                )

    min_samples = max(
        int(np.floor(0.5*compl_thresh*min(seed_refl_per_grain))),
        2
    )
    mean_rpg = int(np.round(np.average(refl_per_grain)))

    return min_samples, mean_rpg


def find_orientations(cfg,
                      hkls=None, clean=False, profile=False,
                      use_direct_testing=False):
    """


    Parameters
    ----------
    cfg : TYPE
        DESCRIPTION.
    hkls : TYPE, optional
        DESCRIPTION. The default is None.
    clean : TYPE, optional
        DESCRIPTION. The default is False.
    profile : TYPE, optional
        DESCRIPTION. The default is False.
    use_direct_search : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    # grab objects from config
    plane_data = cfg.material.plane_data
    imsd = cfg.image_series
    instr = cfg.instrument.hedm
    eta_ranges = cfg.find_orientations.eta.range

    # tolerances
    tth_tol = plane_data.tThWidth
    eta_tol = np.radians(cfg.find_orientations.eta.tolerance)
    ome_tol = np.radians(cfg.find_orientations.omega.tolerance)

    # handle omega period
    ome_period, ome_ranges = _process_omegas(imsd)

    # for multiprocessing
    ncpus = cfg.multiprocessing

    # thresholds
    image_threshold = cfg.find_orientations.orientation_maps.threshold
    on_map_threshold = cfg.find_orientations.threshold
    compl_thresh = cfg.find_orientations.clustering.completeness

    # clustering
    cl_algorithm = cfg.find_orientations.clustering.algorithm
    cl_radius = cfg.find_orientations.clustering.radius

    # =========================================================================
    # ORIENTATION SCORING
    # =========================================================================
    do_grid_search = cfg.find_orientations.use_quaternion_grid is not None

    if use_direct_testing:
        npdiv_DFLT = 2
        params = dict(
                plane_data=plane_data,
                instrument=instr,
                imgser_dict=imsd,
                tth_tol=tth_tol,
                eta_tol=eta_tol,
                ome_tol=ome_tol,
                eta_ranges=np.radians(eta_ranges),
                ome_period=np.radians(ome_period),
                npdiv=npdiv_DFLT,
                threshold=image_threshold)

        logger.info("\tusing direct search on %d processes", ncpus)

        # handle search space
        if cfg.find_orientations.use_quaternion_grid is None:
            # doing seeded search
            logger.info("Will perform seeded search")
            logger.info(
                "\tgenerating search quaternion list using %d processes",
                ncpus
            )
            start = timeit.default_timer()

            # need maps
            eta_ome = load_eta_ome_maps(cfg, plane_data, imsd,
                                        hkls=hkls, clean=clean)

            # generate trial orientations
            qfib = generate_orientation_fibers(cfg, eta_ome)

            logger.info("\t\t...took %f seconds",
                        timeit.default_timer() - start)
        else:
            # doing grid search
            try:
                qfib = np.load(cfg.find_orientations.use_quaternion_grid)
            except(IOError):
                raise RuntimeError(
                    "specified quaternion grid file '%s' not found!"
                    % cfg.find_orientations.use_quaternion_grid
                )

        # execute direct search
        pool = mp.Pool(
            ncpus,
            indexer.test_orientation_FF_init,
            (params, )
        )
        completeness = pool.map(indexer.test_orientation_FF_reduced, qfib.T)
        pool.close()
        pool.join()
    else:
        logger.info("\tusing map search with paintGrid on %d processes", ncpus)

        start = timeit.default_timer()

        # handle eta-ome maps
        eta_ome = load_eta_ome_maps(cfg, plane_data, imsd,
                                    hkls=hkls, clean=clean)

        # handle search space
        if cfg.find_orientations.use_quaternion_grid is None:
            # doing seeded search
            logger.info(
                "\tgenerating search quaternion list using %d processes",
                ncpus
            )
            start = timeit.default_timer()

            qfib = generate_orientation_fibers(cfg, eta_ome)
            logger.info("\t\t...took %f seconds",
                        timeit.default_timer() - start)
        else:
            # doing grid search
            try:
                qfib = np.load(cfg.find_orientations.use_quaternion_grid)
            except(IOError):
                raise RuntimeError(
                    "specified quaternion grid file '%s' not found!"
                    % cfg.find_orientations.use_quaternion_grid
                )
        # do map-based indexing
        start = timeit.default_timer()

        logger.info("will test %d quaternions using %d processes",
                    qfib.shape[1], ncpus)

        completeness = indexer.paintGrid(
            qfib,
            eta_ome,
            etaRange=np.radians(cfg.find_orientations.eta.range),
            omeTol=np.radians(cfg.find_orientations.omega.tolerance),
            etaTol=np.radians(cfg.find_orientations.eta.tolerance),
            omePeriod=np.radians(cfg.find_orientations.omega.period),
            threshold=on_map_threshold,
            doMultiProc=ncpus > 1,
            nCPUs=ncpus
            )
        logger.info("\t\t...took %f seconds",
                    timeit.default_timer() - start)
    completeness = np.array(completeness)

    logger.info("\tSaving %d scored orientations with max completeness %f%%",
                qfib.shape[1], 100*np.max(completeness))

    results = {}
    results['scored_orientations'] = {
        'test_quaternions': qfib,
        'score': completeness
    }

    # =========================================================================
    # CLUSTERING AND GRAINS OUTPUT
    # =========================================================================

    logger.info("\trunning clustering using '%s'", cl_algorithm)

    start = timeit.default_timer()

    if do_grid_search:
        min_samples = 1
        mean_rpg = 1
    else:
        min_samples, mean_rpg = create_clustering_parameters(cfg, eta_ome)

    logger.info("\tmean reflections per grain: %d", mean_rpg)
    logger.info("\tneighborhood size: %d", min_samples)

    qbar, cl = run_cluster(
        completeness, qfib, plane_data.getQSym(), cfg,
        min_samples=min_samples,
        compl_thresh=compl_thresh,
        radius=cl_radius)

    logger.info("\t\t...took %f seconds", (timeit.default_timer() - start))
    logger.info("\tfound %d grains", qbar.shape[1])

    results['qbar'] = qbar

    return results
