import logging
import multiprocessing as mp
import os
import timeit

import numpy as np
# np.seterr(over='ignore', invalid='ignore')

# import tqdm

import scipy.cluster as cluster
from scipy import ndimage

from skimage.feature import blob_dog, blob_log

from hexrd import constants as const
from hexrd import matrixutil as mutil
from hexrd import indexer
from hexrd import instrument
from hexrd import rotations as rot
from hexrd.transforms import xfcapi
from hexrd.xrdutil import EtaOmeMaps

# just require scikit-learn?
have_sklearn = False
try:
    import sklearn
    vstring = sklearn.__version__.split('.')
    if vstring[0] == '0' and int(vstring[1]) >= 14:
        from sklearn.cluster import dbscan
        from sklearn.metrics.pairwise import pairwise_distances
        have_sklearn = True
except ImportError:
    pass


save_as_ascii = False  # FIXME LATER...
fwhm_to_stdev = 1./np.sqrt(8*np.log(2))

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

    # seed_hkl_ids must be consistent with this...
    pd_hkl_ids = eta_ome.iHKLList[seed_hkl_ids]

    # grab angular grid infor from maps
    del_ome = eta_ome.omegas[1] - eta_ome.omegas[0]
    del_eta = eta_ome.etas[1] - eta_ome.etas[0]

    # labeling mask
    structureNDI_label = ndimage.generate_binary_structure(2, 1)

    # crystallography data from the pd object
    pd = eta_ome.planeData
    hkls = pd.hkls
    tTh = pd.getTTh()
    bMat = pd.latVecOps['B']
    csym = pd.getLaueGroup()

    params = dict(
        bMat=bMat,
        chi=chi,
        csym=csym,
        fiber_ndiv=fiber_ndiv)

    # =========================================================================
    # Labeling of spots from seed hkls
    # =========================================================================

    qfib = []
    input_p = []
    numSpots = []
    coms = []
    for i in seed_hkl_ids:
        if method == 'label':
            # First apply filter
            filt_stdev = fwhm_to_stdev * method_kwargs['filter_radius']
            this_map_f = -ndimage.filters.gaussian_laplace(
                eta_ome.dataStore[i], filt_stdev)

            labels_t, numSpots_t = ndimage.label(
                this_map_f > method_kwargs['threshold'],
                structureNDI_label
                )
            coms_t = np.atleast_2d(
                ndimage.center_of_mass(
                    this_map_f,
                    labels=labels_t,
                    index=np.arange(1, np.amax(labels_t) + 1)
                    )
                )
        elif method in ['blob_log', 'blob_dog']:
            # must scale map
            # TODO: we should so a parameter study here
            this_map = eta_ome.dataStore[i]
            this_map[np.isnan(this_map)] = 0.
            this_map -= np.min(this_map)
            scl_map = 2*this_map/np.max(this_map) - 1.

            # TODO: Currently the method kwargs must be explicitly specified
            #       in the config, and there are no checks
            # for 'blob_log': min_sigma=0.5, max_sigma=5,
            #                 num_sigma=10, threshold=0.01, overlap=0.1
            # for 'blob_dog': min_sigma=0.5, max_sigma=5,
            #                 sigma_ratio=1.6, threshold=0.01, overlap=0.1
            if method == 'blob_log':
                blobs = np.atleast_2d(
                    blob_log(scl_map, **method_kwargs)
                )
            else:  # blob_dog
                blobs = np.atleast_2d(
                    blob_dog(scl_map, **method_kwargs)
                )
            numSpots_t = len(blobs)
            coms_t = blobs[:, :2]
        numSpots.append(numSpots_t)
        coms.append(coms_t)
        pass

    for i in range(len(pd_hkl_ids)):
        for ispot in range(numSpots[i]):
            if not np.isnan(coms[i][ispot][0]):
                ome_c = eta_ome.omeEdges[0] + (0.5 + coms[i][ispot][0])*del_ome
                eta_c = eta_ome.etaEdges[0] + (0.5 + coms[i][ispot][1])*del_eta
                input_p.append(
                    np.hstack(
                        [hkls[:, pd_hkl_ids[i]],
                         tTh[pd_hkl_ids[i]], eta_c, ome_c]
                    )
                )
                pass
            pass
        pass

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

    gVec_s = xfcapi.anglesToGVec(
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
                    metric='precomputed'
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
                    metric='minkowski', p=2,
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
                qfib_r[:, cl == i + 1].reshape(4, npts), qsym
            ).flatten()
            pass
        pass

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
                pass
            qbar = tmp
            pass
        pass

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
            available_hkls = pd.hkls.T
            logger.info('loaded eta/ome orientation maps from %s', fn)
            hkls = [str(i) for i in available_hkls[res.iHKLList]]
            logger.info(
                'hkls used to generate orientation maps: %s',
                hkls
            )
            return res
        except (AttributeError, IOError):
            logger.info("specified maps file '%s' not found "
                        + "and clean option specified; "
                        + "recomputing eta/ome orientation maps",
                        fn)
            return generate_eta_ome_maps(cfg, hkls=hkls)
    else:
        logger.info('clean option specified; '
                    + 'recomputing eta/ome orientation maps')
        return generate_eta_ome_maps(cfg, hkls=hkls)


def generate_eta_ome_maps(cfg, hkls=None, save=True):
    """
    Generates the eta-omega maps specified in the input config.
    """
    # extract PlaneData from config and set active hkls
    plane_data = cfg.material.plane_data

    # handle logic for active hkl spec
    # !!!: default to all hkls defined for material,
    #      override with
    #        1) hkls from config, if specified; or
    #        2) hkls from kwarg, if specified
    available_hkls = plane_data.hkls.T
    active_hkls = range(len(available_hkls))
    temp = cfg.find_orientations.orientation_maps.active_hkls
    active_hkls = active_hkls if temp == 'all' else temp
    active_hkls = hkls if hkls is not None else active_hkls

    # logging output
    hklseedstr = ', '.join(
        [str(available_hkls[i]) for i in active_hkls]
    )

    logger.info(
        "building eta_ome maps using hkls: %s",
        hklseedstr
    )

    # grad imageseries dict from cfg
    imsd = cfg.image_series

    # handle omega period
    ome_period, _ = _process_omegas(imsd)

    start = timeit.default_timer()

    # make eta_ome maps
    eta_ome = instrument.GenerateEtaOmeMaps(
        imsd, cfg.instrument.hedm, plane_data,
        active_hkls=active_hkls,
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

    active_hkls = cfg.find_orientations.orientation_maps.active_hkls \
        or eta_ome.iHKLList

    fiber_seeds = cfg.find_orientations.seed_search.hkl_seeds

    # Simulate N random grains to get neighborhood size
    seed_hkl_ids = [
        plane_data.hklDataList[active_hkls[i]]['hklID']
        for i in fiber_seeds
    ]

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
