"""find_orientations command"""
from __future__ import print_function, division, absolute_import

import os

import numpy as np
import timeit

from hexrd import constants as cnst
from hexrd import instrument
from hexrd import indexer
from hexrd.transforms import xfcapi
from .utils import get_eta_ome, generate_orientation_fibers, run_cluster
from .utils import analysis_id

def find_orientations(cfg, hkls=None, clean=False, profile=False):
    print('ready to run find_orientations')
    # %%
    # =============================================================================
    # SEARCH SPACE GENERATION
    # =============================================================================

    hedm = cfg.instrument.hedm
    plane_data = cfg.material.plane_data

    ncpus = cfg.multiprocessing

    # for indexing
    fiber_ndiv = cfg.find_orientations.seed_search.fiber_ndiv
    fiber_seeds = cfg.find_orientations.seed_search.hkl_seeds
    on_map_threshold = cfg.find_orientations.threshold

    # for clustering
    cl_radius = cfg.find_orientations.clustering.radius
    min_compl = cfg.find_orientations.clustering.completeness
    compl_thresh = cfg.find_orientations.clustering.completeness
    min_samples = 15

    print("INFO:\tgenerating search quaternion list using %d processes" % ncpus)
    start = timeit.default_timer()

    eta_ome = get_eta_ome(cfg)
    qfib = generate_orientation_fibers(
        eta_ome, hedm.chi, on_map_threshold,
        fiber_seeds, fiber_ndiv,
        ncpus=ncpus
    )

    # %%
    # =============================================================================
    # ORIENTATION SCORING
    # =============================================================================

    print("INFO:\t\t...took %f seconds" % (timeit.default_timer() - start))
    print("INFO: will test %d quaternions using %d processes"
          % (qfib.shape[1], ncpus))
    print("INFO:\tusing map search with paintGrid on %d processes"
          % ncpus)
    start = timeit.default_timer()

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
    print("INFO:\t\t...took %f seconds" % (timeit.default_timer() - start))
    completeness = np.array(completeness)

    # %%
    # =============================================================================
    # CLUSTERING AND GRAINS OUTPUT
    # =============================================================================

    if not os.path.exists(cfg.analysis_dir):
        os.makedirs(cfg.analysis_dir)
    qbar_filename = 'accepted_orientations_' + analysis_id(cfg) + '.dat'

    print("INFO:\trunning clustering using '%s'"
          % cfg.find_orientations.clustering.algorithm
    )
    start = timeit.default_timer()

    qbar, cl = run_cluster(
        completeness, qfib, plane_data.getQSym(), cfg,
        min_samples=min_samples,
        compl_thresh=compl_thresh,
        radius=cl_radius
    )

    print("INFO:\t\t...took %f seconds" % (timeit.default_timer() - start))
    print("INFO:\tfound %d grains; saved to file: '%s'"
          % (qbar.shape[1], qbar_filename))

    np.savetxt(qbar_filename, qbar.T,
               fmt='%.18e', delimiter='\t')

    gw = instrument.GrainDataWriter(os.path.join(cfg.analysis_dir, 'grains.out'))
    grain_params_list = []
    for gid, q in enumerate(qbar.T):
        phi = 2*np.arccos(q[0])
        n = xfcapi.unitRowVector(q[1:])
        grain_params = np.hstack([phi*n, cnst.zeros_3, cnst.identity_6x1])
        gw.dump_grain(gid, 1., 0., grain_params)
        grain_params_list.append(grain_params)
    gw.close()
