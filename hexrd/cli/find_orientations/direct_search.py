"""Direct search for orientations"""
# Direct Search


def test_orientation_FF_init(params):
    global paramMP
    paramMP = params

def test_orientation_FF_reduced(quat):
    """
    input parameters are [
    plane_data, instrument, imgser_dict,
    tth_tol, eta_tol, ome_tol, npdiv, threshold
    ]
    """
    plane_data = paramMP['plane_data']
    instrument = paramMP['instrument']
    imgser_dict = paramMP['imgser_dict']
    tth_tol = paramMP['tth_tol']
    eta_tol = paramMP['eta_tol']
    ome_tol = paramMP['ome_tol']
    npdiv = paramMP['npdiv']
    threshold = paramMP['threshold']

    phi = 2*np.arccos(quat[0])
    n = xfcapi.unitRowVector(quat[1:])
    grain_params = np.hstack([
        phi*n, cnst.zeros_3, cnst.identity_6x1,
    ])

    compl, scrap = instrument.pull_spots(
        plane_data, grain_params, imgser_dict,
        tth_tol=tth_tol, eta_tol=eta_tol, ome_tol=ome_tol,
        npdiv=npdiv, threshold=threshold,
        eta_ranges=np.radians(cfg.find_orientations.eta.range),
        ome_period=(-np.pi, np.pi),
        check_only=True)

    return sum(compl)/float(len(compl))


def direct_search():

    params = dict(
            plane_data=plane_data,
            instrument=instr,
            imgser_dict=imsd,
            tth_tol=tth_tol,
            eta_tol=eta_tol,
            ome_tol=ome_tol,
            npdiv=npdiv,
            threshold=cfg.fit_grains.threshold)

    print("INFO:\tusing direct seach")
    pool = multiprocessing.Pool(ncpus, test_orientation_FF_init, (params, ))
    completeness = pool.map(test_orientation_FF_reduced, qfib.T)
    pool.close()
