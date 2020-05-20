"""Eta-Omega maps"""
import logging

import numpy as np

from hexrd import constants
from hexrd import crystallography
from hexrd.transforms.xfcapi import mapAngle
from hexrd.valunits import valWUnit

logger = logging.getLogger('hexrd')

class EtaOmeMap(object):
    """Class for eta-omega maps

    data includes:

    self.dataStore
    self.planeData
    self.iHKLList
    self.etaEdges # IN RADIANS
    self.omeEdges # IN RADIANS
    self.etas     # IN RADIANS
    self.omegas   # IN RADIANS
    """
    def __init__(self,
                 image_series_dict=None,
                 instrument=None,
                 plane_data=None, *,
                 load=None,
                 active_hkls=None,
                 eta_step=0.25,
                 threshold=None,
                 ome_period=(0, 360)
    ):
        """Build or load an eta-omega map
        To build the map, call with the three positional/keyword args.
           image_series - OmegaImageSeries class
           instrument  must be a dict (loaded from yaml spec)
           plane_data - crystallography parameters
           eomap = EtaOmeMap(images, instrument, plane_data)

        To load from a file, just pass the *load* keyword arg
           eomap = EtaOmeMap(load=<filename>)
        """
        no_ims = image_series_dict is None
        no_ins = instrument is None
        no_pd = plane_data is None
        if no_ims and no_ins and no_pd:
            self._load(load)
            return

        self._planeData = plane_data

        # ???: change name of iHKLList?
        # ???: can we change the behavior of iHKLList?
        if active_hkls is None:
            n_rings = len(plane_data.getTTh())
            self._iHKLList = range(n_rings)
        else:
            self._iHKLList = active_hkls
            n_rings = len(active_hkls)

        # ???: need to pass a threshold?
        eta_mapping, etas = instrument.extract_polar_maps(
            plane_data, image_series_dict,
            active_hkls=active_hkls, threshold=threshold,
            tth_tol=None, eta_tol=eta_step)

        # grab a det key
        # WARNING: this process assumes that the imageseries for all panels
        # have the same length and omegas
        det_key = list(eta_mapping.keys())[0]
        data_store = []
        for i_ring in range(n_rings):
            full_map = np.zeros_like(eta_mapping[det_key][i_ring])
            nan_mask_full = np.zeros(
                (len(eta_mapping), full_map.shape[0], full_map.shape[1])
            )
            i_p = 0
            for det_key, eta_map in eta_mapping.items():
                nan_mask = ~np.isnan(eta_map[i_ring])
                nan_mask_full[i_p] = nan_mask
                full_map[nan_mask] += eta_map[i_ring][nan_mask]
                i_p += 1
            re_nan_these = np.sum(nan_mask_full, axis=0) == 0
            full_map[re_nan_these] = np.nan
            data_store.append(full_map)
        self._dataStore = data_store

        # handle omegas
        omegas_array = image_series_dict[det_key].metadata['omega']
        self._omegas = mapAngle(
            np.radians(np.average(omegas_array, axis=1)),
            np.radians(ome_period)
        )
        self._omeEdges = mapAngle(
            np.radians(np.r_[omegas_array[:, 0], omegas_array[-1, 1]]),
            np.radians(ome_period)
        )

        # !!! must avoid the case where omeEdges[0] = omeEdges[-1] for the
        # indexer to work properly
        if abs(self._omeEdges[0] - self._omeEdges[-1]) <= constants.sqrt_epsf:
            # !!! SIGNED delta ome
            del_ome = np.radians(omegas_array[0, 1] - omegas_array[0, 0])
            self._omeEdges[-1] = self._omeEdges[-2] + del_ome

        # handle etas
        # WARNING: unlinke the omegas in imageseries metadata,
        # these are in RADIANS and represent bin centers
        self._etas = etas
        self._etaEdges = np.r_[
            etas - 0.5*np.radians(eta_step),
            etas[-1] + 0.5*np.radians(eta_step)]

    @property
    def dataStore(self):
        return self._dataStore

    @property
    def planeData(self):
        return self._planeData

    @property
    def iHKLList(self):
        return np.atleast_1d(self._iHKLList).flatten()

    @property
    def etaEdges(self):
        return self._etaEdges

    @property
    def omeEdges(self):
        return self._omeEdges

    @property
    def etas(self):
        return self._etas

    @property
    def omegas(self):
        return self._omegas

    def save(self, filename):
        """ save map
        self.dataStore
        self.planeData (args to rebuild)
        self.iHKLList
        self.etaEdges
        self.omeEdges
        self.etas
        self.omegas
        """
        args = np.array(self.planeData.getParams())[:4]
        args[2] = valWUnit('wavelength', 'length', args[2], 'angstrom')
        hkls = self.planeData.hkls
        save_dict = {'dataStore': self.dataStore,
                     'etas': self.etas,
                     'etaEdges': self.etaEdges,
                     'iHKLList': self.iHKLList,
                     'omegas': self.omegas,
                     'omeEdges': self.omeEdges,
                     'planeData_args': args,
                     'planeData_hkls': hkls}
        np.savez_compressed(filename, **save_dict)

    def _load(self, fname):
        """load a saved eta-omega map"""
        ome_eta = np.load(fname, allow_pickle=True)

        planeData_args = ome_eta['planeData_args']
        planeData_hkls = ome_eta['planeData_hkls']
        self._planeData = crystallography.PlaneData(planeData_hkls, *planeData_args)

        self._dataStore = ome_eta['dataStore']
        self._iHKLList = ome_eta['iHKLList']
        self._etaEdges = ome_eta['etaEdges']
        self._omeEdges = ome_eta['omeEdges']
        self._etas = ome_eta['etas']
        self._omegas = ome_eta['omegas']
