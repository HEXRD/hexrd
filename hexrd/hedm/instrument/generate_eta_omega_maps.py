from typing import Optional

import numpy as np
from numpy.typing import NDArray

from hexrd.core.instrument.hedm_instrument import HEDMInstrument
from hexrd.core.imageseries.omega import OmegaImageSeries
from hexrd.core import matrixutil as mutil

from hexrd.hedm import xrdutil
from hexrd.core.material.crystallography import PlaneData
from hexrd.core import constants as ct
from hexrd.core.rotations import mapAngle


class GenerateEtaOmeMaps:
    """ eta-ome map class derived from new image_series and YAML config

    self.dataStore
    self.planeData
    self.iHKLList
    self.etaEdges # IN RADIANS
    self.omeEdges # IN RADIANS
    self.etas     # IN RADIANS
    self.omegas   # IN RADIANS

    """

    def __init__(
        self,
        image_series_dict: dict[str, OmegaImageSeries],
        instrument: HEDMInstrument,
        plane_data: PlaneData,
        active_hkls: Optional[NDArray[np.int32] | list[int]] = None,
        eta_step: float = 0.25,
        threshold: Optional[float] = None,
        ome_period: tuple[float, float] = (
            0,
            360,
        ),  # TODO: Remove this - it does nothing.
    ):
        self._planeData = plane_data

        # ???: change name of iHKLList?
        # ???: can we change the behavior of iHKLList?
        if active_hkls is None:
            self._iHKLList = plane_data.getHKLID(plane_data.hkls, master=True)
            assert isinstance(self._iHKLList, list)
            n_rings = len(self._iHKLList)
        else:
            assert hasattr(
                active_hkls, '__len__'
            ), "active_hkls must be an iterable with __len__"
            self._iHKLList = np.asarray(active_hkls, dtype=np.int32).tolist()
            n_rings = len(active_hkls)

        # grab a det key and corresponding imageseries (first will do)
        # !!! assuming that the imageseries for all panels
        #     have the same length and omegas
        this_det_ims = next(iter(image_series_dict.values()))

        # handle omegas
        # !!! for multi wedge, enforncing monotonicity
        # !!! wedges also cannot overlap or span more than 360
        omegas_array = this_det_ims.metadata['omega']  # !!! DEGREES
        frame_mask: Optional[NDArray[np.bool_]] = None
        ome_period = omegas_array[0, 0] + np.r_[0.0, 360.0]  # !!! be careful
        if this_det_ims.omegawedges.nwedges > 1:
            delta_omes = [
                (i['ostop'] - i['ostart']) / i['nsteps']
                for i in this_det_ims.omegawedges.wedges
            ]
            check_wedges = mutil.uniqueVectors(
                np.atleast_2d(delta_omes), tol=1e-6
            ).squeeze()
            assert (
                check_wedges.size == 1
            ), "all wedges must have the same delta omega to 1e-6"
            # grab representative delta ome
            # !!! assuming positive delta consistent with OmegaImageSeries
            delta_ome = delta_omes[0]

            # grab full-range start/stop
            # !!! be sure to map to the same period to enable arithmatic
            # ??? safer to do this way rather than just pulling from
            #     the omegas attribute?
            owedges = this_det_ims.omegawedges.wedges
            ostart = owedges[0]['ostart']  # !!! DEGREES
            ostop = float(mapAngle(owedges[-1]['ostop'], ome_period, units='degrees'))
            # compute total nsteps
            # FIXME: need check for roundoff badness
            nsteps = int((ostop - ostart) / delta_ome)
            ome_edges_full = np.linspace(ostart, ostop, num=nsteps + 1, endpoint=True)
            omegas_array = np.vstack([ome_edges_full[:-1], ome_edges_full[1:]]).T
            ome_centers = np.average(omegas_array, axis=1)

            # use OmegaImageSeries method to determine which bins have data
            # !!! this array has -1 outside a wedge
            # !!! again assuming the valid frame order increases monotonically
            frame_mask = np.array(
                [this_det_ims.omega_to_frame(ome)[0] != -1 for ome in ome_centers]
            )

        # ???: need to pass a threshold?
        eta_mapping, etas = instrument.extract_polar_maps(
            plane_data,
            image_series_dict,
            active_hkls=active_hkls,
            threshold=threshold,
            tth_tol=None,
            eta_tol=eta_step,
        )

        # for convenience grab map shape from first
        map_shape = next(iter(eta_mapping.values())).shape[1:]

        # pack all detectors with masking
        # FIXME: add omega masking
        data_store: list[NDArray[np.float64]] = []
        for i_ring in range(n_rings):
            # first handle etas
            full_map: NDArray[np.float64] = np.zeros(map_shape, dtype=float)
            nan_mask_full: NDArray[np.bool_] = np.zeros(
                (len(eta_mapping), map_shape[0], map_shape[1]), dtype=bool
            )
            i_p = 0
            for eta_map in eta_mapping.values():
                nan_mask = ~np.isnan(eta_map[i_ring])
                nan_mask_full[i_p] = nan_mask
                full_map[nan_mask] += eta_map[i_ring][nan_mask]
                i_p += 1
            re_nan_these: NDArray[np.bool_] = np.sum(nan_mask_full, axis=0) == 0
            full_map[re_nan_these] = np.nan

            # now omegas
            if frame_mask is not None:
                # !!! must expand row dimension to include
                #     skipped omegas
                tmp: NDArray[np.float64] = (
                    np.ones((len(frame_mask), map_shape[1])) * np.nan
                )
                tmp[frame_mask, :] = full_map
                full_map = tmp
            data_store.append(full_map)
        self._dataStore = data_store

        # set required attributes
        self._omegas = mapAngle(
            np.radians(np.average(omegas_array, axis=1)),
            np.radians(ome_period),
        )
        self._omeEdges = mapAngle(
            np.radians(np.r_[omegas_array[:, 0], omegas_array[-1, 1]]),
            np.radians(ome_period),
        )

        # !!! must avoid the case where omeEdges[0] = omeEdges[-1] for the
        #     indexer to work properly
        if abs(self._omeEdges[0] - self._omeEdges[-1]) <= ct.sqrt_epsf:
            # !!! SIGNED delta ome
            del_ome = np.radians(omegas_array[0, 1] - omegas_array[0, 0])
            self._omeEdges[-1] = self._omeEdges[-2] + del_ome

        # handle etas
        # WARNING: unlinke the omegas in imageseries metadata,
        # these are in RADIANS and represent bin centers
        self._etaEdges = etas
        self._etas = self._etaEdges[:-1] + 0.5 * np.radians(eta_step)

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
        xrdutil.EtaOmeMaps.save_eta_ome_maps(self, filename)
