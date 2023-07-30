"""Spots for grain reflections

Currently this is only for saving the reflection image data
for a grain.
"""
from collections import namedtuple
import os
from io import IOBase

import numpy as np
import h5py

from hexrd import matrixutil as mutil


_DataPatch = namedtuple(
    "_DataPatch",
    ["detector_id", "irefl", "peak_id", "hkl_id", "hkl", "tth_edges",
     "eta_edges", "ome_eval", "xyc_arr", "ijs", "frame_indices", "patch_data",
     "ang_centers", "xy_centers", "meas_angs", "meas_xy", "sum_int", "max_int"]
)
"""
     peak_id, hkl_id, hkl, sum_int, max_int,
     ang_centers[i_pt], meas_angs,
     xy_centers[i_pt], meas_xy)
"""
_DataPatch.__doc__ = r"""initialize spot

        Parameters
        ----------
        detector_id: string
           name of detector containing current spot
        irefl: int (nonnegative)
           the reflection number
        peak_id: int
           the peak number, which is the reflection ID or -999
           if there are no peaks associated with this reflection
        hkl_id: in

                detector_iRefl, peak_id, hkl_id, hkl,
                tth_edges, eta_edges, ome_eval,
                xyc_arr, ijs, frame_indices, patch_data,
                ang_centers, xy_centers, meas_angs, meas_xy

        Returns
        -------
        var
           description
"""


class Spot(_DataPatch):
    pass

r""" code from pull_spots
writer.dump_patch(
    detector_id, iRefl, peak_id, hkl_id, hkl,
    tth_edges, eta_edges, np.radians(ome_eval),
    xyc_arr, ijs, frame_indices, patch_data,
    ang_centers[i_pt], xy_centers[i_pt],
    meas_angs, meas_xy)

 writer.dump_patch(
     peak_id, hkl_id, hkl, sum_int, max_int,
     ang_centers[i_pt], meas_angs,
     xy_centers[i_pt], meas_xy)

-        # prepare output if requested
-        if filename is not None and output_format.lower() == 'hdf5':
-            this_filename = os.path.join(dirname, filename)
-            writer = GrainDataWriter_h5(
-                os.path.join(dirname, filename),
-                self.write_config(), grain_params)

"""


class SpotWriter:



    def __init__(self, summary=None, full=None, output_dir=None,
                 instr_cfg=None, grain_params=None):
        """Write spots to files

        Parameters
        ----------
        summary: string or None
           basename of file to write spot summary list
        full: string or None
           basename of file to write full spot list
        output_dir: string
           path to output directory
        instr_cfg: dictionary
           dictionary of instrument config parameters (full output)
        grain_params: list
           grain parameters (full output)


        Returns
        -------
        SpotWriter instance
        """
        self.summary = summary
        self.full = full
        self.output_dir = output_dir

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        if self.summary:
            self._open_summary()


    def _open_summary(self):
        # initialize text-based output writer
        this_filename = os.path.join(self.output_dir, self.summary + ".out")
        self.summary_writer = PatchDataWriter(this_filename)

    def _open_full(self):
        this_filename = os.path.join(output_dir, self.full + ".hdf5")
        self.full_writer = GrainDataWriter_h5(
            this_filename, self.instr_cfg, self.grain_params
        )

    def write_spot(self, spot):
        """Write data for a reflection"""
        if self.summary:
            self.summary_writer.dump_patch(
                spot.peak_id, spot.hkl_id, spot.hkl, spot.sum_int,
                spot.max_int, spot.ang_centers, spot.meas_angs,
                spot.xy_centers, spot.meas_xy
            )

        if self.full:
            pass

    def close(self):
        if self.summary:
            self.summary_writer.close()
        if self.full:
            # self.full_writer.close()
            pass



class GrainDataWriter_h5(object):
    """Class for dumping grain results to an HDF5 archive.

    TODO: add material spec
    """

    def __init__(self, filename, instr_cfg, grain_params, use_attr=False):
        if isinstance(filename, h5py.File):
            self.fid = filename
        else:
            self.fid = h5py.File(filename + ".hdf5", "w")
        icfg = dict(instr_cfg)

        # add instrument groups and attributes
        self.instr_grp = self.fid.create_group('instrument')
        unwrap_dict_to_h5(self.instr_grp, icfg, asattr=use_attr)

        # add grain group
        self.grain_grp = self.fid.create_group('grain')
        rmat_c = makeRotMatOfExpMap(grain_params[:3])
        tvec_c = np.array(grain_params[3:6]).flatten()
        vinv_s = np.array(grain_params[6:]).flatten()
        vmat_s = np.linalg.inv(mutil.vecMVToSymm(vinv_s))

        if use_attr:    # attribute version
            self.grain_grp.attrs.create('rmat_c', rmat_c)
            self.grain_grp.attrs.create('tvec_c', tvec_c)
            self.grain_grp.attrs.create('inv(V)_s', vinv_s)
            self.grain_grp.attrs.create('vmat_s', vmat_s)
        else:    # dataset version
            self.grain_grp.create_dataset('rmat_c', data=rmat_c)
            self.grain_grp.create_dataset('tvec_c', data=tvec_c)
            self.grain_grp.create_dataset('inv(V)_s', data=vinv_s)
            self.grain_grp.create_dataset('vmat_s', data=vmat_s)

        data_key = 'reflection_data'
        self.data_grp = self.fid.create_group(data_key)

        for det_key in self.instr_grp['detectors'].keys():
            self.data_grp.create_group(det_key)

    # FIXME: throws exception when called after close method
    # def __del__(self):
    #    self.close()

    def close(self):
        self.fid.close()

    def dump_patch(self, panel_id,
                   i_refl, peak_id, hkl_id, hkl,
                   tth_edges, eta_edges, ome_centers,
                   xy_centers, ijs, frame_indices,
                   spot_data, pangs, pxy, mangs, mxy, gzip=1):
        """
        to be called inside loop over patches

        default GZIP level for data arrays is 1
        """
        fi = np.array(frame_indices, dtype=int)

        panel_grp = self.data_grp[panel_id]
        spot_grp = panel_grp.create_group("spot_%05d" % i_refl)
        spot_grp.attrs.create('peak_id', int(peak_id))
        spot_grp.attrs.create('hkl_id', int(hkl_id))
        spot_grp.attrs.create('hkl', np.array(hkl, dtype=int))
        spot_grp.attrs.create('predicted_angles', pangs)
        spot_grp.attrs.create('predicted_xy', pxy)
        if mangs is None:
            mangs = np.nan*np.ones(3)
        spot_grp.attrs.create('measured_angles', mangs)
        if mxy is None:
            mxy = np.nan*np.ones(3)
        spot_grp.attrs.create('measured_xy', mxy)

        # get centers crds from edge arrays
        # FIXME: export full coordinate arrays, or just center vectors???
        #
        # ome_crd, eta_crd, tth_crd = np.meshgrid(
        #     ome_centers,
        #     centers_of_edge_vec(eta_edges),
        #     centers_of_edge_vec(tth_edges),
        #     indexing='ij')
        #
        # ome_dim, eta_dim, tth_dim = spot_data.shape

        # !!! for now just exporting center vectors for spot_data
        tth_crd = centers_of_edge_vec(tth_edges)
        eta_crd = centers_of_edge_vec(eta_edges)

        shuffle_data = True  # reduces size by 20%
        spot_grp.create_dataset('tth_crd', data=tth_crd,
                                compression="gzip", compression_opts=gzip,
                                shuffle=shuffle_data)
        spot_grp.create_dataset('eta_crd', data=eta_crd,
                                compression="gzip", compression_opts=gzip,
                                shuffle=shuffle_data)
        spot_grp.create_dataset('ome_crd', data=ome_centers,
                                compression="gzip", compression_opts=gzip,
                                shuffle=shuffle_data)
        spot_grp.create_dataset('xy_centers', data=xy_centers,
                                compression="gzip", compression_opts=gzip,
                                shuffle=shuffle_data)
        spot_grp.create_dataset('ij_centers', data=ijs,
                                compression="gzip", compression_opts=gzip,
                                shuffle=shuffle_data)
        spot_grp.create_dataset('frame_indices', data=fi,
                                compression="gzip", compression_opts=gzip,
                                shuffle=shuffle_data)
        spot_grp.create_dataset('intensities', data=spot_data,
                                compression="gzip", compression_opts=gzip,
                                shuffle=shuffle_data)
        return


class PatchDataWriter(object):
    """Class for dumping Bragg reflection data."""

    def __init__(self, filename):
        self._delim = '  '
        header_items = (
            '# ID', 'PID',
            'H', 'K', 'L',
            'sum(int)', 'max(int)',
            'pred tth', 'pred eta', 'pred ome',
            'meas tth', 'meas eta', 'meas ome',
            'pred X', 'pred Y',
            'meas X', 'meas Y'
        )
        self._header = self._delim.join([
            self._delim.join(np.tile('{:<6}', 5)).format(*header_items[:5]),
            self._delim.join(np.tile('{:<12}', 2)).format(*header_items[5:7]),
            self._delim.join(np.tile('{:<23}', 10)).format(*header_items[7:17])
        ])
        if isinstance(filename, IOBase):
            self.fid = filename
        else:
            self.fid = open(filename, 'w')
        print(self._header, file=self.fid)

    def __del__(self):
        self.close()

    def close(self):
        self.fid.close()

    def dump_patch(self, peak_id, hkl_id,
                   hkl, spot_int, max_int,
                   pangs, mangs, pxy, mxy):
        """
        !!! maybe need to check that last four inputs are arrays
        """
        if mangs is None:
            spot_int = np.nan
            max_int = np.nan
            mangs = np.nan*np.ones(3)
            mxy = np.nan*np.ones(2)

        res = [int(peak_id), int(hkl_id)] \
            + np.array(hkl, dtype=int).tolist() \
            + [spot_int, max_int] \
            + pangs.tolist() \
            + mangs.tolist() \
            + pxy.tolist() \
            + mxy.tolist()

        output_str = self._delim.join(
            [self._delim.join(np.tile('{:<6d}', 5)).format(*res[:5]),
             self._delim.join(np.tile('{:<12e}', 2)).format(*res[5:7]),
             self._delim.join(np.tile('{:<23.16e}', 10)).format(*res[7:])]
        )
        print(output_str, file=self.fid)
        return output_str
