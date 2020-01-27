"""Utilities for output"""
from io import IOBase
import os

import h5py
import numpy as np
from scipy.linalg.matfuncs import logm

from hexrd import matrixutil as mutil
from hexrd.extensions._transforms_CAPI import makeRotMatOfExpMap


class PatchDataWriter(object):
    """
    """
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
            mangs = np.ones(3)*np.nan
            mxy = np.ones(2)*np.nan

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


class GrainDataWriter(object):
    """
    """
    def __init__(self, filename):
        self._delim = '  '
        header_items = (
            '# grain ID', 'completeness', 'chi^2',
            'exp_map_c[0]', 'exp_map_c[1]', 'exp_map_c[2]',
            't_vec_c[0]', 't_vec_c[1]', 't_vec_c[2]',
            'inv(V_s)[0,0]', 'inv(V_s)[1,1]', 'inv(V_s)[2,2]',
            'inv(V_s)[1,2]*sqrt(2)',
            'inv(V_s)[0,2]*sqrt(2)',
            'inv(V_s)[0,2]*sqrt(2)',
            'ln(V_s)[0,0]', 'ln(V_s)[1,1]', 'ln(V_s)[2,2]',
            'ln(V_s)[1,2]', 'ln(V_s)[0,2]', 'ln(V_s)[0,1]'
        )
        self._header = self._delim.join(
            [self._delim.join(
                np.tile('{:<12}', 3)
                ).format(*header_items[:3]),
             self._delim.join(
                np.tile('{:<23}', len(header_items) - 3)
                ).format(*header_items[3:])]
        )
        if isinstance(filename, IOBase):
            self.fid = filename
        else:
            self.fid = open(filename, 'w')
        print(self._header, file=self.fid)

    def __del__(self):
        self.close()

    def close(self):
        self.fid.close()

    def dump_grain(self, grain_id, completeness, chisq,
                   grain_params):
        assert len(grain_params) == 12, \
            "len(grain_params) must be 12, not %d" % len(grain_params)

        # extract strain
        emat = logm(np.linalg.inv(mutil.vecMVToSymm(grain_params[6:])))
        evec = mutil.symmToVecMV(emat, scale=False)

        res = [int(grain_id), completeness, chisq] \
            + grain_params.tolist() \
            + evec.tolist()
        output_str = self._delim.join(
            [self._delim.join(
                ['{:<12d}', '{:<12f}', '{:<12e}']
             ).format(*res[:3]),
             self._delim.join(
                np.tile('{:<23.16e}', len(res) - 3)
             ).format(*res[3:])]
        )
        print(output_str, file=self.fid)
        return output_str


class GrainDataWriter_h5(object):
    """
    TODO: add material spec
    """
    def __init__(self, filename, instr_cfg, grain_params, use_attr=False):

        if isinstance(filename, h5py.File):
            self.fid = filename
        else:
            newfile = filename + ".hdf5"
            self.fid = h5py.File(, "w")
            if os.path.exists(newfile):
                    raise(OSError("file not found: check path"))
        icfg = {}
        icfg.update(instr_cfg)

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


def unwrap_dict_to_h5(grp, d, asattr=True):
    while len(d) > 0:
        key, item = d.popitem()
        if isinstance(item, dict):
            subgrp = grp.create_group(key)
            unwrap_dict_to_h5(subgrp, item)
        else:
            if asattr:
                grp.attrs.create(key, item)
            else:
                data = np.atleast_1d(item)
                try:
                    grp.create_dataset(key, data=data)
                except(TypeError):
                    grp.create_dataset(key, data=str(data[0]))
                grp.create_dataset(key, data=np.atleast_1d(item))


def centers_of_edge_vec(edges):
    assert np.r_[edges].ndim == 1, "edges must be 1-d"
    return np.average(np.vstack([edges[:-1], edges[1:]]), axis=0)
