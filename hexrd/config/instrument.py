import os

import numpy as np

from hexrd import constants
from hexrd import instrument
from hexrd import distortion
from .config import Config


class Instrument(Config):
    # Note: instrument is instantiated with a yaml dictionary; use self
    #       to instantiate classes based on this one
    @property
    def hedm(self):
        return instrument.HEDMInstrument(self.beam,
                                         self.detector_dict,
                                         self.oscillation_stage)

    @property
    def beam(self):
        return Beam(self).beam

    @property
    def oscillation_stage(self):
        oscfg = OscillationStage(self)
        return instrument.oscillation_stage.OscillationStage(oscfg.tvec, oscfg.chi)

    @property
    def detector_dict(self):
        """returns dictionary of detectors"""
        d = dict()
        cfg_dets = self._cfg.get('detectors')
        for id in cfg_dets:
            dcfg = Detector(self, id)
            d[id] = dcfg.detector(self.beam)

        return d


class Detector(Config):

    BASEKEY = 'detectors'

    nrows_DFLT = 2048
    ncols_DFLT = 2048
    pixel_size_DFLT = (0.2, 0.2)

    tilt_angles_DFLT = np.zeros(3)
    t_vec_d_DFLT = np.r_[0., 0., -1000.]

    def __init__(self, cfg, id):
        """Detector with given ID string"""
        super(Detector, self).__init__(cfg)
        self.id = id

    def get(self, key, **kwargs):
        return self._cfg.get(':'.join([self.BASEKEY, self.id, key]), **kwargs)

    def detector(self, beam):
        return instrument.PlanarDetector(
            rows=self._pixel_rows,
            cols=self._pixel_cols,
            pixel_size=self._pixel_size,
            tvec=self._transform_tvec,
            tilt=self._tilt,
            beam=beam,
            evec=self._eta_vec,
            distortion=self._distortion)

    # ========== Input Values
    @property
    def _pixel_rows(self):
        return self.get('pixels:rows', default=self.nrows_DFLT)

    @property
    def _pixel_cols(self):
        return self.get('pixels:columns', default=self.ncols_DFLT)

    @property
    def _pixel_size(self):
        return self.get('pixels:pixel_size', default=self.pixel_size_DFLT)

    @property
    def _transform_tvec(self):
        return self.get('transform:t_vec_d', default=self.t_vec_d_DFLT)

    @property
    def _tilt(self):
        return self.get('transform:tilt_angles', default=self.tilt_angles_DFLT)

    @property
    def _eta_vec(self):
        return self.get('eta_vec', default=constants.eta_vec)

    @property
    def _dparams(self):
        return self.get('distortion:parameters', default=None)

    @property
    def _dfunction(self):
        return self.get('distortion:function_name', default=None)

    @property
    def _distortion(self):
        rval = None
        hasparams = (self._dparams is not None)
        if hasparams and self._dfunction == 'GE_41RT':
            rval = (distortion.GE_41RT, self._dparams)

        return rval


class Beam(Config):

    beam_energy_DFLT = 65.351
    beam_vec_DFLT = constants.beam_vec

    BASEKEY = 'beam'

    def get(self, key, **kwargs):
        """get item with given key"""
        return self._cfg.get(':'.join([self.BASEKEY, key]), **kwargs)

    @property
    def beam(self):
        return instrument.beam.Beam(self._energy, self._vector)

    # ========== Input Values
    @property
    def _energy(self):
        return self.get('energy', default=self.beam_energy_DFLT)

    @property
    def _vector(self):
        d = self.get('vector', default=None)
        if d is None:
            return self.beam_vec_DFLT

        az = d['azimuth']
        pa = d['polar_angle']
        return instrument.beam.Beam.calc_beam_vec(az, pa)


class OscillationStage(Config):

    chi_DFLT = 0.
    tvec_DFLT = np.zeros(3)

    BASEKEY = 'oscillation_stage'

    def get(self, key, **kwargs):
        """get item with given key"""
        return self._cfg.get(':'.join([self.BASEKEY, key]), **kwargs)

    @property
    def tvec(self):
        return self.get('t_vec_s', default=self.tvec_DFLT)

    @property
    def chi(self):
        return self.get('chi', default=self.chi_DFLT)
