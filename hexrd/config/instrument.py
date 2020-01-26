import os

import numpy as np

from hexrd import constants
from hexrd import instrument
from hexrd import distortion
from .config import Config

# Defaults
class Dflt:
    nrows = 2048
    ncols = 2048
    pixel_size = (0.2, 0.2)

    # property:  translation

    @property
    def translation(self):
        """(get-only) default translation"""
        return np.r_[0., 0., -1000.]

    @property
    def tilt(self):
        return np.zeros(3)


class Instrument(Config):
    # Note: instrument is instantiated with a yaml dictionary; use self
    #       to instantiate classes based on this one
    @property
    def hedm(self):
        return instrument.HEDMInstrument(
            beam=self.beam,
            detector_dict=self.detector_dict,
            oscillation_stage=self.oscillation_stage
        )

    @property
    def beam(self):
        # Ensure that a single beam instance is generated here, so that
        # multiple detectors will all share the same instance. We want to
        # avoid having a new instance created on each call.
        if not hasattr(self, '_beam'):
            self._beam = Beam(self).beam
        return self._beam

    @property
    def oscillation_stage(self):
        return OscillationStage(self).oscillation_stage

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

    def __init__(self, cfg, id):
        """Detector with given ID string"""
        super(Detector, self).__init__(cfg)
        self.id = id

    def get(self, key, **kwargs):
        return self._cfg.get(':'.join([self.BASEKEY, self.id, key]), **kwargs)

    def detector(self, beam):
        print("eta vec: ", self._eta_vec)
        return instrument.PlanarDetector(
            rows=self._pixel_rows,
            cols=self._pixel_cols,
            pixel_size=self._pixel_size,
            tvec=self._translation,
            tilt=self._tilt,
            beam=beam,
            evec=self._eta_vec,
            distortion=self._distortion)

    # ========== Input Values
    @property
    def _pixel_rows(self):
        return self.get('pixels:rows', default=Dflt.nrows)

    @property
    def _pixel_cols(self):
        return self.get('pixels:columns', default=Dflt.ncols)

    @property
    def _pixel_size(self):
        return self.get('pixels:pixel_size', default=Dflt.pixel_size)

    @property
    def _translation(self):
        trans = self.get('transform:translation', default=Dflt.translation)
        return np.array(trans)

    @property
    def _tilt(self):
        tilt = self.get('transform:tilt', default=Dflt.tilt)
        return np.array(tilt)

    @property
    def _eta_vec(self):
        return self.get('eta_vec', default=constants.eta_vec.copy())

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
    def oscillation_stage(self):
        return instrument.oscillation_stage.OscillationStage(self._tvec, self._chi)

    # ========== Input Values
    @property
    def _tvec(self):
        return self.get('t_vec_s', default=self.tvec_DFLT)

    @property
    def _chi(self):
        return self.get('chi', default=self.chi_DFLT)
