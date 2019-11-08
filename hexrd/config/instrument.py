import os

import numpy as np

from hexrd import constants
from hexrd import instrument
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
        bcfg = Beam(self)
        return instrument.beam.Beam(bcfg.energy, bcfg.vector)

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
            d[id] = self.get_detector(id)

        return d

    def get_detector(self, id):
        """Get detector by id"""
        dcfg = Detector(self, id)

        rows = dcfg.get('pixels:rows', default=Detector.nrows_DFLT)
        cols = dcfg.get('pixels:columns', default=Detector.ncols_DFLT)
        pixel_size = dcfg.get('pixels:pixel_size', default=Detector.pixel_size_DFLT)
        tvec = dcfg.get('transform:t_vec_d', default=Detector.t_vec_d_DFLT)
        tilt = dcfg.get('transform:tilt_angles', default=Detector.tilt_angles_DFLT)
        bvec = self.beam.vector
        evec = self._cfg.get('eta_vec', default=constants.eta_vec)

        # Distortion
        dparams = dcfg.get('distortion:parameters', default=None)
        distortion = (None, dparams) if dparams is not None else None

        d = instrument.PlanarDetector(
            rows=rows, cols=cols, pixel_size=pixel_size, tvec=tvec, tilt=tilt,
            beam=self.beam, evec=evec, distortion=distortion)

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


class Beam(Config):

    beam_energy_DFLT = 65.351
    beam_vec_DFLT = constants.beam_vec

    BASEKEY = 'beam'

    def get(self, key, **kwargs):
        """get item with given key"""
        return self._cfg.get(':'.join([self.BASEKEY, key]), **kwargs)

    @property
    def energy(self):
        return self.get('energy', default=self.beam_energy_DFLT)

    @property
    def vector(self):
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
