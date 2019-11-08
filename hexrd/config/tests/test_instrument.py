import os

import hexrd.instrument
from .common import TestConfig, test_data
from ..instrument import Instrument, Beam, OscillationStage

reference_data = \
"""
beam: {}
---
beam:
  energy: 2.0
  vector: {azimuth: 0.0, polar_angle: 0.0}
---
oscillation_stage:
  chi: 0.05
  t_vec_s: [1., 2., 3.]
---
detectors:
  GE1:
    distortion:
      function_name: GE_41RT
      parameters: [7.617424115028922e-05, -1.01006559390677e-06, -0.00016461139058911365,
        2.0, 2.0, 2.0]
    pixels:
      columns: 2048
      rows: 2048
      size: [0.2, 0.2]
    saturation_level: 14000.0
    transform:
      t_vec_d: [94.51351402409436, -337.4575337059045, -1921.058935922086]
      tilt_angles: [0.002314455268055846, 6.288758382211901e-05, 1.0938371193555785]
  GE2:
    distortion:
      function_name: GE_41RT
      parameters: [5.245111176545523e-05, -3.165350904260842e-05, -0.00020774139197230943,
        2.0, 2.0, 2.0]
    pixels:
      columns: 2048
      rows: 2048
      size: [0.2, 0.2]
    saturation_level: 14000.0
    transform:
      t_vec_d: [-320.190205619744, -95.95873622987875, -1920.07233414923]
      tilt_angles: [0.00044459111576242654, 0.003958638944891969, -0.47488346109306645]
---
instrument: instrument.yaml
""" % test_data


class TestInstrument(TestConfig):

    @classmethod
    def get_reference_data(cls):
        return reference_data

    def test_beam(self):
        icfg = Instrument(self.cfgs[1])
        b = icfg.beam
        self.assertTrue(isinstance(b, hexrd.instrument.beam.Beam), "Failed to produce a Beam instance")

    def test_oscillation_stage(self):
        icfg = Instrument(self.cfgs[2])
        ostage = icfg.oscillation_stage
        self.assertTrue(isinstance(ostage, hexrd.instrument.oscillation_stage.OscillationStage),
                        "Failed to produce an OscillationStage instance")

    def test_detector(self):
        icfg = Instrument(self.cfgs[3])
        det = icfg.get_detector('GE1')
        self.assertTrue(isinstance(det, hexrd.instrument.PlanarDetector),
                        "Failed to produce an Detector instance")

    def test_detector_dict(self):
        icfg = Instrument(self.cfgs[3])
        dd = icfg.detector_dict
        self.assertTrue(isinstance(dd, dict),
                        "Failed to produce an Detector Dictionary instance")
        for k in dd:
            d = dd[k]
            self.assertTrue(isinstance(d, hexrd.instrument.PlanarDetector),
                            "Detector dictionary values are not detector instances")


class TestBeam(TestConfig):

    @classmethod
    def get_reference_data(cls):
        return reference_data

    def test_beam_energy_dflt(self):
        bcfg = Beam(self.cfgs[0])
        energy = bcfg.energy
        self.assertEqual(energy, Beam.beam_energy_DFLT, "Incorrect default beam energy")

    def test_beam_energy(self):
        bcfg = Beam(self.cfgs[1])
        energy = bcfg.energy
        self.assertEqual(energy, 2.0, "Incorrect beam energy")

    def test_beam_vector_dflt(self):
        bcfg = Beam(self.cfgs[0])
        bvecdflt = Beam.beam_vec_DFLT
        bvec = bcfg.vector

        self.assertEqual(bvec[0], bvecdflt[0], "Incorrect default beam vector")
        self.assertEqual(bvec[1], bvecdflt[1], "Incorrect default beam vector")
        self.assertEqual(bvec[2], bvecdflt[2], "Incorrect default beam vector")

    def test_beam_vector(self):
        bcfg = Beam(self.cfgs[1])
        bvec = bcfg.vector

        self.assertEqual(bvec[0], 0.0, "Incorrect default beam vector")
        self.assertEqual(bvec[1], -1.0, "Incorrect default beam vector")
        self.assertEqual(bvec[2], 0.0, "Incorrect default beam vector")


class TestOscillationStage(TestConfig):

    @classmethod
    def get_reference_data(cls):
        return reference_data

    def test_chi_dflt(self):
        oscfg = OscillationStage(self.cfgs[0])
        self.assertEqual(oscfg.chi, OscillationStage.chi_DFLT, "Incorrect default chi for oscillation stage")

    def test_chi(self):
        oscfg = OscillationStage(self.cfgs[2])
        self.assertEqual(oscfg.chi, 0.05, "Incorrect default chi for oscillation stage")

    def test_tvec_dflt(self):
        oscfg = OscillationStage(self.cfgs[0])
        tvec_dflt = OscillationStage.tvec_DFLT
        tvec = oscfg.tvec

        self.assertEqual(tvec[0], tvec_dflt[0], "Incorrect default translation vector")
        self.assertEqual(tvec[1], tvec_dflt[1], "Incorrect default translation vector")
        self.assertEqual(tvec[2], tvec_dflt[2], "Incorrect default translation vector")

    def test_tvec(self):
        oscfg = OscillationStage(self.cfgs[2])
        tvec = oscfg.tvec

        self.assertEqual(tvec[0], 1., "Incorrect translation vector")
        self.assertEqual(tvec[1], 2., "Incorrect translation vector")
        self.assertEqual(tvec[2], 3., "Incorrect translation vector")
