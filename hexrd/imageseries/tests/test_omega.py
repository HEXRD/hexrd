import numpy as np

from .common import ImageSeriesTest

from hexrd import imageseries
from hexrd.imageseries.omega import OmegaSeriesError, OmegaImageSeries

class TestOmegaSeries(ImageSeriesTest):

    @staticmethod
    def make_ims(nf, meta):
        a = np.zeros((nf, 2, 2))
        ims = imageseries.open(None, 'array', data=a, meta=meta)
        return ims

    def test_no_omega(self):
        ims = self.make_ims(2, {})
        with self.assertRaises(OmegaSeriesError):
            oms = OmegaImageSeries(ims)

    def test_nframes_mismatch(self):
        m = dict(omega=np.zeros((3, 2)))
        ims = self.make_ims(2, m)
        with self.assertRaises(OmegaSeriesError):
            oms = OmegaImageSeries(ims)

    def test_negative_delta(self):
        om = np.zeros((3, 2))
        om[0,1] = -0.5
        m = dict(omega=om, dtype=np.float)
        ims = self.make_ims(3, m)
        with self.assertRaises(OmegaSeriesError):
            oms = OmegaImageSeries(ims)

    def test_one_wedge(self):
        nf = 5
        a = np.linspace(0, nf+1, nf+1)
        om = np.zeros((nf, 2))
        om[:,0] = a[:-1]
        om[:,1] = a[1:]
        m = dict(omega=om, dtype=np.float)
        ims = self.make_ims(nf, m)
        oms = OmegaImageSeries(ims)
        self.assertEqual(oms.nwedges, 1)

    def test_two_wedges(self):
        nf = 5
        a = np.linspace(0, nf+1, nf+1)
        om = np.zeros((nf, 2))
        om[:,0] = a[:-1]
        om[:,1] = a[1:]
        om[3:, :] += 0.1
        m = dict(omega=om, dtype=np.float)
        ims = self.make_ims(nf, m)
        oms = OmegaImageSeries(ims)
        self.assertEqual(oms.nwedges, 2)

    def test_compare_omegas(self):
        nf = 5
        a = np.linspace(0, nf+1, nf+1)
        om = np.zeros((nf, 2))
        om[:,0] = a[:-1]
        om[:,1] = a[1:]
        om[3:, :] += 0.1
        m = dict(omega=om, dtype=np.float)
        ims = self.make_ims(nf, m)
        oms = OmegaImageSeries(ims)
        domega = om - oms.omegawedges.omegas
        dnorm = np.linalg.norm(domega)

        msg='omegas from wedges do not match originals'
        self.assertAlmostEqual(dnorm, 0., msg=msg)

    def test_wedge_delta(self):
        nf = 5
        a = np.linspace(0, nf+1, nf+1)
        om = np.zeros((nf, 2))
        om[:,0] = a[:-1]
        om[:,1] = a[1:]
        om[3:, :] += 0.1
        m = dict(omega=om, dtype=np.float)
        ims = self.make_ims(nf, m)
        oms = OmegaImageSeries(ims)

        mydelta =om[nf - 1, 1] - om[nf - 1, 0]
        d = oms.wedge(oms.nwedges - 1)
        self.assertAlmostEqual(d['delta'], mydelta)

    # end class
