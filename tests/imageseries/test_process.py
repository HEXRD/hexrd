import numpy as np

from .common import ImageSeriesTest, make_array_ims, make_omega_meta, compare

from hexrd import imageseries
from hexrd.imageseries import process, ImageSeries

class TestImageSeriesProcess(ImageSeriesTest):

    def _runfliptest(self, a, flip, aflip):
        is_a = imageseries.open(None, 'array', data=a)
        ops = [('flip', flip)]
        is_p = process.ProcessedImageSeries(is_a, ops)
        is_aflip = imageseries.open(None, 'array', data=aflip)
        diff = compare(is_aflip, is_p)
        msg = "flipped [%s] image series failed" % flip
        self.assertAlmostEqual(diff, 0., msg=msg)

    def test_process(self):
        """Processed image series"""
        _, is_a = make_array_ims()
        is_p = process.ProcessedImageSeries(is_a, [])
        diff = compare(is_a, is_p)
        msg = "processed image series failed to reproduce original"
        self.assertAlmostEqual(diff, 0., msg)

    def test_process_flip_t(self):
        """Processed image series: flip transpose"""
        flip = 't'
        a, _ = make_array_ims()
        aflip = np.transpose(a, (0, 2, 1))
        self._runfliptest(a, flip, aflip)

    def test_process_flip_v(self):
        """Processed image series: flip vertical"""
        flip = 'v'
        a, _ = make_array_ims()
        aflip = a[:, :, ::-1]
        self._runfliptest(a, flip, aflip)

    def test_process_flip_h(self):
        """Processed image series: flip horizontal"""
        flip = 'h'
        a, _ = make_array_ims()
        aflip = a[:, ::-1, :]
        self._runfliptest(a, flip, aflip)

    def test_process_flip_vh(self):
        """Processed image series: flip vertical + horizontal"""
        flip = 'vh'
        a, _ = make_array_ims()
        aflip = a[:, ::-1, ::-1]
        self._runfliptest(a, flip, aflip)

    def test_process_flip_r90(self):
        """Processed image series: flip counterclockwise 90"""
        flip = 'ccw90'
        a, _ = make_array_ims()
        aflip = np.transpose(a, (0, 2, 1))[:, ::-1, :]
        self._runfliptest(a, flip, aflip)

    def test_process_flip_r270(self):
        """Processed image series: flip clockwise 90 """
        flip = 'cw90'
        a, _ = make_array_ims()
        aflip = np.transpose(a, (0, 2, 1))[:, :, ::-1]
        self._runfliptest(a, flip, aflip)

    def test_process_dark(self):
        """Processed image series: dark image"""
        a, _ = make_array_ims()
        dark = np.ones_like(a[0])
        is_a = imageseries.open(None, 'array', data=a)
        apos = np.where(a >= 1, a-1, 0)
        is_a1 = imageseries.open(None, 'array', data=apos)
        ops = [('dark', dark)]
        is_p = process.ProcessedImageSeries(is_a, ops)
        diff = compare(is_a1, is_p)
        self.assertAlmostEqual(diff, 0., msg="dark image failed")

    def test_process_framelist(self):
        a, _ = make_array_ims()
        is_a = imageseries.open(None, 'array', data=a)
        is_a.metadata["omega"] = make_omega_meta(len(is_a))
        ops = []
        frames = [0, 2]
        is_p = process.ProcessedImageSeries(is_a, ops, frame_list=frames)
        is_a2 = imageseries.open(None, 'array', data=a[tuple(frames), ...])
        diff = compare(is_a2, is_p)
        self.assertAlmostEqual(diff, 0., msg="frame list failed")
        self.assertEqual(len(is_p), len(is_p.metadata["omega"]))


    def test_process_shape(self):
        a, _ = make_array_ims()
        is_a = imageseries.open(None, 'array', data=a)
        ops = []
        is_p = process.ProcessedImageSeries(is_a, ops)
        pshape = is_p.shape
        fshape = is_p[0].shape
        for i in range(2):
            self.assertEqual(fshape[i], pshape[i])

    def test_process_dtype(self):
        a, _ = make_array_ims()
        is_a = imageseries.open(None, 'array', data=a)
        ops = []
        is_p = process.ProcessedImageSeries(is_a, ops)
        self.assertEqual(is_p.dtype, is_p[0].dtype)
