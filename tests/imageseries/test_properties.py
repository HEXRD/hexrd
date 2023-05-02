from .common import ImageSeriesTest, make_array_ims


class TestProperties(ImageSeriesTest):
    def setUp(self):
        self._a, self._is_a = make_array_ims()

    def test_prop_nframes(self):
        self.assertEqual(self._a.shape[0], len(self._is_a))

    def test_prop_shape(self):
        self.assertEqual(self._a.shape[1:], self._is_a.shape)

    def test_prop_dtype(self):
        self.assertEqual(self._a.dtype, self._is_a.dtype)
