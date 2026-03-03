import numpy as np
import pytest

from hexrd.core.instrument.hedm_instrument import _parse_imgser_dict


class TestParseImgserDictGroupFallback:
    """Test that _parse_imgser_dict falls back to the group name."""

    def test_group_fallback_with_ndarray(self):
        # Image keyed by group name, not detector name
        img = np.arange(20, dtype=np.float64).reshape(4, 5)
        imgser_dict = {'my_group': img}
        roi = ((1, 3), (0, 5))

        result = _parse_imgser_dict(
            imgser_dict, det_key='det_0', roi=roi, group='my_group'
        )

        expected = img[1:3, 0:5]
        np.testing.assert_array_equal(result, expected)

    def test_group_fallback_not_used_when_det_key_matches(self):
        # Direct det_key match should take priority over group
        img_direct = np.ones((4, 5), dtype=np.float64)
        img_group = np.zeros((4, 5), dtype=np.float64)
        imgser_dict = {'det_0': img_direct, 'my_group': img_group}

        result = _parse_imgser_dict(imgser_dict, det_key='det_0', group='my_group')

        np.testing.assert_array_equal(result, img_direct)

    def test_group_none_raises_when_no_match(self):
        imgser_dict = {'other': np.zeros((4, 5))}

        with pytest.raises(RuntimeError, match="neither"):
            _parse_imgser_dict(imgser_dict, det_key='det_0', group=None)
