import numpy as np
import pytest

from hexrd.hedm.fitgrains import (
    _apply_hkl_exclusions,
    _load_hkl_exclusion_func,
    fit_grain_FF_init,
    fit_grain_FF_cleanup,
)


def _make_results():
    """Create mock pull_spots results with known HKLs.

    Each entry mirrors the 9-element tuple from pull_spots:
        (peak_id, hkl_id, hkl, sum_int, max_int,
         pred_angs, meas_angs, meas_xy, pred_xy)
    """
    return {
        'det1': [
            (0, 0, np.array([1, 0, 0]), 100.0, 50.0,
             np.zeros(3), np.zeros(3), np.zeros(2), np.zeros(2)),
            (1, 1, np.array([1, 1, 0]), 200.0, 75.0,
             np.zeros(3), np.zeros(3), np.zeros(2), np.zeros(2)),
            (2, 2, np.array([1, 1, 1]), 300.0, 100.0,
             np.zeros(3), np.zeros(3), np.zeros(2), np.zeros(2)),
        ],
        'det2': [
            (3, 0, np.array([1, 0, 0]), 150.0, 60.0,
             np.zeros(3), np.zeros(3), np.zeros(2), np.zeros(2)),
            (4, 3, np.array([2, 0, 0]), 250.0, 80.0,
             np.zeros(3), np.zeros(3), np.zeros(2), np.zeros(2)),
        ],
    }


class TestApplyHklExclusions:

    def test_empty_list_returns_same_object(self):
        results = _make_results()
        assert _apply_hkl_exclusions(results, []) is results

    def test_none_returns_same_object(self):
        results = _make_results()
        assert _apply_hkl_exclusions(results, None) is results

    def test_exclude_single_hkl(self):
        results = _make_results()
        filtered = _apply_hkl_exclusions(results, [[1, 0, 0]])
        assert len(filtered['det1']) == 2
        assert len(filtered['det2']) == 1
        hkls = {tuple(e[2]) for e in filtered['det1']}
        assert hkls == {(1, 1, 0), (1, 1, 1)}

    def test_exclude_multiple_hkls(self):
        results = _make_results()
        filtered = _apply_hkl_exclusions(results, [[1, 0, 0], [1, 1, 1]])
        assert len(filtered['det1']) == 1
        assert tuple(filtered['det1'][0][2]) == (1, 1, 0)
        assert len(filtered['det2']) == 1
        assert tuple(filtered['det2'][0][2]) == (2, 0, 0)

    def test_exclude_nonexistent_hkl_is_noop(self):
        results = _make_results()
        filtered = _apply_hkl_exclusions(results, [[5, 5, 5]])
        assert len(filtered['det1']) == 3
        assert len(filtered['det2']) == 2

    def test_exclude_all_hkls(self):
        results = _make_results()
        filtered = _apply_hkl_exclusions(
            results, [[1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 0, 0]]
        )
        assert len(filtered['det1']) == 0
        assert len(filtered['det2']) == 0

    def test_accepts_tuples(self):
        results = _make_results()
        filtered = _apply_hkl_exclusions(results, [(1, 0, 0)])
        assert len(filtered['det1']) == 2

    def test_accepts_numpy_arrays(self):
        results = _make_results()
        filtered = _apply_hkl_exclusions(results, [np.array([1, 0, 0])])
        assert len(filtered['det1']) == 2

    def test_does_not_modify_original(self):
        results = _make_results()
        _apply_hkl_exclusions(results, [[1, 0, 0]])
        assert len(results['det1']) == 3
        assert len(results['det2']) == 2


class TestLoadHklExclusionFunc:

    def test_load_valid_func(self, tmp_path):
        script = tmp_path / "exclusions.py"
        script.write_text(
            "def hkl_exclusion_func(grain_id, grains_table,"
            " plane_data, instrument):\n"
            "    return [[1, 0, 0]]\n"
        )
        func = _load_hkl_exclusion_func(str(script))
        assert callable(func)
        result = func(
            grain_id=0, grains_table=None,
            plane_data=None, instrument=None,
        )
        assert result == [[1, 0, 0]]

    def test_missing_func_raises(self, tmp_path):
        script = tmp_path / "bad.py"
        script.write_text("def wrong_name(): pass\n")
        with pytest.raises(RuntimeError, match="does not define"):
            _load_hkl_exclusion_func(str(script))

    def test_per_grain_logic(self, tmp_path):
        script = tmp_path / "per_grain.py"
        script.write_text(
            "def hkl_exclusion_func(grain_id, grains_table,"
            " plane_data, instrument):\n"
            "    if grain_id == 0:\n"
            "        return [[1, 0, 0]]\n"
            "    elif grain_id == 1:\n"
            "        return [[1, 1, 1], [2, 0, 0]]\n"
            "    return []\n"
        )
        func = _load_hkl_exclusion_func(str(script))
        assert func(
            grain_id=0, grains_table=None,
            plane_data=None, instrument=None,
        ) == [[1, 0, 0]]
        assert func(
            grain_id=1, grains_table=None,
            plane_data=None, instrument=None,
        ) == [[1, 1, 1], [2, 0, 0]]
        assert func(
            grain_id=2, grains_table=None,
            plane_data=None, instrument=None,
        ) == []


class TestFitGrainFFInitLoadsFunc:

    def test_init_loads_func_from_path(self, tmp_path):
        script = tmp_path / "exclusions.py"
        script.write_text(
            "def hkl_exclusion_func(grain_id, grains_table,"
            " plane_data, instrument):\n"
            "    return [[1, 0, 0]]\n"
        )
        params = {'hkl_exclusion_func_path': str(script)}
        fit_grain_FF_init(params)
        try:
            assert callable(params['hkl_exclusion_func'])
            result = params['hkl_exclusion_func'](
                grain_id=0, grains_table=None,
                plane_data=None, instrument=None,
            )
            assert result == [[1, 0, 0]]
        finally:
            fit_grain_FF_cleanup()

    def test_init_sets_none_when_no_path(self):
        params = {}
        fit_grain_FF_init(params)
        try:
            assert params['hkl_exclusion_func'] is None
        finally:
            fit_grain_FF_cleanup()
