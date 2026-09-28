import argparse
import json

import numpy as np
import pytest

from hexrd.hedm.cake_data import CakeDataConfig, CakeMatrix
from hexrd.hedm.cli.cake_data import configure_parser


class FakePanel:
    rows = 2
    cols = 2


class FakeView:
    ntth = 2
    neta = 1
    shape = (1, 2)
    _nan_mask = np.array([[False, False]])
    _coordinate_mapping = {
        "panel": {
            "on_panel_idx": np.array([0, 1]),
            "bilinear_interp_dict": {
                "i_floor_img": np.array([0, 0]),
                "j_floor_img": np.array([0, 0]),
                "i_ceil_img": np.array([1, 1]),
                "j_ceil_img": np.array([1, 1]),
                "cc": np.array([1.0, 0.25]),
                "fc": np.array([0.0, 0.25]),
                "cf": np.array([0.0, 0.25]),
                "ff": np.array([0.0, 0.25]),
            },
        }
    }


def test_cake_matrix_uses_bilinear_matrix_multiplication():
    operator = CakeMatrix.__new__(CakeMatrix)
    operator.instrument = type(
        "FakeInstrument", (), {"detectors": {"panel": FakePanel()}}
    )()
    operator.view = FakeView()
    operator.mask = operator.view._nan_mask.copy()
    operator.matrix = operator._build_matrix()

    result = operator.apply({"panel": np.array([[2.0, 4.0], [6.0, 8.0]])})

    np.testing.assert_array_equal(result.data, [[2.0, 5.0]])


def test_config_file_and_command_line_update(tmp_path):
    path = tmp_path / "cake.json"
    path.write_text(
        json.dumps(
            {
                "data_dir": "raw/{sample_name}/{image_number}",
                "output_dir": "output",
                "sample_name": "ceria",
                "instrument": "instrument.hexrd",
                "par_file": "input.par",
                "cake_width": 10,
            }
        ),
        encoding="utf-8",
    )
    config = CakeDataConfig.from_json(path)
    config.update({"tth_start": 4.0, "cake_width": None})

    assert config.tth_start == 4.0
    assert config.cake_width == 10
    config.validate()


def test_config_rejects_unknown_parameters(tmp_path):
    path = tmp_path / "cake.json"
    path.write_text('{"not_a_parameter": 1}', encoding="utf-8")

    with pytest.raises(ValueError, match="unknown cake-data parameters"):
        CakeDataConfig.from_json(path)


def test_cli_exposes_short_and_long_parameter_forms():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    configure_parser(subparsers)

    short = parser.parse_args(["cake-data", "-b", "4", "-p", "input.par"])
    long = parser.parse_args(
        ["cake-data", "--tth-start", "4", "--par-file", "input.par"]
    )

    assert short.tth_start == long.tth_start == 4
    assert short.par_file == long.par_file == "input.par"
