from pathlib import Path

import numpy as np
import yaml

from hexrd.core.instrument import HEDMInstrument
from hexrd.core.utils.panel_buffer import panel_buffer_from_str


def test_string_panel_buffer(example_repo_path: Path):
    first_ceria_path = example_repo_path / 'eiger/first_ceria'
    monolithic_path = first_ceria_path / 'eiger_ceria_uncalibrated_monolithic.yml'
    composite_path = first_ceria_path / 'eiger_ceria_calibrated_composite.yml'

    with open(monolithic_path, 'r') as rf:
        conf = yaml.safe_load(rf)

    mono_instr = HEDMInstrument(conf)

    with open(composite_path, 'r') as rf:
        conf = yaml.safe_load(rf)

    comp_instr = HEDMInstrument(conf)

    mono_panel = mono_instr.detectors['eiger']
    mono_buffer = panel_buffer_from_str('chess-eiger-stream-v1', mono_panel)

    # Verify a couple of known regions
    assert np.array_equal(
        mono_buffer[506:510, 985:988],
        [[ True,  True,  True],
         [ True,  True,  True],
         [ True, False,  True],
         [ True,  True,  True]],
    )

    assert np.array_equal(
        mono_buffer[2710:2713, 2066:2069],
        [[ True,  True, False],
         [ True,  True, False],
         [False, False, False]],
    )

    # Check a couple of known composite regions too
    panel_0_0 = comp_instr.detectors['eiger_0_0']
    buffer_0_0 = panel_buffer_from_str('chess-eiger-stream-v1', panel_0_0)

    assert np.array_equal(
        buffer_0_0[207:210, 335:338],
        [[ True,  True,  True],
         [ True, False, False],
         [ True,  True, False]],
    )

    panel_6_2 = comp_instr.detectors['eiger_6_2']
    buffer_6_2 = panel_buffer_from_str('chess-eiger-stream-v2', panel_6_2)

    assert np.array_equal(
        buffer_6_2[18:21, 520:523],
        [[False, False, False],
         [False,  True,  True],
         [False,  True,  True]],
    )
