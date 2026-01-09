"""Test rotations module"""

import logging
import numpy as np
import pytest

from hexrd.core.material import symmetry
from hexrd.core import rotations

logger = logging.getLogger(__name__)

def test_misorientations():
    """Use Laue groups to test for zero misorientation"""
    #
    # This tests that all the 11 Laue groups have zero misorientation with
    # their own members.
    #
    laue_groups = [
        "ci",
        "c2h",
        "d2h",
        "c4h",
        "d4h",
        "c3i",
        "d3d",
        "c6h",
        "d6h",
        "th",
        "oh",
    ]
    for lg in laue_groups:
        logger.debug(f"group: {lg}")
        qsym = symmetry.quatOfLaueGroup(lg)
        q1 = qsym[:, -1:]
        ang, mis = rotations.misorientation(q1, qsym, (qsym,))
        assert np.allclose(ang, 0.0)
        assert np.allclose(mis[0, :], 1.0)
        assert np.allclose(mis[1:, :], 0.0)
