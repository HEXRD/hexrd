from hexrd import rotations
import numpy as np


def test_map_angle_degrees():
    """
    Test mapAngle with units being degrees, default range
    """
    np.random.seed(0)
    for _ in range(100):
        angs = np.random.rand(10) * 10000 - 5000
        map_angs_deg = rotations.mapAngle(angs, units='degrees')
        assert (np.abs(map_angs_deg) <= 180).all()
        assert np.allclose(
            np.sin(np.radians(map_angs_deg)), np.sin(np.radians(angs))
        )
        assert np.allclose(
            np.cos(np.radians(map_angs_deg)), np.cos(np.radians(angs))
        )


def test_map_angle_radians():
    """
    Test mapAngle with units being radians, default range
    """
    np.random.seed(0)
    for _ in range(100):
        angs = np.random.rand(10) * 60 - 30
        map_angs_deg = rotations.mapAngle(angs)
        assert (np.abs(map_angs_deg) <= np.pi).all()
        assert np.allclose(np.sin(map_angs_deg), np.sin(angs))
        assert np.allclose(np.cos(map_angs_deg), np.cos(angs))


def test_map_angle_degrees_range():
    """
    Test mapAngle with units being degrees, given a random range
    """
    np.random.seed(0)
    for _ in range(100):
        angs = np.random.rand(10) * 10000 - 5000
        min_val = np.random.rand() * 1000 - 500
        max_val = min_val + 360
        map_angs_deg = rotations.mapAngle(
            angs, [min_val, max_val], units='degrees'
        )

        assert (np.abs(map_angs_deg - min_val - 180) <= 180).all()
        assert np.allclose(
            np.sin(np.radians(map_angs_deg)), np.sin(np.radians(angs))
        )
        assert np.allclose(
            np.cos(np.radians(map_angs_deg)), np.cos(np.radians(angs))
        )


def test_map_angle_radians_range():
    """
    Test mapAngle with units being radians, given a random range
    """
    np.random.seed(0)
    for _ in range(100):
        angs = np.random.rand(10) * 60 - 30
        min_val = np.random.rand() * 30 - 15
        max_val = min_val + 2 * np.pi
        map_angs_deg = rotations.mapAngle(
            angs, [max_val, min_val], units='radians'
        )

        assert (np.abs(map_angs_deg - min_val - np.pi) <= np.pi).all()
        assert np.allclose(np.sin(map_angs_deg), np.sin(angs))
        assert np.allclose(np.cos(map_angs_deg), np.cos(angs))
