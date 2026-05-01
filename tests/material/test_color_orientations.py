from pathlib import Path

import numpy as np

from hexrd.material.material import Material


def test_color_orientations() -> None:
    examples_dir = Path(__file__).resolve().parents[1] / "data" / "materials"
    rotations_file = examples_dir / "color_orientations.npz"

    data = np.load(rotations_file)

    """
    rmats is used to test behavior with single and multiple rotation matrices
    dir3_nh is the nx3 direction for color testing
    dir3_nh_red direction reduced to fundamental triangle
    rgb_nh is the nx3 color for the n directions
    th is n theta values used in calculating color values
    rho is the n azimuth values used in color calculation
    """
    rmats = data["rmats"]
    dir3_nh = data["dir3_nh"]
    dir3_nh_red = data["dir3_nh_red"]
    rgb_nh = data["rgb_nh"]
    th = data["th"]
    rho = data["rho"]

    # check all shapes
    assert rmats.shape == (2, 3, 3)
    assert dir3_nh.shape == (1861, 3)
    assert rgb_nh.shape == (1861, 3)
    assert th.shape == rho.shape == (1861,)

    # check if single matrix and multiple matrix behavior is the same
    mat = Material()
    v1 = np.round(mat.unitcell.color_orientations(rmats), 6)
    v2 = np.vstack(
        (
            np.round(mat.unitcell.color_orientations(rmats[0]), 6),
            np.round(mat.unitcell.color_orientations(rmats[1]), 6),
        )
    )

    assert v1 == v2

    # check actual color values step by step

    # check the coordinates of reduced directions
    dir3_nh_red_calc = mat.unitcell.reduce_dirvector(dir3_nh, switch="super")

    assert dir3_nh_red == dir3_nh_red_calc

    # check theta and rho values
    th_calc = mat.unitcell.sphere_sector.calculate_theta(dir3_nh_red, "super")
    rho_calc = mat.unitcell.sphere_sector.calculate_theta(dir3_nh_red, "super")

    assert th == th_calc
    assert rho == rho_calc

    # check rgb values
    rgb_nh_calc = mat.unitcell.color_directions(dir3_nh, False)

    assert rgb_nh == rgb_nh_calc
