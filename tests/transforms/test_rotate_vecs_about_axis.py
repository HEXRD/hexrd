from hexrd.core.transforms.new_capi.xf_new_capi import rotate_vecs_about_axis
import numpy as np


def test_rotate_vecs_about_axis():
    np.random.seed(0)
    # Test the rotate_vecs_about_axis function
    # Generate some random vectors
    n = 50
    vecs = np.random.rand(3, n) * 4

    # Generate some random rotation axes
    axes = np.random.rand(3, n)
    axes /= np.linalg.norm(axes, axis=0)[np.newaxis, :]

    # Generate some random angles
    angles = np.random.rand(n) * 2 * np.pi

    # Rotate the vectors
    rotated = rotate_vecs_about_axis(angles, axes, vecs)

    # Check that the rotated vectors have the same norm
    assert np.allclose(
        np.linalg.norm(vecs, axis=0), np.linalg.norm(rotated, axis=0)
    )

    # Check that the angle between the rotated vectors and the rotation
    # axes is the same
    for i in range(n):
        assert np.allclose(
            np.arccos(
                np.dot(rotated[:, i], axes[:, i])
                / np.linalg.norm(rotated[:, i])
            ),
            np.arccos(
                np.dot(vecs[:, i], axes[:, i] / np.linalg.norm(vecs[:, i]))
            ),
        )

    # Project all of the vectors on the normal plane to the rotation axes
    vecs_proj = np.zeros((3, n))
    rotated_proj = np.zeros((3, n))
    for i in range(n):
        vecs_proj[:, i] = (
            vecs[:, i] - np.dot(vecs[:, i], axes[:, i]) * axes[:, i]
        )
        rotated_proj[:, i] = (
            rotated[:, i] - np.dot(rotated[:, i], axes[:, i]) * axes[:, i]
        )

    # Normalize so angle checking is easier
    vecs_proj /= np.linalg.norm(vecs_proj, axis=0)
    rotated_proj /= np.linalg.norm(rotated_proj, axis=0)

    # Check that the angle between the projected vectors is the rotation angle
    for i in range(n):
        # The angle between these vectors could be the rotation angle,
        # pi - rotation angle, or other things as well
        # Just call arccos(cos(angle)) to solve this
        ang_fixed = np.arccos(np.cos(angles[i]))
        assert np.allclose(
            np.arccos(np.dot(vecs_proj[:, i], rotated_proj[:, i])), ang_fixed
        )


def test_rotate_vecs_about_axis_zero_vec():
    vec = np.zeros((3, 1))
    axis = np.array([0.0, 0.0, 1.0])
    angle = np.array([np.pi / 2])
    rotated = rotate_vecs_about_axis(angle, axis, vec)
    assert np.allclose(rotated, vec)
