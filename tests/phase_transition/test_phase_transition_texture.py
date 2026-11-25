from pathlib import Path

import numpy as np
import pytest

from hexrd.core.material import load_materials_hdf5
from hexrd.core.valunits import _kev, _nm
from hexrd.phase_transition import variants
from hexrd.phase_transition.orientation_relationship import OrientationRelationship
from hexrd.powder.wppf.phase import Material_Rietveld
from hexrd.powder.wppf.texture import PhaseTransitionTextureModel


@pytest.fixture
def orientation_relationship(test_data_dir: Path) -> OrientationRelationship:
    mats = load_materials_hdf5(
        test_data_dir / 'materials/materials_variants.h5',
        dmin=_nm(0.075),
        kev=_kev(10),
    )
    # lmfit parameter names cannot start with a digit
    mats['4H_SiC_HP'].name = 'SiC_4H'
    parent = Material_Rietveld(material_obj=mats['4H_SiC_HP'])
    daughter = Material_Rietveld(material_obj=mats['SiC_B1'])

    # Same relationship as test_variants.py. The daughter is listed first
    # to make sure the order of the keys does not matter.
    return OrientationRelationship(
        parent,
        daughter,
        parallel_planes={
            'daughter': np.array([0.0, 0.0, 1.0]),
            'parent': np.array([0.0, 0.0, 1.0]),
        },
        parallel_directions={
            'daughter': np.array([1.0, 1.0, 0.0]),
            'parent': np.array([2.0, 1.0, 0.0]),
        },
    )


def test_orientation_relationship(orientation_relationship: OrientationRelationship):
    OR = orientation_relationship
    expected, _ = variants.getPhaseTransformationVariants(
        OR.material_parent,
        OR.material_daughter,
        [OR.plane_parent, OR.plane_daughter],
        [OR.direction_parent, OR.direction_daughter],
        apply_parent_rotation_to_variants=True,
    )
    assert np.shape(expected) == (3, 3, 3)

    # The default parent orientation is the identity
    assert np.allclose(OR.rmat_parent, [np.eye(3)])
    assert np.allclose(OR.rmat_variants_rotated, expected)

    # Every parent orientation gets its own set of variants
    OR.rmat_parent = [np.eye(3), np.diag([1.0, -1.0, -1.0])]
    assert len(OR.rmat_variants_rotated) == 2 * OR.num_variants

    with pytest.raises(ValueError):
        OR.parallel_planes = {'parent': OR.plane_parent, 'foo': OR.plane_daughter}


def test_phase_transition_texture_model(
    orientation_relationship: OrientationRelationship,
):
    OR = orientation_relationship
    truth = PhaseTransitionTextureModel(
        OR,
        parent_kernel_width=5.0,
        daughter_kernel_width=3.0,
        parent_misorientation=np.array([2.0, -1.0, 1.5]),
    )
    true_params = truth.get_parameters()

    # The exact texture index should agree with a Monte Carlo estimate
    J = truth.J(true_params)
    for key, odf in truth.odfs.items():
        assert np.allclose(np.linalg.det(odf.modal_orientations), 1)
        mc = odf.texture_index(n_orientations=200000, seed=0)
        assert J[truth._material(key).name] == pytest.approx(mc, rel=0.05)

    # Refine a fresh model against pole figure data from the truth
    model = PhaseTransitionTextureModel(OR)
    params = model.get_parameters(vary=True)
    for key, view in model._views.items():
        truth_view = truth._views[key]
        # 3 degrees gave one row too few before (int rounded 119.99 down)
        truth_view.calc_pf_rings(true_params, eta_step=np.radians(3.0))
        assert len(truth_view.eta_grid) == 119
        pfdata = {}
        for hkl, ratio in truth_view.intensities_rings.items():
            if np.all(np.isfinite(ratio)):
                gvecs = truth_view.gvecs_rings[hkl]
                untextured = np.ones_like(ratio)
                pfdata[hkl] = np.column_stack((gvecs, ratio, untextured))
        view.pfdata = pfdata

    model._views['parent'].calculate_harmonic_coefficients(params)
    for name, param in true_params.items():
        assert params[name].value == pytest.approx(param.value, abs=1e-3)
