from pathlib import Path
import sys

import numpy as np
import pytest

from hexrd.material import Material
from hexrd.powder.wppf.phase import Material_Rietveld
from hexrd.wppf.tds import TDS, TDS_material
from hexrd.valunits import _nm, _kev, _angstrom
from hexrd.constants import keVToAngstrom
from hexrd.wppf import Rietveld


@pytest.fixture
def tds_examples_path(example_repo_path: Path) -> Path:
    return Path(example_repo_path) / 'tests/wppf/tds'


@pytest.fixture
def tds_material_file(tds_examples_path: Path) -> Path:
    return tds_examples_path / 'Ti.xtal'


@pytest.fixture
def tds_expt_spectrum(tds_examples_path: Path) -> np.ndarray:
    return np.loadtxt(tds_examples_path / 'expt_spec_Ti.xy')


def test_wppf_tds(tds_material_file: Path, tds_expt_spectrum: np.ndarray):
    material_file = tds_material_file
    expt_spec = tds_expt_spectrum

    tth = np.linspace(15, 60, 4500)

    m = Material(
        name="Ti_beta", material_file=material_file, kev=_kev(10.2505), dmin=_nm(0.05)
    )
    exclusions = np.zeros_like(m.planeData.exclusions).astype(bool)
    m.planeData.exclusions = exclusions

    m_r = Material_Rietveld(material_obj=m)

    wavelength = m_r.wavelength * 10  # convert to A

    tds = TDS_material(material=m_r, tth=tth, wavelength=wavelength, smoothing=3)

    kwargs = {
        "expt_spectrum": expt_spec,
        "wavelength": {"XFEL": [_angstrom(keVToAngstrom(10.2505)), 1.0]},
        "bkgmethod": {"chebyshev": 1},
        "peakshape": "pvtch",
        "phases": [m],
    }
    R = Rietveld(**kwargs)

    R.params["scale"].value = 1e-5
    R.params["bkg_0"].value = 0
    R.params["bkg_1"].value = 0
    R.params["U"].value = 0
    R.params["V"].value = 0
    R.params["W"].value = 4000
    R.params["Ti_beta_Ti1_dw"].value = 0.113
    R.params["Ti_beta_X"].value = 0
    R.params["Ti_beta_Y"].value = 10

    kwargs = {
        "model_type": "warren",
        "phases": R.phases,
        "tth": R.tth_list,
        "model_data": None,
        "scale": None,
        "shift": None,
    }

    tds_model = TDS(**kwargs)

    R.tds_model = tds_model

    R.params["scale"].vary = True
    R.Refine()

    assert R.Rwp < 1

    R.params["V"].vary = True
    R.params["W"].vary = True
    R.params["Ti_beta_Y"].vary = True
    R.Refine()

    assert R.Rwp < 0.5

    R.params["Ti_beta_Ti1_dw"].vary = True
    R.Refine()

    assert R.Rwp < 0.2

    R.params["bkg_0"].vary = True
    R.params["bkg_1"].vary = True

    R.Refine()

    assert R.Rwp < 0.1

    # Now try the experimental model.
    # It shouldn't have a big impact. We'll just verify it runs.
    kwargs = {
        "model_type": "experimental",
        "phases": R.phases,
        "tth": R.tth_list,
        "model_data": tds_expt_spectrum,
        "scale": 1,
        "shift": 0,
    }
    tds_model = TDS(**kwargs)
    R.tds_model = tds_model

    R.Refine()

    assert R.Rwp < 0.1

    # Test calculating the equivalent temperatures
    equiv_temp_dict = m_r.calc_temperature({'Ti': 200})
    assert np.isclose(equiv_temp_dict['Ti'], 1511.75)
