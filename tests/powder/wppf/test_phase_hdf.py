import shutil

import h5py
import numpy as np

from hexrd.core.material import Material
from hexrd.core.valunits import _kev, _nm
from hexrd.powder.wppf.phase import Material_Rietveld


def test_rietveld_from_hdf_anisotropic_u(test_data_dir, tmp_path):
    path = tmp_path / 'materials.h5'
    shutil.copy(test_data_dir / 'materials/materials_variants.h5', path)
    with h5py.File(path, 'r+') as f:
        del f['Ta/U']
        f['Ta/U'] = np.array([[0.003, 0.003, 0.003, 0.0005, 0.0, 0.0]]).T

    dmin, kev = _nm(0.1), _kev(80.0)
    mat = Material_Rietveld(fhdf=path, xtal='Ta', dmin=dmin, kev=kev)
    ref = Material('Ta', path, dmin=dmin, kev=kev)
    assert np.allclose(mat.betaij, ref.unitcell.betaij)
