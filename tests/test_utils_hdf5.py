from pathlib import Path

import h5py
import numpy as np
import pytest

from hexrd.core.utils.hdf5 import unwrap_dict_to_h5


def test_hdf5_optional_values_and_attribute_errors(tmp_path: Path) -> None:
    with h5py.File(tmp_path / 'values.h5', 'w') as stream:
        for asattr in (False, True):
            group = stream.create_group(str(asattr))
            unwrap_dict_to_h5(
                group, {'absent': None, 'number': 3, 'text': 'value'}, asattr=asattr
            )
            values = group.attrs if asattr else group
            assert set(values) == {'number', 'text'}
            np.testing.assert_array_equal(values['number'], 3 if asattr else [3])

        # Unsupported attributes must still propagate the original error.
        with pytest.raises(TypeError):
            unwrap_dict_to_h5(stream, {'unicode': np.array(['value'])}, asattr=True)
