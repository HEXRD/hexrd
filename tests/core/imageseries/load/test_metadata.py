from unittest.mock import patch
import numpy as np
import os
from hexrd.core.imageseries.load import metadata


def test_yamlmeta_simple():
    """Test standard key-value pass-through."""
    meta = {'a': 1, 'b': 'two', 'c': [1, 2]}
    res = metadata.yamlmeta(meta)
    assert res == meta


def test_yamlmeta_legacy_array():
    """Test '++np.array' legacy parsing logic."""
    meta = {'data-array': [1, 2, 3], 'data': '++np.array'}

    res = metadata.yamlmeta(meta)

    assert isinstance(res['data'], np.ndarray)
    np.testing.assert_array_equal(res['data'], [1, 2, 3])
    assert 'data-array' not in res


@patch('hexrd.core.imageseries.load.metadata.np.load')
def test_yamlmeta_trigger(mock_load):
    """Test '! load-numpy-array' trigger logic."""
    mock_load.return_value = np.array([10, 20])

    meta = {'myarr': '! load-numpy-array data.npy'}
    res = metadata.yamlmeta(meta)

    assert np.array_equal(res['myarr'], [10, 20])
    mock_load.assert_called_with(os.path.join('.', 'data.npy'))

    res = metadata.yamlmeta(meta, path='/tmp/config.yml')
    mock_load.assert_called_with(os.path.join('/tmp', 'data.npy'))


def test_yamlmeta_ignored_triggers():
    """
    Test logic for strings that look like triggers or have unknown commands.
    Implementation detail: If it matches the trigger pattern (! word ...)
    but isn't a known command, the key is DROPPED.
    """
    meta = {
        'a': '! unknown-command arg',
        'b': '!',
        'c': '!load-numpy-array',
        'd': 'plain string',
    }
    res = metadata.yamlmeta(meta)

    assert 'a' not in res

    assert res['b'] == '!'
    assert res['c'] == '!load-numpy-array'
    assert res['d'] == 'plain string'
