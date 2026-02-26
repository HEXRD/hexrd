from pathlib import Path

import h5py
import numpy as np
import pytest

from hexrd.core.imageseries.load.eiger_stream_v1 import EigerStreamV1ImageSeriesAdapter

from .test_eiger_decompression import _compress_array_csrnpz, _compress_array_lz4

# NOTE: this is in a separate format than the `test_formats.py` file
# so that we can utilize the pytest fixtures.


pytest.importorskip(
    "dectris.compression",
    reason="dectris.compression (Eiger decompression) is not installed",
)


def test_csrnpz_from_hdf5(tmp_path: Path):
    temp_file = tmp_path / "test_csrnpz.h5"
    # Create a random array
    original_array = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    compressed_dict = _compress_array_csrnpz(original_array)

    with h5py.File(temp_file, 'w') as f:
        data = f.create_group('data')
        img_group = data.create_group('0')
        img_group.create_dataset(
            'compression_type', data=compressed_dict['compression_type']
        )
        img_group.create_dataset('dtype', data=compressed_dict['dtype'])
        img_group.create_dataset('shape', data=compressed_dict['shape'])
        img_group.create_dataset('elem_size', data=compressed_dict['elem_size'])
        img_group.create_dataset('data', data=np.void(compressed_dict['data']))

        metadata = f.create_group('metadata')

    adapter = EigerStreamV1ImageSeriesAdapter(temp_file)
    decompressed_array = adapter[0]
    assert np.array_equal(original_array, decompressed_array)


def test_lz4_from_hdf5(tmp_path: Path):
    temp_file = tmp_path / "test_lz4.h5"
    # Create a random array
    original_array = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    compressed_dict = _compress_array_lz4(original_array)

    with h5py.File(temp_file, 'w') as f:
        data = f.create_group('data')
        img_group = data.create_group('0')
        img_group.create_dataset(
            'compression_type', data=compressed_dict['compression_type']
        )
        img_group.create_dataset('dtype', data=compressed_dict['dtype'])
        img_group.create_dataset('shape', data=compressed_dict['shape'])
        img_group.create_dataset('elem_size', data=compressed_dict['elem_size'])
        img_group.create_dataset('data', data=np.void(compressed_dict['data']))

        metadata = f.create_group('metadata')

    adapter = EigerStreamV1ImageSeriesAdapter(temp_file)
    decompressed_array = adapter[0]
    assert np.array_equal(original_array, decompressed_array)
