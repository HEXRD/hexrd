from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.sparse import csr_matrix, save_npz

from hexrd.core.imageseries.load import eiger
from hexrd.core.imageseries.load.eiger_stream_v1 import EigerStreamV1ImageSeriesAdapter


def _compress_array_lz4(array: np.ndarray):
    # Convert array to bytes
    arr_bytes = array.tobytes()
    elem_size = array.dtype.itemsize
    orig_size = len(arr_bytes)
    orig_size_bytes = orig_size.to_bytes(8, "big")
    block_size = orig_size
    block_size_bytes = block_size.to_bytes(4, "big")
    # Compress the data
    compressed = lz4.block.compress(arr_bytes, store_size=False)
    compressed_size = len(compressed)
    compressed_size_bytes = compressed_size.to_bytes(4, "big")
    # Prepend the original size as 8 bytes big-endian
    payload = orig_size_bytes + block_size_bytes + compressed_size_bytes + compressed

    return {
        'compression_type': 'lz4',
        'elem_size': elem_size,
        'data': np.void(payload),
        'shape': array.shape,
        'dtype': array.dtype.str,
    }


def _compress_array_csrnpz(array: np.ndarray):
    # Convert the array to a sparse format (CSR)
    sparse_matrix = csr_matrix(array)
    # Save the sparse matrix to a bytes buffer
    buffer = BytesIO()

    # The compressed arrays from the eiger store the uncompressed size as 8 bytes big-endian before the actual compressed data
    buffer.write(array.nbytes.to_bytes(8, "big"))
    save_npz(buffer, sparse_matrix)
    compressed_data = buffer.getvalue()

    return {
        'compression_type': 'csrnpz',
        'data': np.void(compressed_data),
        'shape': array.shape,
        'dtype': array.dtype.str,
        'elem_size': array.dtype.itemsize,
    }


def test_lz4_decompression():
    pytest.importorskip("lz4", reason="lz4 is not available")
    # Create a random array
    original_array = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    compressed_dict = _compress_array_lz4(original_array)
    decompressed_array = eiger.decompress_frame(compressed_dict)
    assert np.array_equal(original_array, decompressed_array)


def test_csrnpz_decompression():
    # Create a random array
    original_array = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    compressed_dict = _compress_array_csrnpz(original_array)
    decompressed_array = eiger.decompress_frame(compressed_dict)
    assert np.array_equal(original_array, decompressed_array)


def test_uncompressed_decompression():
    # Create a random array
    original_array = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    compressed_dict = {
        'compression_type': None,
        'data': np.void(original_array.tobytes()),
        'shape': original_array.shape,
        'dtype': original_array.dtype.str,
        'elem_size': original_array.dtype.itemsize,
    }
    decompressed_array = eiger.decompress_frame(compressed_dict)
    assert np.array_equal(original_array, decompressed_array)


def test_unsupported_compression_type():
    compressed_dict = {
        'compression_type': 'definitely-not-a-compression-type',
        'data': np.void(b''),
        'shape': (100, 100),
        'dtype': np.uint8,
        'elem_size': 1,
    }
    with pytest.raises(
        ValueError,
        match="Unsupported compression type: definitely-not-a-compression-type",
    ):
        eiger.decompress_frame(compressed_dict)


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
    pytest.importorskip("lz4", reason="lz4 is not available")

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
