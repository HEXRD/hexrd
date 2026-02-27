from io import BytesIO

import numpy as np
from dectris.compression import decompress
from scipy.sparse import csr_array, load_npz


def _decompress_csrnpz(d: dict) -> np.ndarray:
    data = d['data']
    # These are there, but they are included in the NPZ as well.
    # So lets just take them from the NPZ for simplicity.
    # dtype = d['dtype']
    # shape = d['shape']
    # elem_size = d['elem_size']

    # First 8 bytes are the length of the compressed data, which we can ignore for loading
    npz_data = data.tobytes()[8:]
    array: csr_array = load_npz(BytesIO(npz_data))
    return array.toarray()


def decompress_frame(d: dict) -> np.ndarray:
    compression_type = d['compression_type']
    dtype = d['dtype']
    shape = d['shape']
    data = d['data']
    elem_size = d['elem_size']

    if compression_type is None:
        return np.frombuffer(data, dtype=dtype).reshape(shape)

    if compression_type == "csrnpz":
        return _decompress_csrnpz(d)

    if compression_type in ("lz4", "bslz4"):
        decompressed_bytes = decompress(data, compression_type, elem_size=elem_size)
        return np.frombuffer(decompressed_bytes, dtype=dtype).reshape(shape)

    raise ValueError(f"Unsupported compression type: {compression_type}")


__all__ = [
    'decompress_frame',
]
