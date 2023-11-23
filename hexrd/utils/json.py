import base64
import io
import json

import numpy as np

ndarray_key = '!__hexrd_ndarray__'


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # Write it as an npy file
            with io.BytesIO() as bytes_io:
                np.save(bytes_io, obj, allow_pickle=False)
                data = bytes_io.getvalue()

            return {
                # Need to base64 encode it so it is json-valid
                ndarray_key: base64.b64encode(data).decode('ascii')
            }

        # Let the base class default method raise the TypeError
        return super().default(obj)


class NumpyDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        kwargs = {
            'object_hook': self.object_hook,
            **kwargs,
        }
        super().__init__(*args, **kwargs)

    def object_hook(self, obj):
        if ndarray_key in obj:
            data = base64.b64decode(obj[ndarray_key])
            with io.BytesIO(data) as bytes_io:
                return np.load(bytes_io)

        return obj
