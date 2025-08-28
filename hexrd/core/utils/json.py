import io
import json

import numpy as np

ndarray_key = '!__hexrd_ndarray__'


class NumpyToNativeEncoder(json.JSONEncoder):
    # Change all Numpy arrays to native types during JSON encoding
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.generic, np.number)):
            return obj.item()

        return super().default(obj)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            # Write it as an npy file
            with io.BytesIO() as bytes_io:
                np.save(bytes_io, obj, allow_pickle=False)
                data = bytes_io.getvalue()

            return {ndarray_key: data.decode('raw_unicode_escape')}

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
            data = obj[ndarray_key].encode('raw_unicode_escape')
            with io.BytesIO(data) as bytes_io:
                return np.load(bytes_io)

        return obj
