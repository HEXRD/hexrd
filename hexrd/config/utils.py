import copy


def merge_dicts(a, b):
    "Returns a merged dict, updating values from `a` with values from `b`"
    # need to pass a deep copy of a at the top level only:
    return _merge_dicts(copy.deepcopy(a), b)


def _merge_dicts(a, b):
    for k,v in list(b.items()):
        if isinstance(v, dict):
            if a.get(k) is None:
                # happens in cases where all but section head is commented
                a[k] = {}
            _merge_dicts(a[k], v)
        else:
            if v is None and a.get(k) is not None:
                # entire section commented out. Inherit, don't overwrite
                pass
            else:
                a[k] = v
    return a


class Null():
    pass
null = Null()
