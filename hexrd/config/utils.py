from collections import namedtuple
import copy


ExclusionParameters = namedtuple(
    'ExclusionParameters', ["dmin", "dmax", "tthmin", "tthmax",
                            "sfacmin", "sfacmax", "pintmin", "pintmax"]
)


class Null():
    pass


null = Null()


def merge_dicts(a, b):
    """Return a merged dict, updating values from `a` with values from `b`."""
    # need to pass a deep copy of a at the top level only:
    return _merge_dicts(copy.deepcopy(a), b)


def _merge_dicts(a, b):
    for k, v in b.items():
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


def get_exclusion_parameters(cfg, prefix):
    """Return flag use saved parameters and exclusion parameters"""
    #
    # Check for value from old spec for "sfacmin"; use that if it is given,
    # but if the new spec is also there, it will override. Likewise for
    # "tth_max", as used in fit_grains.
    #
    sfmin_dflt = cfg.get(":".join(prefix, "min_sfac_ratio"), None)
    # If this is not None, raise Warning
    reset_exclusions= cfg.get(":".join(prefix, "reset_exclusions"), False)

    return(
        reset_exclusions,
        ExclusionParameters(
            dmin = cfg.get(":".join(prefix, "dmin"), None),
            dmax = cfg.get(":".join(prefix, "dmax"), None),
            tthmin = cfg.get(":".join(prefix, "tthmin"), None),
            tthmax = cfg.get(":".join(prefix, "tthmax"), None),
            sfacmin = cfg.get(":".join(prefix, "sfacmin"), sfmin_dflt),
            sfacmax = cfg.get(":".join(prefix, "sfacmax"), None),
            pintmin = cfg.get(":".join(prefix, "pintmin"), None),
            pintmax = cfg.get(":".join(prefix, "pintmax"), None),
        )
    )
