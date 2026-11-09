from collections import namedtuple
import copy
import warnings


ExclusionParameters = namedtuple(
    'ExclusionParameters',
    [
        "dmin",
        "dmax",
        "tthmin",
        "tthmax",
        "sfacmin",
        "sfacmax",
        "pintmin",
        "pintmax",
    ],
)


class Null:
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
    yaml_key = lambda s: ":".join((prefix, s))
    #
    # Check for value from old spec for "sfacmin"; use that if it is given,
    # but if the new spec is also there, it will override. Likewise for
    # "tth_max", as used in fit_grains.
    # -- Should add a deprecated warning if min_sfac_ratio is used
    #
    sfmin_dflt = cfg.get(yaml_key("min_sfac_ratio"), None)
    if sfmin_dflt is not None:
        warnings.warn(
            '"min_sfac_ratio" is deprecated, use "sfacmin" instead',
            DeprecationWarning,
        )
    # Default for reset_exclusions is True so that old config files will
    # produce the same behavior.
    reset_exclusions = cfg.get(yaml_key("reset_exclusions"), True)

    return (
        reset_exclusions,
        ExclusionParameters(
            dmin=cfg.get(yaml_key("dmin"), None),
            dmax=cfg.get(yaml_key("dmax"), None),
            tthmin=cfg.get(yaml_key("tthmin"), None),
            tthmax=cfg.get(yaml_key("tthmax"), None),
            sfacmin=cfg.get(yaml_key("sfacmin"), sfmin_dflt),
            sfacmax=cfg.get(yaml_key("sfacmax"), None),
            pintmin=cfg.get(yaml_key("pintmin"), None),
            pintmax=cfg.get(yaml_key("pintmax"), None),
        ),
    )
