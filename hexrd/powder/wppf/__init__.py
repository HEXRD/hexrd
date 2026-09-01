def __getattr__(name):
    if name in {'LeBail', 'Rietveld'}:
        from hexrd.powder.wppf import WPPF
        return getattr(WPPF, name)
    raise AttributeError(name)
