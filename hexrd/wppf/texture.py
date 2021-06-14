import numpy as np

"""
========================================================================================================
========================================================================================================

    >> @AUTHOR:     Saransh Singh, Lawrence Livermore National Lab, saransh1@llnl.gov
    >> @DATE:       06/14/2021 SS 1.0 original

    >> @DETAILS:    this module deals with the texture computations for the wppf package.
    two texture models employed are the March-Dollase model and the spherical harmonics
    model. the spherical harmonics model uses the axis distribution function for computing
    the scale factors for each reflection. maybe it makes sense to use symmetrized FEM mesh
    functions for 2-sphere for this. probably use the two theta-eta array from the instrument
    class to precompute the values of k_l^m(y) and we already know what the reflections are
    so k_l^m(h) can also be pre-computed.

    >> @PARAMETERS:     
"""
