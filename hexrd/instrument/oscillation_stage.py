"""Oscillation stage parameters"""

import numpy as np


class OscillationStage(object):

    def __init__(self, tvec, chi):
        self._tvec = np.atleast_1d(tvec).flatten()
        self._chi = chi

    @property
    def chi(self):
        return self._chi

    @chi.setter
    def chi(self, x):
        self._chi = float(x)

    @property
    def tvec(self):
        return self._tvec

    @tvec.setter
    def tvec(self, x):
        assert len(x) == 3
        self._tvec = np.atleast_1d(x).flatten()
