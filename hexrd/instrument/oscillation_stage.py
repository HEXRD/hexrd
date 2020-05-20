"""Oscillation stage parameters"""

import numpy as np

chi_DFLT = 0.
t_vec_s_DFLT = np.zeros(3)


class OscillationStage(object):

    def __init__(self,
                 tvec=t_vec_s_DFLT,
                 chi=chi_DFLT):
        self.tvec = tvec
        self.chi = chi

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
