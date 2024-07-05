import numpy as np
import numba
from numba import prange

from hexrd.sampleOrientations.conversions import cu2ro, ro2qu
from hexrd.sampleOrientations.rfz import insideFZ
from hexrd import constants


@numba.njit(cache=True, nogil=True, parallel=True)
def _sample(pgnum,
            N,
            delta,
            shift,
            ap_2):

    N3 = (2*N+1)**3
    res = np.full((N3, 4), np.nan, dtype=np.float64)

    for ii in prange(-N, N+1):
        xx = (ii + shift) * delta
        for jj in prange(-N, N+1):
            yy = (jj + shift) * delta
            for kk in prange(-N, N+1):
                zz = (kk + shift) * delta
                cu = np.array([xx, yy, zz])
                ma = np.max(np.abs(cu))

                if ma <= ap_2:
                    ro = cu2ro(cu)
                    if insideFZ(ro, pgnum):
                        idx = (ii+N)*(2*N+1)**2 + (jj+N)*(2*N+1) + (kk+N)
                        res[idx,:] = ro2qu(ro)

    return res

class sampleRFZ:

    """This class samples the rodrigues fundamental zone
    of a point group uniformly in the density sense and
    returns a list of orientations which are spaced,
    on an average, to the user specified angular spacing
    @author Saransh Singh, LLNL
    @date   11/28/2022 1.0 original

     Note
    ----
    Details can be found in:
    S. Singh and M. De Graef, "Orientation sampling for 
    dictionary-based diffraction pattern indexing methods". 
    MSMSE 24, 085013 (2016)

    Attributes
    ----------
    pgnum: int
        point group number of crystal
    average_angular_spacing" float
        average angular spacing of sampling (in degrees)

    """

    def __init__(self,
                 pgnum,
                 sampling_type='default',
                 average_angular_spacing=3.0):
        """__init__ method of the sampleRFZ class.


        Parameters
        ----------
        pgnum : int
            point group number
        sampling_type : str
            default sampling with origin
            special which is shifted to mid-points
            of each cubochoric cell
        average_angular_spacing : float
            average angular spacing in degrees

        """

        self.ap_2 = constants.cuA_2
        self.pgnum = pgnum
        self.sampling_type = sampling_type
        self.avg_ang_spacing = average_angular_spacing

    def sampling_N(self):
        """Get the number of sampling steps in the cubochoric
        cube based on the average angular spacing requested.
        Uses eqns. 9 and 10 of S. Singh and M. De Graef MSMSE 24,
        085013 (2016)

        """
        if self.sampling_type.lower() == 'default':
            return np.rint(131.97049 / (self.avg_ang_spacing - 0.03732)).astype(np.int32)
        elif self.sampling_type.lower() == 'special':
            return np.rint(125.70471 / (self.avg_ang_spacing - 0.07127)).astype(np.int32)

    def sample(self):
        res = _sample(self.pgnum,
                                    self.cubN,
                                    self.delta,
                                    self.shift,
                                    self.ap_2)
        mask = ~np.isnan(res[:,0])
        res = res[mask,:]
        self.orientations = res
    def sample_if_possible(self):
        required_attributes = ('pgnum', 'avg_ang_spacing', 'sampling_type')
        if not all(hasattr(self, x) for x in required_attributes):
            return

        self.sample()

    @property
    def pgnum(self):
        return self._pgnum

    @pgnum.setter
    def pgnum(self, pgn):
        self._pgnum = pgn
        self.sample_if_possible()

    @property
    def sampling_type(self):
        return self._sampling_type

    @sampling_type.setter
    def sampling_type(self, stype):
        self._sampling_type = stype
        self.sample_if_possible()

    @property
    def avg_ang_spacing(self):
        return self._avg_ang_spacing


    @avg_ang_spacing.setter
    def avg_ang_spacing(self, ang):
        self._avg_ang_spacing = ang
        self.sample_if_possible()

    @property
    def cubN(self):
        return self.sampling_N()

    @property
    def shift(self):
        if self.sampling_type == 'default':
            return 0.0
        elif self.sampling_type == 'special':
            return 0.5

    @property
    def delta(self):
        return self.ap_2 / self.cubN
