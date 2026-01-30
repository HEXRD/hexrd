"""Handle omega (specimen rotation) metadata

* OmegaWedges class specifies omega metadata in wedges
"""

import numpy as np
from numpy.typing import NDArray

from .baseclass import ImageSeries


class OmegaImageSeries(ImageSeries):
    """ImageSeries with omega metadata"""

    DFLT_TOL = 1.0e-6
    TAU = 360

    def __init__(self, ims: ImageSeries):
        """This class is initialized with an existing imageseries"""
        # check for omega metadata
        if 'omega' in ims.metadata:
            self._omega: NDArray[np.float64] = ims.metadata['omega']
            if len(ims) != self._omega.shape[0]:
                msg = 'omega array mismatch: array has %s frames, expecting %s'
                msg = msg % (self._omega.shape[0], len(ims))
                raise OmegaSeriesError(msg)
        else:
            raise OmegaSeriesError('Imageseries has no omega metadata')

        super(OmegaImageSeries, self).__init__(ims)
        self._make_wedges()

    def _make_wedges(self, tol=DFLT_TOL):
        nf = len(self)
        om = self.omega

        # find the frames where the wedges break
        starts = [0]
        delta = om[0, 1] - om[0, 0]
        omlast = om[0, 1]
        for f in range(1, nf):
            if delta <= 0:
                raise OmegaSeriesError('omega array must be increasing')
            # check whether delta changes or ranges not contiguous
            d = om[f, 1] - om[f, 0]
            if (np.abs(d - delta) > tol) or (np.abs(om[f, 0] - omlast) > tol):
                starts.append(f)
                delta = d
            omlast = om[f, 1]
        starts.append(nf)

        nw = len(starts) - 1
        nf0 = 0
        self._wedge_om = np.zeros((nw, 3))
        self._wedge_f = np.zeros((nw, 2), dtype=int)
        self._omegawedges = OmegaWedges(nf)
        for s in range(nw):
            ostart = om[starts[s], 0]
            ostop = om[starts[s + 1] - 1, 1]
            steps = starts[s + 1] - starts[s]
            self._omegawedges.addwedge(ostart, ostop, steps)
            #
            delta = (ostop - ostart) / steps
            self._wedge_om[s, :] = (ostart, ostop, delta)
            self._wedge_f[s, 0] = nf0
            self._wedge_f[s, 1] = steps
            nf0 += steps
        assert nf0 == nf

    @property
    def omega(self) -> NDArray[np.float64]:
        """return omega range array (nframes, 2)"""
        return self._omega

    @property
    def omegawedges(self):
        """OmegaWedges instance"""
        return self._omegawedges

    @property
    def nwedges(self):
        """number of omega wedges (angular sections)"""
        return self.omegawedges.nwedges

    def wedge(self, i):
        """return i'th wedge as a dictionary"""
        d = self.omegawedges.wedges[i]
        delta = (d['ostop'] - d['ostart']) / d['nsteps']
        d.update(delta=delta)
        return d

    def omega_to_frame(self, om):
        """Return frame and wedge which includes given omega, -1 if not found"""
        f = -1
        w = -1
        for i in range(len(self._wedge_om)):
            omin = self._wedge_om[i, 0]
            omax = self._wedge_om[i, 1]
            omcheck = omin + np.mod(om - omin, self.TAU)
            if omcheck < omax:
                odel = self._wedge_om[i, 2]
                f = self._wedge_f[i, 0] + int(np.floor((omcheck - omin) / odel))
                w = i
                break

        return f, w

    def omegarange_to_frames(self, omin, omax):
        """Return list of frames for range of omegas"""
        noframes = ()
        f0, w0 = self.omega_to_frame(omin)
        if w0 < 0:
            return noframes
        f1, w1 = self.omega_to_frame(omax)
        if w1 < 0:
            return noframes

        # if same wedge, require frames be increasing
        if (w0 == w1) and (f1 > f0):
            return list(range(f0, f1 + 1))

        # case: adjacent wedges with 2pi jump in omega
        w0max = self._wedge_om[w0, 1]
        w1min = self._wedge_om[w1, 0]

        if np.mod(np.abs(w1min - w0max), self.TAU) < self.DFLT_TOL:
            r0 = list(range(f0, self._wedge_f[w0, 0] + self._wedge_f[w0, 1]))
            r1 = list(range(self._wedge_f[w1, 0], f1 + 1))
            return r0 + r1

        return noframes


class OmegaWedges(object):
    """Piecewise Linear Omega Ranges

    PARAMETERS
    ----------
    nframes: int
        number of frames in imageseries
    """

    def __init__(self, nframes):
        self.nframes = nframes
        self._wedges: list[dict[str, int]] = []

    #
    # ============================== API
    #
    @property
    def omegas(self):
        """n x 2 array of omega values, one per frame"""
        if self.nframes != self.wframes:
            msg = (
                "number of frames (%s) does not match "
                "number of wedge frames (%s)" % (self.nframes, self.wframes)
            )
            raise OmegaSeriesError(msg)

        oa = np.zeros((self.nframes, 2))
        wstart = 0
        for w in self.wedges:
            ns = w['nsteps']
            wr = list(range(wstart, wstart + ns))
            wa0 = np.linspace(w['ostart'], w['ostop'], ns + 1)
            oa[wr, 0] = wa0[:-1]
            oa[wr, 1] = wa0[1:]
            wstart += ns

        return oa

    @property
    def nwedges(self):
        """number of wedges"""
        return len(self._wedges)

    @property
    def wedges(self):
        """list of wedges (dictionaries)"""
        return self._wedges

    def addwedge(self, ostart, ostop, nsteps, loc=None):
        """add wedge to list

        PARAMETERS
        ----------
        ostart: float
            starting value of omega for this wedge
        ostop: float
            final value of omega for this wedge
        nsteps: int
            number of steps
        loc: int, optional
            where to insert wedge in the list of wedges; defaults to end
        """
        d = dict(ostart=ostart, ostop=ostop, nsteps=nsteps)
        if loc is None:
            loc = self.nwedges

        self.wedges.insert(loc, d)

    def delwedge(self, i):
        """delete wedge number `i`"""
        self.wedges.pop(i)

    @property
    def wframes(self):
        """number of frames in wedges"""
        wf = [w['nsteps'] for w in self.wedges]
        return int(np.sum(wf))

    def save_omegas(self, fname):
        """save omegas to text file

        PARAMETERS
        ----------
        fname: str or Path
            name of file to save omegas to
        """
        np.save(fname, self.omegas)


class OmegaSeriesError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
