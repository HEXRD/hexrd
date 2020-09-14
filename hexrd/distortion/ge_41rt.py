"""GE4RT Detector Distortion"""
import numpy as np

from hexrd import constants as cnst
from hexrd.constants import USE_NUMBA
if USE_NUMBA:
    import numba

from .distortionabc import DistortionABC
from .registry import _RegisterDistortionClass
from .utils import newton

RHO_MAX = 204.8  # max radius in mm for ge detector

if USE_NUMBA:
    @numba.njit
    def _ge_41rt_inverse_distortion(out, in_, rhoMax, params):
        maxiter = 100
        prec = cnst.epsf

        p0, p1, p2, p3, p4, p5 = params[0:6]
        rxi = 1.0/rhoMax
        for el in range(len(in_)):
            xi, yi = in_[el, 0:2]
            ri = np.sqrt(xi*xi + yi*yi)
            if ri < cnst.sqrt_epsf:
                ri_inv = 0.0
            else:
                ri_inv = 1.0/ri
            sinni = yi*ri_inv
            cosni = xi*ri_inv
            ro = ri
            cos2ni = cosni*cosni - sinni*sinni
            sin2ni = 2*sinni*cosni
            cos4ni = cos2ni*cos2ni - sin2ni*sin2ni
            # newton solver iteration
            for i in range(maxiter):
                ratio = ri*rxi
                fx = (p0*ratio**p3*cos2ni +
                      p1*ratio**p4*cos4ni +
                      p2*ratio**p5 + 1)*ri - ro  # f(x)
                fxp = (p0*ratio**p3*cos2ni*(p3+1) +
                       p1*ratio**p4*cos4ni*(p4+1) +
                       p2*ratio**p5*(p5+1) + 1)  # f'(x)

                delta = fx/fxp
                ri = ri - delta
                # convergence check for newton
                if np.abs(delta) <= prec*np.abs(ri):
                    break

            xi = ri*cosni
            yi = ri*sinni
            out[el, 0] = xi
            out[el, 1] = yi

        return out

    @numba.njit
    def _ge_41rt_distortion(out, in_, rhoMax, params):
        p0, p1, p2, p3, p4, p5 = params[0:6]
        rxi = 1.0/rhoMax

        for el in range(len(in_)):
            xi, yi = in_[el, 0:2]
            ri = np.sqrt(xi*xi + yi*yi)
            if ri < cnst.sqrt_epsf:
                ri_inv = 0.0
            else:
                ri_inv = 1.0/ri
            sinni = yi*ri_inv
            cosni = xi*ri_inv
            cos2ni = cosni*cosni - sinni*sinni
            sin2ni = 2*sinni*cosni
            cos4ni = cos2ni*cos2ni - sin2ni*sin2ni
            ratio = ri*rxi

            ri = (p0*ratio**p3*cos2ni
                  + p1*ratio**p4*cos4ni
                  + p2*ratio**p5
                  + 1)*ri
            xi = ri*cosni
            yi = ri*sinni
            out[el, 0] = xi
            out[el, 1] = yi

        return out
else:
    # non-numba versions for the direct and inverse distortion
    def _ge_41rt_inverse_distortion(out, in_, rhoMax, params):
        maxiter = 100
        prec = cnst.epsf

        p0, p1, p2, p3, p4, p5 = params[0:6]
        rxi = 1.0/rhoMax

        xi, yi = in_[:, 0], in_[:, 1]
        ri = np.sqrt(xi*xi + yi*yi)
        # !!! adding fix TypeError when processings list of coords
        zfix = []
        if np.any(ri) < cnst.sqrt_epsf:
            zfix = ri < cnst.sqrt_epsf
            ri[zfix] = 1.0
        ri_inv = 1.0/ri
        ri_inv[zfix] = 0.

        sinni = yi*ri_inv
        cosni = xi*ri_inv
        ro = ri
        cos2ni = cosni*cosni - sinni*sinni
        sin2ni = 2*sinni*cosni
        cos4ni = cos2ni*cos2ni - sin2ni*sin2ni

        # newton solver iteration
        #
        # FIXME: looks like we hae a problem here,
        #        should iterate over single coord pairs?
        for i in range(maxiter):
            ratio = ri*rxi
            fx = (p0*ratio**p3*cos2ni +
                  p1*ratio**p4*cos4ni +
                  p2*ratio**p5 + 1)*ri - ro  # f(x)
            fxp = (p0*ratio**p3*cos2ni*(p3+1) +
                   p1*ratio**p4*cos4ni*(p4+1) +
                   p2*ratio**p5*(p5+1) + 1)  # f'(x)

            delta = fx/fxp
            ri = ri - delta

            # convergence check for newton
            if np.max(np.abs(delta/ri)) <= prec:
                break

        out[:, 0] = ri*cosni
        out[:, 1] = ri*sinni

        return out

    def _ge_41rt_distortion(out, in_, rhoMax, params):
        p0, p1, p2, p3, p4, p5 = params[0:6]
        rxi = 1.0/rhoMax

        xi, yi = in_[:, 0], in_[:, 1]

        # !!! included fix on ValueError for array--like in_
        ri = np.sqrt(xi*xi + yi*yi)
        ri[ri < cnst.sqrt_epsf] = np.inf
        ri_inv = 1.0/ri

        sinni = yi*ri_inv
        cosni = xi*ri_inv
        cos2ni = cosni*cosni - sinni*sinni
        sin2ni = 2*sinni*cosni
        cos4ni = cos2ni*cos2ni - sin2ni*sin2ni
        ratio = ri*rxi

        ri = (p0*ratio**p3*cos2ni
              + p1*ratio**p4*cos4ni
              + p2*ratio**p5
              + 1)*ri
        out[:, 0] = ri*cosni
        out[:, 1] = ri*sinni

        return out


def _rho_scl_func_inv(ri, ni, ro, rx, p):
    retval = (p[0]*(ri/rx)**p[3] * np.cos(2.0 * ni) +
              p[1]*(ri/rx)**p[4] * np.cos(4.0 * ni) +
              p[2]*(ri/rx)**p[5] + 1)*ri - ro
    return retval


def _rho_scl_dfunc_inv(ri, ni, ro, rx, p):
    retval = p[0]*(ri/rx)**p[3] * np.cos(2.0 * ni) * (p[3] + 1) + \
        p[1]*(ri/rx)**p[4] * np.cos(4.0 * ni) * (p[4] + 1) + \
        p[2]*(ri/rx)**p[5] * (p[5] + 1) + 1
    return retval


def inverse_distortion_numpy(rho0, eta0, rhoMax, params):
    return newton(rho0, _rho_scl_func_inv, _rho_scl_dfunc_inv,
                  (eta0, rho0, rhoMax, params))


class GE_41RT(DistortionABC, metaclass=_RegisterDistortionClass):

    maptype = "GE_41RT"

    def __init__(self, params, **kwargs):
        self._params = np.asarray(params, dtype=float).flatten()

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, x):
        assert len(x) == 6, "parameter list must have len of 6"
        self._params = np.asarray(x, dtype=float).flatten()

    @property
    def is_trivial(self):
        return \
            self.params[0] == 0 and \
            self.params[1] == 0 and \
            self.params[2] == 0

    def apply(self, xy_in):
        if self.is_trivial:
            return xy_in
        else:
            xy_out = np.empty_like(xy_in)
            _ge_41rt_distortion(
                xy_out, xy_in, float(RHO_MAX), np.asarray(self.params)
            )
            return xy_out

    def apply_inverse(self, xy_in):
        if self.is_trivial:
            return xy_in
        else:
            xy_out = np.empty_like(xy_in)
            _ge_41rt_inverse_distortion(
                xy_out, xy_in, float(RHO_MAX), np.asarray(self.params)
            )
            return xy_out
