import numpy as np
from skimage.transform import PiecewiseAffineTransform, warp

from hexrd.core import constants
from hexrd.hedm.xrdutil.utils import zproject_sph_angles


class SphericalView:
    """
    Creates a spherical mapping of detector images.
    """
    MAPPING_TYPES = ('stereographic', 'equal-area')
    VECTOR_TYPES = ('d', 'q')
    PROJ_IMG_DIM = 3.  # 2*np.sqrt(2) rounded up

    def __init__(self, mapping='stereographic', vector_type='d',
                 output_dim=512, rmat=constants.identity_3x3):
        self._mapping = mapping
        self._vector_type = vector_type

        # ??? maybe promote invert_z to a prop for protection?
        if self._vector_type == 'd':
            self.invert_z = False
        elif self._vector_type == 'q':
            self.invert_z = True

        self._output_dim = output_dim
        self._rmat = rmat

    @property
    def mapping(self):
        return self._mapping

    @mapping.setter
    def mapping(self, s):
        if s not in self.MAPPING_TYPES:
            raise RuntimeError("mapping specification '%s' is invalid" % s)

    @property
    def vector_type(self):
        return self._vector_type

    @vector_type.setter
    def vector_type(self, s):
        if s not in self.VECTOR_TYPES:
            raise RuntimeError("vector type specification '%s' is invalid" % s)

    @property
    def output_dim(self):
        return self._output_dim

    @output_dim.setter
    def output_dim(self, x):
        self._output_dim = int(x)

    @property
    def rmat(self):
        return self._rmat

    @rmat.setter
    def rmat(self, x):
        x = np.atleast_2d(x)
        assert x.shape == (3, 3), "rmat must be (3, 3)"
        assert np.linalg.norm(np.dot(x.T, x) - constants.identity_3x3) \
            < constants.ten_epsf, "input matrix is not orthogonal"
        self._rmat = x

    def warp_eta_ome_map(self, eta_ome, map_ids=None, skip=10):
        paxf = PiecewiseAffineTransform()

        nrows_in = len(eta_ome.etas)
        ncols_in = len(eta_ome.omegas)

        # grab tth values
        tths = eta_ome.planeData.getTTh()

        # grab (undersampled) points
        omes = eta_ome.omegas[::skip]
        etas = eta_ome.etas[::skip]

        # make grid of angular values
        op, ep = np.meshgrid(omes,
                             etas,
                             indexing='ij')

        # make grid of output pixel values
        oc, ec = np.meshgrid(np.arange(nrows_in)[::skip],
                             np.arange(ncols_in)[::skip],
                             indexing='ij')

        ps = self.PROJ_IMG_DIM / self.output_dim  # output pixel size

        if map_ids is None:
            map_ids = list(range(len(eta_ome.dataStore)))

        wimgs = []
        for map_id in map_ids:
            img = eta_ome.dataStore[map_id]

            # ??? do we need to use iHKLlist?
            angs = np.vstack([
                tths[map_id]*np.ones_like(ep.flatten()),
                ep.flatten(),
                op.flatten()
            ]).T

            ppts, nmask = zproject_sph_angles(
                angs, method=self.mapping, source=self.vector_type,
                invert_z=self.invert_z, use_mask=True
            )

            # pixel coords in output image
            rp = 0.5*self.output_dim - ppts[:, 1]/ps
            cp = ppts[:, 0]/ps + 0.5*self.output_dim

            # compute piecewise affine transform
            src = np.vstack([ec.flatten(), oc.flatten(), ]).T
            dst = np.vstack([cp.flatten(), rp.flatten(), ]).T
            paxf.estimate(src, dst)

            wimg = warp(
                img,
                inverse_map=paxf.inverse,
                output_shape=(self.output_dim, self.output_dim)
            )
            if len(map_ids) == 1:
                return wimg
            else:
                wimgs.append[wimg]
        return wimgs

    def warp_polar_image(self, pimg, skip=10):
        paxf = PiecewiseAffineTransform()

        img = np.array(pimg['intensities'])

        # remove SNIP bg if there
        if 'snip_background' in pimg:
            # !!! these are float64 so we should be good
            img -= np.array(pimg['snip_background'])

        nrows_in, ncols_in = img.shape

        tth_cen = np.array(pimg['tth_coordinates'])[0, :]
        eta_cen = np.array(pimg['eta_coordinates'])[:, 0]

        tp, ep = np.meshgrid(tth_cen[::skip],
                             eta_cen[::skip])
        tc, ec = np.meshgrid(np.arange(ncols_in)[::skip],
                             np.arange(nrows_in)[::skip])
        op = np.zeros_like(tp.flatten())

        angs = np.radians(
            np.vstack([tp.flatten(),
                       ep.flatten(),
                       op.flatten()]).T
        )

        ppts = zproject_sph_angles(
            angs, method='stereographic', source='d', invert_z=self.invert_z,
            rmat=self.rmat
        )

        # output pixel size
        ps = self.PROJ_IMG_DIM / self.output_dim

        # pixel coords in output image
        rp = 0.5*self.output_dim - ppts[:, 1]/ps
        cp = ppts[:, 0]/ps + 0.5*self.output_dim

        src = np.vstack([tc.flatten(), ec.flatten(), ]).T
        dst = np.vstack([cp.flatten(), rp.flatten(), ]).T
        paxf.estimate(src, dst)

        wimg = warp(
            img,
            inverse_map=paxf.inverse,
            output_shape=(self.output_dim, self.output_dim)
        )

        return wimg
