import numpy as np

from hexrd import constants as ct
from hexrd.gridutil import cellIndices
from hexrd import matrixutil as mutil
from hexrd import xrdutil
from hexrd.crystallography import PlaneData
from hexrd.extensions._transforms_CAPI import\
    anglesToGVec, \
    detectorXYToGvec, \
    gvecToDetectorXY, \
    makeRotMatOfExpMap, \
    mapAngle, \
    oscillAnglesOfHKLs,\
    rowNorm
from skimage.draw import polygon

panel_calibration_flags_DFLT = np.array(
    [1, 1, 1, 1, 1, 1],
    dtype=bool
)
pixel_size_DFLT = [0.2, 0.2]

class PlanarDetector(object):
    """
    base class for 2D planar, rectangular row-column detector
    """

    __pixelPitchUnit = 'mm'

    def __init__(self,
                 rows=2048, cols=2048,
                 pixel_size=pixel_size_DFLT,
                 tvec=np.r_[0., 0., -1000.],
                 tilt=ct.zeros_3,
                 name='default',
                 bvec=ct.beam_vec,
                 evec=ct.eta_vec,
                 saturation_level=None,
                 panel_buffer=None,
                 roi=None,
                 distortion=None):
        """
        panel buffer is in pixels...

        """
        self._name = name

        self._rows = rows
        self._cols = cols

        self._pixel_size_row = pixel_size[0]
        self._pixel_size_col = pixel_size[1]

        self._saturation_level = saturation_level

        if panel_buffer is None:
            self._panel_buffer = 20*np.r_[self._pixel_size_col,
                                          self._pixel_size_row]

        self._roi = roi

        self._tvec = np.array(tvec).flatten()
        self._tilt = np.array(tilt).flatten()

        self._bvec = np.array(bvec).flatten()
        self._evec = np.array(evec).flatten()

        self._distortion = distortion

        #
        # set up calibration parameter list and refinement flags
        #
        # order for a single detector will be
        #
        #     [tilt, translation, <distortion>]
        dparams = []
        if self._distortion is not None:
            # need dparams
            # FIXME: must update when we fix distortion
            dparams.append(np.atleast_1d(self._distortion[1]).flatten())
        dparams = np.array(dparams).flatten()
        self._calibration_parameters = np.hstack(
                [self._tilt, self._tvec, dparams]
            )
        self._calibration_flags = np.hstack(
                [panel_calibration_flags_DFLT,
                 np.zeros(len(dparams), dtype=bool)]
            )
        return

    # detector ID
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, s):
        assert isinstance(s, (str, unicode)), "requires string input"
        self._name = s

    # properties for physical size of rectangular detector
    @property
    def rows(self):
        return self._rows

    @rows.setter
    def rows(self, x):
        assert isinstance(x, int)
        self._rows = x

    @property
    def cols(self):
        return self._cols

    @cols.setter
    def cols(self, x):
        assert isinstance(x, int)
        self._cols = x

    @property
    def pixel_size_row(self):
        return self._pixel_size_row

    @pixel_size_row.setter
    def pixel_size_row(self, x):
        self._pixel_size_row = float(x)

    @property
    def pixel_size_col(self):
        return self._pixel_size_col

    @pixel_size_col.setter
    def pixel_size_col(self, x):
        self._pixel_size_col = float(x)

    @property
    def pixel_area(self):
        return self.pixel_size_row * self.pixel_size_col

    @property
    def saturation_level(self):
        return self._saturation_level

    @saturation_level.setter
    def saturation_level(self, x):
        if x is not None:
            assert np.isreal(x)
        self._saturation_level = x

    @property
    def panel_buffer(self):
        return self._panel_buffer

    @panel_buffer.setter
    def panel_buffer(self, x):
        """if not None, a buffer in mm (x, y)"""
        if x is not None:
            assert len(x) == 2 or x.ndim == 2
        self._panel_buffer = x

    @property
    def roi(self):
        return self._roi

    @roi.setter
    def roi(self, vertex_array):
        """
        vertex array must be

        [[r0, c0], [r1, c1], ..., [rn, cn]]

        and have len >= 3

        does NOT need to repeat start vertex for closure
        """
        if vertex_array is not None:
            assert len(vertex_array) >= 3
        self._roi = vertex_array

    @property
    def row_dim(self):
        return self.rows * self.pixel_size_row

    @property
    def col_dim(self):
        return self.cols * self.pixel_size_col

    @property
    def row_pixel_vec(self):
        return self.pixel_size_row*(0.5*(self.rows-1)-np.arange(self.rows))

    @property
    def row_edge_vec(self):
        return self.pixel_size_row*(0.5*self.rows-np.arange(self.rows+1))

    @property
    def col_pixel_vec(self):
        return self.pixel_size_col*(np.arange(self.cols)-0.5*(self.cols-1))

    @property
    def col_edge_vec(self):
        return self.pixel_size_col*(np.arange(self.cols+1)-0.5*self.cols)

    @property
    def corner_ul(self):
        return np.r_[-0.5 * self.col_dim,  0.5 * self.row_dim]

    @property
    def corner_ll(self):
        return np.r_[-0.5 * self.col_dim, -0.5 * self.row_dim]

    @property
    def corner_lr(self):
        return np.r_[0.5 * self.col_dim, -0.5 * self.row_dim]

    @property
    def corner_ur(self):
        return np.r_[0.5 * self.col_dim,  0.5 * self.row_dim]

    @property
    def tvec(self):
        return self._tvec

    @tvec.setter
    def tvec(self, x):
        x = np.array(x).flatten()
        assert len(x) == 3, 'input must have length = 3'
        self._tvec = x

    @property
    def tilt(self):
        return self._tilt

    @tilt.setter
    def tilt(self, x):
        assert len(x) == 3, 'input must have length = 3'
        self._tilt = np.array(x).squeeze()

    @property
    def bvec(self):
        return self._bvec

    @bvec.setter
    def bvec(self, x):
        x = np.array(x).flatten()
        assert len(x) == 3 and sum(x*x) > 1-ct.sqrt_epsf, \
            'input must have length = 3 and have unit magnitude'
        self._bvec = x

    @property
    def evec(self):
        return self._evec

    @evec.setter
    def evec(self, x):
        x = np.array(x).flatten()
        assert len(x) == 3 and sum(x*x) > 1-ct.sqrt_epsf, \
            'input must have length = 3 and have unit magnitude'
        self._evec = x

    @property
    def distortion(self):
        return self._distortion

    @distortion.setter
    def distortion(self, x):
        """
        Probably should make distortion a class...
        ***FIX THIS***
        """
        assert len(x) == 2 and hasattr(x[0], '__call__'), \
            'distortion must be a tuple: (<func>, params)'
        self._distortion = x

    @property
    def rmat(self):
        return makeRotMatOfExpMap(self.tilt)

    @property
    def normal(self):
        return self.rmat[:, 2]

    @property
    def beam_position(self):
        """
        returns the coordinates of the beam in the cartesian detector
        frame {Xd, Yd, Zd}.  NaNs if no intersection.
        """
        output = np.nan * np.ones(2)
        b_dot_n = np.dot(self.bvec, self.normal)
        if np.logical_and(
            abs(b_dot_n) > ct.sqrt_epsf,
            np.sign(b_dot_n) == -1
        ):
            u = np.dot(self.normal, self.tvec) / b_dot_n
            p2_l = u*self.bvec
            p2_d = np.dot(self.rmat.T, p2_l - self.tvec)
            output = p2_d[:2]
        return output

    # ...memoize???
    @property
    def pixel_coords(self):
        pix_i, pix_j = np.meshgrid(
            self.row_pixel_vec, self.col_pixel_vec,
            indexing='ij')
        return pix_i, pix_j

    @property
    def calibration_parameters(self):
        #
        # set up calibration parameter list and refinement flags
        #
        # order for a single detector will be
        #
        #     [tilt, translation, <distortion>]
        dparams = []
        if self.distortion is not None:
            # need dparams
            # FIXME: must update when we fix distortion
            dparams.append(np.atleast_1d(self.distortion[1]).flatten())
        dparams = np.array(dparams).flatten()
        self._calibration_parameters = np.hstack(
                [self.tilt, self.tvec, dparams]
            )
        return self._calibration_parameters

    @property
    def calibration_flags(self):
        return self._calibration_flags

    @calibration_flags.setter
    def calibration_flags(self, x):
        x = np.array(x, dtype=bool).flatten()
        if len(x) != len(self._calibration_flags):
            raise RuntimeError(
                "length of parameter list must be %d; you gave %d"
                % (len(self._calibration_flags), len(x))
            )
        self._calibration_flags = x

    # =========================================================================
    # METHODS
    # =========================================================================

    def config_dict(self, chi, t_vec_s, sat_level=None):
        """
        """
        if sat_level is None:
            sat_level = self.saturation_level

        t_vec_s = np.atleast_1d(t_vec_s)

        d = dict(
            detector=dict(
                transform=dict(
                    tilt=self.tilt.tolist(),
                    translation=self.tvec.tolist(),
                ),
                pixels=dict(
                    rows=self.rows,
                    columns=self.cols,
                    size=[self.pixel_size_row, self.pixel_size_col],
                ),
            ),
            oscillation_stage=dict(
                chi=chi,
                translation=t_vec_s.tolist(),
            ),
        )

        if sat_level is not None:
            d['detector']['saturation_level'] = sat_level

        if self.distortion is not None:
            """...HARD CODED DISTORTION! FIX THIS!!!"""
            dist_d = dict(
                function_name='GE_41RT',
                parameters=np.r_[self.distortion[1]].tolist()
            )
            d['detector']['distortion'] = dist_d
        return d

    def pixel_angles(self, origin=ct.zeros_3):
        assert len(origin) == 3, "origin must have 3 elemnts"
        pix_i, pix_j = self.pixel_coords
        xy = np.ascontiguousarray(
            np.vstack([
                pix_j.flatten(), pix_i.flatten()
                ]).T
            )
        angs, g_vec = detectorXYToGvec(
            xy, self.rmat, ct.identity_3x3,
            self.tvec, ct.zeros_3, origin,
            beamVec=self.bvec, etaVec=self.evec)
        del(g_vec)
        tth = angs[0].reshape(self.rows, self.cols)
        eta = angs[1].reshape(self.rows, self.cols)
        return tth, eta

    def cartToPixel(self, xy_det, pixels=False):
        """
        Convert vstacked array or list of [x,y] points in the center-based
        cartesian frame {Xd, Yd, Zd} to (i, j) edge-based indices

        i is the row index, measured from the upper-left corner
        j is the col index, measured from the upper-left corner

        if pixels=True, then (i,j) are integer pixel indices.
        else (i,j) are continuous coords
        """
        xy_det = np.atleast_2d(xy_det)

        npts = len(xy_det)

        tmp_ji = xy_det - np.tile(self.corner_ul, (npts, 1))
        i_pix = -tmp_ji[:, 1] / self.pixel_size_row - 0.5
        j_pix = tmp_ji[:, 0] / self.pixel_size_col - 0.5

        ij_det = np.vstack([i_pix, j_pix]).T
        if pixels:
            ij_det = np.array(np.round(ij_det), dtype=int)
        return ij_det

    def pixelToCart(self, ij_det):
        """
        Convert vstacked array or list of [i,j] pixel indices
        (or UL corner-based points) and convert to (x,y) in the
        cartesian frame {Xd, Yd, Zd}
        """
        ij_det = np.atleast_2d(ij_det)

        x = (ij_det[:, 1] + 0.5)*self.pixel_size_col\
            + self.corner_ll[0]
        y = (self.rows - ij_det[:, 0] - 0.5)*self.pixel_size_row\
            + self.corner_ll[1]
        return np.vstack([x, y]).T

    def angularPixelSize(self, xy, rMat_s=None, tVec_s=None, tVec_c=None):
        """
        Wraps xrdutil.angularPixelSize
        """
        # munge kwargs
        if rMat_s is None:
            rMat_s = ct.identity_3x3
        if tVec_s is None:
            tVec_s = ct.zeros_3x1
        if tVec_c is None:
            tVec_c = ct.zeros_3x1

        # call function
        ang_ps = xrdutil.angularPixelSize(
            xy, (self.pixel_size_row, self.pixel_size_col),
            self.rmat, rMat_s,
            self.tvec, tVec_s, tVec_c,
            distortion=self.distortion,
            beamVec=self.bvec, etaVec=self.evec)
        return ang_ps

    def clip_to_panel(self, xy, buffer_edges=True):
        """
        if self.roi is not None, uses it by default

        TODO: check if need shape kwarg
        TODO: optimize ROI search better than list comprehension below
        TODO: panel_buffer can be a 2-d boolean mask, but needs testing

        """
        xy = np.atleast_2d(xy)

        if self.roi is not None:
            ij_crds = self.cartToPixel(xy, pixels=True)
            ii, jj = polygon(self.roi[:, 0], self.roi[:, 1],
                             shape=(self.rows, self.cols))
            on_panel_rows = [i in ii for i in ij_crds[:, 0]]
            on_panel_cols = [j in jj for j in ij_crds[:, 1]]
            on_panel = np.logical_and(on_panel_rows, on_panel_cols)
        else:
            xlim = 0.5*self.col_dim
            ylim = 0.5*self.row_dim
            if buffer_edges and self.panel_buffer is not None:
                if self.panel_buffer.ndim == 2:
                    pix = self.cartToPixel(xy, pixels=True)

                    roff = np.logical_or(pix[:, 0] < 0, pix[:, 0] >= self.rows)
                    coff = np.logical_or(pix[:, 1] < 0, pix[:, 1] >= self.cols)

                    idx = np.logical_or(roff, coff)

                    pix[idx, :] = 0

                    on_panel = self.panel_buffer[pix[:, 0], pix[:, 1]]
                    on_panel[idx] = False
                else:
                    xlim -= self.panel_buffer[0]
                    ylim -= self.panel_buffer[1]
                    on_panel_x = np.logical_and(
                        xy[:, 0] >= -xlim, xy[:, 0] <= xlim
                    )
                    on_panel_y = np.logical_and(
                        xy[:, 1] >= -ylim, xy[:, 1] <= ylim
                    )
                    on_panel = np.logical_and(on_panel_x, on_panel_y)
            elif not buffer_edges:
                on_panel_x = np.logical_and(
                    xy[:, 0] >= -xlim, xy[:, 0] <= xlim
                )
                on_panel_y = np.logical_and(
                    xy[:, 1] >= -ylim, xy[:, 1] <= ylim
                )
                on_panel = np.logical_and(on_panel_x, on_panel_y)
        return xy[on_panel, :], on_panel

    def cart_to_angles(self, xy_data):
        """
        TODO: distortion
        """
        rmat_s = ct.identity_3x3
        tvec_s = ct.zeros_3
        tvec_c = ct.zeros_3
        angs, g_vec = detectorXYToGvec(
            xy_data, self.rmat, rmat_s,
            self.tvec, tvec_s, tvec_c,
            beamVec=self.bvec, etaVec=self.evec)
        tth_eta = np.vstack([angs[0], angs[1]]).T
        return tth_eta, g_vec

    def angles_to_cart(self, tth_eta):
        """
        TODO: distortion
        """
        rmat_s = rmat_c = ct.identity_3x3
        tvec_s = tvec_c = ct.zeros_3

        angs = np.hstack([tth_eta, np.zeros((len(tth_eta), 1))])

        xy_det = gvecToDetectorXY(
            anglesToGVec(angs, bHat_l=self.bvec, eHat_l=self.evec),
            self.rmat, rmat_s, rmat_c,
            self.tvec, tvec_s, tvec_c,
            beamVec=self.bvec)
        return xy_det

    def interpolate_nearest(self, xy, img, pad_with_nans=True):
        """
        TODO: revisit normalization in here?

        """
        is_2d = img.ndim == 2
        right_shape = img.shape[0] == self.rows and img.shape[1] == self.cols
        assert is_2d and right_shape,\
            "input image must be 2-d with shape (%d, %d)"\
            % (self.rows, self.cols)

        # initialize output with nans
        if pad_with_nans:
            int_xy = np.nan*np.ones(len(xy))
        else:
            int_xy = np.zeros(len(xy))

        # clip away points too close to or off the edges of the detector
        xy_clip, on_panel = self.clip_to_panel(xy, buffer_edges=True)

        # get pixel indices of clipped points
        i_src = cellIndices(self.row_pixel_vec, xy_clip[:, 1])
        j_src = cellIndices(self.col_pixel_vec, xy_clip[:, 0])

        # next interpolate across cols
        int_vals = img[i_src, j_src]
        int_xy[on_panel] = int_vals
        return int_xy

    def interpolate_bilinear(self, xy, img, pad_with_nans=True):
        """
        TODO: revisit normalization in here?
        """
        is_2d = img.ndim == 2
        right_shape = img.shape[0] == self.rows and img.shape[1] == self.cols
        assert is_2d and right_shape,\
            "input image must be 2-d with shape (%d, %d)"\
            % (self.rows, self.cols)

        # initialize output with nans
        if pad_with_nans:
            int_xy = np.nan*np.ones(len(xy))
        else:
            int_xy = np.zeros(len(xy))

        # clip away points too close to or off the edges of the detector
        xy_clip, on_panel = self.clip_to_panel(xy, buffer_edges=True)

        # grab fractional pixel indices of clipped points
        ij_frac = self.cartToPixel(xy_clip)

        # get floors/ceils from array of pixel _centers_
        i_floor = cellIndices(self.row_pixel_vec, xy_clip[:, 1])
        j_floor = cellIndices(self.col_pixel_vec, xy_clip[:, 0])
        i_ceil = i_floor + 1
        j_ceil = j_floor + 1

        # first interpolate at top/bottom rows
        row_floor_int = \
            (j_ceil - ij_frac[:, 1])*img[i_floor, j_floor] \
            + (ij_frac[:, 1] - j_floor)*img[i_floor, j_ceil]
        row_ceil_int = \
            (j_ceil - ij_frac[:, 1])*img[i_ceil, j_floor] \
            + (ij_frac[:, 1] - j_floor)*img[i_ceil, j_ceil]

        # next interpolate across cols
        int_vals = \
            (i_ceil - ij_frac[:, 0])*row_floor_int \
            + (ij_frac[:, 0] - i_floor)*row_ceil_int
        int_xy[on_panel] = int_vals
        return int_xy

    def make_powder_rings(
            self, pd, merge_hkls=False, delta_tth=None,
            delta_eta=10., eta_period=None,
            rmat_s=ct.identity_3x3,  tvec_s=ct.zeros_3,
            tvec_c=ct.zeros_3, full_output=False):
        """
        !!! it is assuming that rmat_s is built from (chi, ome)
        !!! as it the case for HEDM
        """
        # in case you want to give it tth angles directly
        if hasattr(pd, '__len__'):
            tth = np.array(pd).flatten()
            if delta_tth is None:
                raise RuntimeError(
                    "If supplying a 2theta list as first arg, "
                    + "must supply a delta_tth")
            sector_vertices = np.tile(
                0.5*np.radians([-delta_tth, -delta_eta,
                                -delta_tth, delta_eta,
                                delta_tth, delta_eta,
                                delta_tth, -delta_eta,
                                0.0, 0.0]), (len(tth), 1)
                )
        else:
            # Okay, we have a PlaneData object
            try:
                pd = PlaneData.makeNew(pd)    # make a copy to munge
            except(TypeError):
                # !!! have some other object here, likely a dummy plane data
                # object of some sort...
                pass

            if delta_tth is not None:
                pd.tThWidth = np.radians(delta_tth)
            else:
                delta_tth = np.degrees(pd.tThWidth)

            # conversions, meh...
            del_eta = np.radians(delta_eta)

            # do merging if asked
            if merge_hkls:
                _, tth_ranges = pd.getMergedRanges()
                tth = np.array([0.5*sum(i) for i in tth_ranges])
            else:
                tth_ranges = pd.getTThRanges()
                tth = pd.getTTh()
            tth_pm = tth_ranges - np.tile(tth, (2, 1)).T
            sector_vertices = np.vstack(
                [[i[0], -del_eta,
                  i[0], del_eta,
                  i[1], del_eta,
                  i[1], -del_eta,
                  0.0, 0.0]
                 for i in tth_pm])

        # for generating rings, make eta vector in correct period
        if eta_period is None:
            eta_period = (-np.pi, np.pi)
        neta = int(360./float(delta_eta))

        # this is the vector of ETA EDGES
        eta_edges = mapAngle(
            np.radians(
                delta_eta*np.linspace(0., neta, num=neta + 1)
            ) + eta_period[0],
            eta_period
        )

        # get eta bin centers from edges
        """
        # !!! this way is probably overkill, since we have delta eta
        eta_centers = np.average(
            np.vstack([eta[:-1], eta[1:]),
            axis=0)
        """
        # !!! should be safe as eta_edges are monotonic
        eta_centers = eta_edges[:-1] + del_eta

        # !!! get chi and ome from rmat_s
        # chi = np.arctan2(rmat_s[2, 1], rmat_s[1, 1])
        ome = np.arctan2(rmat_s[0, 2], rmat_s[0, 0])

        # make list of angle tuples
        angs = [
            np.vstack(
                [i*np.ones(neta), eta_centers, ome*np.ones(neta)]
            ) for i in tth
        ]

        # need xy coords and pixel sizes
        valid_ang = []
        valid_xy = []
        map_indices = []
        npp = 5  # [ll, ul, ur, lr, center]
        for i_ring in range(len(angs)):
            # expand angles to patch vertices
            these_angs = angs[i_ring].T
            patch_vertices = (
                np.tile(these_angs[:, :2], (1, npp))
                + np.tile(sector_vertices[i_ring], (neta, 1))
            ).reshape(npp*neta, 2)

            # duplicate ome array
            ome_dupl = np.tile(
                these_angs[:, 2], (npp, 1)
            ).T.reshape(npp*neta, 1)

            # find vertices that all fall on the panel
            gVec_ring_l = anglesToGVec(
                np.hstack([patch_vertices, ome_dupl]),
                bHat_l=self.bvec)
            all_xy = gvecToDetectorXY(
                gVec_ring_l,
                self.rmat, rmat_s, ct.identity_3x3,
                self.tvec, tvec_s, tvec_c,
                beamVec=self.bvec)
            _, on_panel = self.clip_to_panel(all_xy)

            # all vertices must be on...
            patch_is_on = np.all(on_panel.reshape(neta, npp), axis=1)
            patch_xys = all_xy.reshape(neta, 5, 2)[patch_is_on]

            # the surving indices
            idx = np.where(patch_is_on)[0]

            # form output arrays
            valid_ang.append(these_angs[patch_is_on, :2])
            valid_xy.append(patch_xys[:, -1, :].squeeze())
            map_indices.append(idx)
            pass
        # ??? is this option necessary?
        if full_output:
            return valid_ang, valid_xy, map_indices, eta_edges
        else:
            return valid_ang, valid_xy

    def map_to_plane(self, pts, rmat, tvec):
        """
        map detctor points to specified plane

        by convention

        n * (u*pts_l - tvec) = 0

        [pts]_l = rmat*[pts]_m + tvec
        """
        # arg munging
        pts = np.atleast_2d(pts)
        npts = len(pts)

        # map plane normal & translation vector, LAB FRAME
        nvec_map_lab = rmat[:, 2].reshape(3, 1)
        tvec_map_lab = np.atleast_2d(tvec).reshape(3, 1)
        tvec_d_lab = np.atleast_2d(self.tvec).reshape(3, 1)

        # put pts as 3-d in panel CS and transform to 3-d lab coords
        pts_det = np.hstack([pts, np.zeros((npts, 1))])
        pts_lab = np.dot(self.rmat, pts_det.T) + tvec_d_lab

        # scaling along pts vectors to hit map plane
        u = np.dot(nvec_map_lab.T, tvec_map_lab) \
            / np.dot(nvec_map_lab.T, pts_lab)

        # pts on map plane, in LAB FRAME
        pts_map_lab = np.tile(u, (3, 1)) * pts_lab

        return np.dot(rmat.T, pts_map_lab - tvec_map_lab)[:2, :].T

    def simulate_rotation_series(self, plane_data, grain_param_list,
                                 eta_ranges=[(-np.pi, np.pi), ],
                                 ome_ranges=[(-np.pi, np.pi), ],
                                 ome_period=(-np.pi, np.pi),
                                 chi=0., tVec_s=ct.zeros_3,
                                 wavelength=None):
        """
        """

        # grab B-matrix from plane data
        bMat = plane_data.latVecOps['B']

        # reconcile wavelength
        #   * added sanity check on exclusions here; possible to
        #   * make some reflections invalid (NaN)
        if wavelength is None:
            wavelength = plane_data.wavelength
        else:
            if plane_data.wavelength != wavelength:
                plane_data.wavelength = ct.keVToAngstrom(wavelength)
        assert not np.any(np.isnan(plane_data.getTTh())),\
            "plane data exclusions incompatible with wavelength"

        # vstacked G-vector id, h, k, l
        full_hkls = xrdutil._fetch_hkls_from_planedata(plane_data)

        """ LOOP OVER GRAINS """
        valid_ids = []
        valid_hkls = []
        valid_angs = []
        valid_xys = []
        ang_pixel_size = []
        for gparm in grain_param_list:

            # make useful parameters
            rMat_c = makeRotMatOfExpMap(gparm[:3])
            tVec_c = gparm[3:6]
            vInv_s = gparm[6:]

            # All possible bragg conditions as vstacked [tth, eta, ome]
            # for each omega solution
            angList = np.vstack(
                oscillAnglesOfHKLs(
                    full_hkls[:, 1:], chi,
                    rMat_c, bMat, wavelength,
                    vInv=vInv_s,
                    )
                )

            # filter by eta and omega ranges
            # ??? get eta range from detector?
            allAngs, allHKLs = xrdutil._filter_hkls_eta_ome(
                full_hkls, angList, eta_ranges, ome_ranges
                )
            allAngs[:, 2] = mapAngle(allAngs[:, 2], ome_period)

            # find points that fall on the panel
            det_xy, rMat_s = xrdutil._project_on_detector_plane(
                allAngs,
                self.rmat, rMat_c, chi,
                self.tvec, tVec_c, tVec_s,
                self.distortion)
            xys_p, on_panel = self.clip_to_panel(det_xy)
            valid_xys.append(xys_p)

            # grab hkls and gvec ids for this panel
            valid_hkls.append(allHKLs[on_panel, 1:])
            valid_ids.append(allHKLs[on_panel, 0])

            # reflection angles (voxel centers) and pixel size in (tth, eta)
            valid_angs.append(allAngs[on_panel, :])
            ang_pixel_size.append(self.angularPixelSize(xys_p))
        return valid_ids, valid_hkls, valid_angs, valid_xys, ang_pixel_size

    def simulate_laue_pattern(self, crystal_data,
                              minEnergy=5., maxEnergy=35.,
                              rmat_s=None, tvec_s=None,
                              grain_params=None,
                              beam_vec=None):
        """
        """
        if isinstance(crystal_data, PlaneData):

            plane_data = crystal_data

            # grab the expanded list of hkls from plane_data
            hkls = np.hstack(plane_data.getSymHKLs())

            # and the unit plane normals (G-vectors) in CRYSTAL FRAME
            gvec_c = np.dot(plane_data.latVecOps['B'], hkls)
        elif len(crystal_data) == 2:
            # !!! should clean this up
            hkls = np.array(crystal_data[0])
            bmat = crystal_data[1]
            gvec_c = np.dot(bmat, hkls)
        else:
            raise(RuntimeError, 'argument list not understood')
        nhkls_tot = hkls.shape[1]

        # parse energy ranges
        # TODO: allow for spectrum parsing
        multipleEnergyRanges = False
        if hasattr(maxEnergy, '__len__'):
            assert len(maxEnergy) == len(minEnergy), \
                'energy cutoff ranges must have the same length'
            multipleEnergyRanges = True
            lmin = []
            lmax = []
            for i in range(len(maxEnergy)):
                lmin.append(ct.keVToAngstrom(maxEnergy[i]))
                lmax.append(ct.keVToAngstrom(minEnergy[i]))
        else:
            lmin = ct.keVToAngstrom(maxEnergy)
            lmax = ct.keVToAngstrom(minEnergy)

        # parse grain parameters kwarg
        if grain_params is None:
            grain_params = np.atleast_2d(
                np.hstack([np.zeros(6), ct.identity_6x1])
            )
        n_grains = len(grain_params)

        # sample rotation
        if rmat_s is None:
            rmat_s = ct.identity_3x3

        # dummy translation vector... make input
        if tvec_s is None:
            tvec_s = ct.zeros_3

        # beam vector
        if beam_vec is None:
            beam_vec = ct.beam_vec

        # =========================================================================
        # LOOP OVER GRAINS
        # =========================================================================

        # pre-allocate output arrays
        xy_det = np.nan*np.ones((n_grains, nhkls_tot, 2))
        hkls_in = np.nan*np.ones((n_grains, 3, nhkls_tot))
        angles = np.nan*np.ones((n_grains, nhkls_tot, 2))
        dspacing = np.nan*np.ones((n_grains, nhkls_tot))
        energy = np.nan*np.ones((n_grains, nhkls_tot))
        for iG, gp in enumerate(grain_params):
            rmat_c = makeRotMatOfExpMap(gp[:3])
            tvec_c = gp[3:6].reshape(3, 1)
            vInv_s = mutil.vecMVToSymm(gp[6:].reshape(6, 1))

            # stretch them: V^(-1) * R * Gc
            gvec_s_str = np.dot(vInv_s, np.dot(rmat_c, gvec_c))
            ghat_c_str = mutil.unitVector(np.dot(rmat_c.T, gvec_s_str))

            # project
            dpts = gvecToDetectorXY(ghat_c_str.T,
                                    self.rmat, rmat_s, rmat_c,
                                    self.tvec, tvec_s, tvec_c,
                                    beamVec=beam_vec)

            # check intersections with detector plane
            canIntersect = ~np.isnan(dpts[:, 0])
            npts_in = sum(canIntersect)

            if np.any(canIntersect):
                dpts = dpts[canIntersect, :].reshape(npts_in, 2)
                dhkl = hkls[:, canIntersect].reshape(3, npts_in)

                # back to angles
                tth_eta, gvec_l = detectorXYToGvec(
                    dpts,
                    self.rmat, rmat_s,
                    self.tvec, tvec_s, tvec_c,
                    beamVec=beam_vec)
                tth_eta = np.vstack(tth_eta).T

                # warp measured points
                if self.distortion is not None:
                    if len(self.distortion) == 2:
                        dpts = self.distortion[0](
                            dpts, self.distortion[1],
                            invert=True)
                    else:
                        raise(RuntimeError,
                              "something is wrong with the distortion")

                # plane spacings and energies
                dsp = 1. / rowNorm(gvec_s_str[:, canIntersect].T)
                wlen = 2*dsp*np.sin(0.5*tth_eta[:, 0])

                # clip to detector panel
                _, on_panel = self.clip_to_panel(dpts, buffer_edges=True)

                if multipleEnergyRanges:
                    validEnergy = np.zeros(len(wlen), dtype=bool)
                    for i in range(len(lmin)):
                        in_energy_range = np.logical_and(
                                wlen >= lmin[i],
                                wlen <= lmax[i])
                        validEnergy = validEnergy | in_energy_range
                        pass
                else:
                    validEnergy = np.logical_and(wlen >= lmin, wlen <= lmax)
                    pass

                # index for valid reflections
                keepers = np.where(np.logical_and(on_panel, validEnergy))[0]

                # assign output arrays
                xy_det[iG][keepers, :] = dpts[keepers, :]
                hkls_in[iG][:, keepers] = dhkl[:, keepers]
                angles[iG][keepers, :] = tth_eta[keepers, :]
                dspacing[iG, keepers] = dsp[keepers]
                energy[iG, keepers] = ct.keVToAngstrom(wlen[keepers])
                pass    # close conditional on valids
            pass    # close loop on grains
        return xy_det, hkls_in, angles, dspacing, energy
