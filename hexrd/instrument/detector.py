from abc import abstractmethod
import copy
import os
from typing import Optional

from hexrd.instrument.constants import (
    COATING_DEFAULT, FILTER_DEFAULTS, PHOSPHOR_DEFAULT
)
from hexrd.instrument.physics_package import AbstractPhysicsPackage
import numpy as np
import numba

from hexrd import constants as ct
from hexrd import distortion as distortion_pkg
from hexrd import matrixutil as mutil
from hexrd import xrdutil
from hexrd.rotations import mapAngle

from hexrd.material import crystallography
from hexrd.material.crystallography import PlaneData

from hexrd.transforms.xfcapi import (
    xy_to_gvec,
    gvec_to_xy,
    make_beam_rmat,
    make_rmat_of_expmap,
    oscill_angles_of_hkls,
    angles_to_dvec,
)

from hexrd.utils.decorators import memoize
from hexrd.gridutil import cellIndices
from hexrd.instrument import detector_coatings

distortion_registry = distortion_pkg.Registry()

max_workers_DFLT = max(1, os.cpu_count() - 1)

panel_calibration_flags_DFLT = np.array([1, 1, 1, 1, 1, 1], dtype=bool)

beam_energy_DFLT = 65.351

# Memoize these, so each detector can avoid re-computing if nothing
# has changed.
_lorentz_factor = memoize(crystallography.lorentz_factor)
_polarization_factor = memoize(crystallography.polarization_factor)


class Detector:
    """
    Base class for 2D detectors with functions and properties
    common to planar and cylindrical detectors. This class
    will be inherited by both those classes.
    """

    __pixelPitchUnit = 'mm'

    # Abstract methods that must be redefined in derived classes
    @property
    @abstractmethod
    def detector_type(self):
        raise NotImplementedError

    @abstractmethod
    def cart_to_angles(
        self,
        xy_data,
        rmat_s=None,
        tvec_s=None,
        tvec_c=None,
        apply_distortion=False,
    ):
        """
        Transform cartesian coordinates to angular.

        Parameters
        ----------
        xy_data : TYPE
            The (n, 2) array of n (x, y) coordinates to be transformed in
            either the raw or ideal cartesian plane (see `apply_distortion`
            kwarg below).
        rmat_s : array_like, optional
            The (3, 3) COB matrix for the sample frame. The default is None.
        tvec_s : array_like, optional
            The (3, ) translation vector for the sample frame.
            The default is None.
        tvec_c : array_like, optional
            The (3, ) translation vector for the crystal frame.
            The default is None.
        apply_distortion : bool, optional
            If True, apply distortion to the inpout cartesian coordinates.
            The default is False.

        Returns
        -------
        tth_eta : TYPE
            DESCRIPTION.
        g_vec : TYPE
            DESCRIPTION.

        """
        raise NotImplementedError

    @abstractmethod
    def angles_to_cart(
        self,
        tth_eta,
        rmat_s=None,
        tvec_s=None,
        rmat_c=None,
        tvec_c=None,
        apply_distortion=False,
    ):
        """
        Transform angular coordinates to cartesian.

        Parameters
        ----------
        tth_eta : array_like
            The (n, 2) array of n (tth, eta) coordinates to be transformed.
        rmat_s : array_like, optional
            The (3, 3) COB matrix for the sample frame. The default is None.
        tvec_s : array_like, optional
            The (3, ) translation vector for the sample frame.
            The default is None.
        rmat_c : array_like, optional
            (3, 3) COB matrix for the crystal frame.
            The default is None.
        tvec_c : array_like, optional
            The (3, ) translation vector for the crystal frame.
            The default is None.
        apply_distortion : bool, optional
            If True, apply distortion to take cartesian coordinates to the
            "warped" configuration. The default is False.

        Returns
        -------
        xy_det : array_like
            The (n, 2) array on the n input coordinates in the .

        """
        raise NotImplementedError

    @abstractmethod
    def cart_to_dvecs(self, xy_data):
        """Convert cartesian coordinates to dvectors"""
        raise NotImplementedError

    @abstractmethod
    def pixel_angles(self, origin=ct.zeros_3):
        raise NotImplementedError

    @abstractmethod
    def pixel_tth_gradient(self, origin=ct.zeros_3):
        raise NotImplementedError

    @abstractmethod
    def pixel_eta_gradient(self, origin=ct.zeros_3):
        raise NotImplementedError

    @abstractmethod
    def calc_filter_coating_transmission(self, energy):
        pass

    @property
    @abstractmethod
    def beam_position(self):
        """
        returns the coordinates of the beam in the cartesian detector
        frame {Xd, Yd, Zd}.  NaNs if no intersection.
        """
        raise NotImplementedError

    @property
    def extra_config_kwargs(self):
        return {}

    # End of abstract methods

    def __init__(
        self,
        rows=2048,
        cols=2048,
        pixel_size=(0.2, 0.2),
        tvec=np.r_[0.0, 0.0, -1000.0],
        tilt=ct.zeros_3,
        name='default',
        bvec=ct.beam_vec,
        xrs_dist=None,
        evec=ct.eta_vec,
        saturation_level=None,
        panel_buffer=None,
        tth_distortion=None,
        roi=None,
        group=None,
        distortion=None,
        max_workers=max_workers_DFLT,
        detector_filter: Optional[detector_coatings.Filter] = None,
        detector_coating: Optional[detector_coatings.Coating] = None,
        phosphor: Optional[detector_coatings.Phosphor] = None,
    ):
        """
        Instantiate a PlanarDetector object.

        Parameters
        ----------
        rows : TYPE, optional
            DESCRIPTION. The default is 2048.
        cols : TYPE, optional
            DESCRIPTION. The default is 2048.
        pixel_size : TYPE, optional
            DESCRIPTION. The default is (0.2, 0.2).
        tvec : TYPE, optional
            DESCRIPTION. The default is np.r_[0., 0., -1000.].
        tilt : TYPE, optional
            DESCRIPTION. The default is ct.zeros_3.
        name : TYPE, optional
            DESCRIPTION. The default is 'default'.
        bvec : TYPE, optional
            DESCRIPTION. The default is ct.beam_vec.
        evec : TYPE, optional
            DESCRIPTION. The default is ct.eta_vec.
        saturation_level : TYPE, optional
            DESCRIPTION. The default is None.
        panel_buffer : TYPE, optional
            If a scalar or len(2) array_like, the interpretation is a border
            in mm. If an array with shape (nrows, ncols), interpretation is a
            boolean with True marking valid pixels.  The default is None.
        roi : TYPE, optional
            DESCRIPTION. The default is None.
        group : TYPE, optional
            DESCRIPTION. The default is None.
        distortion : TYPE, optional
            DESCRIPTION. The default is None.
        detector_filter : detector_coatings.Filter, optional
            filter specifications including material type,
            density and thickness. Used for absorption correction
            calculations.
        detector_coating : detector_coatings.Coating, optional
            coating specifications including material type,
            density and thickness. Used for absorption correction
            calculations.
        phosphor : detector_coatings.Phosphor, optional
            phosphor specifications including material type,
            density and thickness. Used for absorption correction
            calculations.

        Returns
        -------
        None.

        """
        self._name = name

        self._rows = rows
        self._cols = cols

        self._pixel_size_row = pixel_size[0]
        self._pixel_size_col = pixel_size[1]

        self._saturation_level = saturation_level

        self._panel_buffer = panel_buffer

        self._tth_distortion = tth_distortion

        if roi is None:
            self._roi = roi
        else:
            assert len(roi) == 2, "roi is set via (start_row, start_col)"
            self._roi = (
                (roi[0], roi[0] + self._rows),
                (roi[1], roi[1] + self._cols),
            )

        self._tvec = np.array(tvec).flatten()
        self._tilt = np.array(tilt).flatten()

        self._bvec = np.array(bvec).flatten()
        self._xrs_dist = xrs_dist

        self._evec = np.array(evec).flatten()

        self._distortion = distortion

        self.max_workers = max_workers

        self.group = group

        if detector_filter is None:
            detector_filter = detector_coatings.Filter(
                **FILTER_DEFAULTS.TARDIS)
        self.filter = detector_filter

        if detector_coating is None:
            detector_coating = detector_coatings.Coating(**COATING_DEFAULT)
        self.coating = detector_coating

        if phosphor is None:
            phosphor = detector_coatings.Phosphor(**PHOSPHOR_DEFAULT)
        self.phosphor = phosphor

        #
        # set up calibration parameter list and refinement flags
        #
        # order for a single detector will be
        #
        #     [tilt, translation, <distortion>]
        dparams = []
        if self._distortion is not None:
            dparams = self._distortion.params
        self._calibration_parameters = np.hstack(
            [self._tilt, self._tvec, dparams]
        )
        self._calibration_flags = np.hstack(
            [panel_calibration_flags_DFLT, np.zeros(len(dparams), dtype=bool)]
        )

    # detector ID
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, s):
        assert isinstance(s, str), "requires string input"
        self._name = s

    @property
    def lmfit_name(self):
        # lmfit requires underscores instead of dashes
        return self.name.replace('-', '_')

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
    def tth_distortion(self):
        return self._tth_distortion

    @tth_distortion.setter
    def tth_distortion(self, x):
        """if not None, a buffer in mm (x, y)"""
        if x is not None:
            assert x.ndim == 2 and x.shape == self.shape
        self._tth_distortion = x

    @property
    def roi(self):
        return self._roi

    @roi.setter
    def roi(self, vertex_array):
        """
        !!! vertex array must be (r0, c0)
        """
        if vertex_array is not None:
            assert (
                len(vertex_array) == 2
            ), "roi is set via (start_row, start_col)"
        self._roi = (
            (vertex_array[0], vertex_array[0] + self.rows),
            (vertex_array[1], vertex_array[1] + self.cols),
        )

    @property
    def row_dim(self):
        return self.rows * self.pixel_size_row

    @property
    def col_dim(self):
        return self.cols * self.pixel_size_col

    @property
    def row_pixel_vec(self):
        return self.pixel_size_row * (
            0.5 * (self.rows - 1) - np.arange(self.rows)
        )

    @property
    def row_edge_vec(self):
        return _row_edge_vec(self.rows, self.pixel_size_row)

    @property
    def col_pixel_vec(self):
        return self.pixel_size_col * (
            np.arange(self.cols) - 0.5 * (self.cols - 1)
        )

    @property
    def col_edge_vec(self):
        return _col_edge_vec(self.cols, self.pixel_size_col)

    @property
    def corner_ul(self):
        return np.r_[-0.5 * self.col_dim, 0.5 * self.row_dim]

    @property
    def corner_ll(self):
        return np.r_[-0.5 * self.col_dim, -0.5 * self.row_dim]

    @property
    def corner_lr(self):
        return np.r_[0.5 * self.col_dim, -0.5 * self.row_dim]

    @property
    def corner_ur(self):
        return np.r_[0.5 * self.col_dim, 0.5 * self.row_dim]

    @property
    def shape(self):
        return (self.rows, self.cols)

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
        assert (
            len(x) == 3 and sum(x * x) > 1 - ct.sqrt_epsf
        ), 'input must have length = 3 and have unit magnitude'
        self._bvec = x

    @property
    def xrs_dist(self):
        return self._xrs_dist

    @xrs_dist.setter
    def xrs_dist(self, x):
        assert x is None or np.isscalar(
            x
        ), f"'source_distance' must be None or scalar; you input '{x}'"
        self._xrs_dist = x

    @property
    def evec(self):
        return self._evec

    @evec.setter
    def evec(self, x):
        x = np.array(x).flatten()
        assert (
            len(x) == 3 and sum(x * x) > 1 - ct.sqrt_epsf
        ), 'input must have length = 3 and have unit magnitude'
        self._evec = x

    @property
    def distortion(self):
        return self._distortion

    @distortion.setter
    def distortion(self, x):
        if x is not None:
            check_arg = np.zeros(len(distortion_registry), dtype=bool)
            for i, dcls in enumerate(distortion_registry.values()):
                check_arg[i] = isinstance(x, dcls)
            assert np.any(check_arg), 'input distortion is not in registry!'
        self._distortion = x

    @property
    def rmat(self):
        return make_rmat_of_expmap(self.tilt)

    @property
    def normal(self):
        return self.rmat[:, 2]

    # ...memoize???
    @property
    def pixel_coords(self):
        pix_i, pix_j = np.meshgrid(
            self.row_pixel_vec, self.col_pixel_vec, indexing='ij'
        )
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
            dparams = self.distortion.params
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

    @property
    def calibration_flags_to_lmfit_names(self):
        # Create a list identical in length to `self.calibration_flags`
        # where the entries in the list are the corresponding lmfit
        # parameter names.
        name = self.lmfit_name
        flags = [
            f'{name}_euler_z',
            f'{name}_euler_xp',
            f'{name}_euler_zpp',
            f'{name}_tvec_x',
            f'{name}_tvec_y',
            f'{name}_tvec_z',
        ]
        if self.distortion is not None:
            for i in range(len(self.distortion.params)):
                flags.append(f'{name}_distortion_param_{i}')

        return flags

    # =========================================================================
    # METHODS
    # =========================================================================

    def polarization_factor(self, f_hor, f_vert, unpolarized=False):
        """
        Calculated the polarization factor for every pixel.

        Parameters
        ----------
        f_hor : float
             the fraction of horizontal polarization. for XFELs
             this is close to 1.
        f_vert : TYPE
            the fraction of vertical polarization, which is ~0 for XFELs.

        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        s = f_hor + f_vert
        if np.abs(s - 1) > ct.sqrt_epsf:
            msg = (
                "sum of fraction of "
                "horizontal and vertical polarizations "
                "must be equal to 1."
            )
            raise RuntimeError(msg)

        if f_hor < 0 or f_vert < 0:
            msg = (
                "fraction of polarization in horizontal "
                "or vertical directions can't be negative."
            )
            raise RuntimeError(msg)

        tth, eta = self.pixel_angles()
        kwargs = {
            'tth': tth,
            'eta': eta,
            'f_hor': f_hor,
            'f_vert': f_vert,
            'unpolarized': unpolarized,
        }

        return _polarization_factor(**kwargs)

    def lorentz_factor(self):
        """
        calculate the lorentz factor for every pixel

        Parameters
        ----------
        None

        Raises
        ------
        None

        Returns
        -------
        numpy.ndarray
            returns an array the same size as the detector panel
            with each element containg the lorentz factor of the
            corresponding pixel
        """
        tth, eta = self.pixel_angles()
        return _lorentz_factor(tth)

    def config_dict(
        self,
        chi=0,
        tvec=ct.zeros_3,
        beam_energy=beam_energy_DFLT,
        beam_vector=ct.beam_vec,
        sat_level=None,
        panel_buffer=None,
        style='yaml',
    ):
        """
        Return a dictionary of detector parameters.

        Optional instrument level parameters.  This is a convenience function
        to work with the APIs in several functions in xrdutil.

        Parameters
        ----------
        chi : float, optional
            DESCRIPTION. The default is 0.
        tvec : array_like (3,), optional
            DESCRIPTION. The default is ct.zeros_3.
        beam_energy : float, optional
            DESCRIPTION. The default is beam_energy_DFLT.
        beam_vector : aray_like (3,), optional
            DESCRIPTION. The default is ct.beam_vec.
        sat_level : scalar, optional
            DESCRIPTION. The default is None.
        panel_buffer : scalar, array_like (2,), optional
            DESCRIPTION. The default is None.

        Returns
        -------
        config_dict : dict
            DESCRIPTION.

        """
        assert style.lower() in ['yaml', 'hdf5'], (
            "style must be either 'yaml', or 'hdf5'; you gave '%s'" % style
        )

        config_dict = {}

        # =====================================================================
        # DETECTOR PARAMETERS
        # =====================================================================
        # transform and pixels
        #
        # assign local vars; listify if necessary
        tilt = self.tilt
        translation = self.tvec
        roi = (
            None
            if self.roi is None
            else np.array([self.roi[0][0], self.roi[1][0]]).flatten()
        )
        if style.lower() == 'yaml':
            tilt = tilt.tolist()
            translation = translation.tolist()
            tvec = tvec.tolist()
            roi = None if roi is None else roi.tolist()

        det_dict = dict(
            detector_type=self.detector_type,
            transform=dict(
                tilt=tilt,
                translation=translation,
            ),
            pixels=dict(
                rows=int(self.rows),
                columns=int(self.cols),
                size=[float(self.pixel_size_row), float(self.pixel_size_col)],
            ),
        )

        if roi is not None:
            # Only add roi if it is not None
            det_dict['pixels']['roi'] = roi

        if self.group is not None:
            # Only add group if it is not None
            det_dict['group'] = self.group

        # distortion
        if self.distortion is not None:
            dparams = self.distortion.params
            if style.lower() == 'yaml':
                dparams = dparams.tolist()
            dist_d = dict(
                function_name=self.distortion.maptype, parameters=dparams
            )
            det_dict['distortion'] = dist_d

        # saturation level
        if sat_level is None:
            sat_level = self.saturation_level
        det_dict['saturation_level'] = float(sat_level)

        # panel buffer
        if panel_buffer is None:
            # could be none, a 2-element list, or a 2-d array (rows, cols)
            panel_buffer = copy.deepcopy(self.panel_buffer)
        # !!! now we have to do some style-dependent munging of panel_buffer
        if isinstance(panel_buffer, np.ndarray):
            if panel_buffer.ndim == 1:
                assert len(panel_buffer) == 2, "length of 1-d buffer must be 2"
                # if here is a 2-element array
                if style.lower() == 'yaml':
                    panel_buffer = panel_buffer.tolist()
            elif panel_buffer.ndim == 2:
                if style.lower() == 'yaml':
                    # !!! can't practically write array-like buffers to YAML
                    #     so forced to clobber
                    print("clobbering panel buffer array in yaml-ready output")
                    panel_buffer = [0.0, 0.0]
            else:
                raise RuntimeError(
                    "panel buffer ndim must be 1 or 2; you specified %d"
                    % panel_buffer.ndmin
                )
        elif panel_buffer is None:
            # still None on self
            # !!! this gets handled by unwrap_dict_to_h5 now

            # if style.lower() == 'hdf5':
            #     # !!! can't write None to hdf5; substitute with zeros
            #     panel_buffer = np.r_[0., 0.]
            pass
        det_dict['buffer'] = panel_buffer

        det_dict.update(self.extra_config_kwargs)

        # =====================================================================
        # SAMPLE STAGE PARAMETERS
        # =====================================================================
        stage_dict = dict(chi=chi, translation=tvec)

        # =====================================================================
        # BEAM PARAMETERS
        # =====================================================================
        # !!! make_reflection_patches is still using the vector
        # azim, pola = calc_angles_from_beam_vec(beam_vector)
        # beam_dict = dict(
        #     energy=beam_energy,
        #     vector=dict(
        #         azimuth=azim,
        #         polar_angle=pola
        #     )
        # )
        beam_dict = dict(energy=beam_energy, vector=beam_vector)

        config_dict['detector'] = det_dict
        config_dict['oscillation_stage'] = stage_dict
        config_dict['beam'] = beam_dict

        return config_dict

    def cartToPixel(self, xy_det, pixels=False, apply_distortion=False):
        """
        Coverts cartesian coordinates to pixel coordinates

        Parameters
        ----------
        xy_det : array_like
            The (n, 2) vstacked array of (x, y) pairs in the reference
            cartesian frame (possibly subject to distortion).
        pixels : bool, optional
            If True, return discrete pixel indices; otherwise fractional pixel
            coordinates are returned. The default is False.
        apply_distortion : bool, optional
            If True, apply self.distortion to the input (if applicable).
            The default is False.

        Returns
        -------
        ij_det : array_like
            The (n, 2) array of vstacked (i, j) coordinates in the pixel
            reference frame where i is the (slow) row dimension and j is the
            (fast) column dimension.

        """
        xy_det = np.atleast_2d(xy_det)
        if apply_distortion and self.distortion is not None:
            xy_det = self.distortion.apply(xy_det)

        npts = len(xy_det)

        tmp_ji = xy_det - np.tile(self.corner_ul, (npts, 1))
        i_pix = -tmp_ji[:, 1] / self.pixel_size_row - 0.5
        j_pix = tmp_ji[:, 0] / self.pixel_size_col - 0.5

        ij_det = np.vstack([i_pix, j_pix]).T
        if pixels:
            # Hide any runtime warnings in this conversion. Their output values
            # will certainly be off the detector, which is fine.
            with np.errstate(invalid='ignore'):
                ij_det = np.array(np.round(ij_det), dtype=int)

        return ij_det

    def pixelToCart(self, ij_det):
        """
        Convert vstacked array or list of [i,j] pixel indices
        (or UL corner-based points) and convert to (x,y) in the
        cartesian frame {Xd, Yd, Zd}
        """
        ij_det = np.atleast_2d(ij_det)

        x = (ij_det[:, 1] + 0.5) * self.pixel_size_col + self.corner_ll[0]
        y = (
            self.rows - ij_det[:, 0] - 0.5
        ) * self.pixel_size_row + self.corner_ll[1]
        return np.vstack([x, y]).T

    def angularPixelSize(self, xy, rMat_s=None, tVec_s=None, tVec_c=None):
        """
        Notes
        -----
        !!! assumes xy are in raw (distorted) frame, if applicable
        """
        # munge kwargs
        if rMat_s is None:
            rMat_s = ct.identity_3x3
        if tVec_s is None:
            tVec_s = ct.zeros_3x1
        if tVec_c is None:
            tVec_c = ct.zeros_3x1

        # FIXME: perhaps not necessary, but safe...
        xy = np.atleast_2d(xy)

        '''
        # ---------------------------------------------------------------------
        # TODO: needs testing and memoized gradient arrays!
        # ---------------------------------------------------------------------
        # need origin arg
        origin = np.dot(rMat_s, tVec_c).flatten() + tVec_s.flatten()

        # get pixel indices
        i_crds = cellIndices(self.row_edge_vec, xy[:, 1])
        j_crds = cellIndices(self.col_edge_vec, xy[:, 0])

        ptth_grad = self.pixel_tth_gradient(origin=origin)[i_crds, j_crds]
        peta_grad = self.pixel_eta_gradient(origin=origin)[i_crds, j_crds]

        return np.vstack([ptth_grad, peta_grad]).T
        '''
        # call xrdutil function
        ang_ps = xrdutil.angularPixelSize(
            xy,
            (self.pixel_size_row, self.pixel_size_col),
            self.rmat,
            rMat_s,
            self.tvec,
            tVec_s,
            tVec_c,
            distortion=self.distortion,
            beamVec=self.bvec,
            etaVec=self.evec,
        )
        return ang_ps

    def clip_to_panel(self, xy, buffer_edges=True):
        """
        if self.roi is not None, uses it by default

        TODO: check if need shape kwarg
        TODO: optimize ROI search better than list comprehension below
        TODO: panel_buffer can be a 2-d boolean mask, but needs testing

        """
        xy = np.atleast_2d(xy)

        '''
        # !!! THIS LOGIC IS OBSOLETE
        if self.roi is not None:
            ij_crds = self.cartToPixel(xy, pixels=True)
            ii, jj = polygon(self.roi[:, 0], self.roi[:, 1],
                             shape=(self.rows, self.cols))
            on_panel_rows = [i in ii for i in ij_crds[:, 0]]
            on_panel_cols = [j in jj for j in ij_crds[:, 1]]
            on_panel = np.logical_and(on_panel_rows, on_panel_cols)
        else:
        '''
        xlim = 0.5 * self.col_dim
        ylim = 0.5 * self.row_dim
        if buffer_edges and self.panel_buffer is not None:
            if self.panel_buffer.ndim == 2:
                pix = self.cartToPixel(xy, pixels=True)

                roff = np.logical_or(pix[:, 0] < 0, pix[:, 0] >= self.rows)
                coff = np.logical_or(pix[:, 1] < 0, pix[:, 1] >= self.cols)

                idx = np.logical_or(roff, coff)

                on_panel = np.full(pix.shape[0], False)
                valid_pix = pix[~idx, :]
                on_panel[~idx] = self.panel_buffer[
                    valid_pix[:, 0], valid_pix[:, 1]
                ]
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
        elif not buffer_edges or self.panel_buffer is None:
            on_panel_x = np.logical_and(xy[:, 0] >= -xlim, xy[:, 0] <= xlim)
            on_panel_y = np.logical_and(xy[:, 1] >= -ylim, xy[:, 1] <= ylim)
            on_panel = np.logical_and(on_panel_x, on_panel_y)
        return xy[on_panel, :], on_panel

    def interpolate_nearest(self, xy, img, pad_with_nans=True):
        """
        TODO: revisit normalization in here?

        """
        is_2d = img.ndim == 2
        right_shape = img.shape[0] == self.rows and img.shape[1] == self.cols
        assert (
            is_2d and right_shape
        ), "input image must be 2-d with shape (%d, %d)" % (
            self.rows,
            self.cols,
        )

        # initialize output with nans
        if pad_with_nans:
            int_xy = np.nan * np.ones(len(xy))
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

    def interpolate_bilinear(self, xy, img, pad_with_nans=True,
                             clip_to_panel=True,
                             on_panel: Optional[np.ndarray] = None):
        """
        Interpolate an image array at the specified cartesian points.

        Parameters
        ----------
        xy : array_like, (n, 2)
            Array of cartesian coordinates in the image plane at which
            to evaluate intensity.
        img : array_like
            2-dimensional image array.
        pad_with_nans : bool, optional
            Toggle for assigning NaN to points that fall off the detector.
            The default is True.
        on_panel : np.ndarray, optional
            If you want to skip clip_to_panel() for performance reasons,
            just provide an array of which pixels are on the panel.

        Returns
        -------
        int_xy : array_like, (n,)
            The array of interpolated intensities at each of the n input
            coordinates.

        Notes
        -----
        TODO: revisit normalization in here?
        """

        is_2d = img.ndim == 2
        right_shape = img.shape[0] == self.rows and img.shape[1] == self.cols
        assert (
            is_2d and right_shape
        ), "input image must be 2-d with shape (%d, %d)" % (
            self.rows,
            self.cols,
        )

        # initialize output with nans
        if pad_with_nans:
            int_xy = np.nan * np.ones(len(xy))
        else:
            int_xy = np.zeros(len(xy))

        if on_panel is None:
            # clip away points too close to or off the edges of the detector
            xy_clip, on_panel = self.clip_to_panel(xy, buffer_edges=True)
        else:
            xy_clip = xy[on_panel]

        # grab fractional pixel indices of clipped points
        ij_frac = self.cartToPixel(xy_clip)

        # get floors/ceils from array of pixel _centers_
        # and fix indices running off the pixel centers
        # !!! notice we already clipped points to the panel!
        i_floor = cellIndices(self.row_pixel_vec, xy_clip[:, 1])
        i_floor_img = _fix_indices(i_floor, 0, self.rows - 1)

        j_floor = cellIndices(self.col_pixel_vec, xy_clip[:, 0])
        j_floor_img = _fix_indices(j_floor, 0, self.cols - 1)

        # ceilings from floors
        i_ceil = i_floor + 1
        i_ceil_img = _fix_indices(i_ceil, 0, self.rows - 1)

        j_ceil = j_floor + 1
        j_ceil_img = _fix_indices(j_ceil, 0, self.cols - 1)

        # first interpolate at top/bottom rows
        row_floor_int = (j_ceil - ij_frac[:, 1]) * img[
            i_floor_img, j_floor_img
        ] + (ij_frac[:, 1] - j_floor) * img[i_floor_img, j_ceil_img]
        row_ceil_int = (j_ceil - ij_frac[:, 1]) * img[
            i_ceil_img, j_floor_img
        ] + (ij_frac[:, 1] - j_floor) * img[i_ceil_img, j_ceil_img]

        # next interpolate across cols
        int_vals = (i_ceil - ij_frac[:, 0]) * row_floor_int + (
            ij_frac[:, 0] - i_floor
        ) * row_ceil_int
        int_xy[on_panel] = int_vals
        return int_xy

    def make_powder_rings(
        self,
        pd,
        merge_hkls=False,
        delta_tth=None,
        delta_eta=10.0,
        eta_period=None,
        eta_list=None,
        rmat_s=ct.identity_3x3,
        tvec_s=ct.zeros_3,
        tvec_c=ct.zeros_3,
        full_output=False,
        tth_distortion=None,
    ):
        """
        Generate points on Debye_Scherrer rings over the detector.

        !!! it is assuming that rmat_s is built from (chi, ome) as it the case
            for HEDM!

        Parameters
        ----------
        pd : TYPE
            DESCRIPTION.
        merge_hkls : TYPE, optional
            DESCRIPTION. The default is False.
        delta_tth : TYPE, optional
            DESCRIPTION. The default is None.
        delta_eta : TYPE, optional
            DESCRIPTION. The default is 10..
        eta_period : TYPE, optional
            DESCRIPTION. The default is None.
        eta_list : TYPE, optional
            DESCRIPTION. The default is None.
        rmat_s : TYPE, optional
            DESCRIPTION. The default is ct.identity_3x3.
        tvec_s : TYPE, optional
            DESCRIPTION. The default is ct.zeros_3.
        tvec_c : TYPE, optional
            DESCRIPTION. The default is ct.zeros_3.
        full_output : TYPE, optional
            DESCRIPTION. The default is False.
        tth_distortion : special class, optional
            Special distortion class.  The default is None.

        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if tth_distortion is not None:
            tnorms = mutil.rowNorm(np.vstack([tvec_s, tvec_c]))
            assert (
                np.all(tnorms) < ct.sqrt_epsf
            ), "If using distrotion function, translations must be zero"

        # in case you want to give it tth angles directly
        if isinstance(pd, PlaneData):
            pd = PlaneData(None, pd)
            if delta_tth is not None:
                pd.tThWidth = np.radians(delta_tth)
            else:
                delta_tth = np.degrees(pd.tThWidth)

            # !!! conversions, meh...
            del_eta = np.radians(delta_eta)

            # do merging if asked
            if merge_hkls:
                _, tth_ranges = pd.getMergedRanges(cullDupl=True)
                tth = np.average(tth_ranges, axis=1)
            else:
                tth_ranges = pd.getTThRanges()
                tth = pd.getTTh()
            tth_pm = tth_ranges - np.tile(tth, (2, 1)).T
            sector_vertices = np.vstack(
                [
                    [
                        i[0],
                        -del_eta,
                        i[0],
                        del_eta,
                        i[1],
                        del_eta,
                        i[1],
                        -del_eta,
                        0.0,
                        0.0,
                    ]
                    for i in tth_pm
                ]
            )
        else:
            # Okay, we have a array-like tth specification
            tth = np.array(pd).flatten()
            if delta_tth is None:
                raise RuntimeError(
                    "If supplying a 2theta list as first arg, "
                    + "must supply a delta_tth"
                )
            tth_pm = 0.5 * delta_tth * np.r_[-1.0, 1.0]
            tth_ranges = np.radians([i + tth_pm for i in tth])  # !!! units
            sector_vertices = np.tile(
                0.5
                * np.radians(
                    [
                        -delta_tth,
                        -delta_eta,
                        -delta_tth,
                        delta_eta,
                        delta_tth,
                        delta_eta,
                        delta_tth,
                        -delta_eta,
                        0.0,
                        0.0,
                    ]
                ),
                (len(tth), 1),
            )
            # !! conversions, meh...
            tth = np.radians(tth)
            del_eta = np.radians(delta_eta)

        # for generating rings, make eta vector in correct period
        if eta_period is None:
            eta_period = (-np.pi, np.pi)

        if eta_list is None:
            neta = int(360.0 / float(delta_eta))
            # this is the vector of ETA EDGES
            eta_edges = mapAngle(
                np.radians(delta_eta * np.linspace(0.0, neta, num=neta + 1))
                + eta_period[0],
                eta_period,
            )

            # get eta bin centers from edges
            """
            # !!! this way is probably overkill, since we have delta eta
            eta_centers = np.average(
                np.vstack([eta[:-1], eta[1:]),
                axis=0)
            """
            # !!! should be safe as eta_edges are monotonic
            eta_centers = eta_edges[:-1] + 0.5 * del_eta
        else:
            eta_centers = np.radians(eta_list).flatten()
            neta = len(eta_centers)
            eta_edges = (
                np.tile(eta_centers, (2, 1))
                + np.tile(0.5 * del_eta * np.r_[-1, 1], (neta, 1)).T
            ).T.flatten()

        # get chi and ome from rmat_s
        # !!! API ambiguity
        # !!! this assumes rmat_s was made from the composition
        # !!! rmat_s = R(Xl, chi) * R(Yl, ome)
        ome = np.arctan2(rmat_s[0, 2], rmat_s[0, 0])

        # make list of angle tuples
        angs = [
            np.vstack([i * np.ones(neta), eta_centers, ome * np.ones(neta)])
            for i in tth
        ]

        # need xy coords and pixel sizes
        valid_ang = []
        valid_xy = []
        map_indices = []
        npp = 5  # [ll, ul, ur, lr, center]
        for i_ring in range(len(angs)):
            # expand angles to patch vertices
            these_angs = angs[i_ring].T

            # push to vertices to see who falls off
            # FIXME: clipping is not checking if masked regions are on the
            #        patch interior
            patch_vertices = (
                np.tile(these_angs[:, :2], (1, npp))
                + np.tile(sector_vertices[i_ring], (neta, 1))
            ).reshape(npp * neta, 2)

            # find vertices that all fall on the panel
            # !!! not API ambiguity regarding rmat_s above
            all_xy = self.angles_to_cart(
                patch_vertices,
                rmat_s=rmat_s,
                tvec_s=tvec_s,
                rmat_c=None,
                tvec_c=tvec_c,
                apply_distortion=True,
            )

            _, on_panel = self.clip_to_panel(all_xy)

            # all vertices must be on...

            patch_is_on = np.all(on_panel.reshape(neta, npp), axis=1)
            patch_xys = all_xy.reshape(neta, 5, 2)[patch_is_on]

            # !!! Have to apply after clipping, distortion can get wonky near
            #     the edeg of the panel, and it is assumed to be <~1 deg
            # !!! The tth_ranges are NOT correct!
            if tth_distortion is not None:
                patch_valid_angs = tth_distortion.apply(
                    self.angles_to_cart(these_angs[patch_is_on, :2]),
                    return_nominal=True,
                )
                patch_valid_xys = self.angles_to_cart(
                    patch_valid_angs, apply_distortion=True
                )
            else:
                patch_valid_angs = these_angs[patch_is_on, :2]
                patch_valid_xys = patch_xys[:, -1, :].squeeze()

            # form output arrays
            valid_ang.append(patch_valid_angs)
            valid_xy.append(patch_valid_xys)
            map_indices.append(patch_is_on)
        # ??? is this option necessary?
        if full_output:
            return valid_ang, valid_xy, tth_ranges, map_indices, eta_edges
        else:
            return valid_ang, valid_xy, tth_ranges

    def map_to_plane(self, pts, rmat, tvec):
        """
        Map detctor points to specified plane.

        Parameters
        ----------
        pts : TYPE
            DESCRIPTION.
        rmat : TYPE
            DESCRIPTION.
        tvec : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        Notes
        -----
        by convention:

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
        u = np.dot(nvec_map_lab.T, tvec_map_lab) / np.dot(
            nvec_map_lab.T, pts_lab
        )

        # pts on map plane, in LAB FRAME
        pts_map_lab = np.tile(u, (3, 1)) * pts_lab

        return np.dot(rmat.T, pts_map_lab - tvec_map_lab)[:2, :].T

    def simulate_rotation_series(
        self,
        plane_data,
        grain_param_list,
        eta_ranges=[
            (-np.pi, np.pi),
        ],
        ome_ranges=[
            (-np.pi, np.pi),
        ],
        ome_period=(-np.pi, np.pi),
        chi=0.0,
        tVec_s=ct.zeros_3,
        wavelength=None,
    ):
        """
        Simulate a monochromatic rotation series for a list of grains.

        Parameters
        ----------
        plane_data : TYPE
            DESCRIPTION.
        grain_param_list : TYPE
            DESCRIPTION.
        eta_ranges : TYPE, optional
            DESCRIPTION. The default is [(-np.pi, np.pi), ].
        ome_ranges : TYPE, optional
            DESCRIPTION. The default is [(-np.pi, np.pi), ].
        ome_period : TYPE, optional
            DESCRIPTION. The default is (-np.pi, np.pi).
        chi : TYPE, optional
            DESCRIPTION. The default is 0..
        tVec_s : TYPE, optional
            DESCRIPTION. The default is ct.zeros_3.
        wavelength : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        valid_ids : TYPE
            DESCRIPTION.
        valid_hkls : TYPE
            DESCRIPTION.
        valid_angs : TYPE
            DESCRIPTION.
        valid_xys : TYPE
            DESCRIPTION.
        ang_pixel_size : TYPE
            DESCRIPTION.

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
        assert not np.any(
            np.isnan(plane_data.getTTh())
        ), "plane data exclusions incompatible with wavelength"

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
            rMat_c = make_rmat_of_expmap(gparm[:3])
            tVec_c = gparm[3:6]
            vInv_s = gparm[6:]

            # All possible bragg conditions as vstacked [tth, eta, ome]
            # for each omega solution
            angList = np.vstack(
                oscill_angles_of_hkls(
                    full_hkls[:, 1:],
                    chi,
                    rMat_c,
                    bMat,
                    wavelength,
                    v_inv=vInv_s,
                )
            )

            # filter by eta and omega ranges
            # ??? get eta range from detector?
            allAngs, allHKLs = xrdutil._filter_hkls_eta_ome(
                full_hkls, angList, eta_ranges, ome_ranges
            )
            allAngs[:, 2] = mapAngle(allAngs[:, 2], ome_period)

            # find points that fall on the panel
            det_xy, rMat_s, on_plane = xrdutil._project_on_detector_plane(
                allAngs,
                self.rmat,
                rMat_c,
                chi,
                self.tvec,
                tVec_c,
                tVec_s,
                self.distortion,
            )
            xys_p, on_panel = self.clip_to_panel(det_xy)
            valid_xys.append(xys_p)

            # filter angs and hkls that are on the detector plane
            # !!! check this -- seems unnecessary but the results of
            #     _project_on_detector_plane() can have len < the input?
            #     the output of _project_on_detector_plane has been modified to
            #     hand back the index array to remedy this JVB 2020-05-27
            if np.any(~on_plane):
                allAngs = np.atleast_2d(allAngs[on_plane, :])
                allHKLs = np.atleast_2d(allHKLs[on_plane, :])

            # grab hkls and gvec ids for this panel
            valid_hkls.append(allHKLs[on_panel, 1:])
            valid_ids.append(allHKLs[on_panel, 0])

            # reflection angles (voxel centers) and pixel size in (tth, eta)
            valid_angs.append(allAngs[on_panel, :])
            ang_pixel_size.append(self.angularPixelSize(xys_p))
        return valid_ids, valid_hkls, valid_angs, valid_xys, ang_pixel_size

    def simulate_laue_pattern(
        self,
        crystal_data,
        minEnergy=5.0,
        maxEnergy=35.0,
        rmat_s=None,
        tvec_s=None,
        grain_params=None,
        beam_vec=None,
    ):
        """ """
        if isinstance(crystal_data, PlaneData):

            plane_data = crystal_data

            # grab the expanded list of hkls from plane_data
            hkls = np.hstack(plane_data.getSymHKLs())

            # and the unit plane normals (G-vectors) in CRYSTAL FRAME
            gvec_c = np.dot(plane_data.latVecOps['B'], hkls)

            # Filter out g-vectors going in the wrong direction. `gvec_to_xy()` used
            # to do this, but not anymore.
            to_keep = np.dot(gvec_c.T, self.bvec) <= 0

            hkls = hkls[:, to_keep]
            gvec_c = gvec_c[:, to_keep]
        elif len(crystal_data) == 2:
            # !!! should clean this up
            hkls = np.array(crystal_data[0])
            bmat = crystal_data[1]
            gvec_c = np.dot(bmat, hkls)
        else:
            raise RuntimeError(
                f'argument list not understood: {crystal_data=}'
            )
        nhkls_tot = hkls.shape[1]

        # parse energy ranges
        # TODO: allow for spectrum parsing
        multipleEnergyRanges = False
        if hasattr(maxEnergy, '__len__'):
            assert len(maxEnergy) == len(
                minEnergy
            ), 'energy cutoff ranges must have the same length'
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
        xy_det = np.nan * np.ones((n_grains, nhkls_tot, 2))
        hkls_in = np.nan * np.ones((n_grains, 3, nhkls_tot))
        angles = np.nan * np.ones((n_grains, nhkls_tot, 2))
        dspacing = np.nan * np.ones((n_grains, nhkls_tot))
        energy = np.nan * np.ones((n_grains, nhkls_tot))
        for iG, gp in enumerate(grain_params):
            rmat_c = make_rmat_of_expmap(gp[:3])
            tvec_c = gp[3:6].reshape(3, 1)
            vInv_s = mutil.vecMVToSymm(gp[6:].reshape(6, 1))

            # stretch them: V^(-1) * R * Gc
            gvec_s_str = np.dot(vInv_s, np.dot(rmat_c, gvec_c))
            ghat_c_str = mutil.unitVector(np.dot(rmat_c.T, gvec_s_str))

            # project
            dpts = gvec_to_xy(
                ghat_c_str.T,
                self.rmat,
                rmat_s,
                rmat_c,
                self.tvec,
                tvec_s,
                tvec_c,
                beam_vec=beam_vec,
            )

            # check intersections with detector plane
            canIntersect = ~np.isnan(dpts[:, 0])
            npts_in = sum(canIntersect)

            if np.any(canIntersect):
                dpts = dpts[canIntersect, :].reshape(npts_in, 2)
                dhkl = hkls[:, canIntersect].reshape(3, npts_in)

                rmat_b = make_beam_rmat(beam_vec, ct.eta_vec)
                # back to angles
                tth_eta, gvec_l = xy_to_gvec(
                    dpts,
                    self.rmat,
                    rmat_s,
                    self.tvec,
                    tvec_s,
                    tvec_c,
                    rmat_b=rmat_b,
                )
                tth_eta = np.vstack(tth_eta).T

                # warp measured points
                if self.distortion is not None:
                    dpts = self.distortion.apply_inverse(dpts)

                # plane spacings and energies
                dsp = 1.0 / mutil.rowNorm(gvec_s_str[:, canIntersect].T)
                wlen = 2 * dsp * np.sin(0.5 * tth_eta[:, 0])

                # clip to detector panel
                _, on_panel = self.clip_to_panel(dpts, buffer_edges=True)

                if multipleEnergyRanges:
                    validEnergy = np.zeros(len(wlen), dtype=bool)
                    for i in range(len(lmin)):
                        in_energy_range = np.logical_and(
                            wlen >= lmin[i], wlen <= lmax[i]
                        )
                        validEnergy = validEnergy | in_energy_range
                else:
                    validEnergy = np.logical_and(wlen >= lmin, wlen <= lmax)

                # index for valid reflections
                keepers = np.where(np.logical_and(on_panel, validEnergy))[0]

                # assign output arrays
                xy_det[iG][keepers, :] = dpts[keepers, :]
                hkls_in[iG][:, keepers] = dhkl[:, keepers]
                angles[iG][keepers, :] = tth_eta[keepers, :]
                dspacing[iG, keepers] = dsp[keepers]
                energy[iG, keepers] = ct.keVToAngstrom(wlen[keepers])
        return xy_det, hkls_in, angles, dspacing, energy

    @staticmethod
    def update_memoization_sizes(all_panels):
        funcs = [
            _polarization_factor,
            _lorentz_factor,
        ]

        min_size = len(all_panels)
        return Detector.increase_memoization_sizes(funcs, min_size)

    @staticmethod
    def increase_memoization_sizes(funcs, min_size):
        for f in funcs:
            cache_info = f.cache_info()
            if cache_info['maxsize'] < min_size:
                f.set_cache_maxsize(min_size)

    def calc_physics_package_transmission(self, energy: np.floating,
                                          rMat_s: np.array,
                                          physics_package: AbstractPhysicsPackage) -> np.float64:
        """get the transmission from the physics package
        need to consider HED and HEDM samples separately
        """
        bvec = self.bvec
        sample_normal = np.dot(rMat_s, [0., 0., -1.])
        seca = 1./np.dot(bvec, sample_normal)

        tth, eta = self.pixel_angles()
        angs = np.vstack((tth.flatten(), eta.flatten(),
                          np.zeros(tth.flatten().shape))).T

        dvecs = angles_to_dvec(angs, beam_vec=bvec)

        secb = np.abs(1./np.dot(dvecs, sample_normal).reshape(self.shape))

        T_sample = self.calc_transmission_sample(
            seca, secb, energy, physics_package)
        T_window = self.calc_transmission_window(secb, energy, physics_package)

        transmission_physics_package = T_sample * T_window
        return transmission_physics_package

    def calc_transmission_sample(self, seca: np.array,
                                 secb: np.array, energy: np.floating,
                                 physics_package: AbstractPhysicsPackage) -> np.array:
        thickness_s = physics_package.sample_thickness  # in microns
        # in microns^-1
        mu_s = 1./physics_package.sample_absorption_length(energy)
        x = (mu_s*thickness_s)
        pre = 1./x/(secb - seca)
        num = np.exp(-x*seca) - np.exp(-x*secb)
        return pre * num

    def calc_transmission_window(self, secb: np.array, energy: np.floating,
                                 physics_package: AbstractPhysicsPackage) -> np.array:
        thickness_w = physics_package.window_thickness  # in microns
        # in microns^-1
        mu_w = 1./physics_package.window_absorption_length(energy)
        return np.exp(-thickness_w*mu_w*secb)

    def calc_effective_pinhole_area(self, physics_package: AbstractPhysicsPackage) -> np.array:
        """get the effective pinhole area correction
        """
        effective_pinhole_area = np.ones(self.shape)

        if (not np.isclose(physics_package.pinhole_diameter, 0)
            and not np.isclose(physics_package.pinhole_thickness, 0)):

            hod = (physics_package.pinhole_thickness /
                   physics_package.pinhole_diameter)
            bvec = self.bvec

            tth, eta = self.pixel_angles()
            angs = np.vstack((tth.flatten(), eta.flatten(),
                              np.zeros(tth.flatten().shape))).T
            dvecs = angles_to_dvec(angs, beam_vec=bvec)

            cth = -dvecs[:, 2].reshape(self.shape)
            tanth = np.tan(np.arccos(cth))
            f = hod*tanth
            f[np.abs(f) > 1.] = np.nan
            asinf = np.arcsin(f)
            effective_pinhole_area = (
                (2/np.pi) * cth * (np.pi/2 - asinf - f*np.cos(asinf)))

        return effective_pinhole_area

    def calc_transmission_generic(self,
                                  secb: np.array,
                                  thickness: np.floating,
                                  absorption_length: np.floating) -> np.array:
        mu = 1./absorption_length  # in microns^-1
        return np.exp(-thickness*mu*secb)

    def calc_transmission_phosphor(self,
                                   secb: np.array,
                                   thickness: np.floating,
                                   readout_length: np.floating,
                                   absorption_length: np.floating,
                                   energy: np.floating) -> np.array:

        f1 = absorption_length*thickness
        f2 = absorption_length*readout_length
        arg = (secb + 1/f2)
        return energy*((1.0 - np.exp(-f1*arg))/arg)

# =============================================================================
# UTILITY METHODS
# =============================================================================


def _fix_indices(idx, lo, hi):
    nidx = np.array(idx)
    off_lo = nidx < lo
    off_hi = nidx > hi
    nidx[off_lo] = lo
    nidx[off_hi] = hi
    return nidx


def _row_edge_vec(rows, pixel_size_row):
    return pixel_size_row * (0.5 * rows - np.arange(rows + 1))


def _col_edge_vec(cols, pixel_size_col):
    return pixel_size_col * (np.arange(cols + 1) - 0.5 * cols)


# FIXME find a better place for this, and maybe include loop over pixels
@numba.njit(nogil=True, cache=True)
def _solid_angle_of_triangle(vtx_list):
    norms = np.sqrt(np.sum(vtx_list * vtx_list, axis=1))
    norms_prod = norms[0] * norms[1] * norms[2]
    scalar_triple_product = np.dot(
        vtx_list[0], np.cross(vtx_list[2], vtx_list[1])
    )
    denominator = (
        norms_prod
        + norms[0] * np.dot(vtx_list[1], vtx_list[2])
        + norms[1] * np.dot(vtx_list[2], vtx_list[0])
        + norms[2] * np.dot(vtx_list[0], vtx_list[1])
    )

    return 2.0 * np.arctan2(scalar_triple_product, denominator)
