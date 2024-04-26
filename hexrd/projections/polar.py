import numpy as np

from hexrd import constants
from hexrd.material.crystallography import PlaneData
from hexrd.xrdutil.utils import (
    _project_on_detector_cylinder,
    _project_on_detector_plane,
)


class PolarView:
    """
    Create (two-theta, eta) plot of detector images.
    """

    def __init__(self, plane_data, instrument,
                 eta_min=0., eta_max=360.,
                 pixel_size=(0.1, 0.25)):
        """
        Instantiates a PolarView class.

        Parameters
        ----------
        plane_data : PlaneData object or array_like
            Specification for the 2theta ranges.  If a PlaneData instance, the
            min/max is set to the lowermost/uppermost values from the tThTanges
            as defined but the active hkls and the tThWidth (or strainMag).
            If array_like, the input must be (2, ) specifying the [min, maz]
            2theta values explicitly in degrees.
        instrument : hexrd.instrument.HEDMInstrument
            The instruemnt object.
        eta_min : scalar, optional
            The minimum azimuthal extent in degrees. The default is 0.
        eta_max : scalar, optional
            The minimum azimuthal extent in degrees. The default is 360.
        pixel_size : array_like, optional
            The angular pixels sizes (2theta, eta) in degrees.
            The default is (0.1, 0.25).

        Returns
        -------
        None.

        Notes
        -----
        Currently there is no check on the eta range, which should be strictly
        less than 360 degrees.

        """
        # tth stuff
        if isinstance(plane_data, PlaneData):
            tth_ranges = plane_data.getTThRanges()
            self._tth_min = np.min(tth_ranges)
            self._tth_max = np.max(tth_ranges)
        else:
            self._tth_min = np.radians(plane_data[0])
            self._tth_max = np.radians(plane_data[1])

        # etas
        self._eta_min = np.radians(eta_min)
        self._eta_max = np.radians(eta_max)

        assert np.all(np.asarray(pixel_size) > 0), \
            'pixel sizes must be non-negative'
        self._tth_pixel_size = pixel_size[0]
        self._eta_pixel_size = pixel_size[1]

        self._instrument = instrument

    @property
    def instrument(self):
        return self._instrument

    @property
    def detectors(self):
        return self._instrument.detectors

    @property
    def tvec(self):
        return self._instrument.tvec

    @property
    def chi(self):
        return self._instrument.chi

    @property
    def tth_min(self):
        return self._tth_min

    @tth_min.setter
    def tth_min(self, x):
        assert x < self.tth_max, f'tth_min must be < tth_max ({self._tth_max})'
        self._tth_min = x

    @property
    def tth_max(self):
        return self._tth_max

    @tth_max.setter
    def tth_max(self, x):
        assert x > self.tth_min, f'tth_max must be < tth_min ({self._tth_min})'
        self._tth_max = x

    @property
    def tth_range(self):
        return self.tth_max - self.tth_min

    @property
    def tth_pixel_size(self):
        return self._tth_pixel_size

    @tth_pixel_size.setter
    def tth_pixel_size(self, x):
        assert x > 0, "pixel size must be non-negative"
        self._tth_pixel_size = float(x)

    @property
    def eta_min(self):
        return self._eta_min

    @eta_min.setter
    def eta_min(self, x):
        assert x < self.eta_max, f'eta_min must be < eta_max ({self.eta_max})'
        self._eta_min = x

    @property
    def eta_max(self):
        return self._eta_max

    @eta_max.setter
    def eta_max(self, x):
        assert x > self.eta_min, f'eta_max must be > eta_min ({self.eta_min})'
        self._eta_max = x

    @property
    def eta_range(self):
        return self.eta_max - self.eta_min

    @property
    def eta_pixel_size(self):
        return self._eta_pixel_size

    @eta_pixel_size.setter
    def eta_pixel_size(self, x):
        assert x > 0, "pixel size must be non-negative"
        self._eta_pixel_size = float(x)

    @property
    def ntth(self):
        # return int(np.ceil(np.degrees(self.tth_range)/self.tth_pixel_size))
        return int(round(np.degrees(self.tth_range)/self.tth_pixel_size))

    @property
    def neta(self):
        # return int(np.ceil(np.degrees(self.eta_range)/self.eta_pixel_size))
        return int(round(np.degrees(self.eta_range)/self.eta_pixel_size))

    @property
    def shape(self):
        return (self.neta, self.ntth)

    @property
    def angular_grid(self):
        tth_vec = np.radians(self.tth_pixel_size*(np.arange(self.ntth)))\
            + self.tth_min + 0.5*np.radians(self.tth_pixel_size)
        eta_vec = np.radians(self.eta_pixel_size*(np.arange(self.neta)))\
            + self.eta_min + 0.5*np.radians(self.eta_pixel_size)
        return np.meshgrid(eta_vec, tth_vec, indexing='ij')

    @property
    def extent(self):
        ev, tv = self.angular_grid
        heps = np.radians(0.5*self.eta_pixel_size)
        htps = np.radians(0.5*self.tth_pixel_size)
        return [np.min(tv) - htps, np.max(tv) + htps,
                np.max(ev) + heps, np.min(ev) - heps]

    def _func_project_on_detector(self, detector):
        '''
        helper function to decide which function to
        use for mapping of g-vectors to detector
        '''
        if detector.detector_type == 'cylindrical':
            return _project_on_detector_cylinder
        else:
            return _project_on_detector_plane

    def _args_project_on_detector(self, gvec_angs, detector):
        kwargs = {'beamVec': detector.bvec}
        arg = (gvec_angs,
               detector.rmat,
               constants.identity_3x3,
               self.chi,
               detector.tvec,
               constants.zeros_3,
               self.tvec,
               detector.distortion)
        if detector.detector_type == 'cylindrical':
            arg = (gvec_angs,
                   self.chi,
                   detector.tvec,
                   detector.caxis,
                   detector.paxis,
                   detector.radius,
                   detector.physical_size,
                   detector.angle_extent,
                   detector.distortion)

        return arg, kwargs

    # =========================================================================
    #                         ####### METHODS #######
    # =========================================================================
    def warp_image(self, image_dict, pad_with_nans=False,
                   do_interpolation=True):
        """
        Performs the polar mapping of the input images.

        Parameters
        ----------
        image_dict : dict
            DIctionary of image arrays, 1 per detector.

        Returns
        -------
        wimg : numpy.ndarray
            The composite polar mapping of the detector images.  Dimensions are
            self.shape

        Notes
        -----
        Tested ouput using Maud.

        """

        gvec_angs = np.vstack([
            self.angular_grid[1].flatten(),
            self.angular_grid[0].flatten(),
            np.zeros((self.ntth*self.neta))]).T
        xypts = np.empty((len(gvec_angs), 2))

        img_dict = dict.fromkeys(self.detectors)
        for detector_id, panel in self.detectors.items():
            _project_on_detector = self._func_project_on_detector(panel)

            args, kwargs = self._args_project_on_detector(gvec_angs, panel)

            valid_xys, _, on_plane = _project_on_detector(*args, **kwargs)
            xypts[on_plane, :] = valid_xys

            if do_interpolation:
                this_img = panel.interpolate_bilinear(
                    xypts, image_dict[detector_id],
                    pad_with_nans=pad_with_nans).reshape(self.shape)
            else:
                this_img = panel.interpolate_nearest(
                    xypts, image_dict[detector_id],
                    pad_with_nans=pad_with_nans).reshape(self.shape)
            nan_mask = np.isnan(this_img)
            img_dict[detector_id] = np.ma.masked_array(
                data=this_img, mask=nan_mask, fill_value=0.
            )

        return np.ma.sum(np.ma.stack(img_dict.values()), axis=0)

    def tth_to_pixel(self, tth):
        """
        convert two-theta value to pixel value (float) along two-theta axis
        """
        return np.degrees(tth - self.tth_min)/self.tth_pixel_size
