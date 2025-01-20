import numpy as np

from hexrd.core import constants
from hexrd.core.material.crystallography import PlaneData

# TODO: Resolve extra-core-dependency
from hexrd.hedm.xrdutil.utils import (
    _project_on_detector_cylinder,
    _project_on_detector_plane,
)
from hexrd.core.utils.panel_buffer import panel_buffer_as_2d_array


class PolarView:
    """
    Create (two-theta, eta) plot of detector images.
    """

    def __init__(
        self,
        plane_data,
        instrument,
        eta_min=0.0,
        eta_max=360.0,
        pixel_size=(0.1, 0.25),
        cache_coordinate_map=False,
    ):
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
        instrument : hexrd.hedm.instrument.HEDMInstrument
            The instruemnt object.
        eta_min : scalar, optional
            The minimum azimuthal extent in degrees. The default is 0.
        eta_max : scalar, optional
            The minimum azimuthal extent in degrees. The default is 360.
        pixel_size : array_like, optional
            The angular pixels sizes (2theta, eta) in degrees.
            The default is (0.1, 0.25).
        cache_coordinate_map : bool, optional
            If True, the coordinate map will be cached so that calls to
            `warp_image()` will be *significantly* faster.
            If set to True, the caller *must* ensure that no parameters
            on the instrument that would affect polar view generation,
            and no parameters set on this class, will be modified,
            because doing so would result in an incorrect `warp_image()`.

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

        assert np.all(
            np.asarray(pixel_size) > 0
        ), 'pixel sizes must be non-negative'
        self._tth_pixel_size = pixel_size[0]
        self._eta_pixel_size = pixel_size[1]

        self._instrument = instrument

        self._coordinate_mapping = None
        self._cache_coordinate_map = cache_coordinate_map
        if cache_coordinate_map:
            # It is important to generate the cached map now, rather than
            # later, because this object might be sent to other processes
            # for parallelization, and it will be faster if the mapping
            # is already generated.
            self._coordinate_mapping = self._generate_coordinate_mapping()

    @property
    def instrument(self):
        return self._instrument

    @property
    def detectors(self):
        return self._instrument.detectors

    @property
    def cache_coordinate_map(self):
        return self._cache_coordinate_map

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
        return int(round(np.degrees(self.tth_range) / self.tth_pixel_size))

    @property
    def neta(self):
        # return int(np.ceil(np.degrees(self.eta_range)/self.eta_pixel_size))
        return int(round(np.degrees(self.eta_range) / self.eta_pixel_size))

    @property
    def shape(self):
        return (self.neta, self.ntth)

    @property
    def angular_grid(self):
        tth_vec = (
            np.radians(self.tth_pixel_size * (np.arange(self.ntth)))
            + self.tth_min
            + 0.5 * np.radians(self.tth_pixel_size)
        )
        eta_vec = (
            np.radians(self.eta_pixel_size * (np.arange(self.neta)))
            + self.eta_min
            + 0.5 * np.radians(self.eta_pixel_size)
        )
        return np.meshgrid(eta_vec, tth_vec, indexing='ij')

    @property
    def extent(self):
        ev, tv = self.angular_grid
        heps = np.radians(0.5 * self.eta_pixel_size)
        htps = np.radians(0.5 * self.tth_pixel_size)
        return [
            np.min(tv) - htps,
            np.max(tv) + htps,
            np.max(ev) + heps,
            np.min(ev) - heps,
        ]

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
        arg = (
            gvec_angs,
            detector.rmat,
            constants.identity_3x3,
            self.chi,
            detector.tvec,
            constants.zeros_3,
            self.tvec,
            detector.distortion,
        )
        if detector.detector_type == 'cylindrical':
            arg = (
                gvec_angs,
                self.chi,
                detector.tvec,
                detector.caxis,
                detector.paxis,
                detector.radius,
                detector.physical_size,
                detector.angle_extent,
                detector.distortion,
            )

        return arg, kwargs

    # =========================================================================
    #                         ####### METHODS #######
    # =========================================================================
    def warp_image(
        self, image_dict, pad_with_nans=False, do_interpolation=True
    ):
        """
        Performs the polar mapping of the input images.

        Note: this function has the potential to run much faster if
        `cache_coordinate_map` is set to `True` on the `PolarView`
        initialization.

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
        if self.cache_coordinate_map:
            # The mapping should have already been generated.
            mapping = self._coordinate_mapping
        else:
            # Otherwise, we must generate it every time
            mapping = self._generate_coordinate_mapping()

        return self._warp_image_from_coordinate_map(
            image_dict,
            mapping,
            pad_with_nans=pad_with_nans,
            do_interpolation=do_interpolation,
        )

    def _generate_coordinate_mapping(self) -> dict[str, dict[str, np.ndarray]]:
        """Generate mapping of detector coordinates to generate polar view

        This function is, in general, the most time consuming part of creating
        the polar view. Its results can be cached
        If you plan to generate the polar view many times in a row using the
        same instrument configuration, but different data files, this
        function can be called once at the beginning to generate a mapping
        of the detectors to the cartesian coordinates for each angular pixel,
        followed by warp_image_from_mapping() to create the polar view.

        This can be significantly faster than calling `warp_image()` every
        time

        The dictionary that returns has detector IDs as the first key, and
        another dict as the second key.

        The nested dict has "xypts" and "on_panel" as keys, and the
        respective arrays as the values.
        """
        angpts = self.angular_grid
        dummy_ome = np.zeros((self.ntth * self.neta))

        mapping = {}
        for detector_id, panel in self.detectors.items():
            _project_on_detector = self._func_project_on_detector(panel)
            gvec_angs = np.vstack(
                [angpts[1].flatten(), angpts[0].flatten(), dummy_ome]
            ).T

            args, kwargs = self._args_project_on_detector(gvec_angs, panel)

            xypts = np.nan * np.ones((len(gvec_angs), 2))
            valid_xys, rmats_s, on_plane = _project_on_detector(
                *args, **kwargs
            )
            xypts[on_plane, :] = valid_xys

            _, on_panel = panel.clip_to_panel(xypts, buffer_edges=True)

            mapping[detector_id] = {
                'xypts': xypts,
                'on_panel': on_panel,
            }

        return mapping

    def _warp_image_from_coordinate_map(
        self,
        image_dict: dict[str, np.ndarray],
        coordinate_map: dict[str, dict[str, np.ndarray]],
        pad_with_nans: bool = False,
        do_interpolation=True,
    ) -> np.ma.MaskedArray:

        panel_buffer_fill_value = np.nan
        img_dict = dict.fromkeys(self.detectors)
        nan_mask = None
        for detector_id, panel in self.detectors.items():
            # Make a copy since we may modify
            img = image_dict[detector_id].copy()

            # Before warping, mask out any pixels that are invalid,
            # so that they won't affect the results.
            buffer = panel_buffer_as_2d_array(panel)
            if np.issubdtype(
                type(panel_buffer_fill_value), np.floating
            ) and not np.issubdtype(img.dtype, np.floating):
                # Convert to float. This is especially important
                # for nan, since it is a float...
                img = img.astype(float)

            img[~buffer] = panel_buffer_fill_value

            xypts = coordinate_map[detector_id]['xypts']
            on_panel = coordinate_map[detector_id]['on_panel']

            if do_interpolation:
                this_img = panel.interpolate_bilinear(
                    xypts, img, pad_with_nans=pad_with_nans, on_panel=on_panel
                ).reshape(self.shape)
            else:
                this_img = panel.interpolate_nearest(
                    xypts, img, pad_with_nans=pad_with_nans
                ).reshape(self.shape)

            # It is faster to keep track of the global nans like this
            # rather than the previous way we were doing it...
            img_nans = np.isnan(this_img)
            if nan_mask is None:
                nan_mask = img_nans
            else:
                nan_mask = np.logical_and(img_nans, nan_mask)

            this_img[img_nans] = 0
            img_dict[detector_id] = this_img

        summed_img = np.sum(list(img_dict.values()), axis=0)
        return np.ma.masked_array(
            data=summed_img, mask=nan_mask, fill_value=0.0
        )

    def tth_to_pixel(self, tth):
        """
        convert two-theta value to pixel value (float) along two-theta axis
        """
        return np.degrees(tth - self.tth_min) / self.tth_pixel_size
