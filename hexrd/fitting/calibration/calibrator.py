from abc import ABC, abstractmethod


class Calibrator(ABC):
    """Base class for calibrators for use by InstrumentCalibrator"""

    @property
    @abstractmethod
    def type(self):
        """The type of the calibrator"""
        pass

    @abstractmethod
    def create_lmfit_params(self, current_params):
        """Create lmfit parameters that this class will use and return them

        current_params is passed only so that this class can ensure that
        it does not produce any name conflicts with existing parameters.

        It is recommended to save the names of the parameters created,
        for example, `self.param_names`, to remember them when updating
        in the future.

        For example, the powder calibrator creates lmfit parameters
        for the lattice parameters. The Laue calibrator creates lmfit
        parameters for crystal parameters.
        """
        pass

    @abstractmethod
    def update_from_lmfit_params(self, params_dict):
        """Update any needed properties based on the params_dict

        The entries in the params_dict should have the same names as those
        that were created via `create_lmfit_params()`. Update any local
        parameters based upon the new contents of the params_dict.

        For example, the powder calibrator will update the lattice parameters
        on the material. The Laue calibrator will update crystal parameters.
        """
        pass

    @abstractmethod
    def residual(self, calibration_data=None):
        """Compute a residual using the calibration data.

        Calibration data may have already been stored as an attribute on
        the calibration class instead, in which case, calibration_data can
        be `None`.
        """
        pass

    @property
    @abstractmethod
    def calibration_picks(self):
        """A way to retrieve picks in calibration format

        Often, the calibrators store the picks in a different format
        internally. But the calibrators must have a way to set/get
        from the calibration_picks format. The calibration_picks
        format looks like this:

        {
            "det_key1": {
                "hkl1": picks1,
                "hkl2": picks2,
                ...
            },
            "det_key2": ...
        }

        Where "det_key" is the detector key. "hkl" is a space-delimited
        string of the hkl. And "picks" are either a list of points (powder)
        or a single point (laue). The picks are in cartesian coordinates.
        """
        pass

    @calibration_picks.setter
    @abstractmethod
    def calibration_picks(self, val):
        """Setter for calibration_picks. See getter docs for details."""
        pass
