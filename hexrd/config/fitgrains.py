import logging
import os

from .config import Config
from .utils import get_exclusion_parameters


logger = logging.getLogger('hexrd.config')


class ToleranceConfig(Config):

    @property
    def eta(self):
        temp = self._cfg.get('fit_grains:tolerance:eta')
        if isinstance(temp, (int, float)):
            temp = [temp, temp]
        return temp

    @property
    def omega(self):
        temp = self._cfg.get('fit_grains:tolerance:omega')
        if isinstance(temp, (int, float)):
            temp = [temp, temp]
        return temp

    @property
    def tth(self):
        temp = self._cfg.get('fit_grains:tolerance:tth')
        if isinstance(temp, (int, float)):
            temp = [temp, temp]
        return temp


class FitGrainsConfig(Config):

    def _active_material_str(self):
        return self.parent.material.active.strip().replace(' ', '-')

    def __init__(self, cfg):
        super().__init__(cfg)
        re, ep = get_exclusion_parameters(self._cfg, 'fit_grains')
        self._reset_exclusions, self._exclusion_parameters = re, ep

    @property
    def logfile(self):
        """Name of log file"""
        return self.parent.analysis_dir / "fit-grains.log"

    @property
    def grains_file(self):
        return self.parent.analysis_dir / "grains.out"

    @property
    def reset_exclusions(self):
        """Flag to use hkls saved in the material"""
        return self._reset_exclusions

    @property
    def exclusion_parameters(self):
        return self._exclusion_parameters

    @property
    def do_fit(self):
        return self._cfg.get('fit_grains:do_fit', True)

    @property
    def estimate(self):
        key = 'fit_grains:estimate'
        temp = self._cfg.get(key, None)
        if temp is None:
            return temp
        if not os.path.isabs(temp):
            temp = os.path.join(self._cfg.working_dir, temp)
        if os.path.isfile(temp):
            return temp
        logger.warning('"%s": "%s" does not exist', key, temp)

    @property
    def npdiv(self):
        return self._cfg.get('fit_grains:npdiv', 2)

    @property
    def threshold(self):
        return self._cfg.get('fit_grains:threshold')

    @property
    def tolerance(self):
        return ToleranceConfig(self._cfg)

    @property
    def refit(self):
        key = 'fit_grains:refit'
        temp = self._cfg.get(key, None)
        if temp is None:
            return temp
        else:
            if not isinstance(temp, (int, float, list)):
                raise RuntimeError(
                    '"%s" must be None, a scalar, or a list, got "%s"'
                    % (key, temp)
                    )
            if isinstance(temp, (int, float)):
                temp = [temp, temp]
            return temp

    """
    TODO: evaluate the need for this
    """
    @property
    def skip_on_estimate(self):
        key = 'fit_grains:skip_on_estimate'
        temp = self._cfg.get(key, False)
        if temp in (True, False):
            return temp
        raise RuntimeError(
            '"%s" must be true or false, got "%s"' % (key, temp)
            )

    @property
    def fit_only(self):
        key = 'fit_grains:fit_only'
        temp = self._cfg.get(key, False)
        if temp in (True, False):
            return temp
        raise RuntimeError(
            '"%s" must be true or false, got "%s"' % (key, temp)
            )

    @property
    def tth_max(self):
        key = 'fit_grains:tth_max'
        temp = self._cfg.get(key, True)
        if isinstance(temp, bool):
            return temp
        if isinstance(temp, (int, float)):
            if temp > 0:
                return temp
        raise RuntimeError(
            '"%s" must be > 0, true, or false, got "%s"' % (key, temp)
            )

    @property
    def spots_data_file(self):
        if getattr(self, '_spots_data_file', None):
            return self._spots_data_file

        key = 'fit_grains:spots_data_file'
        return self._cfg.get(key, None)

    @spots_data_file.setter
    def spots_data_file(self, filepath):
        self._spots_data_file = filepath
