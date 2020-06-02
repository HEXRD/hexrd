import os

import logging

import numpy as np

from .config import Config


logger = logging.getLogger('hexrd.config')

# TODO: set these as defaults
seed_search_methods = {
    'label': dict(filter_radius=1, threshold=1),
    'blob_log': dict(min_sigma=0.5, max_sigma=5,
                     num_sigma=10, threshold=0.01,
                     overlap=0.1),
    'blob_dog': dict(min_sigma=0.5, max_sigma=5,
                     sigma_ratio=1.6,
                     threshold=0.01, overlap=0.1)
}


class FindOrientationsConfig(Config):

    # Subsections
    @property
    def orientation_maps(self):
        return OrientationMapsConfig(self._cfg)

    @property
    def seed_search(self):
        return SeedSearchConfig(self._cfg)

    @property
    def clustering(self):
        return ClusteringConfig(self._cfg)

    @property
    def eta(self):
        return EtaConfig(self._cfg)

    @property
    def omega(self):
        return OmegaConfig(self._cfg)

    # Simple Values
    @property
    def threshold(self):
        return self._cfg.get('find_orientations:threshold', 1)

    @property
    def use_quaternion_grid(self):
        key = 'find_orientations:use_quaternion_grid'
        temp = self._cfg.get(key, None)
        if temp is None:
            return temp
        if not os.path.isabs(temp):
            temp = os.path.join(self._cfg.working_dir, temp)
        if os.path.isfile(temp):
            return temp
        raise IOError(
            '"%s": "%s" does not exist' % (key, temp)
            )

    @property
    def extract_measured_g_vectors(self):
        return self._cfg.get(
            'find_orientations:extract_measured_g_vectors',
            False
            )


class ClusteringConfig(Config):

    @property
    def algorithm(self):
        key = 'find_orientations:clustering:algorithm'
        choices = ['dbscan', 'ort-dbscan', 'sph-dbscan', 'fclusterdata']
        temp = self._cfg.get(key, 'dbscan').lower()
        if temp in choices:
            return temp
        raise RuntimeError(
            '"%s": "%s" not recognized, must be one of %s'
            % (key, temp, choices)
            )

    @property
    def completeness(self):
        key = 'find_orientations:clustering:completeness'
        temp = self._cfg.get(key, None)
        if temp is not None:
            return temp
        raise RuntimeError(
            '"%s" must be specified' % key
            )

    @property
    def radius(self):
        key = 'find_orientations:clustering:radius'
        temp = self._cfg.get(key, None)
        if temp is not None:
            return temp
        raise RuntimeError(
            '"%s" must be specified' % key
            )


class OmegaConfig(Config):

    tolerance_dflt = 0.5

    @property
    def period(self):
        # ??? maybe should get from image_series like before in v0.3.x
        key = 'find_orientations:omega:period'
        temp = self._cfg.get(key, [-180., 180])
        range = np.abs(temp[1]-temp[0])
        logger.warning('omega period specification is deprecated')
        if range != 360:
            raise RuntimeError(
                '"%s": range must be 360 degrees, range of %s is %g'
                % (key, temp, range)
                )
        return temp

    @property
    def tolerance(self):
        return self._cfg.get(
            'find_orientations:omega:tolerance',
            self.tolerance_dflt
            )


class EtaConfig(Config):

    tolerance_dflt = 0.5

    @property
    def tolerance(self):
        return self._cfg.get(
            'find_orientations:eta:tolerance',
            self.tolerance_dflt
            )

    @property
    def mask(self):
        return self._cfg.get('find_orientations:eta:mask', 5)

    @property
    def range(self):
        mask = self.mask
        if mask is None:
            return mask
        return np.array([[-90. + mask, 90. - mask], [90. + mask, 270. - mask]])


class SeedSearchConfig(Config):

    @property
    def hkl_seeds(self):
        key = 'find_orientations:seed_search:hkl_seeds'
        try:
            temp = self._cfg.get(key)
            if isinstance(temp, int):
                temp = [temp, ]
            return temp
        except:
            if self._cfg.find_orientations.use_quaternion_grid is None:
                raise RuntimeError(
                    '"%s" must be defined for seeded search' % key
                    )

    @property
    def fiber_step(self):
        return self._cfg.get(
            'find_orientations:seed_search:fiber_step',
            self._cfg.find_orientations.omega.tolerance
            )

    @property
    def method(self):
        key = 'find_orientations:seed_search:method'
        try:
            temp = self._cfg.get(key)
            assert len(temp) == 1., \
                "method must have exactly one key"
            if isinstance(temp, dict):
                method_spec = next(iter(list(temp.keys())))
                if method_spec.lower() not in seed_search_methods:
                    raise RuntimeError(
                        'invalid seed search method "%s"'
                        % method_spec
                    )
                else:
                    return temp
        except:
            raise RuntimeError(
                '"%s" must be defined for seeded search' % key
            )

    @property
    def fiber_ndiv(self):
        return int(360.0 / self.fiber_step)


class OrientationMapsConfig(Config):

    @property
    def active_hkls(self):
        temp = self._cfg.get(
            'find_orientations:orientation_maps:active_hkls', default='all'
            )
        if isinstance(temp, int):
            temp = [temp]
        if temp == 'all':
            temp = None
        return temp

    @property
    def bin_frames(self):
        return self._cfg.get(
            'find_orientations:orientation_maps:bin_frames', default=1
            )

    @property
    def file(self):
        temp = self._cfg.get('find_orientations:orientation_maps:file',
                             default=None)
        if temp is not None:
            if not os.path.isabs(temp):
                temp = os.path.join(self._cfg.working_dir, temp)
        return temp

    @property
    def threshold(self):
        return self._cfg.get('find_orientations:orientation_maps:threshold')
