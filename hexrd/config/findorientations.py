import os
from pathlib import Path
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

    _find_ori = "find-orientations"

    def _active_material_str(self):
        return self.parent.material.active.strip().replace(' ', '-')

    @property
    def logfile(self):
        """Name of log file"""
        actmat = self._active_material_str()
        return self.parent.analysis_dir / f"{self._find_ori}-{actmat}.log"

    def accepted_orientations_file(self, to_load=False):
        """Path of accepted_orientations file

        PARAMETERS
        ----------
        to_load: bool, default = False
           if True, check whether possible file names exist

        RETURNS
        -------
        Path or None:
            if to_load is False, it returns the path to write the file; if
            True, it checks for existing files (new name first) and returns
            an existing file name or None
        """
        actmat = self._active_material_str()
        newname = f"accepted-orientations-{actmat}.dat"
        aof_new = self.parent.analysis_dir / newname

        if not to_load:
            fname = aof_new
        else:
            fname = None
            if aof_new.exists():
                fname = aof_new
            else:
                oldname = (
                    'accepted_orientations_%s.dat' % self.parent.analysis_id
                )
                aof_old = self.parent.working_dir / oldname
                if aof_old.exists():
                    fname = aof_old

        return fname

    @property
    def grains_file(self):
        return self.parent.analysis_dir / "grains.out"

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
        # FIXME: this is deprecated and now set from the imageseries
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
    def eta_step(self):
        return self._cfg.get(
            'find_orientations:orientation_maps:eta_step', default=0.25
            )

    def file(self, to_load=False):
        """Path of eta-omega maps file

        This function implements the newer file placement for the eta-omega
        maps file in the analysis directory, instead of the old placement in
        the working directory. New files will be saved using the new
        placement. For loading existing files, the new placement will be
        checked first, and then the old placement. This ensures that old
        data sets will still find existing map files.

        PARAMETERS
        ----------
        to_load: bool, default = False
           if True, check whether possible file names exist

        RETURNS
        -------
        Path or None:
            if `to_load` is False, it returns the path to write the file; if
            True, it checks for existing files (new name first) and returns
            the name of an existing file or None
        """
        #
        # Get file name.  Because users often set file to "null", which
        # returns a valid value of None, we have to check twice before setting
        # it to the default value.
        #
        actmat = self.parent.find_orientations._active_material_str()
        dflt = f"eta-ome-maps-{actmat}.npz"
        temp = self._cfg.get(
            'find_orientations:orientation_maps:file',
            default=None
        )
        if temp is None:
            ome_new = self.parent.analysis_dir / dflt
            old_name = '_'.join([self.parent.analysis_id, "eta-ome_maps.npz"])
            ome_old = Path(self.parent.working_dir) / old_name

        else:
            # User specified value.
            if temp.suffix != ".npz":
                temp = temp.with_suffix(".npz")
            ome_new = ome_old = temp

        # Now, we whether to use the old or the new and set the correct
        # directory.

        if ome_new.is_absolute():
            ome_path = ome_new
        else:
            if not to_load:
                ome_path = ome_new
            else:
                if ome_new.exists():
                    ome_path = ome_new
                elif ome_old.exists():
                    ome_path = ome_old
                else:
                    ome_path = None

        return ome_path

    @property
    def scored_orientations_file(self):
        root = self.parent
        adir = root.analysis_dir
        actmat = root.find_orientations._active_material_str()
        return Path(adir) / f'scored-orientations-{actmat}.npz'

    @property
    def threshold(self):
        return self._cfg.get('find_orientations:orientation_maps:threshold')

    @property
    def filter_maps(self):
        return self._cfg.get('find_orientations:orientation_maps:filter_maps',
                             default=False)
