import os
from pathlib import Path
import logging
import multiprocessing as mp

from hexrd.core.constants import shared_ims_key
from hexrd.core import imageseries

from ...core.config.config import Config
from .instrument import Instrument
from .findorientations import FindOrientationsConfig
from .fitgrains import FitGrainsConfig
from ...core.config.material import MaterialConfig

logger = logging.getLogger('hexrd.config')


class RootConfig(Config):

    @property
    def working_dir(self):
        """Working directory, either specified in file or current directory

        If the directory is not specified in the config file, then it will
        default to the current working directory. If it is specified, the
        directory must exist, or it will throw an IOError.
        """
        wdir = Path(self.get('working_dir', default=Path.cwd()))
        if not wdir.exists():
            raise IOError(f'"working_dir": {str(wdir)} does not exist')
        return wdir

    @working_dir.setter
    def working_dir(self, val):
        val = Path(val)
        if not val.is_dir():
            raise IOError('"working_dir": "%s" does not exist' % str(val))
        self.set('working_dir', val)

    @property
    def analysis_name(self):
        """Name of the analysis

        This will be used to set up the output directory. The name can
        contain slash ("/") characters, which will generate a subdirectory
        structure in the `analysis_dir`.
        """
        return str(self.get('analysis_name', default='analysis'))

    @analysis_name.setter
    def analysis_name(self, val):
        self.set('analysis_name', val)

    @property
    def analysis_dir(self):
        """Analysis directory, where output files go

        The name is derived from `working_dir` and `analysis_name`. This
        property returns a Path object. The directory and any intermediate
        directories can be created with the `mkdir()` method, e.g.

        >>> analysis_dir.mkdir(parents=True, exist_ok=True)
        """
        adir = Path(self.working_dir) / self.analysis_name
        return adir

    @property
    def analysis_id(self):
        return '_'.join(
            [self.analysis_name.strip().replace(' ', '-'),
             self.material.active.strip().replace(' ', '-')]
        )

    @property
    def new_file_placement(self):
        """Use new file placements for find-orientations and fit-grains

        The new file placement rules put several files in the `analysis_dir`
        instead of the `working_dir`.
        """
        return self.get('new_file_placement', default=False)

    @property
    def find_orientations(self):
        return FindOrientationsConfig(self)

    @property
    def fit_grains(self):
        if not hasattr(self, "_fitgrain_config"):
            self._fitgrain_config = FitGrainsConfig(self)
        return self._fitgrain_config

    @property
    def instrument(self):
        if not hasattr(self, '_instr_config'):
            instr_file = self.get('instrument', None)
            if instr_file is not None:
                instr_file = self.check_filename(instr_file, self.working_dir)
            self._instr_config = Instrument(self, instr_file)
        return self._instr_config

    @instrument.setter
    def instrument(self, instr_config):
        self._instr_config = instr_config

    @property
    def material(self):
        if not hasattr(self, '_material_config'):
            self._material_config = MaterialConfig(self)

        if self.instrument.configuration is not None:
            # !!! must make matl beam energy consistent with the instrument
            beam_energy = self.instrument.hedm.beam_energy
            self._material_config.beam_energy = beam_energy

        return self._material_config

    @material.setter
    def material(self, material_config):
        self._material_config = material_config

    @property
    def multiprocessing(self):
        # determine number of processes to run in parallel
        multiproc = self.get('multiprocessing', default=-1)
        ncpus = mp.cpu_count()
        if multiproc == 'all':
            res = ncpus
        elif multiproc == 'half':
            temp = ncpus // 2
            res = temp if temp else 1
        elif isinstance(multiproc, int):
            if multiproc >= 0:
                if multiproc > ncpus:
                    logger.warning(
                        'Resuested %s processes, %d available',
                        multiproc, ncpus
                        )
                    res = ncpus
                else:
                    res = multiproc if multiproc else 1
            else:
                temp = ncpus + multiproc
                if temp < 1:
                    logger.warning(
                        'Cannot use less than 1 process, requested %d of %d',
                        temp, ncpus
                        )
                    res = 1
                else:
                    res = temp
        else:
            temp = ncpus - 1
            logger.warning(
                "Invalid value %s for multiprocessing",
                multiproc
                )
            res = temp
        return res

    @multiprocessing.setter
    def multiprocessing(self, val):
        isint = isinstance(val, int)
        if val in ('half', 'all', -1):
            self.set('multiprocessing', val)
        elif (isint and val >= 0 and val <= mp.cpu_count()):
            self.set('multiprocessing', int(val))
        else:
            raise RuntimeError(
                '"multiprocessing": must be 1:%d, got %s'
                % (mp.cpu_count(), val)
                )

    @property
    def image_series(self):
        """Return the imageseries dictionary."""
        if not hasattr(self, '_image_dict'):
            self._image_dict = dict()
            fmt = self.get('image_series:format')
            imsdata = self.get('image_series:data')
            for ispec in imsdata:
                fname = self.check_filename(ispec['file'], self.working_dir)
                args = ispec['args']
                ims = imageseries.open(fname, fmt, **args)
                oms = imageseries.omega.OmegaImageSeries(ims)
                try:
                    panel = ispec['panel']
                    if isinstance(panel, (tuple, list)):
                        panel = '_'.join(panel)
                    elif panel is None:
                        panel = shared_ims_key
                except(KeyError):
                    try:
                        panel = oms.metadata['panel']
                    except(KeyError):
                        panel = shared_ims_key
                self._image_dict[panel] = oms

        return self._image_dict

    @image_series.setter
    def image_series(self, ims_dict):
        self._image_dict = ims_dict
