import os
import logging
import multiprocessing as mp

from hexrd.constants import shared_ims_key
from hexrd import imageseries

from .config import Config
from .instrument import Instrument
from .findorientations import FindOrientationsConfig
from .fitgrains import FitGrainsConfig
from .material import MaterialConfig

logger = logging.getLogger('hexrd.config')


class RootConfig(Config):

    @property
    def analysis_name(self):
        return str(self.get('analysis_name', default='analysis'))

    @analysis_name.setter
    def analysis_name(self, val):
        self.set('analysis_name', val)

    @property
    def analysis_dir(self):
        return os.path.join(self.working_dir, self.analysis_name)

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
    def analysis_id(self):
        return '_'.join(
            [self.analysis_name.strip().replace(' ', '-'),
             self.material.active.strip().replace(' ', '-')]
        )

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
    def working_dir(self):
        try:
            temp = self.get('working_dir')
            if not os.path.exists(temp):
                raise IOError(
                    '"working_dir": "%s" does not exist', temp
                    )
            return temp
        except RuntimeError:
            temp = os.getcwd()
            was_dirty = self.dirty
            self.working_dir = temp
            if not was_dirty:
                self._dirty = False
            logger.info(
                '"working_dir" not specified, defaulting to "%s"' % temp
                )
            return temp

    @working_dir.setter
    def working_dir(self, val):
        val = os.path.abspath(val)
        if not os.path.isdir(val):
            raise IOError('"working_dir": "%s" does not exist' % val)
        self.set('working_dir', val)

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
