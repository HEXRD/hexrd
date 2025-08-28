import h5py
import yaml

from .config import Config
from .loader import NumPyIncludeLoader

from hexrd.core import instrument


class Instrument(Config):
    """Handle HEDM instrument config."""

    def __init__(self, cfg, instr_file=None):
        super().__init__(cfg)
        self._configuration = instr_file
        self._max_workers = self._cfg.multiprocessing

    # Note: instrument is instantiated with a yaml dictionary; use self
    #       to instantiate classes based on this one
    @property
    def configuration(self):
        """Return the YAML config filename."""
        return self._configuration

    @property
    def hedm(self):
        """Return the HEDMInstrument class."""
        if not hasattr(self, '_hedm'):
            if self.configuration is None:
                raise RuntimeError("No instrument file was given")

            try:
                icfg = h5py.File(self.configuration, 'r')
            except OSError:
                with open(self.configuration, 'r') as f:
                    icfg = yaml.load(f, Loader=NumPyIncludeLoader)

            kwargs = {
                'instrument_config': icfg,
                'max_workers': self._max_workers,
            }
            self._hedm = instrument.HEDMInstrument(**kwargs)
        return self._hedm

    @hedm.setter
    def hedm(self, icfg_fname):
        """Set the HEDMInstrument class."""
        try:
            icfg = h5py.File(icfg_fname, 'r')
        except OSError:
            with open(icfg_fname, 'r') as f:
                icfg = yaml.load(f, Loader=NumPyIncludeLoader)

        kwargs = {
            'instrument_config': icfg,
            'max_workers': self._max_workers,
        }
        self._hedm = instrument.HEDMInstrument(**kwargs)

    @property
    def detector_dict(self):
        """Return dictionary of detectors."""
        return self.hedm.detectors
