import yaml

from .config import Config

from hexrd import instrument


class Instrument(Config):
    """Handle HEDM instrument config."""

    def __init__(self, instr_file):
        self._configuration = instr_file
        with open(instr_file, 'r') as f:
            icfg = yaml.safe_load(f)
        self._hedm = instrument.HEDMInstrument(icfg)

    # Note: instrument is instantiated with a yaml dictionary; use self
    #       to instantiate classes based on this one
    @property
    def configuration(self):
        """Return the YAML config filename."""
        return self._configuration

    @property
    def hedm(self):
        """Return the HEDMInstrument class."""
        return self._hedm

    @hedm.setter
    def hedm(self, yml):
        """Set the HEDMInstrument class."""
        with open(yml, 'r') as f:
            icfg = yaml.safe_load(f)
        self._hedm = instrument.HEDMInstrument(icfg)

    @property
    def detector_dict(self):
        """Return dictionary of detectors."""
        return self.hedm.detectors
