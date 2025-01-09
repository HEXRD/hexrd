from dataclasses import dataclass, field
import glob
import os
import yaml
from hexrd.preprocess.argument_classes_factory import (
    ArgumentClassesFactory,
    autoregister,
)
from hexrd.preprocess.yaml_internals import (
    HexrdPPScriptArgumentsDumper,
    HexrdPPScriptArgumentsSafeLoader,
)
from typing import Union


# Classes holding script arguments and their defaults based on the detector.
# Each subclass can overwrite defaults or add more options


# To add a new Argument class :
# 1. Derive from HexrdPPScript_Arguments or one of its children.
# 2. Add @autoregister decorator to the new class
# 3. Make sure the class has a unique "yaml_tag" and "profile_name".
# yaml_tag will be used in the configuration file and profile_name is
# how we refer to the argument class in the cli.


class HexrdPPScript_Arguments(yaml.YAMLObject):
    # yaml tag to help deserialiser pick the right type write It is used in the
    # YAML file to indicate the class that should be created when reading a
    # file.  Override this when deriving new classes.
    yaml_tag = "!HexrdPPScript_Arguments"
    # "profile_name" is the name to use in the command line
    # when referring to these class.
    # Override this when deriving new classes.
    profile_name = "none"

    # Allow this and derived class to be read using yaml.safe_load
    yaml_loader = yaml.SafeLoader

    @classmethod
    def known_formats(_):
        """Get all know argument formats registered so far"""
        return list(ArgumentClassesFactory().get_registered())

    def dump_config(self) -> str:
        """Create a yaml string representation of the values hold in this
        dataclass"""
        return yaml.dump(self, Dumper=HexrdPPScriptArgumentsDumper)

    @classmethod
    def create_default_config(_, name) -> str:
        """Create argument class of type name using kwargs to set the
        dataclass values"""
        return ArgumentClassesFactory().get_args(name)().dump_config()

    @classmethod
    def create_args(_, name, **kwargs):
        """Create argument class of type name using kwargs to set the
        dataclass values"""
        return ArgumentClassesFactory().get_args(name)(**kwargs)

    @classmethod
    def load_from_config(cls, buffer: str):
        """Create an HexrdPPScript_Arguments instance from yaml string"""
        try:
            args = yaml.load(buffer, Loader=HexrdPPScriptArgumentsSafeLoader)
        except Exception as e:
            raise RuntimeError(f"Could not read config from buffer: {e}")
        return args


@dataclass
@autoregister
class Eiger_Arguments(HexrdPPScript_Arguments):
    yaml_tag = "!Eiger_Arguments"
    profile_name = "eiger"
    # fields
    base_dir: str = None
    expt_name: str = None
    samp_name: str = None
    scan_number: int = None
    num_frames: int = 1440
    start_frame: int = 0
    threshold: int = 5
    ome_start: float = 0.0
    ome_end: float = 360.0
    absolute_path: str = None
    panel_opts: dict[str, any] = field(default_factory=dict)
    style: str = "npz"
    output: str = "test"

    # class helpers
    # we gather all argument messages here to allow for automated help
    # generation but also easier derivation of new dataclass arguments
    # through inheritance
    help_messages = {
        "base_dir": "raw data path on chess daq",
        "expt_name": "experiment name",
        "samp_name": "sample name",
        "scan_number": "ff scan number",
        "num_frames": "number of frames to read",
        "start_frame": "index of first data frame",
        "threshold": "threshold for frame caches",
        "ome_start": "start omega",
        "ome_end": "end omega",
        "absolute_path": "absolute path to image file",
        "panel_opts": "detector-specific options",
        "style": "format for saving framecache",
        "output": "output filename",
    }

    short_switches = {
        "num_frames": "n",
        "start_frame": "s",
        "threshold": "t",
        "ome_start": "o",
        "ome_end": "e",
        "absolute_path": "ap",
    }

    @property
    def ostep(self) -> float:
        return (self.ome_end - self.ome_start) / float(self.num_frames)

    @property
    def file_name(self) -> str:
        if self.absolute_path:
            file_name = os.path.abspath(self.absolute_path)
        else:
            file_name = glob.glob(
                os.path.join(
                    self.base_dir,
                    self.expt_name,
                    self.samp_name,
                    str(self.scan_number),
                    "ff",
                    "*.h5",
                )
            )
        if not os.path.exists(file_name):
            raise RuntimeError(f"File {file_name} does not exist!")

        return file_name

@dataclass
@autoregister
class Dexelas_Arguments(Eiger_Arguments):
    yaml_tag = "!Dexelas_Arguments"
    profile_name = "dexelas"
    # !!!: hard coded options for each dexela for April 2017
    panel_opts: dict[str, any] = field(
        default_factory=lambda: {
            "FF1": [
                ("add-row", 1944),
                ("add-column", 1296),
                ("flip", "v"),
            ],
            "FF2": [("add-row", 1944), ("add-column", 1296), ("flip", "h")],
        }
    )

    num_frames: int = 1441
    start_frame: int = 4  # usually 4 for normal CHESS stuff
    threshold: int = 50
