from dataclasses import dataclass, field, fields
import glob
import os
import yaml
from hexrd.hedm.preprocess.argument_classes_factory import (
    ArgumentClassesFactory,
    autoregister,
)
from hexrd.hedm.preprocess.yaml_internals import HexrdPPScriptArgumentsDumper
from typing import Any, Union, Optional, cast


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
    def known_formats(cls) -> list[str]:
        """Get all know argument formats registered so far"""
        return ArgumentClassesFactory().get_registered()

    def dump_config(self) -> str:
        """Create a yaml string representation of the values hold in this
        dataclass"""
        return yaml.dump(self, Dumper=HexrdPPScriptArgumentsDumper)

    @classmethod
    def create_default_config(cls, name: str) -> str:
        """Create argument class of type name using kwargs to set the
        dataclass values"""
        return ArgumentClassesFactory().get_args(name)().dump_config()

    @classmethod
    def create_args(cls, name: str, **kwargs: Any) -> 'HexrdPPScript_Arguments':
        """Create argument class of type name using kwargs to set the
        dataclass values"""
        return ArgumentClassesFactory().get_args(name)(**kwargs)

    @classmethod
    def load_from_config(cls, buffer: str) -> 'HexrdPPScript_Arguments':
        """Create an HexrdPPScript_Arguments instance from yaml string"""
        try:
            args = yaml.safe_load(buffer)
        except Exception as e:
            raise RuntimeError(f"Could not read config from buffer: {e}")
        return args

    def validate_arguments(self) -> None:
        pass


@dataclass
class Chess_Arguments(HexrdPPScript_Arguments):
    num_frames: int = 1440
    start_frame: int = 0
    threshold: int = 5
    ome_start: float = 0.0
    ome_end: float = 360.0
    style: str = "npz"
    output: str = "test"

    # class helpers

    # we gather all argument messages here to allow for automated help
    # generation
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
        "style": "format for saving framecache",
        "output": "output filename",
    }

    short_switches = {
        "num_frames": "n",
        "start_frame": "s",
        "threshold": "t",
        "ome_start": "o",
        "ome_end": "e",
    }

    @property
    def ostep(self) -> float:
        return (self.ome_end - self.ome_start) / float(self.num_frames)

    def validate_arguments(self) -> None:
        super().validate_arguments()
        """ Make sure that we set all the required (i.e. = None) arguments """
        collect_none = []
        for f in fields(self):
            if getattr(self, f.name) is None:
                collect_none.append(f.name)
        if len(collect_none) != 0:
            raise RuntimeError(f"Required argument are missing a value: {collect_none}")


@dataclass
@autoregister
class Eiger_Arguments(Chess_Arguments):
    yaml_tag = "!Eiger_Arguments"
    profile_name = "eiger"
    # fields
    absolute_path: Optional[str] = None
    eiger_stream_v2_threshold: str = 'threshold_1'
    eiger_stream_v2_multiplier: float = 1.0

    help_messages = {
        **Chess_Arguments.help_messages,
        "absolute_path": "absolute path to image file",
        "eiger_stream_v2_threshold": "Threshold to use for eiger-stream-v2 input file. Options are 'threshold_1', 'threshold_2', or 'man_diff', which is defined as `threshold_1 - multiplier * threshold_2`",
        "eiger_stream_v2_multiplier": "Multiplier to use for threshold setting 'man_diff'. Unused otherwise.",
    }

    short_switches = {
        **Chess_Arguments.short_switches,
    }

    def validate_arguments(self) -> None:
        super().validate_arguments()
        if not os.path.exists(self.file_name):
            raise RuntimeError(f'File {self.file_name} does not exist!')

    @property
    def file_name(self) -> str:
        value = cast(
            str, self.absolute_path
        )  # validate_arguments ensures that this is not None
        return value


@dataclass
@autoregister
class Dexelas_Arguments(Chess_Arguments):
    yaml_tag = "!Dexelas_Arguments"
    profile_name = "dexelas"
    # fields
    base_dir: Optional[str] = None
    expt_name: Optional[str] = None
    samp_name: Optional[str] = None
    scan_number: Optional[int] = None
    num_frames: int = 1441
    start_frame: int = 4  # usually 4 for normal CHESS stuff
    threshold: int = 50
    # !!!: hard coded options for each dexela for April 2017
    panel_opts: dict[str, list[list[Union[str, int]]]] = field(
        default_factory=lambda: {
            "FF1": [
                ["add-row", 1944],
                ["add-column", 1296],
                ["flip", "v"],
            ],
            "FF2": [["add-row", 1944], ["add-column", 1296], ["flip", "h"]],
        }
    )

    help_messages = {
        "base_dir": "raw data path on chess daq",
        "expt_name": "experiment name",
        "samp_name": "sample name",
        "scan_number": "ff scan number",
        "panel_opts": "detector-specific options",
        **Chess_Arguments.help_messages,
    }

    def validate_arguments(self) -> None:
        super().validate_arguments()
        check_files_exist = [os.path.exists(file_name) for file_name in self.file_names]
        if not all(check_files_exist):
            raise RuntimeError("files don't exist!")

    @property
    def file_names(self) -> list[str]:
        # validate_arguments() ensures that none of these are None
        base_dir = cast(str, self.base_dir)
        expt_name = cast(str, self.expt_name)
        samp_name = cast(str, self.samp_name)
        scan_number = cast(int, self.scan_number)
        names = glob.glob(
            os.path.join(
                base_dir,
                expt_name,
                samp_name,
                str(scan_number),
                'ff',
                '*.h5',
            )
        )
        return names
