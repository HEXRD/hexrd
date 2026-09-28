"""Command-line interface for the CakeData workflow."""

from __future__ import annotations

import argparse

from hexrd.hedm.cake_data import CakeDataConfig, cake_data, config_as_json


DESCRIPTION = "Create azimuthally segmented powder profiles from detector data"


OPTIONS = (
    ("data_dir", "d", str, "raw-data directory template"),
    ("output_dir", "o", str, "directory for the output files"),
    ("sample_name", "s", str, "sample name and output file stem"),
    ("instrument", "i", str, "HEXRD instrument YAML or HDF5 file"),
    ("par_file", "p", str, "input CHESS par file"),
    ("output_par", "P", str, "output par filename"),
    ("tth_start", "b", float, "starting two-theta angle"),
    ("tth_end", "t", float, "ending two-theta angle"),
    ("cake_width", "c", float, "eta wedge width"),
    ("eta_start", "a", float, "starting eta angle"),
    ("eta_end", "z", float, "ending eta angle"),
    ("ome_start", "w", float, "starting omega angle"),
    ("ome_end", "e", float, "ending omega angle"),
    ("dome", "D", float, "omega integration width"),
    ("skip_frames", "f", int, "number of initial raw frames to skip"),
    ("refinement", "r", float, "two-theta pixel refinement factor"),
    ("start_layer", "l", int, "first par-file layer to process"),
    ("par_version", "v", int, "CHESS par-file version"),
    ("max_input_frames", "m", int, "maximum raw frames per layer"),
    ("image_pattern", "g", str, "glob used to locate an image file"),
    ("image_format", "F", str, "HEXRD image-series format"),
)


def configure_parser(sub_parsers):
    parser = sub_parsers.add_parser(
        "cake-data",
        description=DESCRIPTION,
        help=DESCRIPTION,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-j", "--config", help="JSON file containing cake-data parameters"
    )
    parser.add_argument(
        "-G",
        "--generate-default-config",
        action="store_true",
        help="print a default JSON configuration and exit",
    )
    for name, short, value_type, help_text in OPTIONS:
        parser.add_argument(
            f"-{short}",
            f"--{name.replace('_', '-')}",
            dest=name,
            type=value_type,
            default=None,
            help=help_text,
        )
    parser.set_defaults(func=execute)


def execute(args, _parser):
    if args.generate_default_config:
        print(config_as_json())
        return
    config = CakeDataConfig.from_json(args.config) if args.config else CakeDataConfig()
    config.update({name: getattr(args, name) for name, *_ in OPTIONS})
    cake_data(config)
