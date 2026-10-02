"""Command-line interface for the CakeData workflow."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields, replace
from typing import get_args, get_origin, get_type_hints

import yaml

from hexrd.powder.cake_data import CakeDataConfig, cake_data, config_as_yaml


DESCRIPTION = "Create azimuthally segmented powder profiles from detector data"


def configure_parser(sub_parsers: argparse._SubParsersAction) -> None:
    parser = sub_parsers.add_parser(
        "cake-data",
        description=DESCRIPTION,
        help=DESCRIPTION,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-y", "--config", help="YAML file containing cake-data parameters"
    )
    parser.add_argument(
        "-G",
        "--generate-default-config",
        action="store_true",
        help="print a default YAML configuration and exit",
    )
    types = get_type_hints(CakeDataConfig)
    for item in fields(CakeDataConfig):
        name = item.name
        value_type = types[name]
        if type(None) in get_args(value_type):
            value_type = get_args(value_type)[0]
        if get_origin(value_type) is dict:
            value_type = json.loads
        parser.add_argument(
            f"-{item.metadata['short']}",
            f"--{name.replace('_', '-')}",
            dest=name,
            type=value_type,
            default=None,
            help=item.metadata["help"],
        )
    parser.set_defaults(func=execute)


def execute(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.generate_default_config:
        print(config_as_yaml(), end="")
        return
    try:
        config = (
            CakeDataConfig.from_yaml(args.config) if args.config else CakeDataConfig()
        )
        overrides = {item.name: getattr(args, item.name) for item in fields(config)}
        config = replace(
            config, **{k: v for k, v in overrides.items() if v is not None}
        )
        cake_data(config)
    except (ValueError, OSError, KeyError, RuntimeError, yaml.YAMLError) as exc:
        if getattr(args, "debug", False):
            raise
        parser.error(str(exc))
