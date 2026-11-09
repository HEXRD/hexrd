"""Command-line interface for the CakeData workflow"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields

import yaml

from hexrd.powder.cake_data import (
    GEOMETRY_DEFAULTS,
    CakeDataConfig,
    _field_types,
    cake_data,
    config_as_yaml,
)


DESCRIPTION = 'Create azimuthally segmented powder profiles from detector data'


def configure_parser(sub_parsers: argparse._SubParsersAction) -> None:
    parser = sub_parsers.add_parser(
        'cake-data',
        description=DESCRIPTION,
        help=DESCRIPTION,
        epilog='Relative paths resolve against the working directory.',
    )
    parser.add_argument(
        '-y', '--config', help='YAML file containing cake-data parameters'
    )
    parser.add_argument(
        '-G',
        '--generate-default-config',
        action='store_true',
        help='print a default YAML configuration and exit',
    )

    types = _field_types()
    defaults = CakeDataConfig()
    for item in fields(CakeDataConfig):
        name = item.name
        value_type = (
            json.loads if dict in types[name] and str not in types[name]
            else types[name][0]
        )
        help_text = item.metadata['help']
        if name in GEOMETRY_DEFAULTS:
            help_text += f' (instrument default: {GEOMETRY_DEFAULTS[name]})'
        elif getattr(defaults, name) is not None:
            help_text += f' (default: {getattr(defaults, name)})'

        parser.add_argument(
            f"-{item.metadata['short']}",
            f"--{name.replace('_', '-')}",
            dest=name,
            type=value_type,
            default=None,
            help=help_text,
        )

    parser.set_defaults(func=execute)


def execute(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.generate_default_config:
        print(config_as_yaml(), end='')
        return

    try:
        config = CakeDataConfig()
        if args.config is not None:
            config = CakeDataConfig.from_yaml(args.config)

        # Command-line values override the YAML configuration
        for item in fields(config):
            value = getattr(args, item.name)
            if value is not None:
                setattr(config, item.name, value)

        cake_data(config)
    except (ValueError, OSError, KeyError, RuntimeError, yaml.YAMLError) as exc:
        if getattr(args, 'debug', False):
            raise

        parser.error(str(exc))
