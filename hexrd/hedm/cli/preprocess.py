import dataclasses
import logging
from hexrd.hedm.preprocess.profiles import HexrdPPScript_Arguments
from hexrd.hedm.preprocess.preprocessors import preprocess
from dataclasses import fields
import json
import copy
from typing import get_origin, get_args, Union

import argparse

logger = logging.getLogger(__name__)

_description = 'Preprocess detector images'
_help = "Preprocess data from detector and attach metadata"


def configure_parser(sub_parsers: argparse._SubParsersAction) -> None:
    parser = sub_parsers.add_parser(
        'preprocess', description=_description, help=_help
    )

    subparsers = parser.add_subparsers(
        dest="profile", required=True, help="Select detector profile"
    )

    for fmt in HexrdPPScript_Arguments.known_formats():
        klass = HexrdPPScript_Arguments.create_args(fmt)
        add_profile_subparser(subparsers, fmt, klass)

    parser.set_defaults(func=execute)


def execute(args: argparse.Namespace, _: argparse.ArgumentParser) -> None:
    kwargs, extra = _remove_non_dataclass_args(vars(args))

    if extra["generate_default_config"]:
        logger.info(HexrdPPScript_Arguments.create_default_config(extra["profile"]))
    else:
        if extra["config"] is not None:
            args_object = HexrdPPScript_Arguments.load_from_config(
                extra["config"].read()
            )
        else:
            args_object = HexrdPPScript_Arguments.create_args(
                extra["profile"], **kwargs
            )
        args_object.validate_arguments()
        preprocess(args_object)


def add_profile_subparser(
    subparsers: argparse._SubParsersAction,
    name: str,
    klass: HexrdPPScript_Arguments,
) -> None:
    """Create a subparser with the options related to detector `name` using
    `klass` to retrieve default values"""

    subparser = subparsers.add_parser(
        name,
        help=f"Preprocess data for detector profile: {name}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    help_messages = getattr(klass, "help_messages")
    short_switches = getattr(klass, "short_switches")

    for field in fields(klass):
        default_value = None
        if field.default_factory != dataclasses.MISSING:
            default_value = field.default_factory()
        else:
            default_value = field.default
        tp, default_value = _get_supported_type(field.type, default_value)
        # fields with default value = None are treated as posiitonal
        if default_value is not None:
            switches = [f"--{field.name}"]
            if field.name in short_switches:
                switch = short_switches[field.name]
                switches.insert(0, f"-{switch}")
            subparser.add_argument(
                *switches,
                type=tp,
                default=default_value,
                help=help_messages[field.name],
            )
        else:
            subparser.add_argument(
                field.name,
                type=tp,
                help=help_messages[field.name],
            )

    subparser.add_argument(
        "--generate-default-config",
        action="store_true",
        help="Generate config file with default values",
    )
    subparser.add_argument(
        "--config",
        type=argparse.FileType("r"),
        required=False,
        help="Read arguments from .pp config file",
    )


def _remove_non_dataclass_args(args_dict: dict) -> tuple[dict, dict]:
    """Remove args that do not belong to any dataclass. These are standard args
    we manually inserted or application args to allow the rest
    of the arguments to initialize dataclass"""

    # keep orignal intact
    args = copy.deepcopy(args_dict)

    # these are defined in main.py
    # if we ever add more we will need to update this list
    hexrd_app_args = ['debug', 'inst_profile', 'cmd', 'func']
    for key in hexrd_app_args:
        del args[key]

    # extra are added by the preprocess subparser
    extra = {}
    for key in ["profile", "config", "generate_default_config"]:
        v = args.get(key, None)
        extra[key] = v
        del args[key]
    return args, extra


def _get_supported_type(tp, default_value=None):
    """Replace any type not supported by argparse in the command line with an
    alternative. Also, return the new default value in the appropriate format.

    For now we just replace dictionaries with json strings this
    allows to pass a dict as '{"key1":value1, "key2":value2}'
    """
    # second condition is required in case the dataclass field is defined using
    # members of the typing module.
    if tp is dict or get_origin(tp) is dict:
        return json.loads, f"'{json.dumps(default_value)}'"
    elif is_optional(tp):
        return get_args(tp)[0], None
    else:
        return tp, default_value


def is_optional(field):
    return get_origin(field) is Union and type(None) in get_args(field)
