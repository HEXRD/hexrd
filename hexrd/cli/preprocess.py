import dataclasses
from hexrd.preprocess.profiles import HexrdPPScript_Arguments
from hexrd.preprocess.preprocessors import preprocess
from dataclasses import fields

import argparse

_description = 'Preprocess detector images'
_help = "Preprocess data from detector and attach metadata"


def configure_parser(sub_parsers):
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


def execute(args, parser):
    kwargs, extra = _remove_non_dataclass_args(vars(args))

    if extra["generate_default_config"]:
        s = HexrdPPScript_Arguments.create_default_config(extra["profile"])
        print(s)
    else:
        if extra["config"] is not None:
            args_object = HexrdPPScript_Arguments.load_from_config(
                extra["config"].read()
            )
        else:
            args_object = HexrdPPScript_Arguments.create_args(
                extra["profile"], **kwargs
            )
        preprocess(args_object)


def add_profile_subparser(subparsers, name, klass):
    """Create a subparser with the options related to detector `name` using
    `klass` to retrieve default values"""

    subparser = subparsers.add_parser(
        name,
        help=f"{name} help",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    help_messages = getattr(klass, "help_messages")
    short_switches = getattr(klass, "short_switches")

    for field in fields(klass):
        switches = [f"--{field.name}"]
        if field.name in short_switches:
            switch = short_switches[field.name]
            switches.insert(0, f"-{switch}")
        default_value = None
        if field.default_factory != dataclasses.MISSING:
            default_value = field.default_factory()
        else:
            default_value = field.default

        subparser.add_argument(
            *switches,
            type=field.type,
            default=default_value,
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


def _remove_non_dataclass_args(args_dict):
    """Remove args that do not belong to any dataclass. These are standard args
    we manually inserted and now remove to allow the rest of the arguments to
    initialize dataclass"""

    extra = {}
    for key in ["profile", "config", "generate_default_config"]:
        v = args_dict.get(key, None)
        extra[key] = v
        if v is not None:
            del args_dict[key]
    return args_dict, extra
