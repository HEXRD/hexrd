import yaml
from hexrd.preprocess.argument_classes_factory import ArgumentClassesFactory
from hexrd.preprocess.yaml_internals import HexrdScriptArgumentsDumper, HexrdScriptArgumentsSafeLoader


# Classes holding script arguments and their defaults based on the detector.
# Each subclass can overwrite defaults or add more options


# To add a new Argument class :
# 1. Derive from HexrdScript_Arguments or one of its children.
# 2. Add @autoregister decorator to the new class
# 3. Make sure the class has a unique "yaml_tag" and "profile_name".
# yaml_tag will be used in the configuration file and profile_name is
# how we refer to the argument class in the cli.

class HexrdScript_Arguments(yaml.YAMLObject):
    # yaml tag to help deserialiser pick the right type write It is used in the
    # YAML file to indicate the class that should be created when reading a
    # file.  Override this when deriving new classes.
    yaml_tag = "!HexrdScript_Arguments"
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
        """Create a yaml string representation of the values hold in this dataclass"""
        return yaml.dump(self, Dumper=HexrdScriptArgumentsDumper)

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
        """Create an HexrdScript_Arguments instance from yaml string"""
        try:
            args = yaml.load(buffer, Loader=HexrdScriptArgumentsSafeLoader)
        except:
            raise RuntimeError("Could not read config from buffer")
        return args
