import yaml

class HexrdPPScriptArgumentsDumper(yaml.Dumper):
    # skip aliases, this makes sure that no references are used when using
    # immutable objects. i.e. referring to the same tuple twice it will just
    # serialize the tuple twice instead of once and using references to the
    # first instance.
    def _ignore(*args):
        return True

    ignore_aliases = _ignore
