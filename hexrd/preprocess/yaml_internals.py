import yaml

# Create custom yaml loader and  dumper that know about all types in
# ArgumentsClasses and can handle python tuples.


# custom loader that can deserialize python tuples tagged as such while using safeloader
class HexrdScriptArgumentsSafeLoader(yaml.SafeLoader):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        def construct_python_tuple(self, node):
            return tuple(self.construct_sequence(node))

        tag = "tag:yaml.org,2002:python/tuple"
        if tag not in HexrdScriptArgumentsSafeLoader.yaml_constructors:
            HexrdScriptArgumentsSafeLoader.add_constructor(
                "tag:yaml.org,2002:python/tuple",
                construct_python_tuple,
            )


class HexrdScriptArgumentsDumper(yaml.Dumper):
    # skip aliases, this makes sure that no references are used when using
    # immutable objects. i.e. referring to the same tuple twice it will just
    # serialize the tuple twice instead of once and using references to the first
    # instance.
    ignore_aliases = lambda *args: True
