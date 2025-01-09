class ArgumentClassesFactory:
    """A factory to collect all Argument classes"""

    _creators = {}

    @classmethod
    def register(cls, klass):
        cls._creators[klass.profile_name] = klass

    @classmethod
    def get_registered(cls):
        return cls._creators.keys()

    @classmethod
    def get_args(cls, profile_name):
        creator = cls._creators.get(profile_name)
        if not creator:
            raise ValueError(format)
        return creator


def autoregister(cls):
    """decorator that registers cls with ArgumentClassesFactory"""
    ArgumentClassesFactory().register(cls)
    return cls
