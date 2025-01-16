import hexrd.preprocess.profiles as profiles
from typing import Type


class ArgumentClassesFactory:
    """A factory to collect all Argument classes"""

    _creators: dict[str, Type["profiles.HexrdPPScript_Arguments"]] = {}

    @classmethod
    def register(cls, klass: Type["profiles.HexrdPPScript_Arguments"]) -> None:
        cls._creators[klass.profile_name] = klass

    @classmethod
    def get_registered(cls) -> list[str]:
        return list(cls._creators.keys())

    @classmethod
    def get_registered_types(
        cls,
    ) -> tuple[Type["profiles.HexrdPPScript_Arguments"], ...]:
        return tuple(cls._creators.values())

    @classmethod
    def get_args(
        cls, profile_name: str
    ) -> Type["profiles.HexrdPPScript_Arguments"]:
        creator = cls._creators.get(profile_name)
        if not creator:
            raise ValueError(format)
        return creator


def autoregister(
    cls: Type["profiles.HexrdPPScript_Arguments"],
) -> Type["profiles.HexrdPPScript_Arguments"]:
    """decorator that registers cls with ArgumentClassesFactory"""
    ArgumentClassesFactory().register(cls)
    return cls
