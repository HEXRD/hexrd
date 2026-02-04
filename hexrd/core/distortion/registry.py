"""Distortion package (python 3)"""

import abc

__all__ = ['maptypes', 'get_mapping']


class _RegisterDistortionClass(abc.ABCMeta):
    maptype: str | None

    def __init__(cls, name, bases, attrs):
        abc.ABCMeta.__init__(cls, name, bases, attrs)
        Registry.register(cls)


class Registry(object):
    """Registry for imageseries adapters"""

    distortion_registry: dict[str, _RegisterDistortionClass] = dict()

    @classmethod
    def register(cls, acls: _RegisterDistortionClass):
        """Register adapter class"""
        if acls.__name__ != 'DistortionBase':
            assert acls.maptype is not None
            cls.distortion_registry[acls.maptype] = acls
