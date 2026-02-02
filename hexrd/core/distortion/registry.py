"""Distortion package (python 3)"""

import abc

from hexrd.core.distortion.distortionabc import DistortionABC

__all__ = ['maptypes', 'get_mapping']


class _RegisterDistortionClass(abc.ABCMeta):

    def __init__(cls, name, bases, attrs):
        abc.ABCMeta.__init__(cls, name, bases, attrs)
        Registry.register(cls)


class Registry(object):
    """Registry for imageseries adapters"""

    distortion_registry: dict[str, DistortionABC] = dict()

    @classmethod
    def register(cls, acls: DistortionABC):
        """Register adapter class"""
        if acls.__name__ != 'DistortionBase':
            assert acls.maptype is not None
            cls.distortion_registry[acls.maptype] = acls
