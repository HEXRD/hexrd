"""Distortion package (python 3)"""
import abc

__all__ = ['maptypes', 'get_mapping']

class _RegisterDistortionClass(abc.ABCMeta):

    def __init__(cls, name, bases, attrs):
        abc.ABCMeta.__init__(cls, name, bases, attrs)
        Registry.register(cls)


class Registry(object):
    """Registry for imageseries adapters"""
    distortion_registry = dict()

    @classmethod
    def register(cls, acls):
        """Register adapter class"""
        if acls.__name__ is not 'DistortionBase':
            cls.distortion_registry[acls.maptype] = acls

    pass  # end class
