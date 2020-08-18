"""Adapter registry
"""
class Registry(object):
    """Registry for imageseries adapters"""
    adapter_registry = dict()

    @classmethod
    def register(cls, acls):
        """Register adapter class"""
        if acls.__name__ != 'ImageSeriesAdapter':
            cls.adapter_registry[acls.format] = acls

    pass  # end class
