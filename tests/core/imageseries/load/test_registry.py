from unittest.mock import patch
from hexrd.core.imageseries.load.registry import Registry


def test_register_valid_adapter():
    """Test registering a valid adapter class."""

    class MockAdapter:
        format = 'mock-format'

    with patch.dict(Registry.adapter_registry, {}, clear=True):
        Registry.register(MockAdapter)

        assert 'mock-format' in Registry.adapter_registry
        assert Registry.adapter_registry['mock-format'] is MockAdapter


def test_register_base_class_ignored():
    """Test that the base class 'ImageSeriesAdapter' is skipped."""

    class ImageSeriesAdapter:
        format = 'base-format'

    with patch.dict(Registry.adapter_registry, {}, clear=True):
        Registry.register(ImageSeriesAdapter)

        # Should NOT be added
        assert 'base-format' not in Registry.adapter_registry
        assert len(Registry.adapter_registry) == 0


def test_register_overwrite():
    """Test that registering a new class with the same format overwrites the old one."""

    class AdapterA:
        format = 'common-format'

    class AdapterB:
        format = 'common-format'

    with patch.dict(Registry.adapter_registry, {}, clear=True):
        Registry.register(AdapterA)
        assert Registry.adapter_registry['common-format'] is AdapterA

        Registry.register(AdapterB)
        assert Registry.adapter_registry['common-format'] is AdapterB
