import hexrd.core.distortion as distortion


def test_distortion_initialization():
    result = distortion.maptypes()
    assert any(result)


def test_get_mapping():
    maptype = distortion.maptypes()[0]
    params = [0.0] * 8
    distortion_instance = distortion.get_mapping(maptype, params)
    assert distortion_instance is not None
    assert hasattr(distortion_instance, "maptype")
    assert distortion_instance.maptype == maptype
