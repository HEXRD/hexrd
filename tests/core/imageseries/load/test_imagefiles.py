import os
import yaml
import pickle

import numpy as np
import pytest

import hexrd.core.imageseries.load.imagefiles as imf


class FakeFabioImage:
    """Lightweight fake for fabio image context manager."""

    def __init__(self, filename):
        self._fname = filename
        # behavior varies by filename so tests can exercise branches
        if filename.endswith(".gel"):
            # small uint8 array for gel processing
            self.data = np.array([[0, 255], [128, 64]], dtype=np.uint8)
            self.classname = "GELCLASS"
            self.nframes = 1
        elif "multi" in filename:
            # multi-frame: 3 frames of 2x2
            self.data = np.arange(12).reshape((3, 2, 2))
            self.classname = "MULTICLASS"
            self.nframes = 3
        elif "big" in filename:
            # intentionally large value to trigger dtype truncation detection
            self.data = np.array([[300]], dtype=np.int32)
            self.classname = "BIG"
            self.nframes = 1
        else:
            # default simple 2D image
            self.data = np.arange(6).reshape((2, 3))
            self.classname = "IMGCLASS"
            self.nframes = 1

    def getframe(self, idx):
        arr = self.data[idx]
        return type("G", (), {"data": arr})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


@pytest.fixture(autouse=True)
def patch_fabio_open(monkeypatch):
    """Monkeypatch fabio.open to return our FakeFabioImage based on filename."""

    def fake_open(fname):
        return FakeFabioImage(fname)

    monkeypatch.setattr(imf.fabio, "open", fake_open)
    return fake_open


def write_yaml_file(tmp_path, directory, patterns, options=None, meta=None):
    doc = {
        "image-files": {"directory": str(directory), "files": patterns},
        "options": options or {},
        "meta": meta or {},
    }
    fn = tmp_path / "imgs.yml"
    fn.write_text(yaml.safe_dump(doc))
    return str(fn), doc


def touch_files(tmp_path, names):
    for n in names:
        (tmp_path / n).write_text("dummy")
    return [str(tmp_path / n) for n in names]


def test_yaml_string_input_and_basic_singleframe_behavior(
    tmp_path, patch_fabio_open
):
    files = touch_files(tmp_path, ["a.img", "b.img"])
    ydict = {
        "image-files": {"directory": str(tmp_path), "files": "*.img"},
        "options": {},
        "meta": {"info": "test"},
    }
    ycontent = yaml.safe_dump(ydict)

    adapter = imf.ImageFilesImageSeriesAdapter(ycontent)
    try:
        assert adapter.fname == ycontent

        md = adapter.metadata
        assert "info" in md

        assert len(adapter._files) >= 2
        assert adapter.singleframes is True
        assert len(adapter) == adapter._nframes == len(adapter._files)

        arr0 = adapter[0]
        assert isinstance(arr0, np.ndarray)
        np.testing.assert_array_equal(arr0, FakeFabioImage(files[0]).data)

        s = str(adapter)
        assert "imageseries from file list" in s
        assert "nframes" in s
        assert "fabio class" in s
    finally:
        del adapter


def test_dtype_truncation_detection(tmp_path):
    touch_files(tmp_path, ["big.img"])
    yfn, _ = write_yaml_file(
        tmp_path, tmp_path, "big.img", options={"dtype": "uint8"}
    )
    adapter = imf.ImageFilesImageSeriesAdapter(yfn)

    with pytest.raises(RuntimeError):
        _ = adapter[0]


def test_multi_frame_indexing_and_file_and_frame_logic(tmp_path):
    touch_files(tmp_path, ["multi.img"])
    yfn, _ = write_yaml_file(tmp_path, tmp_path, "multi.img")
    adapter = imf.ImageFilesImageSeriesAdapter(yfn)
    try:
        assert adapter.singleframes is False
        assert len(adapter) == 3
        np.testing.assert_array_equal(
            adapter[0], FakeFabioImage("multi.img").getframe(0).data
        )
        np.testing.assert_array_equal(
            adapter[2], FakeFabioImage("multi.img").getframe(2).data
        )
        np.testing.assert_array_equal(
            adapter[-1], FakeFabioImage("multi.img").getframe(2).data
        )

        with pytest.raises(LookupError):
            adapter._file_and_frame(100)
    finally:
        del adapter


def test__checkvalue_consistency_and_error():
    assert imf.ImageFilesImageSeriesAdapter._checkvalue(None, 5, "msg") == 5
    assert imf.ImageFilesImageSeriesAdapter._checkvalue(5, 5, "msg") == 5

    with pytest.raises(ValueError):
        imf.ImageFilesImageSeriesAdapter._checkvalue(3, 4, "inconsistent")


def test_FileInfo_properties_and_errors(tmp_path):
    fname = str(tmp_path / "file1.img")
    open(fname, "w").close()
    info = imf.FileInfo(fname, empty=0, max_frames=0)

    assert info.filename == fname
    assert info.fabioclass == "IMGCLASS"
    assert info.nframes == 1
    assert info.shape == info.dat.shape
    assert info.dtype == info.dat.dtype

    assert str(info) is not None

    with pytest.raises(ValueError):
        imf.FileInfo(fname, empty=10, max_frames=0)


def test_process_gel_data_and_scale_factor():
    arr = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    out = imf._process_gel_data(arr)
    assert out.dtype == np.float64
    assert out.shape == arr.shape
    assert np.all(out >= 0)
    expected = (np.invert(arr).astype(np.float64) ** 2) * imf.GEL_SCALE_FACTOR
    np.testing.assert_allclose(out, expected)
