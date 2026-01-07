import pytest
from unittest.mock import MagicMock, patch, mock_open, call
import numpy as np
import os
import warnings
from hexrd.core.imageseries import save

# --- Helpers & Fixtures ---


class FakeH5File:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def create_group(self, name):
        pass

    def create_dataset(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, key):
        pass

    attrs = {}
    filename = "dummy.h5"


@pytest.fixture
def mock_h5_env():
    with patch('hexrd.core.imageseries.save.h5py') as m_h5:
        file_instance = MagicMock(spec=FakeH5File)
        file_instance.__enter__.return_value = file_instance
        file_instance.__exit__.return_value = None
        file_instance.attrs = MagicMock()

        constructor_spy = MagicMock(return_value=file_instance)
        FakeH5File.__new__ = lambda cls, *args, **kwargs: constructor_spy(
            *args, **kwargs
        )
        m_h5.File = FakeH5File

        yield m_h5, constructor_spy, file_instance


@pytest.fixture
def mock_ims():
    ims = MagicMock()
    ims.__len__.return_value = 2
    ims.shape = (10, 10)
    ims.dtype = np.float32
    ims.metadata = {'key': np.str_('value')}

    def getitem(key):
        if key == 0:
            return np.zeros((10, 10), dtype=np.float32)
        if key == 1:
            return np.ones((10, 10), dtype=np.float32)
        raise IndexError

    ims.__getitem__.side_effect = getitem

    return ims


# --- Top Level Tests ---


@patch('hexrd.core.imageseries.save._Registry.getwriter')
def test_write_dispatch(mock_getwriter, mock_ims):
    mock_writer_cls = MagicMock()
    mock_writer_instance = mock_writer_cls.return_value
    mock_getwriter.return_value = mock_writer_cls

    save.write(mock_ims, "test.h5", "hdf5", option='val')

    mock_getwriter.assert_called_with("hdf5")
    mock_writer_cls.assert_called_with(mock_ims, "test.h5", option='val')
    mock_writer_instance.write.assert_called_once()


# --- WriteH5 Tests ---


def test_writeh5_basic(mock_h5_env, mock_ims):
    _, spy, file_inst = mock_h5_env
    mock_group = file_inst.create_group.return_value
    mock_dataset = mock_group.create_dataset.return_value

    writer = save.WriteH5(mock_ims, "out.h5", path="images")
    writer.write()

    spy.assert_called_with("out.h5", "w")
    file_inst.create_group.assert_called_with("images")
    mock_group.create_dataset.assert_called_with(
        'images',
        (2, 10, 10),
        np.float32,
        shuffle=True,
        compression='gzip',
        compression_opts=1,
        chunks=(1, 10, 10),
    )
    assert mock_dataset.__setitem__.call_count == 2
    mock_group.attrs.__setitem__.assert_called_with('key', b'value')


def test_writeh5_options(mock_ims):
    writer = save.WriteH5(
        mock_ims, "t.h5", path="p", gzip=5, chunk_rows=2, shuffle=False
    )
    opts = writer.h5opts
    assert opts['compression'] == 'gzip'
    assert opts['compression_opts'] == 5
    assert opts['chunks'] == (1, 2, 10)
    assert opts['shuffle'] is False

    with pytest.raises(ValueError):
        save.WriteH5(mock_ims, "t.h5", path="p", gzip=10).h5opts


# --- WriteFrameCache Tests ---


def test_framecache_paths(mock_ims):
    w = save.WriteFrameCache(mock_ims, "test.npz", threshold=0)
    assert w.cache == "test.npz"

    w = save.WriteFrameCache(
        mock_ims, "dir/test.yml", threshold=0, cache_file="data.npz"
    )
    assert w.cache == os.path.join("dir", "data.npz")


@patch('hexrd.core.imageseries.save.np.save')
@patch('hexrd.core.imageseries.save.os.path.exists', return_value=False)
def test_framecache_process_meta(mock_exists, mock_save, mock_ims):
    mock_ims.metadata = {
        'scalar': 1,
        'nested': {'a': 1},
        'array': np.array([1, 2]),
    }
    writer = save.WriteFrameCache(mock_ims, "base.npz", threshold=0)

    with patch('sys.stderr') as mock_stderr:
        meta_out = writer._process_meta(save_omegas=True)
        assert "! load-numpy-array" in meta_out['array']
        mock_save.assert_called()


# --- Coverage for _write_yml ---


@patch('hexrd.core.imageseries.save.yaml.safe_dump')
@patch('builtins.open', new_callable=mock_open)
def test_write_yml(mock_file, mock_yaml, mock_ims):
    writer = save.WriteFrameCache(mock_ims, "out.yml", threshold=0)

    if hasattr(writer, 'cachename') and not hasattr(writer, '_cachename'):
        writer._cachename = writer.cachename

    with patch.object(writer, '_process_meta', return_value={'k': 'v'}):
        writer._write_yml()

    mock_file.assert_called_with("out.yml", "w")
    args, _ = mock_yaml.call_args
    info = args[0]

    assert info['data']['file'] == "out.yml"
    assert info['data']['nframes'] == 2
    assert info['meta']['k'] == 'v'


# --- Coverage for _write_frames dispatch ---


def test_write_frames_dispatch(mock_ims):
    writer = save.WriteFrameCache(mock_ims, "out", style='npz', threshold=0)
    with patch.object(writer, '_write_frames_npz') as mock_npz:
        writer._write_frames()
        mock_npz.assert_called_once()

    writer = save.WriteFrameCache(mock_ims, "out", style='fch5', threshold=0)
    with patch.object(writer, '_write_frames_fch5') as mock_fch5:
        writer._write_frames()
        mock_fch5.assert_called_once()


@patch('hexrd.core.imageseries.save.np.savez_compressed')
@patch('hexrd.core.imageseries.save.extract_ijv')
def test_write_frames_npz_threading(mock_extract, mock_savez, mock_ims):
    def side_effect(img, thresh, rows, cols, vals):
        rows[0] = 1
        cols[0] = 2
        vals[0] = 99.0
        return 1

    mock_extract.side_effect = side_effect

    writer = save.WriteFrameCache(mock_ims, "out.npz", threshold=10)
    writer._write_frames_npz()
    assert mock_extract.call_count == 2

    args, kwargs = mock_savez.call_args

    assert args[0] == "out.npz"
    assert kwargs['0_row'][0] == 1
    assert kwargs['0_col'][0] == 2
    assert kwargs['0_data'][0] == 99.0


def test_check_sparsity_logic(mock_ims):
    writer = save.WriteFrameCache(mock_ims, "out.npz", threshold=0)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        writer._check_sparsity(0, 1, 100)
        assert len(w) == 0

    with pytest.warns(UserWarning, match="frame 0 is 80.00% sparse"):
        writer._check_sparsity(0, 20, 100)


def test_write_frames_fch5(mock_h5_env, mock_ims):
    _, _, file_inst = mock_h5_env
    with patch('hexrd.core.imageseries.save.extract_ijv', return_value=1):
        writer = save.WriteFrameCache(
            mock_ims, "out.h5", style='fch5', threshold=0
        )
        writer._write_frames_fch5()

    calls = [c[0][0] for c in file_inst.create_dataset.call_args_list]
    assert 'data' in calls
    assert 'indices' in calls
    assert 'frame_ids' in calls
    file_inst.create_dataset.return_value.resize.assert_called()


def test_write_yaml_deprecation(mock_ims):
    writer = save.WriteFrameCache(mock_ims, "out.yml", threshold=0)
    if hasattr(writer, 'cachename') and not hasattr(writer, '_cachename'):
        writer._cachename = writer.cachename

    with patch.object(writer, '_write_frames'), patch.object(
        writer, '_write_yml'
    ):
        with pytest.warns(DeprecationWarning, match="deprecated"):
            writer.write(output_yaml=True)


def test_unsupported_style(mock_ims):
    with pytest.raises(TypeError, match="Unknown file style"):
        save.WriteFrameCache(mock_ims, "out", threshold=0, style="bad_style")
