import pytest
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
import yaml
from hexrd.core.imageseries.load.rawimage import RawImageSeriesAdapter

# --- Fixtures ---


@pytest.fixture
def mock_yaml_content():
    """Standard YAML config content for raw image loader."""
    return {
        'filename': 'data.bin',
        'shape': '10 10',
        'skip': 100,
        'scalar': {
            'type': 'i',
            'bytes': 2,
            'signed': False,
            'endian': 'little',
        },
    }


# --- Initialization & Dtype Tests ---


@patch('builtins.open', new_callable=mock_open)
@patch('hexrd.core.imageseries.load.rawimage.os.path.getsize')
def test_init_basic(mock_getsize, mock_file, mock_yaml_content):
    """Test basic initialization parameters."""
    mock_file.side_effect = [
        mock_open(read_data=yaml.dump(mock_yaml_content)).return_value,
        MagicMock(),
    ]

    mock_getsize.return_value = 500

    adapter = RawImageSeriesAdapter("config.yml")

    assert len(adapter) == 2
    assert adapter.shape == (10, 10)
    assert adapter.dtype == np.dtype('<H')
    assert adapter.skipbytes == 100

    mock_file.assert_called_with('data.bin', 'r')


@patch('builtins.open', new_callable=mock_open)
@patch('hexrd.core.imageseries.load.rawimage.os.path.getsize')
def test_big_endian(mock_getsize, mock_file, mock_yaml_content):
    """Test big endian logic."""
    mock_yaml_content['scalar']['endian'] = 'big'

    mock_file.side_effect = [
        mock_open(read_data=yaml.dump(mock_yaml_content)).return_value,
        MagicMock(),
    ]
    mock_getsize.return_value = 500
    adapter = RawImageSeriesAdapter("config.yml")

    assert adapter.dtype == np.dtype('>H')


@patch('builtins.open', new_callable=mock_open)
@patch('hexrd.core.imageseries.load.rawimage.os.path.getsize')
def test_init_file_size_error(mock_getsize, mock_file, mock_yaml_content):
    """Test error when file size is invalid for frame config."""
    mock_file.side_effect = [
        mock_open(read_data=yaml.dump(mock_yaml_content)).return_value,
        MagicMock(),
    ]

    mock_getsize.return_value = 501

    with pytest.raises(ValueError, match="Total number of bytes"):
        RawImageSeriesAdapter("config.yml")


def test_typechars_logic():
    """Test static method for dtype string generation."""
    assert RawImageSeriesAdapter.typechars('i', 1, True, True) == '<b'
    assert RawImageSeriesAdapter.typechars('i', 2, False, False) == '>H'
    assert RawImageSeriesAdapter.typechars('i', 4, True, True) == '<i'
    assert RawImageSeriesAdapter.typechars('i', 8, True, True) == '<l'

    assert RawImageSeriesAdapter.typechars('f', 4, True, True) == '<f'
    assert RawImageSeriesAdapter.typechars('d', 8, True, True) == '<d'

    assert RawImageSeriesAdapter.typechars('b', 1, True, True) == '<?'


@patch('builtins.open', new_callable=mock_open)
@patch('hexrd.core.imageseries.load.rawimage.os.path.getsize')
def test_dtype_endian_error(mock_getsize, mock_file, mock_yaml_content):
    """Test invalid endianness in YAML."""
    mock_yaml_content['scalar']['endian'] = 'mixed'
    mock_file.side_effect = [
        mock_open(read_data=yaml.dump(mock_yaml_content)).return_value,
        MagicMock(),
    ]
    mock_getsize.return_value = 300

    with pytest.raises(ValueError, match='endian must be "big"'):
        RawImageSeriesAdapter("config.yml")


# --- Data Access Tests ---


@patch('hexrd.core.imageseries.load.rawimage.np.fromfile')
@patch('builtins.open', new_callable=mock_open)
@patch('hexrd.core.imageseries.load.rawimage.os.path.getsize')
def test_getitem_access(
    mock_getsize, mock_file, mock_fromfile, mock_yaml_content
):
    """Test seeking and reading frames."""
    mock_file.side_effect = [
        mock_open(read_data=yaml.dump(mock_yaml_content)).return_value,
        MagicMock(),
    ]

    mock_getsize.return_value = 700
    mock_fromfile.return_value = np.zeros(100, dtype=np.uint16)

    adapter = RawImageSeriesAdapter("config.yml")
    assert len(adapter) == 3
    assert adapter[1].shape == (10, 10)

    adapter.f.seek.assert_called_with(300, 0)
    mock_fromfile.assert_called_with(adapter.f, adapter.dtype, count=100)


@patch('hexrd.core.imageseries.load.rawimage.np.fromfile')
@patch('builtins.open', new_callable=mock_open)
@patch('hexrd.core.imageseries.load.rawimage.os.path.getsize')
def test_getitem_fancy_indexing(
    mock_getsize, mock_file, mock_fromfile, mock_yaml_content
):
    """Test support for tuple keys (frame, slice)."""
    mock_file.side_effect = [
        mock_open(read_data=yaml.dump(mock_yaml_content)).return_value,
        MagicMock(),
    ]
    mock_getsize.return_value = 300

    flat_data = np.arange(100, dtype=np.uint16)
    mock_fromfile.return_value = flat_data

    adapter = RawImageSeriesAdapter("config.yml")

    assert adapter[0, 0, 0] == 0
    assert adapter[0, 0:5].shape == (5, 10)


def test_iterator_support(mock_yaml_content):
    """Test iteration."""
    with patch(
        'builtins.open', mock_open(read_data=yaml.dump(mock_yaml_content))
    ) as m_open, patch(
        'hexrd.core.imageseries.load.rawimage.os.path.getsize',
        return_value=500,
    ), patch(
        'hexrd.core.imageseries.load.rawimage.np.fromfile',
        return_value=np.zeros(100),
    ):

        m_open.side_effect = [
            mock_open(read_data=yaml.dump(mock_yaml_content)).return_value,
            MagicMock(),
        ]

        adapter = RawImageSeriesAdapter("config.yml")
        frames = list(adapter)

        assert len(frames) == 2
        assert frames[0].shape == (10, 10)


@patch('builtins.open', new_callable=mock_open)
@patch('hexrd.core.imageseries.load.rawimage.os.path.getsize')
def test_metadata_property(mock_getsize, mock_file, mock_yaml_content):
    """Test metadata property access."""
    mock_file.side_effect = [
        mock_open(read_data=yaml.dump(mock_yaml_content)).return_value,
        MagicMock(),
    ]
    mock_getsize.return_value = 300

    adapter = RawImageSeriesAdapter("config.yml")

    assert adapter.metadata == {}
