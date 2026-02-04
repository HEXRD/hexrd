from pathlib import Path
import tempfile

import pytest

from hexrd.hedm.preprocess.profiles import (
    Eiger_Arguments,
    Dexelas_Arguments,
    HexrdPPScript_Arguments,
)
from hexrd.hedm.preprocess.preprocessors import preprocess


# FIXME: this is not the appropriate file to test the preprocess scrips since
# it is missing metadata. Until we add a file we just test that
# validate_arguments() works as expected
@pytest.fixture
def eiger_examples_path(example_repo_path: Path) -> Path:
    return Path(example_repo_path) / 'eiger'


@pytest.fixture
def ceria_examples_path(eiger_examples_path: Path) -> Path:
    return eiger_examples_path / 'first_ceria' / 'ff_000_data_000001.h5'


@pytest.fixture
def eiger_stream_v2_examples_path(eiger_examples_path: Path) -> Path:
    return eiger_examples_path / 'eiger_stream_v2/eiger_stream_v2_test_dataset.h5'


# test load/safe with defaults
def test_save_load_defaults_eiger():
    eargs = Eiger_Arguments()
    buffer = eargs.dump_config()
    args = HexrdPPScript_Arguments.load_from_config(buffer)
    assert eargs == args


def test_save_load_defaults_dexelas():
    eargs = Dexelas_Arguments()
    buffer = eargs.dump_config()
    args = HexrdPPScript_Arguments.load_from_config(buffer)
    assert eargs == args


# test load/safe with modified parameters
def test_save_load_eiger(ceria_examples_path):
    eargs = Eiger_Arguments()
    buffer = eargs.dump_config()
    args = HexrdPPScript_Arguments.load_from_config(buffer)
    assert eargs == args
    with pytest.raises(RuntimeError):  # required argument 'absolute_path' is missing
        args.validate_arguments()
    args.absolute_path = str(ceria_examples_path)
    args.validate_arguments()


def test_save_load_eiger_stream_v2(eiger_stream_v2_examples_path: Path):
    eargs = Eiger_Arguments()
    buffer = eargs.dump_config()
    args = HexrdPPScript_Arguments.load_from_config(buffer)
    assert eargs == args
    with pytest.raises(RuntimeError):  # required argument 'absolute_path' is missing
        args.validate_arguments()
    args.absolute_path = str(eiger_stream_v2_examples_path)
    args.eiger_stream_v2_threshold = 'man_diff'
    args.eiger_stream_v2_multiplier = 0.75
    args.num_frames = 1

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Write the output file in the temporary directory
        output_file = Path(tmpdirname) / 'test'
        args.output = output_file

        args.validate_arguments()

        # Write out the example file!
        preprocess(args)

        # Verify that there is an output file
        assert len(list(Path(tmpdirname).glob(f"*.npz"))) > 0


def test_save_load_dexelas():
    eargs = Dexelas_Arguments()
    eargs.base_dir = "/data"
    eargs.num_frames = 1
    eargs.panel_opts["FF1"] = [["add-row", 1000]]
    buffer = eargs.dump_config()
    args = HexrdPPScript_Arguments.load_from_config(buffer)
    assert eargs == args
    with pytest.raises(RuntimeError):  # many required arguments are missing
        args.validate_arguments()
