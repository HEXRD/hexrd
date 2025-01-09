from hexrd.preprocess.profiles import (
    Eiger_Arguments,
    Dexelas_Arguments,
    HexrdPPScript_Arguments,
)


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
def test_save_load_eiger():
    eargs = Eiger_Arguments()
    eargs.base_dir = "/data"
    eargs.num_frames = 1
    buffer = eargs.dump_config()
    args = HexrdPPScript_Arguments.load_from_config(buffer)
    assert eargs == args


def test_save_load_dexelas():
    eargs = Dexelas_Arguments()
    eargs.base_dir = "/data"
    eargs.num_frames = 1
    eargs.panel_opts["FF1"] = {("add-row", 1000)}
    buffer = eargs.dump_config()
    args = HexrdPPScript_Arguments.load_from_config(buffer)
    assert eargs == args
