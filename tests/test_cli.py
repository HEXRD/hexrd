import importlib
import sys

import pytest

from hexrd.cli.main import main


@pytest.mark.parametrize(
    'command',
    [
        '',
        'help',
        'documentation',
        'find-orientations',
        'fit-grains',
        'preprocess',
        'cake-data',
    ],
)
def test_command_help(command, monkeypatch, capsys):
    argv = ['hexrd', command, '--help'] if command else ['hexrd']
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(sys.modules['__main__'], '__spec__', None)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert f'usage: hexrd {command}' in output
    if command == 'cake-data':
        help_text = ' '.join(output.split())
        assert 'eta wedge width (default: 5.0)' in help_text
        assert 'starting two-theta angle (instrument default: 2.0)' in help_text
        assert 'default: None' not in help_text


def test_legacy_cli_imports():
    for name in ('help', 'documentation'):
        assert importlib.import_module(f'hexrd.hedm.cli.{name}') is (
            importlib.import_module(f'hexrd.cli.{name}')
        )
    for name in ('find_orientations', 'fit_grains', 'preprocess'):
        assert importlib.import_module(f'hexrd.cli.{name}') is (
            importlib.import_module(f'hexrd.hedm.cli.{name}')
        )
