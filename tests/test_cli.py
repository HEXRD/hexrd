import importlib
import sys

import pytest

from hexrd.cli.main import main


@pytest.mark.parametrize(
    'command',
    [
        '',
        'help',
        'test',
        'documentation',
        'find-orientations',
        'fit-grains',
        'pickle23',
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
    assert f'usage: hexrd {command}' in capsys.readouterr().out


def test_legacy_cli_imports():
    from hexrd.hedm.cli.main import main as legacy_main

    assert legacy_main is main
    for name in ('help', 'test', 'documentation', 'pickle23'):
        assert importlib.import_module(f'hexrd.hedm.cli.{name}') is (
            importlib.import_module(f'hexrd.cli.{name}')
        )
    for name in ('find_orientations', 'fit_grains', 'preprocess'):
        assert importlib.import_module(f'hexrd.cli.{name}') is (
            importlib.import_module(f'hexrd.hedm.cli.{name}')
        )


@pytest.mark.parametrize(
    'args, status',
    [
        (['--debug'], 2),
        (['help', 'cake-data'], 0),
        (['help', 'missing-command'], 2),
    ],
)
def test_dispatch_errors_and_command_help(monkeypatch, capsys, args, status):
    monkeypatch.setattr(sys, 'argv', ['hexrd', *args])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == status
    output = capsys.readouterr()
    assert 'error:' in output.err if status else '--matrix' in output.out
