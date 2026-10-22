import importlib
import subprocess
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
def test_command_help(
    command: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
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


def test_legacy_cli_imports() -> None:
    for name in ('help', 'documentation'):
        assert importlib.import_module(f'hexrd.hedm.cli.{name}') is (
            importlib.import_module(f'hexrd.cli.{name}')
        )
    for name in ('find_orientations', 'fit_grains', 'preprocess'):
        assert importlib.import_module(f'hexrd.cli.{name}') is (
            importlib.import_module(f'hexrd.hedm.cli.{name}')
        )


def test_module_entry_point() -> None:
    for args in (['--help'], ['help', 'cake-data']):
        result = subprocess.run(
            [sys.executable, '-m', 'hexrd.cli.main', *args],
            check=True, capture_output=True, text=True,
        )
        assert 'cake-data' in result.stdout
        assert 'conda' not in result.stdout
    result = subprocess.run(
        [sys.executable, '-m', 'hexrd.cli.main', 'help', 'not-a-command'],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert 'invalid choice' in result.stderr
    assert 'Traceback' not in result.stderr
