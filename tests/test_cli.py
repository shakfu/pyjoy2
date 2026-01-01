"""Tests for CLI entry point."""

import subprocess
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from pyjoy2.__main__ import main


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run pyjoy2 CLI with given arguments."""
    return subprocess.run(
        [sys.executable, "-m", "pyjoy2", *args],
        capture_output=True,
        text=True,
    )


class TestCLI:
    """Tests for command-line interface."""

    def test_version(self):
        result = run_cli("--version")
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_eval_expression(self):
        result = run_cli("-e", "3 4 +")
        assert result.returncode == 0
        assert "7" in result.stdout

    def test_eval_multiple_results(self):
        result = run_cli("-e", "1 2 3")
        assert result.returncode == 0
        assert "1" in result.stdout
        assert "2" in result.stdout
        assert "3" in result.stdout

    def test_command_alias(self):
        result = run_cli("-c", "5 dup *")
        assert result.returncode == 0
        assert "25" in result.stdout

    def test_eval_empty_result(self):
        result = run_cli("-e", "1 pop")
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_eval_error(self):
        result = run_cli("-e", "1 2 undefined_word")
        assert result.returncode == 1
        assert "Error" in result.stderr

    def test_eval_error_with_debug(self):
        result = run_cli("--debug", "-e", "pop")  # pop on empty stack
        assert result.returncode == 1
        assert "Error" in result.stderr
        assert "Traceback" in result.stderr

    def test_file_execution(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".joy", delete=False) as f:
            f.write("10 20 +\n")
            f.flush()
            try:
                result = run_cli(f.name)
                assert result.returncode == 0
                assert "30" in result.stdout
            finally:
                Path(f.name).unlink()

    def test_file_not_found(self):
        result = run_cli("nonexistent_file.joy")
        assert result.returncode == 1
        assert "file not found" in result.stderr

    def test_file_error(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".joy", delete=False) as f:
            f.write("undefined_word\n")
            f.flush()
            try:
                result = run_cli(f.name)
                assert result.returncode == 1
                assert "Error" in result.stderr
            finally:
                Path(f.name).unlink()

    def test_file_error_with_debug(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".joy", delete=False) as f:
            f.write("pop\n")  # pop on empty stack
            f.flush()
            try:
                result = run_cli("--debug", f.name)
                assert result.returncode == 1
                assert "Traceback" in result.stderr
            finally:
                Path(f.name).unlink()

    def test_help(self):
        result = run_cli("--help")
        assert result.returncode == 0
        assert "pyjoy2" in result.stdout
        assert "--eval" in result.stdout


class TestMainFunction:
    """Tests that call main() directly for coverage."""

    def test_eval_expression_direct(self, capsys):
        with patch.object(sys, "argv", ["pyjoy2", "-e", "2 3 +"]):
            main()
        captured = capsys.readouterr()
        assert "5" in captured.out

    def test_eval_empty_result_direct(self, capsys):
        with patch.object(sys, "argv", ["pyjoy2", "-e", "1 pop"]):
            main()
        captured = capsys.readouterr()
        assert captured.out.strip() == ""

    def test_eval_multiple_results_direct(self, capsys):
        with patch.object(sys, "argv", ["pyjoy2", "-e", "1 2 3"]):
            main()
        captured = capsys.readouterr()
        assert "1" in captured.out
        assert "2" in captured.out
        assert "3" in captured.out

    def test_command_alias_direct(self, capsys):
        with patch.object(sys, "argv", ["pyjoy2", "-c", "4 4 *"]):
            main()
        captured = capsys.readouterr()
        assert "16" in captured.out

    def test_eval_error_direct(self, capsys):
        with patch.object(sys, "argv", ["pyjoy2", "-e", "pop"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_eval_error_with_debug_direct(self, capsys):
        with patch.object(sys, "argv", ["pyjoy2", "--debug", "-e", "pop"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Traceback" in captured.err

    def test_file_execution_direct(self, capsys, tmp_path):
        joy_file = tmp_path / "test.joy"
        joy_file.write_text("5 5 *\n")
        with patch.object(sys, "argv", ["pyjoy2", str(joy_file)]):
            main()
        captured = capsys.readouterr()
        assert "25" in captured.out

    def test_file_empty_result_direct(self, capsys, tmp_path):
        joy_file = tmp_path / "test.joy"
        joy_file.write_text("1 pop\n")
        with patch.object(sys, "argv", ["pyjoy2", str(joy_file)]):
            main()
        captured = capsys.readouterr()
        assert captured.out.strip() == ""

    def test_file_not_found_direct(self, capsys):
        with patch.object(sys, "argv", ["pyjoy2", "nonexistent.joy"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "file not found" in captured.err

    def test_file_error_direct(self, capsys, tmp_path):
        joy_file = tmp_path / "test.joy"
        joy_file.write_text("undefined_xyz\n")
        with patch.object(sys, "argv", ["pyjoy2", str(joy_file)]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_file_error_with_debug_direct(self, capsys, tmp_path):
        joy_file = tmp_path / "test.joy"
        joy_file.write_text("pop\n")
        with patch.object(sys, "argv", ["pyjoy2", "--debug", str(joy_file)]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Traceback" in captured.err
