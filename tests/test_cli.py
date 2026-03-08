# tests/test_cli.py
"""Tests for CLI."""
import json
import os
import subprocess
import sys
from pathlib import Path

# Ensure subprocess can find the drumsep package via PYTHONPATH
_PROJECT_ROOT = str(Path(__file__).parent.parent)
_SRC_DIR = str(Path(__file__).parent.parent / "src")


def _run_cli(*args: str, **kwargs) -> subprocess.CompletedProcess:
    """Run drumsep CLI in a subprocess with PYTHONPATH set."""
    env = os.environ.copy()
    env["PYTHONPATH"] = _SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "drumsep", *args],
        capture_output=True, text=True, cwd=_PROJECT_ROOT, env=env, **kwargs,
    )


def test_cli_separate(drums_audio_path, tmp_path):
    output = str(tmp_path / "cli_output")
    result = _run_cli(drums_audio_path, "-o", output)
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert Path(output, "kick.wav").exists()
    assert Path(output, "snare.wav").exists()


def test_cli_analyze(kick_audio_path):
    result = _run_cli("analyze", kick_audio_path)
    assert result.returncode == 0, f"stderr: {result.stderr}"
    data = json.loads(result.stdout)
    assert "fundamental_freq" in data


def test_cli_no_args():
    result = _run_cli()
    assert result.returncode != 0


def test_cli_version():
    result = _run_cli("--version")
    assert result.returncode == 0
    assert "0.1.0" in result.stdout
