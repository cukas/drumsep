# tests/test_separator.py
"""Tests for DrumSeparator."""
import os
from pathlib import Path

from drumsep.separator import DrumSeparator
from drumsep.types import SeparationResult


def test_separator_produces_five_stems(drums_audio_path, output_dir):
    sep = DrumSeparator()
    result = sep.separate(drums_audio_path, output_dir)
    assert isinstance(result, SeparationResult)
    assert len(result.stems) == 5
    assert set(result.stems.keys()) == {"kick", "snare", "hihat", "cymbals", "toms"}


def test_separator_creates_wav_files(drums_audio_path, output_dir):
    sep = DrumSeparator()
    result = sep.separate(drums_audio_path, output_dir)
    for name, path in result.stems.items():
        assert os.path.exists(path), f"{name} stem file not created"
        assert path.endswith(".wav")


def test_separator_result_metadata(drums_audio_path, output_dir):
    sep = DrumSeparator()
    result = sep.separate(drums_audio_path, output_dir)
    assert result.model_name == "native_drumsep"
    assert result.processing_time > 0


def test_separator_stereo_input(stereo_drums_audio_path, output_dir):
    sep = DrumSeparator()
    result = sep.separate(stereo_drums_audio_path, output_dir)
    assert len(result.stems) == 5
    for path in result.stems.values():
        assert os.path.exists(path)


def test_separator_creates_output_dir(drums_audio_path, tmp_path):
    new_dir = str(tmp_path / "new_output")
    sep = DrumSeparator()
    result = sep.separate(drums_audio_path, new_dir)
    assert os.path.isdir(new_dir)
    assert len(result.stems) == 5


def test_separator_without_enhanced(drums_audio_path, output_dir):
    sep = DrumSeparator(enhanced=False)
    result = sep.separate(drums_audio_path, output_dir)
    assert len(result.stems) == 5


def test_separator_progress_callback(drums_audio_path, output_dir):
    progress_calls = []
    def on_progress(percent, message):
        progress_calls.append((percent, message))
    sep = DrumSeparator()
    sep.separate(drums_audio_path, output_dir, on_progress=on_progress)
    assert len(progress_calls) > 0
    assert progress_calls[-1][0] == 100


def test_separator_cancellation(drums_audio_path, output_dir):
    import threading
    from drumsep.types import CancellationError
    import pytest
    cancel = threading.Event()
    cancel.set()
    sep = DrumSeparator(cancel_event=cancel)
    with pytest.raises(CancellationError):
        sep.separate(drums_audio_path, output_dir)


def test_separator_invalid_path(output_dir):
    sep = DrumSeparator()
    import pytest
    with pytest.raises(Exception):
        sep.separate("/nonexistent/drums.wav", output_dir)


def test_separator_stem_names():
    sep = DrumSeparator()
    assert sep.stem_names == ["kick", "snare", "hihat", "cymbals", "toms"]


def test_separator_silent_audio(silent_audio_path, output_dir):
    sep = DrumSeparator()
    result = sep.separate(silent_audio_path, output_dir)
    assert len(result.stems) == 5
