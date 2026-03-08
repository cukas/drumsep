# tests/test_debleed.py
"""Tests for kick debleeding."""
import numpy as np
import soundfile as sf

from drumsep.debleed import debleed_kick


def test_debleed_returns_same_length(kick_audio_path, bass_audio_path, sample_rate):
    kick, _ = sf.read(kick_audio_path)
    result = debleed_kick(kick, bass_audio_path, sample_rate)
    assert len(result) == len(kick)


def test_debleed_reduces_bass_similarity(kick_audio_path, bass_audio_path, sample_rate):
    kick, _ = sf.read(kick_audio_path)
    result = debleed_kick(kick, bass_audio_path, sample_rate)
    assert not np.array_equal(kick, result)


def test_debleed_strength_zero_returns_original(kick_audio_path, bass_audio_path, sample_rate):
    kick, _ = sf.read(kick_audio_path)
    result = debleed_kick(kick, bass_audio_path, sample_rate, strength=0.0)
    np.testing.assert_allclose(result, kick[:len(result)], atol=1e-5)


def test_debleed_handles_length_mismatch(tmp_path, sample_rate):
    kick = np.random.randn(sample_rate * 3).astype(np.float32)
    bass = np.random.randn(sample_rate * 2).astype(np.float32)
    bass_path = str(tmp_path / "short_bass.wav")
    sf.write(bass_path, bass, sample_rate)
    result = debleed_kick(kick, bass_path, sample_rate)
    assert len(result) == len(kick)
