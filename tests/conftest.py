"""Shared test fixtures — synthetic audio generators."""
from __future__ import annotations

import numpy as np
import soundfile as sf
import pytest


@pytest.fixture
def tmp_audio_dir(tmp_path):
    return tmp_path


@pytest.fixture
def sample_rate():
    return 44100


@pytest.fixture
def duration():
    return 2.0


def _generate_sine(freq, sr, duration, amplitude=0.5):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _generate_transient_burst(freq, sr, duration, hit_interval=0.5, decay_ms=50.0):
    n_samples = int(sr * duration)
    audio = np.zeros(n_samples, dtype=np.float32)
    hit_samples = int(sr * hit_interval)
    decay_samples = int(sr * decay_ms / 1000)
    for start in range(0, n_samples, hit_samples):
        end = min(start + decay_samples, n_samples)
        length = end - start
        t = np.linspace(0, length / sr, length, endpoint=False)
        envelope = np.exp(-t * 1000 / decay_ms)
        audio[start:end] += (0.8 * envelope * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return audio


def _generate_noise_burst(sr, duration, low_hz, high_hz, hit_interval=0.25):
    from scipy.signal import butter, sosfilt
    n_samples = int(sr * duration)
    audio = np.zeros(n_samples, dtype=np.float32)
    hit_samples = int(sr * hit_interval)
    burst_len = int(sr * 0.02)
    nyq = sr / 2
    sos = butter(4, [low_hz / nyq, min(high_hz / nyq, 0.99)], btype="band", output="sos")
    for start in range(0, n_samples, hit_samples):
        end = min(start + burst_len, n_samples)
        length = end - start
        noise = np.random.randn(length).astype(np.float32) * 0.5
        filtered = sosfilt(sos, noise).astype(np.float32)
        envelope = np.exp(-np.linspace(0, 5, length))
        audio[start:end] += (filtered * envelope).astype(np.float32)
    return audio


@pytest.fixture
def drums_audio_path(tmp_audio_dir, sample_rate, duration):
    kick = _generate_transient_burst(50, sample_rate, duration, hit_interval=0.5, decay_ms=80)
    snare = _generate_transient_burst(200, sample_rate, duration, hit_interval=0.5, decay_ms=40)
    hihat = _generate_noise_burst(sample_rate, duration, 6000, 12000, hit_interval=0.25)
    mix = kick + snare * 0.6 + hihat * 0.4
    mix = mix / np.max(np.abs(mix)) * 0.9
    path = tmp_audio_dir / "drums.wav"
    sf.write(str(path), mix, sample_rate)
    return str(path)


@pytest.fixture
def stereo_drums_audio_path(tmp_audio_dir, sample_rate, duration):
    kick = _generate_transient_burst(50, sample_rate, duration, hit_interval=0.5, decay_ms=80)
    snare = _generate_transient_burst(200, sample_rate, duration, hit_interval=0.5, decay_ms=40)
    hihat = _generate_noise_burst(sample_rate, duration, 6000, 12000, hit_interval=0.25)
    left = kick + snare * 0.6 + hihat * 0.3
    right = kick + snare * 0.4 + hihat * 0.5
    stereo = np.vstack([left, right])
    stereo = stereo / np.max(np.abs(stereo)) * 0.9
    path = tmp_audio_dir / "drums_stereo.wav"
    sf.write(str(path), stereo.T, sample_rate)
    return str(path)


@pytest.fixture
def bass_audio_path(tmp_audio_dir, sample_rate, duration):
    bass = _generate_sine(80, sample_rate, duration, amplitude=0.7)
    path = tmp_audio_dir / "bass.wav"
    sf.write(str(path), bass, sample_rate)
    return str(path)


@pytest.fixture
def kick_audio_path(tmp_audio_dir, sample_rate, duration):
    kick = _generate_transient_burst(50, sample_rate, duration, hit_interval=0.5, decay_ms=80)
    path = tmp_audio_dir / "kick.wav"
    sf.write(str(path), kick, sample_rate)
    return str(path)


@pytest.fixture
def silent_audio_path(tmp_audio_dir, sample_rate):
    silence = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
    path = tmp_audio_dir / "silence.wav"
    sf.write(str(path), silence, sample_rate)
    return str(path)


@pytest.fixture
def output_dir(tmp_audio_dir):
    out = tmp_audio_dir / "output"
    out.mkdir()
    return str(out)
