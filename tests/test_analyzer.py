# tests/test_analyzer.py
"""Tests for DrumAnalyzer."""
from drumsep.analyzer import DrumAnalyzer
from drumsep.types import KickAnalysis


def test_analyze_kick_returns_kick_analysis(kick_audio_path):
    analyzer = DrumAnalyzer()
    result = analyzer.analyze_kick(kick_audio_path)
    assert isinstance(result, KickAnalysis)


def test_analyze_kick_frequency_range(kick_audio_path):
    analyzer = DrumAnalyzer()
    result = analyzer.analyze_kick(kick_audio_path)
    assert 20 <= result.fundamental_freq <= 100


def test_analyze_kick_metrics_valid(kick_audio_path):
    analyzer = DrumAnalyzer()
    result = analyzer.analyze_kick(kick_audio_path)
    assert isinstance(result.sub_bass_energy, float)
    assert result.attack_timing_ms >= 0
    assert result.decay_time_ms >= 0
    assert 0 <= result.transient_ratio <= 1
    assert result.spectral_centroid > 0
    assert result.onsets_per_second >= 0


def test_analyze_kick_to_dict(kick_audio_path):
    analyzer = DrumAnalyzer()
    result = analyzer.analyze_kick(kick_audio_path)
    d = result.to_dict()
    assert "fundamental_freq" in d
    assert "onsets_per_second" in d
    assert len(d) == 7


def test_analyze_silent_audio(silent_audio_path):
    analyzer = DrumAnalyzer()
    result = analyzer.analyze_kick(silent_audio_path)
    assert isinstance(result, KickAnalysis)
