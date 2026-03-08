"""Tests for drumsep type definitions."""
from drumsep.types import SeparationResult, KickAnalysis, DrumSepError, CancellationError


def test_separation_result_creation():
    result = SeparationResult(
        stems={"kick": "/tmp/kick.wav", "snare": "/tmp/snare.wav"},
        model_name="native_drumsep",
        processing_time=1.5,
    )
    assert result.stems["kick"] == "/tmp/kick.wav"
    assert result.model_name == "native_drumsep"
    assert result.processing_time == 1.5


def test_kick_analysis_creation():
    analysis = KickAnalysis(
        fundamental_freq=52.3, sub_bass_energy=-12.5, attack_timing_ms=4.2,
        decay_time_ms=45.0, transient_ratio=0.78, spectral_centroid=120.5,
        onsets_per_second=2.1,
    )
    assert analysis.fundamental_freq == 52.3
    assert analysis.transient_ratio == 0.78


def test_kick_analysis_to_dict():
    analysis = KickAnalysis(
        fundamental_freq=50.0, sub_bass_energy=-10.0, attack_timing_ms=3.0,
        decay_time_ms=40.0, transient_ratio=0.5, spectral_centroid=100.0,
        onsets_per_second=2.0,
    )
    d = analysis.to_dict()
    assert isinstance(d, dict)
    assert d["fundamental_freq"] == 50.0
    assert "onsets_per_second" in d


def test_drumsep_error_is_exception():
    assert issubclass(DrumSepError, Exception)


def test_cancellation_error_is_exception():
    assert issubclass(CancellationError, Exception)
