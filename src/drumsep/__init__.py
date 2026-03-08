# src/drumsep/__init__.py
"""drumsep -- Separate drums into kick, snare, hi-hat, cymbals, and toms.

No ML models required. Uses frequency analysis with HPSS, transient detection,
and spectral gating.

Usage:
    from drumsep import separate, analyze_kick

    result = separate("drums.wav", output_dir="./stems/")
    analysis = analyze_kick("./stems/kick.wav")
"""
from __future__ import annotations

from typing import Optional, Callable

from .types import SeparationResult, KickAnalysis, DrumSepError, CancellationError
from .separator import DrumSeparator
from .analyzer import DrumAnalyzer
from .debleed import debleed_kick as _debleed_kick

try:
    from importlib.metadata import version as _version
    __version__ = _version("drumsep")
except Exception:
    __version__ = "0.1.0"
__all__ = [
    "separate",
    "analyze_kick",
    "DrumSeparator",
    "DrumAnalyzer",
    "SeparationResult",
    "KickAnalysis",
    "DrumSepError",
    "CancellationError",
    "debleed_kick",
]


def separate(
    drums_path: str,
    output_dir: str = "./stems",
    bass_path: Optional[str] = None,
    enhanced: bool = True,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> SeparationResult:
    """Separate a drums audio file into 5 sub-stems."""
    separator = DrumSeparator(enhanced=enhanced)
    return separator.separate(drums_path, output_dir, on_progress=on_progress, bass_path=bass_path)


def analyze_kick(audio_path: str) -> KickAnalysis:
    """Analyze a kick drum audio file."""
    analyzer = DrumAnalyzer()
    return analyzer.analyze_kick(audio_path)


def debleed_kick(kick_audio, bass_stem_path: str, sr: int, **kwargs):
    """Remove bass bleed from kick audio."""
    return _debleed_kick(kick_audio, bass_stem_path, sr, **kwargs)
