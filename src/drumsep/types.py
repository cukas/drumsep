"""Type definitions for drumsep."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class SeparationResult:
    """Result of drum stem separation."""
    stems: dict[str, str]
    model_name: str
    processing_time: float


@dataclass
class KickAnalysis:
    """Detailed kick drum analysis results."""
    fundamental_freq: float
    sub_bass_energy: float
    attack_timing_ms: float
    decay_time_ms: float
    transient_ratio: float
    spectral_centroid: float
    onsets_per_second: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DrumSepError(Exception):
    pass


class CancellationError(DrumSepError):
    pass
