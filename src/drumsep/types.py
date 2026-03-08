"""Type definitions for drumsep."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class SeparationResult:
    """Result of drum stem separation."""
    stems: Dict[str, str]
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DrumSepError(Exception):
    pass


class CancellationError(Exception):
    pass
