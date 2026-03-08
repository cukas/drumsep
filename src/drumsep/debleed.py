# src/drumsep/debleed.py
"""Cross-stem debleeding using Wiener-filter inspired soft masking.

Compares extracted kick against the bass stem in the 20-150Hz overlap band.
High frame-wise cosine similarity = shared content (bleed) = attenuated from kick.
"""
from __future__ import annotations

import numpy as np
import librosa


def debleed_kick(
    kick_audio: np.ndarray,
    bass_stem_path: str,
    sr: int | float,
    n_fft: int = 4096,
    hop_length: int = 512,
    low_hz: float = 20.0,
    high_hz: float = 150.0,
    strength: float = 0.5,
) -> np.ndarray:
    """Remove bass bleed from extracted kick using soft masking.

    Args:
        kick_audio: Mono kick audio array
        bass_stem_path: Path to the bass stem audio file
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length for STFT
        low_hz: Lower frequency bound for overlap band
        high_hz: Upper frequency bound for overlap band
        strength: Attenuation strength (0=no debleed, 1=full removal)

    Returns:
        Debleeded kick audio (same length as input)
    """
    y_bass, _ = librosa.load(bass_stem_path, sr=sr, mono=True)

    min_len = min(len(kick_audio), len(y_bass))
    kick_trimmed = kick_audio[:min_len]
    bass_trimmed = y_bass[:min_len]

    S_kick = librosa.stft(kick_trimmed, n_fft=n_fft, hop_length=hop_length)
    S_bass = librosa.stft(bass_trimmed, n_fft=n_fft, hop_length=hop_length)

    mag_kick = np.abs(S_kick)
    mag_bass = np.abs(S_bass)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    overlap_bins = (freqs >= low_hz) & (freqs <= high_hz)

    kick_overlap = mag_kick[overlap_bins, :]
    bass_overlap = mag_bass[overlap_bins, :]

    kick_norm = np.linalg.norm(kick_overlap, axis=0) + 1e-10
    bass_norm = np.linalg.norm(bass_overlap, axis=0) + 1e-10

    similarity = np.sum(kick_overlap * bass_overlap, axis=0) / (kick_norm * bass_norm)
    similarity = np.clip(similarity, 0.0, 1.0)

    attenuation = np.ones_like(mag_kick)
    attenuation[overlap_bins, :] = 1.0 - strength * similarity[np.newaxis, :]

    S_debleeded = S_kick * attenuation
    kick_debleeded = librosa.istft(S_debleeded, hop_length=hop_length, length=min_len)

    if len(kick_audio) > min_len:
        result = np.copy(kick_audio)
        result[:min_len] = kick_debleeded
        return result

    return kick_debleeded
