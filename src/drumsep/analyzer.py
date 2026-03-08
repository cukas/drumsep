# src/drumsep/analyzer.py
"""Enhanced drum analysis for kick detection and characterization."""
from __future__ import annotations

import numpy as np
import librosa

from .types import KickAnalysis


class DrumAnalyzer:
    """Specialized analyzer for drum elements with enhanced kick detection."""

    def analyze_kick(self, audio_path: str) -> KickAnalysis:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        fundamental_freq = self._detect_fundamental(y, sr)
        sub_bass_energy = self._measure_sub_bass_energy(y, sr)
        attack_ms, decay_ms = self._measure_envelope(y, sr)
        transient_ratio = self._measure_transient_ratio(y, sr)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid = float(centroid.mean())
        onsets_per_second = self._detect_kick_rate(y, sr)
        return KickAnalysis(
            fundamental_freq=round(fundamental_freq, 2),
            sub_bass_energy=round(sub_bass_energy, 2),
            attack_timing_ms=round(attack_ms, 2),
            decay_time_ms=round(decay_ms, 2),
            transient_ratio=round(transient_ratio, 3),
            spectral_centroid=round(spectral_centroid, 2),
            onsets_per_second=round(onsets_per_second, 2),
        )

    def _detect_fundamental(self, y, sr):
        S = np.abs(librosa.stft(y, n_fft=4096))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
        mask = (freqs >= 20) & (freqs <= 100)
        masked_spectrum = S[mask]
        if masked_spectrum.size == 0:
            return 50.0
        low_freq_spectrum = masked_spectrum.mean(axis=1)
        if low_freq_spectrum.size == 0 or low_freq_spectrum.max() == 0:
            return 50.0
        peak_idx = low_freq_spectrum.argmax()
        return float(freqs[mask][peak_idx])

    def _measure_sub_bass_energy(self, y, sr):
        S = np.abs(librosa.stft(y, n_fft=4096))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
        mask = (freqs >= 20) & (freqs <= 80)
        masked_spectrum = S[mask]
        if masked_spectrum.size == 0:
            return -60.0
        sub_bass_magnitude = masked_spectrum.mean()
        if sub_bass_magnitude == 0:
            return -60.0
        energy_db = librosa.amplitude_to_db(np.array([sub_bass_magnitude]))[0]
        return float(energy_db)

    def _measure_envelope(self, y, sr):
        hop_length = 64
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        if len(rms) == 0:
            return (0.0, 0.0)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        if len(onsets) == 0:
            return (0.0, 0.0)
        first_onset = onsets[0]
        onset_slice = rms[first_onset:min(first_onset + 100, len(rms))]
        if len(onset_slice) == 0:
            return (0.0, 0.0)
        peak_idx = first_onset + onset_slice.argmax()
        attack_samples = (peak_idx - first_onset) * hop_length
        attack_ms = (attack_samples / sr) * 1000
        peak_energy = rms[peak_idx]
        half_energy = peak_energy * 0.5
        decay_idx = peak_idx
        for i in range(peak_idx, min(peak_idx + 500, len(rms))):
            if rms[i] < half_energy:
                decay_idx = i
                break
        decay_samples = (decay_idx - peak_idx) * hop_length
        decay_ms = (decay_samples / sr) * 1000
        return (float(attack_ms), float(decay_ms))

    def _measure_transient_ratio(self, y, sr):
        S = np.abs(librosa.stft(y))
        flux = np.diff(S, axis=1)
        transient_energy = np.sum(flux[flux > 0])
        total_energy = np.sum(S)
        if total_energy > 0:
            return float(min(transient_energy / total_energy, 1.0))
        return 0.0

    def _detect_kick_rate(self, y, sr):
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        duration = len(y) / sr
        if duration > 0:
            return float(len(onsets) / duration)
        return 0.0
