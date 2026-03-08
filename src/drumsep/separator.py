"""Drum sub-stem separation using frequency analysis.

Separates a drums audio stem into 5 sub-stems:
- kick: 20-100Hz (low-frequency transients)
- snare: 150-300Hz + 2-4kHz (transient + body)
- hihat: 6-12kHz (high-frequency, short decay)
- cymbals: 3-16kHz (crashes/rides, long decay)
- toms: 80-400Hz (mid-frequency transients)

No ML models required — uses HPSS, frequency masking, transient detection,
and spectral gating.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import sys
import gc
import threading

import numpy as np
import librosa
import soundfile as sf

from .types import SeparationResult, DrumSepError, CancellationError


class DrumSeparator:
    """Separates a drums stem into kick, snare, hihat, cymbals, and toms."""

    STEMS: List[str] = ["kick", "snare", "hihat", "cymbals", "toms"]

    def __init__(
        self,
        enhanced: bool = True,
        cancel_event: Optional[threading.Event] = None,
    ):
        self.enhanced = enhanced
        self.cancel_event = cancel_event

    @property
    def stem_names(self) -> List[str]:
        return self.STEMS.copy()

    def separate(
        self,
        drums_path: str,
        output_dir: str,
        on_progress: Optional[Callable[[int, str], None]] = None,
        bass_path: Optional[str] = None,
    ) -> SeparationResult:
        start_time = time.time()
        audio_path = Path(drums_path)
        out_path = Path(output_dir)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {drums_path}")

        out_path.mkdir(parents=True, exist_ok=True)

        self._check_cancelled()
        self._progress(on_progress, 5, "Loading drums audio...")

        try:
            y, sr = librosa.load(str(audio_path), sr=None, mono=False)

            if y.ndim == 1:
                y_mono = y
                is_stereo = False
            else:
                y_mono = librosa.to_mono(y)
                is_stereo = True

            self._progress(on_progress, 12, "Analyzing drum frequency content...")

            if self.enhanced:
                y_percussive, _ = librosa.effects.hpss(y_mono, margin=(1.0, 5.0))
            else:
                y_percussive = y_mono

            self._progress(on_progress, 18, "Computing STFTs...")

            S_perc = librosa.stft(y_percussive, n_fft=4096, hop_length=512)
            mag_perc = np.abs(S_perc)
            phase_perc = np.angle(S_perc)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

            S_full = librosa.stft(y_mono, n_fft=4096, hop_length=512)
            mag_full = np.abs(S_full)
            phase_full = np.angle(S_full)

            self._check_cancelled()
            self._progress(on_progress, 25, "Computing drum frequency masks...")

            if self.enhanced:
                kick_mask = self._create_kick_mask(freqs, mag_perc, sr, y_percussive)
            else:
                kick_mask = self._create_frequency_mask(freqs, 20, 100)

            snare_mask_low = self._create_frequency_mask(freqs, 150, 300)
            snare_mask_high = self._create_frequency_mask(freqs, 2000, 4000)
            snare_mask = np.maximum(snare_mask_low, snare_mask_high * 0.5)

            hihat_mask = self._create_hihat_mask(freqs, mag_full, sr, y_mono)

            cymbals_mask = self._create_frequency_mask(freqs, 3000, 16000)
            cymbals_mask = cymbals_mask * (1 - hihat_mask * 0.8)

            toms_mask = self._create_frequency_mask(freqs, 80, 400)
            toms_mask = toms_mask * (1 - kick_mask * 0.7) * (1 - snare_mask_low * 0.5)

            self._check_cancelled()
            self._progress(on_progress, 45, "Building drum STFTs...")

            exp_phase_perc = np.exp(1j * phase_perc)
            exp_phase_full = np.exp(1j * phase_full)

            complex_stfts = {
                "kick": mag_perc * kick_mask * exp_phase_perc,
                "snare": mag_perc * snare_mask * exp_phase_perc,
                "toms": mag_perc * toms_mask * exp_phase_perc,
                "hihat": mag_full * hihat_mask * exp_phase_full,
                "cymbals": mag_full * cymbals_mask * exp_phase_full,
            }

            if self.enhanced:
                complex_stfts["kick"] = self._apply_spectral_gate(
                    complex_stfts["kick"], sr
                )

            del kick_mask, snare_mask, snare_mask_low, snare_mask_high
            del hihat_mask, cymbals_mask, toms_mask
            del exp_phase_perc, exp_phase_full, mag_perc, mag_full
            del phase_perc, phase_full, S_perc, S_full

            self._check_cancelled()
            self._progress(on_progress, 55, "Synthesizing drum sub-stems...")

            def _istft(name: str) -> tuple:
                return (name, librosa.istft(complex_stfts[name], hop_length=512))

            audio_results: Dict[str, np.ndarray] = {}
            try:
                with ThreadPoolExecutor(max_workers=5) as pool:
                    for name, audio in pool.map(_istft, self.STEMS):
                        audio_results[name] = audio
            except (MemoryError, Exception) as e:
                print(
                    f"[WARNING] Parallel ISTFT failed ({e}), falling back to sequential",
                    file=sys.stderr,
                )
                audio_results.clear()
                for name in self.STEMS:
                    audio_results[name] = librosa.istft(
                        complex_stfts[name], hop_length=512
                    )

            del freqs, complex_stfts
            gc.collect()

            if self.enhanced and bass_path:
                try:
                    from .debleed import debleed_kick

                    audio_results["kick"] = debleed_kick(
                        audio_results["kick"], bass_path, sr
                    )
                except Exception as e:
                    print(f"[WARNING] Kick debleed failed: {e}", file=sys.stderr)

            self._check_cancelled()
            self._progress(on_progress, 90, "Saving drum sub-stems...")

            substems = dict(audio_results)
            if is_stereo:
                substems = self._restore_stereo(substems, y)

            stem_paths = {}
            for i, (name, audio) in enumerate(substems.items()):
                output_file = out_path / f"{name}.wav"
                sf.write(
                    str(output_file),
                    audio.T if audio.ndim > 1 else audio,
                    sr,
                    subtype="FLOAT",
                )
                stem_paths[name] = str(output_file)
                progress = 90 + int((i + 1) / len(substems) * 10)
                self._progress(on_progress, progress, f"Saved {name} sub-stem...")

            self._progress(on_progress, 100, "Drum separation complete!")

            return SeparationResult(
                stems=stem_paths,
                model_name="native_drumsep",
                processing_time=time.time() - start_time,
            )

        except CancellationError:
            raise
        except Exception as e:
            raise DrumSepError(f"Drum separation failed: {e}") from e

    def _progress(self, callback, percent, message):
        if callback:
            callback(percent, message)

    def _check_cancelled(self):
        if self.cancel_event and self.cancel_event.is_set():
            raise CancellationError("Separation cancelled")

    def _create_frequency_mask(self, freqs, low_hz, high_hz):
        mask = ((freqs >= low_hz) & (freqs <= high_hz)).astype(float)
        transition = 20
        low_trans = (freqs >= low_hz - transition) & (freqs < low_hz)
        high_trans = (freqs > high_hz) & (freqs <= high_hz + transition)
        mask[low_trans] = (freqs[low_trans] - (low_hz - transition)) / transition
        mask[high_trans] = 1 - (freqs[high_trans] - high_hz) / transition
        return mask[:, np.newaxis]

    def _create_kick_mask(self, freqs, magnitude, sr, y_percussive):
        freq_mask = np.zeros_like(freqs, dtype=float)
        low_ramp = (freqs >= 0) & (freqs < 20)
        freq_mask[low_ramp] = freqs[low_ramp] / 20.0
        freq_mask[(freqs >= 20) & (freqs <= 100)] = 1.0
        rolloff = (freqs > 100) & (freqs <= 150)
        if np.any(rolloff):
            freq_mask[rolloff] = 1.0 - (freqs[rolloff] - 100) / 50.0
        freq_mask = freq_mask[:, np.newaxis]

        onset_env = librosa.onset.onset_strength(
            y=y_percussive, sr=sr, fmax=200, hop_length=512
        )
        if len(onset_env) == 0 or onset_env.max() == 0:
            return freq_mask

        onset_norm = onset_env / onset_env.max()

        kick_bins = (freqs >= 20) & (freqs <= 150)
        kick_mag = magnitude[kick_bins, :]
        flux = np.zeros(magnitude.shape[1])
        if magnitude.shape[1] > 1:
            flux[1:] = np.maximum(
                0, np.sum(kick_mag[:, 1:] - kick_mag[:, :-1], axis=0)
            )
        flux_max = flux.max()
        flux_norm = flux / flux_max if flux_max > 0 else flux

        min_len = min(len(onset_norm), len(flux_norm))
        onset_norm = onset_norm[:min_len]
        flux_norm = flux_norm[:min_len]

        transient_gate = np.maximum(onset_norm * 0.7, flux_norm * 0.3)
        transient_gate = np.clip(transient_gate, 0.1, 1.0)

        gate_full = np.ones(magnitude.shape[1])
        gate_full[:min_len] = transient_gate
        gate_full = gate_full[np.newaxis, :]

        return freq_mask * gate_full

    def _apply_spectral_gate(self, kick_stft, sr, gate_floor=0.3):
        magnitude = np.abs(kick_stft)
        frame_energy = np.sum(magnitude, axis=0)
        if frame_energy.max() == 0:
            return kick_stft

        onset_env = frame_energy / frame_energy.max()
        peaks = librosa.util.peak_pick(
            onset_env,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=3,
            delta=0.3,
            wait=4,
        )

        gate = np.full(len(frame_energy), gate_floor)
        attack_frames = 3
        release_frames = 8

        for peak in peaks:
            for i in range(attack_frames):
                idx = peak - attack_frames + i + 1
                if 0 <= idx < len(gate):
                    ramp = (i + 1) / attack_frames
                    gate[idx] = max(
                        gate[idx], gate_floor + (1.0 - gate_floor) * ramp
                    )

            if 0 <= peak < len(gate):
                gate[peak] = 1.0

            for i in range(release_frames):
                idx = peak + i + 1
                if 0 <= idx < len(gate):
                    decay = np.exp(-3.0 * (i + 1) / release_frames)
                    val = gate_floor + (1.0 - gate_floor) * decay
                    gate[idx] = max(gate[idx], val)

        return kick_stft * gate[np.newaxis, :]

    def _create_hihat_mask(self, freqs, magnitude, sr, y):
        freq_mask = self._create_frequency_mask(freqs, 6000, 12000)

        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        if len(onset_env) == 0 or onset_env.max() == 0:
            onset_mask = np.zeros((1, magnitude.shape[1]))
        else:
            onset_mask = np.maximum(0, onset_env / onset_env.max())
            onset_mask = onset_mask[np.newaxis, :]

        return freq_mask * (0.3 + 0.7 * onset_mask)

    def _restore_stereo(self, substems, original_stereo):
        stereo_substems = {}
        for name, mono_audio in substems.items():
            sample_len = min(1000, len(mono_audio), original_stereo.shape[1])
            if sample_len < 10:
                stereo_substems[name] = np.vstack(
                    [mono_audio * 0.5, mono_audio * 0.5]
                )
                continue

            left_corr = np.correlate(
                mono_audio[:sample_len],
                original_stereo[0, :sample_len],
                mode="valid",
            )[0]
            right_corr = np.correlate(
                mono_audio[:sample_len],
                original_stereo[1, :sample_len],
                mode="valid",
            )[0]

            total_corr = abs(left_corr) + abs(right_corr)
            if total_corr > 0:
                left_gain = abs(left_corr) / total_corr
                right_gain = abs(right_corr) / total_corr
            else:
                left_gain = right_gain = 0.5

            stereo_audio = np.zeros((2, len(mono_audio)))
            stereo_audio[0] = mono_audio * left_gain
            stereo_audio[1] = mono_audio * right_gain
            stereo_substems[name] = stereo_audio

        return stereo_substems
