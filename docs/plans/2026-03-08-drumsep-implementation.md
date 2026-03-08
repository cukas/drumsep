# drumsep — Standalone Drum Separator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract AudioFacets drum separation into a standalone open-source Python library + CLI at `/Users/nicolascukas/Web/drumsep`.

**Architecture:** Pure Python library using librosa for STFT/HPSS/onset detection, numpy for array ops, soundfile for WAV I/O. No ML models. Separator class with frequency masking + transient detection produces 5 sub-stems (kick, snare, hihat, cymbals, toms). Optional kick debleed against bass stem. Drum analyzer provides detailed kick metrics.

**Tech Stack:** Python 3.10+, numpy, librosa, soundfile, pytest, argparse

---

### Task 1: Project Scaffolding

**Files:**
- Create: `/Users/nicolascukas/Web/drumsep/pyproject.toml`
- Create: `/Users/nicolascukas/Web/drumsep/.gitignore`
- Create: `/Users/nicolascukas/Web/drumsep/LICENSE`
- Create: `/Users/nicolascukas/Web/drumsep/src/drumsep/__init__.py` (empty placeholder)

**Step 1: Initialize git repo**

```bash
cd /Users/nicolascukas/Web/drumsep
git init
```

**Step 2: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "drumsep"
version = "0.1.0"
description = "Separate drums into kick, snare, hi-hat, cymbals, and toms — no ML models required"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [
    { name = "Nicolas Cukas" },
]
keywords = ["audio", "drums", "separation", "stems", "music", "dsp"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Multimedia :: Sound/Audio :: Analysis",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "numpy>=1.24.0",
    "librosa>=0.10.0",
    "soundfile>=0.12.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1.0",
]

[project.scripts]
drumsep = "drumsep.cli:main"

[project.urls]
Homepage = "https://github.com/nicklascukas/drumsep"
Repository = "https://github.com/nicklascukas/drumsep"
Issues = "https://github.com/nicklascukas/drumsep/issues"

[tool.hatch.build.targets.wheel]
packages = ["src/drumsep"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
target-version = "py310"
line-length = 100
```

**Step 3: Create .gitignore**

```
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
.eggs/
*.egg
.venv/
venv/
env/
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/
*.wav
*.mp3
*.flac
*.ogg
.DS_Store
*.ipynb_checkpoints/
```

**Step 4: Create LICENSE**

```
MIT License

Copyright (c) 2026 Nicolas Cukas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Step 5: Create empty package**

```bash
mkdir -p /Users/nicolascukas/Web/drumsep/src/drumsep
touch /Users/nicolascukas/Web/drumsep/src/drumsep/__init__.py
```

**Step 6: Commit**

```bash
cd /Users/nicolascukas/Web/drumsep
git add pyproject.toml .gitignore LICENSE src/drumsep/__init__.py docs/
git commit -m "chore: initial project scaffolding"
```

---

### Task 2: Types Module

**Files:**
- Create: `/Users/nicolascukas/Web/drumsep/src/drumsep/types.py`
- Create: `/Users/nicolascukas/Web/drumsep/tests/test_types.py`

**Step 1: Write the failing test**

```python
# tests/test_types.py
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
        fundamental_freq=52.3,
        sub_bass_energy=-12.5,
        attack_timing_ms=4.2,
        decay_time_ms=45.0,
        transient_ratio=0.78,
        spectral_centroid=120.5,
        onsets_per_second=2.1,
    )
    assert analysis.fundamental_freq == 52.3
    assert analysis.transient_ratio == 0.78


def test_kick_analysis_to_dict():
    analysis = KickAnalysis(
        fundamental_freq=50.0,
        sub_bass_energy=-10.0,
        attack_timing_ms=3.0,
        decay_time_ms=40.0,
        transient_ratio=0.5,
        spectral_centroid=100.0,
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
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/nicolascukas/Web/drumsep && python -m pytest tests/test_types.py -v`
Expected: FAIL (ModuleNotFoundError)

**Step 3: Write implementation**

```python
# src/drumsep/types.py
"""Type definitions for drumsep."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class SeparationResult:
    """Result of drum stem separation.

    Attributes:
        stems: Dict mapping stem name to output file path
        model_name: Name of the separation method used
        processing_time: Time taken in seconds
    """
    stems: Dict[str, str]
    model_name: str
    processing_time: float


@dataclass
class KickAnalysis:
    """Detailed kick drum analysis results.

    Attributes:
        fundamental_freq: Dominant low-frequency peak in Hz
        sub_bass_energy: Energy in 20-80Hz band in dB
        attack_timing_ms: Time from onset to peak in milliseconds
        decay_time_ms: Time from peak to 50% energy in milliseconds
        transient_ratio: Ratio of transient to total energy (0-1)
        spectral_centroid: Brightness measure in Hz
        onsets_per_second: Average kick hits per second
    """
    fundamental_freq: float
    sub_bass_energy: float
    attack_timing_ms: float
    decay_time_ms: float
    transient_ratio: float
    spectral_centroid: float
    onsets_per_second: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class DrumSepError(Exception):
    """Raised when drum separation fails."""
    pass


class CancellationError(Exception):
    """Raised when separation is cancelled."""
    pass
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/nicolascukas/Web/drumsep && python -m pytest tests/test_types.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add src/drumsep/types.py tests/test_types.py
git commit -m "feat: add type definitions (SeparationResult, KickAnalysis, errors)"
```

---

### Task 3: Test Fixtures (Synthetic Audio)

**Files:**
- Create: `/Users/nicolascukas/Web/drumsep/tests/conftest.py`

**Step 1: Create synthetic audio fixtures**

These fixtures generate short WAV files with known frequency content for testing. No real audio needed.

```python
# tests/conftest.py
"""Shared test fixtures — synthetic audio generators."""
from __future__ import annotations

import numpy as np
import soundfile as sf
import pytest
from pathlib import Path


@pytest.fixture
def tmp_audio_dir(tmp_path):
    """Temporary directory for audio files."""
    return tmp_path


@pytest.fixture
def sample_rate():
    """Standard sample rate for test audio."""
    return 44100


@pytest.fixture
def duration():
    """Duration in seconds for test audio."""
    return 2.0


def _generate_sine(freq: float, sr: int, duration: float, amplitude: float = 0.5) -> np.ndarray:
    """Generate a sine wave."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _generate_transient_burst(
    freq: float, sr: int, duration: float, hit_interval: float = 0.5, decay_ms: float = 50.0
) -> np.ndarray:
    """Generate repeated transient bursts (like drum hits)."""
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


def _generate_noise_burst(
    sr: int, duration: float, low_hz: float, high_hz: float, hit_interval: float = 0.25
) -> np.ndarray:
    """Generate band-limited noise bursts (like hi-hats)."""
    from scipy.signal import butter, sosfilt

    n_samples = int(sr * duration)
    audio = np.zeros(n_samples, dtype=np.float32)
    hit_samples = int(sr * hit_interval)
    burst_len = int(sr * 0.02)  # 20ms bursts

    # Band-pass filter
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
    """Generate a synthetic drums mix with kick, snare-like, and hihat-like content."""
    kick = _generate_transient_burst(50, sample_rate, duration, hit_interval=0.5, decay_ms=80)
    snare = _generate_transient_burst(200, sample_rate, duration, hit_interval=0.5, decay_ms=40)
    hihat = _generate_noise_burst(sample_rate, duration, 6000, 12000, hit_interval=0.25)

    mix = kick + snare * 0.6 + hihat * 0.4
    mix = mix / np.max(np.abs(mix)) * 0.9  # Normalize

    path = tmp_audio_dir / "drums.wav"
    sf.write(str(path), mix, sample_rate)
    return str(path)


@pytest.fixture
def stereo_drums_audio_path(tmp_audio_dir, sample_rate, duration):
    """Generate stereo synthetic drums."""
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
    """Generate a synthetic bass stem (low-frequency sine)."""
    bass = _generate_sine(80, sample_rate, duration, amplitude=0.7)
    path = tmp_audio_dir / "bass.wav"
    sf.write(str(path), bass, sample_rate)
    return str(path)


@pytest.fixture
def kick_audio_path(tmp_audio_dir, sample_rate, duration):
    """Generate a synthetic kick drum (50Hz transient bursts)."""
    kick = _generate_transient_burst(50, sample_rate, duration, hit_interval=0.5, decay_ms=80)
    path = tmp_audio_dir / "kick.wav"
    sf.write(str(path), kick, sample_rate)
    return str(path)


@pytest.fixture
def silent_audio_path(tmp_audio_dir, sample_rate):
    """Generate a silent audio file."""
    silence = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
    path = tmp_audio_dir / "silence.wav"
    sf.write(str(path), silence, sample_rate)
    return str(path)


@pytest.fixture
def output_dir(tmp_audio_dir):
    """Output directory for separated stems."""
    out = tmp_audio_dir / "output"
    out.mkdir()
    return str(out)
```

**Step 2: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add synthetic audio fixtures for drum testing"
```

---

### Task 4: Separator Module

**Files:**
- Create: `/Users/nicolascukas/Web/drumsep/src/drumsep/separator.py`
- Create: `/Users/nicolascukas/Web/drumsep/tests/test_separator.py`

**Step 1: Write the failing tests**

```python
# tests/test_separator.py
"""Tests for DrumSeparator."""
import os
from pathlib import Path

from drumsep.separator import DrumSeparator
from drumsep.types import SeparationResult


def test_separator_produces_five_stems(drums_audio_path, output_dir):
    sep = DrumSeparator()
    result = sep.separate(drums_audio_path, output_dir)

    assert isinstance(result, SeparationResult)
    assert len(result.stems) == 5
    assert set(result.stems.keys()) == {"kick", "snare", "hihat", "cymbals", "toms"}


def test_separator_creates_wav_files(drums_audio_path, output_dir):
    sep = DrumSeparator()
    result = sep.separate(drums_audio_path, output_dir)

    for name, path in result.stems.items():
        assert os.path.exists(path), f"{name} stem file not created"
        assert path.endswith(".wav")


def test_separator_result_metadata(drums_audio_path, output_dir):
    sep = DrumSeparator()
    result = sep.separate(drums_audio_path, output_dir)

    assert result.model_name == "native_drumsep"
    assert result.processing_time > 0


def test_separator_stereo_input(stereo_drums_audio_path, output_dir):
    sep = DrumSeparator()
    result = sep.separate(stereo_drums_audio_path, output_dir)

    assert len(result.stems) == 5
    for path in result.stems.values():
        assert os.path.exists(path)


def test_separator_creates_output_dir(drums_audio_path, tmp_path):
    new_dir = str(tmp_path / "new_output")
    sep = DrumSeparator()
    result = sep.separate(drums_audio_path, new_dir)

    assert os.path.isdir(new_dir)
    assert len(result.stems) == 5


def test_separator_without_enhanced(drums_audio_path, output_dir):
    sep = DrumSeparator(enhanced=False)
    result = sep.separate(drums_audio_path, output_dir)

    assert len(result.stems) == 5


def test_separator_progress_callback(drums_audio_path, output_dir):
    progress_calls = []

    def on_progress(percent, message):
        progress_calls.append((percent, message))

    sep = DrumSeparator()
    sep.separate(drums_audio_path, output_dir, on_progress=on_progress)

    assert len(progress_calls) > 0
    assert progress_calls[-1][0] == 100


def test_separator_cancellation(drums_audio_path, output_dir):
    import threading
    from drumsep.types import CancellationError

    cancel = threading.Event()
    cancel.set()  # Cancel immediately

    sep = DrumSeparator(cancel_event=cancel)

    import pytest
    with pytest.raises(CancellationError):
        sep.separate(drums_audio_path, output_dir)


def test_separator_invalid_path(output_dir):
    sep = DrumSeparator()

    import pytest
    with pytest.raises(Exception):
        sep.separate("/nonexistent/drums.wav", output_dir)


def test_separator_stem_names():
    sep = DrumSeparator()
    assert sep.stem_names == ["kick", "snare", "hihat", "cymbals", "toms"]


def test_separator_silent_audio(silent_audio_path, output_dir):
    sep = DrumSeparator()
    result = sep.separate(silent_audio_path, output_dir)
    # Should complete without error even on silence
    assert len(result.stems) == 5
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/nicolascukas/Web/drumsep && python -m pytest tests/test_separator.py -v`
Expected: FAIL (ModuleNotFoundError)

**Step 3: Write implementation**

This is adapted from AudioFacets' `drumsep.py` — stripped of security validators, i18n, abstract base, and plugin system.

```python
# src/drumsep/separator.py
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
    """Separates a drums stem into kick, snare, hihat, cymbals, and toms.

    Uses frequency-based analysis with HPSS pre-processing, transient-aware
    kick detection, and spectral gating. No ML models required.

    Args:
        enhanced: Enable HPSS + transient kick mask + spectral gate (default True)
        cancel_event: Optional threading.Event for cancellation support
    """

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
        """List of sub-stem names produced by separation."""
        return self.STEMS.copy()

    def separate(
        self,
        drums_path: str,
        output_dir: str,
        on_progress: Optional[Callable[[int, str], None]] = None,
        bass_path: Optional[str] = None,
    ) -> SeparationResult:
        """Separate drums stem into 5 sub-stems.

        Args:
            drums_path: Path to the drums audio file (WAV, FLAC, MP3, etc.)
            output_dir: Directory to save separated sub-stems
            on_progress: Optional callback(percent, message) for progress reporting
            bass_path: Optional bass stem path for kick debleeding

        Returns:
            SeparationResult with stem file paths and metadata

        Raises:
            DrumSepError: If separation fails
            CancellationError: If cancelled via cancel_event
            FileNotFoundError: If drums_path doesn't exist
        """
        start_time = time.time()

        audio_path = Path(drums_path)
        out_path = Path(output_dir)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {drums_path}")

        out_path.mkdir(parents=True, exist_ok=True)

        self._check_cancelled()
        self._progress(on_progress, 5, "Loading drums audio...")

        try:
            # Load audio
            y, sr = librosa.load(str(audio_path), sr=None, mono=False)

            if y.ndim == 1:
                y_mono = y
                is_stereo = False
            else:
                y_mono = librosa.to_mono(y)
                is_stereo = True

            self._progress(on_progress, 12, "Analyzing drum frequency content...")

            # HPSS pre-processing
            if self.enhanced:
                y_percussive, _ = librosa.effects.hpss(y_mono, margin=(1.0, 5.0))
            else:
                y_percussive = y_mono

            self._progress(on_progress, 18, "Computing STFTs...")

            # STFT for percussive signal (kick/snare/toms)
            S_perc = librosa.stft(y_percussive, n_fft=4096, hop_length=512)
            mag_perc = np.abs(S_perc)
            phase_perc = np.angle(S_perc)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

            # STFT for full signal (hihat/cymbals preserve harmonic shimmer)
            S_full = librosa.stft(y_mono, n_fft=4096, hop_length=512)
            mag_full = np.abs(S_full)
            phase_full = np.angle(S_full)

            self._check_cancelled()
            self._progress(on_progress, 25, "Computing drum frequency masks...")

            # --- Compute all masks ---
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

            # --- Build complex STFTs ---
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
                complex_stfts["kick"] = self._apply_spectral_gate(complex_stfts["kick"], sr)

            # Free memory
            del kick_mask, snare_mask, snare_mask_low, snare_mask_high
            del hihat_mask, cymbals_mask, toms_mask
            del exp_phase_perc, exp_phase_full, mag_perc, mag_full
            del phase_perc, phase_full, S_perc, S_full

            self._check_cancelled()
            self._progress(on_progress, 55, "Synthesizing drum sub-stems...")

            # --- Parallel ISTFT ---
            def _istft(name: str) -> tuple:
                return (name, librosa.istft(complex_stfts[name], hop_length=512))

            audio_results: Dict[str, np.ndarray] = {}
            try:
                with ThreadPoolExecutor(max_workers=5) as pool:
                    for name, audio in pool.map(_istft, self.STEMS):
                        audio_results[name] = audio
            except (MemoryError, Exception) as e:
                print(f"[WARNING] Parallel ISTFT failed ({e}), falling back to sequential",
                      file=sys.stderr)
                audio_results.clear()
                for name in self.STEMS:
                    audio_results[name] = librosa.istft(complex_stfts[name], hop_length=512)

            del freqs, complex_stfts
            gc.collect()

            # Cross-stem debleed
            if self.enhanced and bass_path:
                try:
                    from .debleed import debleed_kick
                    audio_results["kick"] = debleed_kick(audio_results["kick"], bass_path, sr)
                except Exception as e:
                    print(f"[WARNING] Kick debleed failed: {e}", file=sys.stderr)

            self._check_cancelled()
            self._progress(on_progress, 90, "Saving drum sub-stems...")

            # Stereo restoration
            substems = dict(audio_results)
            if is_stereo:
                substems = self._restore_stereo(substems, y)

            # Save WAV files
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

    def _progress(
        self,
        callback: Optional[Callable[[int, str], None]],
        percent: int,
        message: str,
    ) -> None:
        if callback:
            callback(percent, message)

    def _check_cancelled(self) -> None:
        if self.cancel_event and self.cancel_event.is_set():
            raise CancellationError("Separation cancelled")

    def _create_frequency_mask(self, freqs: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
        """Create soft frequency mask with 20Hz transition roll-off."""
        mask = ((freqs >= low_hz) & (freqs <= high_hz)).astype(float)

        transition = 20
        low_trans = (freqs >= low_hz - transition) & (freqs < low_hz)
        high_trans = (freqs > high_hz) & (freqs <= high_hz + transition)

        mask[low_trans] = (freqs[low_trans] - (low_hz - transition)) / transition
        mask[high_trans] = 1 - (freqs[high_trans] - high_hz) / transition

        return mask[:, np.newaxis]

    def _create_kick_mask(
        self, freqs: np.ndarray, magnitude: np.ndarray, sr: int, y_percussive: np.ndarray
    ) -> np.ndarray:
        """Transient-aware kick mask: frequency + onset + spectral flux."""
        # Frequency mask: 20-150Hz with soft rolloff above 100Hz
        freq_mask = np.zeros_like(freqs, dtype=float)
        low_ramp = (freqs >= 0) & (freqs < 20)
        freq_mask[low_ramp] = freqs[low_ramp] / 20.0
        freq_mask[(freqs >= 20) & (freqs <= 100)] = 1.0
        rolloff = (freqs > 100) & (freqs <= 150)
        if np.any(rolloff):
            freq_mask[rolloff] = 1.0 - (freqs[rolloff] - 100) / 50.0
        freq_mask = freq_mask[:, np.newaxis]

        # Low-frequency onset envelope
        onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr, fmax=200, hop_length=512)
        if len(onset_env) == 0 or onset_env.max() == 0:
            return freq_mask

        onset_norm = onset_env / onset_env.max()

        # Spectral flux in kick range (20-150Hz)
        kick_bins = (freqs >= 20) & (freqs <= 150)
        kick_mag = magnitude[kick_bins, :]
        flux = np.zeros(magnitude.shape[1])
        if magnitude.shape[1] > 1:
            flux[1:] = np.maximum(0, np.sum(kick_mag[:, 1:] - kick_mag[:, :-1], axis=0))
        flux_max = flux.max()
        flux_norm = flux / flux_max if flux_max > 0 else flux

        # Align lengths
        min_len = min(len(onset_norm), len(flux_norm))
        onset_norm = onset_norm[:min_len]
        flux_norm = flux_norm[:min_len]

        # Combined transient gate
        transient_gate = np.maximum(onset_norm * 0.7, flux_norm * 0.3)
        transient_gate = np.clip(transient_gate, 0.1, 1.0)

        gate_full = np.ones(magnitude.shape[1])
        gate_full[:min_len] = transient_gate
        gate_full = gate_full[np.newaxis, :]

        return freq_mask * gate_full

    def _apply_spectral_gate(
        self, kick_stft: np.ndarray, sr: int, gate_floor: float = 0.3
    ) -> np.ndarray:
        """Attack/release spectral gate for kick. Prevents bass bleed between hits."""
        magnitude = np.abs(kick_stft)
        frame_energy = np.sum(magnitude, axis=0)
        if frame_energy.max() == 0:
            return kick_stft

        onset_env = frame_energy / frame_energy.max()
        peaks = librosa.util.peak_pick(
            onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.3, wait=4
        )

        gate = np.full(len(frame_energy), gate_floor)
        attack_frames = 3
        release_frames = 8

        for peak in peaks:
            for i in range(attack_frames):
                idx = peak - attack_frames + i + 1
                if 0 <= idx < len(gate):
                    ramp = (i + 1) / attack_frames
                    gate[idx] = max(gate[idx], gate_floor + (1.0 - gate_floor) * ramp)
            if 0 <= peak < len(gate):
                gate[peak] = 1.0
            for i in range(release_frames):
                idx = peak + i + 1
                if 0 <= idx < len(gate):
                    decay = np.exp(-3.0 * (i + 1) / release_frames)
                    val = gate_floor + (1.0 - gate_floor) * decay
                    gate[idx] = max(gate[idx], val)

        return kick_stft * gate[np.newaxis, :]

    def _create_hihat_mask(
        self, freqs: np.ndarray, magnitude: np.ndarray, sr: int, y: np.ndarray
    ) -> np.ndarray:
        """Hi-hat mask using frequency (6-12kHz) + transient characteristics."""
        freq_mask = self._create_frequency_mask(freqs, 6000, 12000)

        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)

        if len(onset_env) == 0 or onset_env.max() == 0:
            onset_mask = np.zeros((1, magnitude.shape[1]))
        else:
            onset_mask = np.maximum(0, onset_env / onset_env.max())
            onset_mask = onset_mask[np.newaxis, :]

        return freq_mask * (0.3 + 0.7 * onset_mask)

    def _restore_stereo(
        self, substems: Dict[str, np.ndarray], original_stereo: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Restore stereo image using correlation with original channels."""
        stereo_substems = {}

        for name, mono_audio in substems.items():
            sample_len = min(1000, len(mono_audio), original_stereo.shape[1])
            if sample_len < 10:
                stereo_substems[name] = np.vstack([mono_audio * 0.5, mono_audio * 0.5])
                continue

            left_corr = np.correlate(
                mono_audio[:sample_len], original_stereo[0, :sample_len], mode="valid"
            )[0]
            right_corr = np.correlate(
                mono_audio[:sample_len], original_stereo[1, :sample_len], mode="valid"
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
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/nicolascukas/Web/drumsep && python -m pytest tests/test_separator.py -v`
Expected: 11 passed

**Step 5: Commit**

```bash
git add src/drumsep/separator.py tests/test_separator.py
git commit -m "feat: add DrumSeparator with frequency masking and transient detection"
```

---

### Task 5: Debleed Module

**Files:**
- Create: `/Users/nicolascukas/Web/drumsep/src/drumsep/debleed.py`
- Create: `/Users/nicolascukas/Web/drumsep/tests/test_debleed.py`

**Step 1: Write the failing test**

```python
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

    # Debleed should change the audio (reduce bass content)
    result = debleed_kick(kick, bass_audio_path, sample_rate)
    assert not np.array_equal(kick, result)


def test_debleed_strength_zero_returns_original(kick_audio_path, bass_audio_path, sample_rate):
    kick, _ = sf.read(kick_audio_path)
    result = debleed_kick(kick, bass_audio_path, sample_rate, strength=0.0)
    np.testing.assert_allclose(result, kick[:len(result)], atol=1e-5)


def test_debleed_handles_length_mismatch(tmp_path, sample_rate):
    # Kick is longer than bass
    kick = np.random.randn(sample_rate * 3).astype(np.float32)
    bass = np.random.randn(sample_rate * 2).astype(np.float32)

    bass_path = str(tmp_path / "short_bass.wav")
    sf.write(bass_path, bass, sample_rate)

    result = debleed_kick(kick, bass_path, sample_rate)
    assert len(result) == len(kick)
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/nicolascukas/Web/drumsep && python -m pytest tests/test_debleed.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
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
    sr: int,
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
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/nicolascukas/Web/drumsep && python -m pytest tests/test_debleed.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/drumsep/debleed.py tests/test_debleed.py
git commit -m "feat: add kick debleed with Wiener-filter soft masking"
```

---

### Task 6: Analyzer Module

**Files:**
- Create: `/Users/nicolascukas/Web/drumsep/src/drumsep/analyzer.py`
- Create: `/Users/nicolascukas/Web/drumsep/tests/test_analyzer.py`

**Step 1: Write the failing test**

```python
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

    # Synthetic kick is 50Hz sine burst
    assert 20 <= result.fundamental_freq <= 100


def test_analyze_kick_metrics_valid(kick_audio_path):
    analyzer = DrumAnalyzer()
    result = analyzer.analyze_kick(kick_audio_path)

    assert result.sub_bass_energy < 0  # dB, should be negative
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

    # Should return defaults without crashing
    assert isinstance(result, KickAnalysis)
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/nicolascukas/Web/drumsep && python -m pytest tests/test_analyzer.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/drumsep/analyzer.py
"""Enhanced drum analysis for kick detection and characterization."""
from __future__ import annotations

import numpy as np
import librosa

from .types import KickAnalysis


class DrumAnalyzer:
    """Specialized analyzer for drum elements with enhanced kick detection."""

    def analyze_kick(self, audio_path: str) -> KickAnalysis:
        """Analyze a kick drum stem for detailed metrics.

        Args:
            audio_path: Path to kick drum audio file

        Returns:
            KickAnalysis with fundamental frequency, attack/decay timing,
            transient ratio, spectral centroid, and onset rate
        """
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

    def _detect_fundamental(self, y: np.ndarray, sr: int) -> float:
        """Detect dominant low-frequency peak (20-100Hz)."""
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

    def _measure_sub_bass_energy(self, y: np.ndarray, sr: int) -> float:
        """Measure energy in sub-bass range (20-80Hz) in dB."""
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

    def _measure_envelope(self, y: np.ndarray, sr: int) -> tuple:
        """Measure attack and decay timing in milliseconds."""
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

    def _measure_transient_ratio(self, y: np.ndarray, sr: int) -> float:
        """Ratio of transient (attack) energy to total energy (0-1)."""
        S = np.abs(librosa.stft(y))
        flux = np.diff(S, axis=1)
        transient_energy = np.sum(flux[flux > 0])
        total_energy = np.sum(S)

        if total_energy > 0:
            return float(min(transient_energy / total_energy, 1.0))

        return 0.0

    def _detect_kick_rate(self, y: np.ndarray, sr: int) -> float:
        """Average kick drum hits per second."""
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

        duration = len(y) / sr
        if duration > 0:
            return float(len(onsets) / duration)

        return 0.0
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/nicolascukas/Web/drumsep && python -m pytest tests/test_analyzer.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add src/drumsep/analyzer.py tests/test_analyzer.py
git commit -m "feat: add DrumAnalyzer with kick frequency/transient/envelope metrics"
```

---

### Task 7: Public API (__init__.py)

**Files:**
- Modify: `/Users/nicolascukas/Web/drumsep/src/drumsep/__init__.py`

**Step 1: Write implementation**

```python
# src/drumsep/__init__.py
"""drumsep — Separate drums into kick, snare, hi-hat, cymbals, and toms.

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
    """Separate a drums audio file into 5 sub-stems.

    Args:
        drums_path: Path to drums audio (WAV, FLAC, MP3, etc.)
        output_dir: Directory for output WAV files (default: ./stems)
        bass_path: Optional bass stem for kick debleeding
        enhanced: Use HPSS + transient detection + spectral gate (default True)
        on_progress: Optional callback(percent, message)

    Returns:
        SeparationResult with paths to kick.wav, snare.wav, hihat.wav,
        cymbals.wav, toms.wav

    Example:
        >>> result = separate("drums.wav")
        >>> print(result.stems["kick"])
        ./stems/kick.wav
    """
    separator = DrumSeparator(enhanced=enhanced)
    return separator.separate(drums_path, output_dir, on_progress=on_progress, bass_path=bass_path)


def analyze_kick(audio_path: str) -> KickAnalysis:
    """Analyze a kick drum audio file.

    Args:
        audio_path: Path to kick drum audio file

    Returns:
        KickAnalysis with fundamental_freq, sub_bass_energy, attack_timing_ms,
        decay_time_ms, transient_ratio, spectral_centroid, onsets_per_second

    Example:
        >>> analysis = analyze_kick("kick.wav")
        >>> print(f"Fundamental: {analysis.fundamental_freq}Hz")
        Fundamental: 52.3Hz
    """
    analyzer = DrumAnalyzer()
    return analyzer.analyze_kick(audio_path)


def debleed_kick(kick_audio, bass_stem_path: str, sr: int, **kwargs):
    """Remove bass bleed from kick audio. See drumsep.debleed.debleed_kick for full docs."""
    return _debleed_kick(kick_audio, bass_stem_path, sr, **kwargs)
```

**Step 2: Verify imports work**

Run: `cd /Users/nicolascukas/Web/drumsep && python -c "from drumsep import separate, analyze_kick, DrumSeparator, DrumAnalyzer; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/drumsep/__init__.py
git commit -m "feat: add public API with separate() and analyze_kick() convenience functions"
```

---

### Task 8: CLI Module

**Files:**
- Create: `/Users/nicolascukas/Web/drumsep/src/drumsep/cli.py`
- Create: `/Users/nicolascukas/Web/drumsep/tests/test_cli.py`

**Step 1: Write the failing test**

```python
# tests/test_cli.py
"""Tests for CLI."""
import json
import subprocess
import sys
from pathlib import Path


def test_cli_separate(drums_audio_path, tmp_path):
    output = str(tmp_path / "cli_output")
    result = subprocess.run(
        [sys.executable, "-m", "drumsep", drums_audio_path, "-o", output],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0
    assert Path(output, "kick.wav").exists()
    assert Path(output, "snare.wav").exists()


def test_cli_analyze(kick_audio_path):
    result = subprocess.run(
        [sys.executable, "-m", "drumsep", "analyze", kick_audio_path],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "fundamental_freq" in data


def test_cli_no_args():
    result = subprocess.run(
        [sys.executable, "-m", "drumsep"],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode != 0


def test_cli_version():
    result = subprocess.run(
        [sys.executable, "-m", "drumsep", "--version"],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0
    assert "0.1.0" in result.stdout
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/nicolascukas/Web/drumsep && python -m pytest tests/test_cli.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/drumsep/cli.py
"""Command-line interface for drumsep."""
from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="drumsep",
        description="Separate drums into kick, snare, hi-hat, cymbals, and toms",
    )
    parser.add_argument("--version", action="version", version="drumsep 0.1.0")

    subparsers = parser.add_subparsers(dest="command")

    # Default: separate
    parser.add_argument("input", nargs="?", help="Path to drums audio file")
    parser.add_argument("-o", "--output", default="./stems", help="Output directory (default: ./stems)")
    parser.add_argument("--bass", help="Bass stem path for kick debleeding")
    parser.add_argument("--no-enhanced", action="store_true", help="Disable HPSS/transient enhancements")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")

    # analyze subcommand
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a kick drum stem")
    analyze_parser.add_argument("input", help="Path to kick drum audio file")

    # batch subcommand
    batch_parser = subparsers.add_parser("batch", help="Process a folder of drum stems")
    batch_parser.add_argument("input_dir", help="Directory containing drum audio files")
    batch_parser.add_argument("-o", "--output", default="./batch_output", help="Output directory")
    batch_parser.add_argument("--bass", help="Bass stem path for kick debleeding")
    batch_parser.add_argument("--no-enhanced", action="store_true", help="Disable enhancements")

    args = parser.parse_args(argv)

    if args.command == "analyze":
        return _cmd_analyze(args)
    elif args.command == "batch":
        return _cmd_batch(args)
    elif args.input:
        return _cmd_separate(args)
    else:
        parser.print_usage(sys.stderr)
        return 1


def _cmd_separate(args) -> int:
    from drumsep import separate

    def on_progress(pct, msg):
        if not args.quiet:
            print(f"\r[{pct:3d}%] {msg}", end="", flush=True)

    try:
        result = separate(
            args.input,
            output_dir=args.output,
            bass_path=args.bass,
            enhanced=not args.no_enhanced,
            on_progress=on_progress,
        )
        if not args.quiet:
            print()  # newline after progress
            print(f"\nSeparated into {len(result.stems)} stems in {result.processing_time:.1f}s:")
            for name, path in result.stems.items():
                print(f"  {name}: {path}")
        return 0
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


def _cmd_analyze(args) -> int:
    from drumsep import analyze_kick

    try:
        analysis = analyze_kick(args.input)
        print(json.dumps(analysis.to_dict(), indent=2))
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _cmd_batch(args) -> int:
    from drumsep import separate

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: {args.input_dir} is not a directory", file=sys.stderr)
        return 1

    audio_extensions = {".wav", ".flac", ".mp3", ".ogg", ".aif", ".aiff"}
    files = [f for f in sorted(input_dir.iterdir()) if f.suffix.lower() in audio_extensions]

    if not files:
        print(f"No audio files found in {args.input_dir}", file=sys.stderr)
        return 1

    output_base = Path(args.output)
    errors = 0

    for i, audio_file in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Processing {audio_file.name}...")
        out_dir = output_base / audio_file.stem
        try:
            separate(
                str(audio_file),
                output_dir=str(out_dir),
                bass_path=args.bass,
                enhanced=not args.no_enhanced,
            )
            print(f"  -> {out_dir}")
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            errors += 1

    print(f"\nDone: {len(files) - errors}/{len(files)} files processed")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
```

Also create `__main__.py` so `python -m drumsep` works:

```python
# src/drumsep/__main__.py
"""Allow running drumsep as a module: python -m drumsep"""
from drumsep.cli import main
import sys

sys.exit(main())
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/nicolascukas/Web/drumsep && python -m pytest tests/test_cli.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/drumsep/cli.py src/drumsep/__main__.py tests/test_cli.py
git commit -m "feat: add CLI with separate, analyze, and batch commands"
```

---

### Task 9: Examples

**Files:**
- Create: `/Users/nicolascukas/Web/drumsep/examples/basic_usage.py`
- Create: `/Users/nicolascukas/Web/drumsep/examples/with_debleed.py`
- Create: `/Users/nicolascukas/Web/drumsep/examples/analyze_kick.py`
- Create: `/Users/nicolascukas/Web/drumsep/examples/batch_process.py`

**Step 1: Create examples**

```python
# examples/basic_usage.py
"""Basic drum separation — 5 lines of code."""
from drumsep import separate

result = separate("drums.wav", output_dir="./stems/")

print(f"Done in {result.processing_time:.1f}s")
for name, path in result.stems.items():
    print(f"  {name}: {path}")
```

```python
# examples/with_debleed.py
"""Drum separation with bass debleeding for cleaner kick."""
from drumsep import separate

# Pass the bass stem to remove bass bleed from the kick
result = separate(
    "drums.wav",
    output_dir="./stems/",
    bass_path="bass.wav",  # Optional: debleeds the kick
)

print(f"Done in {result.processing_time:.1f}s")
for name, path in result.stems.items():
    print(f"  {name}: {path}")
```

```python
# examples/analyze_kick.py
"""Analyze kick drum characteristics."""
from drumsep import analyze_kick

analysis = analyze_kick("kick.wav")

print(f"Fundamental frequency: {analysis.fundamental_freq} Hz")
print(f"Sub-bass energy: {analysis.sub_bass_energy} dB")
print(f"Attack: {analysis.attack_timing_ms} ms")
print(f"Decay: {analysis.decay_time_ms} ms")
print(f"Transient ratio: {analysis.transient_ratio}")
print(f"Spectral centroid: {analysis.spectral_centroid} Hz")
print(f"Kicks per second: {analysis.onsets_per_second}")

# Export as JSON
import json
print(json.dumps(analysis.to_dict(), indent=2))
```

```python
# examples/batch_process.py
"""Process a folder of drum stems."""
from pathlib import Path
from drumsep import separate

input_dir = Path("./drum_stems/")
output_dir = Path("./separated/")

for audio_file in sorted(input_dir.glob("*.wav")):
    print(f"Processing {audio_file.name}...")
    out = output_dir / audio_file.stem
    result = separate(str(audio_file), output_dir=str(out))
    print(f"  -> {len(result.stems)} stems in {result.processing_time:.1f}s")
```

**Step 2: Commit**

```bash
git add examples/
git commit -m "docs: add usage examples (basic, debleed, analyze, batch)"
```

---

### Task 10: Jupyter Notebook

**Files:**
- Create: `/Users/nicolascukas/Web/drumsep/notebooks/drumsep_demo.ipynb`

**Step 1: Create notebook**

Create a Jupyter notebook with these cells:

Cell 1 (markdown):
```markdown
# drumsep Demo
Separate drums into kick, snare, hi-hat, cymbals, and toms — no ML models required.
```

Cell 2 (code):
```python
# Install if needed
# !pip install drumsep

from drumsep import separate, analyze_kick
```

Cell 3 (markdown):
```markdown
## Separate Drums
```

Cell 4 (code):
```python
# Replace with your drums file
result = separate("drums.wav", output_dir="./demo_stems/")

print(f"Separated in {result.processing_time:.1f}s")
for name, path in result.stems.items():
    print(f"  {name}: {path}")
```

Cell 5 (markdown):
```markdown
## Visualize Frequency Content
```

Cell 6 (code):
```python
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(5, 1, figsize=(12, 15))

for ax, (name, path) in zip(axes, result.stems.items()):
    y, sr = librosa.load(path, sr=None)
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(S, sr=sr, hop_length=512, x_axis='time', y_axis='hz', ax=ax)
    ax.set_title(name.capitalize())
    ax.set_ylim(0, 16000)

plt.tight_layout()
plt.show()
```

Cell 7 (markdown):
```markdown
## Analyze Kick
```

Cell 8 (code):
```python
analysis = analyze_kick(result.stems["kick"])

print(f"Fundamental: {analysis.fundamental_freq} Hz")
print(f"Sub-bass energy: {analysis.sub_bass_energy} dB")
print(f"Attack: {analysis.attack_timing_ms} ms")
print(f"Decay: {analysis.decay_time_ms} ms")
print(f"Transient ratio: {analysis.transient_ratio}")
print(f"Kicks/sec: {analysis.onsets_per_second}")
```

**Step 2: Commit**

```bash
git add notebooks/
git commit -m "docs: add Jupyter notebook demo with spectrogram visualization"
```

---

### Task 11: README

**Files:**
- Create: `/Users/nicolascukas/Web/drumsep/README.md`

**Step 1: Write README**

```markdown
# drumsep

Separate drums into kick, snare, hi-hat, cymbals, and toms — no ML models required.

Uses frequency analysis with HPSS, transient detection, and spectral gating. Pure Python, runs on CPU, works offline.

## Install

```bash
pip install drumsep
```

## Quick Start

### Python

```python
from drumsep import separate, analyze_kick

# Separate drums into 5 sub-stems
result = separate("drums.wav", output_dir="./stems/")
# -> kick.wav, snare.wav, hihat.wav, cymbals.wav, toms.wav

# Optional: debleed kick against bass
result = separate("drums.wav", output_dir="./stems/", bass_path="bass.wav")

# Analyze kick characteristics
analysis = analyze_kick("./stems/kick.wav")
print(f"Fundamental: {analysis.fundamental_freq}Hz")
print(f"Attack: {analysis.attack_timing_ms}ms")
```

### CLI

```bash
# Separate
drumsep drums.wav -o ./stems/

# With bass debleeding
drumsep drums.wav -o ./stems/ --bass bass.wav

# Analyze kick
drumsep analyze kick.wav

# Batch process
drumsep batch ./drum_folder/ -o ./output/
```

## How It Works

1. **HPSS pre-processing** — Harmonic-Percussive Source Separation isolates transients from sustained content
2. **Dual STFT** — Percussive component for kick/snare/toms, full signal for hihat/cymbals (preserves harmonic shimmer)
3. **Frequency masking** — Soft masks with 20Hz transition roll-offs target each instrument's range
4. **Transient-aware kick detection** — Onset envelope + spectral flux gate passes full energy during hits, attenuates bass bleed between hits
5. **Spectral gate** — Attack/release envelope on kick (3-frame attack, 8-frame exponential decay)
6. **Cross-stem debleed** — Optional Wiener-filter soft masking removes bass content from kick using cosine similarity
7. **Stereo restoration** — Correlation-based L/R gain recovery from original stereo image

## Sub-stems

| Stem | Range | Description |
|------|-------|-------------|
| kick | 20-100Hz | Low-frequency transients with transient gate |
| snare | 150-300Hz + 2-4kHz | Body + crack (dual-band) |
| hihat | 6-12kHz | High-frequency transient bursts |
| cymbals | 3-16kHz | Crashes/rides (hihat-subtracted) |
| toms | 80-400Hz | Mid-frequency transients (kick/snare-subtracted) |

## API Reference

### `separate(drums_path, output_dir="./stems", bass_path=None, enhanced=True, on_progress=None)`

Separate drums audio into 5 sub-stems. Returns `SeparationResult` with `.stems` dict and `.processing_time`.

### `analyze_kick(audio_path)`

Analyze kick drum audio. Returns `KickAnalysis` with:
- `fundamental_freq` — Dominant frequency in Hz
- `sub_bass_energy` — Energy in 20-80Hz band (dB)
- `attack_timing_ms` — Onset to peak time
- `decay_time_ms` — Peak to 50% energy time
- `transient_ratio` — Attack vs total energy (0-1)
- `spectral_centroid` — Brightness (Hz)
- `onsets_per_second` — Kick rate

### `DrumSeparator(enhanced=True, cancel_event=None)`

Low-level class for separation with cancellation support.

### `DrumAnalyzer()`

Low-level class for drum analysis.

### `debleed_kick(kick_audio, bass_stem_path, sr, strength=0.5)`

Remove bass bleed from kick audio array using Wiener-filter soft masking.

## Requirements

- Python 3.10+
- numpy
- librosa
- soundfile

No GPU required. No model downloads.

## Development

```bash
git clone https://github.com/nicklascukas/drumsep.git
cd drumsep
pip install -e ".[dev]"
pytest
```

## License

MIT
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with install, usage, API reference, and algorithm overview"
```

---

### Task 12: GitHub Actions CI

**Files:**
- Create: `/Users/nicolascukas/Web/drumsep/.github/workflows/ci.yml`

**Step 1: Create CI config**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install system dependencies
        run: sudo apt-get update && sudo apt-get install -y libsndfile1

      - name: Install package
        run: pip install -e ".[dev]"

      - name: Lint
        run: ruff check src/ tests/

      - name: Test
        run: pytest -v --tb=short

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install ruff
      - run: ruff check src/ tests/
```

**Step 2: Commit**

```bash
mkdir -p /Users/nicolascukas/Web/drumsep/.github/workflows
git add .github/
git commit -m "ci: add GitHub Actions workflow for Python 3.10/3.11/3.12"
```

---

### Task 13: Run All Tests and Final Commit

**Step 1: Install dev dependencies**

```bash
cd /Users/nicolascukas/Web/drumsep
pip install -e ".[dev]"
```

**Step 2: Run full test suite**

```bash
cd /Users/nicolascukas/Web/drumsep
python -m pytest -v
```

Expected: All tests pass (20+ tests)

**Step 3: Run linter**

```bash
cd /Users/nicolascukas/Web/drumsep
ruff check src/ tests/
```

Fix any issues.

**Step 4: Final verification**

```bash
# Test CLI
python -m drumsep --version

# Test import
python -c "from drumsep import separate, analyze_kick; print('All imports OK')"
```

**Step 5: Tag release**

```bash
git tag v0.1.0
```
