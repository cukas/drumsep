# drumsep

[![CI](https://github.com/cukas/drumsep/actions/workflows/ci.yml/badge.svg)](https://github.com/cukas/drumsep/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/drumsep.svg)](https://pypi.org/project/drumsep/)

Separate drums into kick, snare, hi-hat, cymbals, and toms — **no ML models required**.

Uses frequency analysis with HPSS, transient detection, and spectral gating. Pure Python, runs on CPU, works offline.

## Why No ML?

Most drum separation tools (Demucs, HTDemucs) rely on large neural networks that need GPUs, model downloads, and heavy dependencies. drumsep takes a different approach:

| | drumsep | ML-based (Demucs etc.) |
|---|---|---|
| **Dependencies** | 3 (numpy, librosa, soundfile) | 10+ (torch, torchaudio, ...) |
| **Install size** | ~50 MB | ~2 GB+ |
| **GPU required** | No | Recommended |
| **Model download** | None | 80-300 MB per model |
| **Startup time** | <100ms | 2-10s (model loading) |
| **Deterministic** | Yes | Yes |
| **Best for** | Pre-isolated drum stems | Full mix separation |

drumsep is designed for a specific use case: splitting an **already-isolated drums stem** into its sub-components. If you need to separate drums from a full mix, use Demucs first, then pipe the drums stem through drumsep.

## Install

```bash
pip install drumsep
```

Requires `libsndfile` system library (`brew install libsndfile` on macOS, `apt install libsndfile1` on Ubuntu).

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

See [`notebooks/drumsep_demo.ipynb`](notebooks/drumsep_demo.ipynb) for an interactive walkthrough with spectrogram visualizations.

## How It Works

```
drums.wav
    │
    ▼
┌──────────┐
│   HPSS   │──── Harmonic-Percussive Source Separation
└────┬─────┘
     │
     ├─── percussive ──► Dual STFT (4096-point FFT)
     │                       │
     │                  ┌────┴────────────────────┐
     │                  │   Frequency Masking      │
     │                  │   (soft 20Hz roll-offs)  │
     │                  └────┬────────────────────-┘
     │                       │
     │               ┌──────┼──────┬──────┐
     │               ▼      ▼      ▼      ▼
     │             kick   snare   toms   (full signal)
     │              │                      │
     │         transient              ┌────┴────┐
     │          gate +                ▼         ▼
     │         spectral             hihat    cymbals
     │          gate          (onset-weighted)  │
     │              │                      subtract
     │              ▼                      hihat
     │         [optional]
     │         debleed vs
     │         bass stem
     │
     └─── original stereo ──► correlation-based L/R gain recovery
                                        │
                                        ▼
                               5 stereo WAV files
```

1. **HPSS pre-processing** — isolates transients from sustained content
2. **Dual STFT** — percussive component for kick/snare/toms, full signal for hihat/cymbals (preserves harmonic shimmer)
3. **Frequency masking** — soft masks with 20Hz transition roll-offs target each instrument's range
4. **Transient-aware kick detection** — onset envelope + spectral flux gate passes full energy during hits, attenuates bass bleed between hits
5. **Spectral gate** — attack/release envelope on kick (3-frame attack, 8-frame exponential decay)
6. **Cross-stem debleed** — optional Wiener-filter soft masking removes bass content from kick using cosine similarity
7. **Stereo restoration** — correlation-based L/R gain recovery from original stereo image

## Sub-stems

| Stem | Range | Source | Description |
|------|-------|--------|-------------|
| kick | 20-100Hz | Percussive STFT | Low-frequency transients with transient + spectral gate |
| snare | 150-300Hz + 2-4kHz | Percussive STFT | Body + crack (dual-band) |
| hihat | 6-12kHz | Full STFT | High-frequency transient bursts, onset-weighted |
| cymbals | 3-16kHz | Full STFT | Crashes/rides (hihat-subtracted) |
| toms | 80-400Hz | Percussive STFT | Mid-frequency transients (kick/snare-subtracted) |

## API Reference

### `separate(drums_path, output_dir="./stems", bass_path=None, enhanced=True, on_progress=None)`

Separate drums audio into 5 sub-stems. Returns `SeparationResult` with `.stems` dict and `.processing_time`.

- `drums_path` — path to input drums WAV/FLAC/MP3
- `output_dir` — directory for output stems (created if needed)
- `bass_path` — optional bass stem for kick debleeding
- `enhanced` — enable HPSS + transient detection (default: `True`)
- `on_progress` — callback `(percent: int, message: str) -> None`, percent ranges 0-100

### `analyze_kick(audio_path)`

Analyze kick drum audio. Returns `KickAnalysis` with:
- `fundamental_freq` — dominant frequency in Hz
- `sub_bass_energy` — energy in 20-80Hz band (dB)
- `attack_timing_ms` — onset to peak time
- `decay_time_ms` — peak to 50% energy time
- `transient_ratio` — attack vs total energy (0-1)
- `spectral_centroid` — brightness (Hz)
- `onsets_per_second` — kick rate

### `DrumSeparator(enhanced=True, cancel_event=None)`

Low-level class for separation with threading cancellation support via `threading.Event`.

### `DrumAnalyzer()`

Low-level class for drum analysis.

### `debleed_kick(kick_audio, bass_stem_path, sr, strength=0.5)`

Remove bass bleed from kick audio array using Wiener-filter soft masking.

## Known Limitations

- **Designed for pre-isolated drum stems** — not a full-mix separator. Use Demucs/HTDemucs first if you need to extract drums from a full mix.
- **Stereo restoration is approximate** — uses correlation-based L/R gain split, not true panning/phase reconstruction. Works well for center-panned drums, less so for hard-panned elements.
- **Frequency overlap** — some bleed between adjacent stems is expected (e.g., low toms into kick range). The `enhanced` mode and debleed feature help minimize this.
- **Optimized for rock/pop/electronic drums** — acoustic jazz kits with heavy cymbal wash may produce less clean separation.

## Requirements

- Python 3.10+
- numpy, librosa, soundfile
- System: `libsndfile`

No GPU required. No model downloads.

## Development

```bash
git clone https://github.com/cukas/drumsep.git
cd drumsep
pip install -e ".[dev]"
make check   # lint + typecheck + test with coverage
```

Or run individually:

```bash
make test       # pytest with coverage
make lint       # ruff
make typecheck  # mypy
```

## Contributing

Contributions are welcome. Please:

1. Fork the repo and create a feature branch
2. Add tests for new functionality
3. Run `make check` before submitting a PR
4. Keep PRs focused — one feature or fix per PR

## License

MIT
