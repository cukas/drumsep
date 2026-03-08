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
