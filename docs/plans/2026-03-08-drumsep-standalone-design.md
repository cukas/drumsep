# drumsep — Standalone Drum Sub-Stem Separator

**Date:** 2026-03-08
**Status:** Approved

## Overview

Extract the drum separation code from AudioFacets into a standalone open-source Python library and CLI tool. Separates a drums audio stem into 5 sub-stems (kick, snare, hi-hat, cymbals, toms) using frequency analysis — no ML models required.

## Key Decisions

- **License:** MIT
- **Python:** 3.10+
- **Dependencies:** numpy, librosa, soundfile (minimal — no torch, no scipy direct)
- **Approach:** Refactored library with examples, notebook, tests, CI (Option B)
- **Includes:** Separator + kick debleed + drum analyzer
- **Excludes:** debleed_piano (not drum-related), abstract plugin system, i18n, tier logic

## Architecture

### Source Files (from AudioFacets → drumsep)

| AudioFacets Source | drumsep Target | Changes |
|---|---|---|
| `separation/drumsep.py` | `src/drumsep/separator.py` | Strip security validators, inline base class, simplify progress |
| `separation/debleed.py` | `src/drumsep/debleed.py` | Drop `debleed_piano`, keep `debleed_kick` only |
| `separation/base.py` | `src/drumsep/types.py` | Keep `SeparationResult` dataclass, drop abstract class |
| `analysis/drum_analyzer.py` | `src/drumsep/analyzer.py` | Direct copy, minimal changes |
| (new) | `src/drumsep/__init__.py` | Public API: `separate()`, `analyze_kick()` |
| (new) | `src/drumsep/cli.py` | CLI entry point with argparse |

### Project Structure

```
drumsep/
├── src/drumsep/
│   ├── __init__.py          # Public API
│   ├── separator.py         # DrumSeparator
│   ├── debleed.py           # debleed_kick()
│   ├── analyzer.py          # DrumAnalyzer + KickAnalysis
│   ├── types.py             # SeparationResult dataclass
│   └── cli.py               # CLI entry point
├── examples/
│   ├── basic_usage.py
│   ├── with_debleed.py
│   ├── analyze_kick.py
│   └── batch_process.py
├── notebooks/
│   └── drumsep_demo.ipynb
├── tests/
│   ├── conftest.py          # Synthetic audio fixtures
│   ├── test_separator.py
│   ├── test_debleed.py
│   ├── test_analyzer.py
│   └── test_cli.py
├── pyproject.toml
├── README.md
├── LICENSE
├── .github/workflows/ci.yml
└── .gitignore
```

## Public API

```python
from drumsep import separate, analyze_kick

result = separate("drums.wav", output_dir="./stems/")
result = separate("drums.wav", output_dir="./stems/", bass_path="bass.wav")
analysis = analyze_kick("./stems/kick.wav")
```

```bash
drumsep drums.wav -o ./stems/
drumsep drums.wav -o ./stems/ --bass bass.wav
drumsep analyze drums.wav
drumsep batch ./drum_folder/ -o ./output/
```

## Separation Approach

1. HPSS pre-processing to separate transients from bass bleed
2. Dual STFT: percussive component (kick/snare/toms) + full signal (hihat/cymbals)
3. Frequency masks with soft roll-off transitions
4. Transient-aware kick mask (onset envelope + spectral flux)
5. Spectral gate on kick (attack/release envelope)
6. Parallel ISTFT synthesis (5 workers, sequential fallback)
7. Optional cross-stem debleed (kick vs bass)
8. Stereo restoration from correlation analysis

## Sub-stems Produced

| Stem | Frequency Range | Source STFT | Special Processing |
|------|----------------|-------------|-------------------|
| kick | 20-100Hz | percussive | transient mask + spectral gate + debleed |
| snare | 150-300Hz + 2-4kHz | percussive | dual-band mask |
| hihat | 6-12kHz | full | transient-weighted |
| cymbals | 3-16kHz | full | hihat-subtracted |
| toms | 80-400Hz | percussive | kick/snare-subtracted |

## Testing

- Synthetic audio fixtures (numpy-generated sine waves, noise bursts)
- Verify: output files exist, correct stem count, frequency content in expected bands
- Debleed: verify similarity reduction
- Analyzer: verify valid metric ranges
- CI: GitHub Actions with Python 3.10/3.11/3.12

## Stripped from AudioFacets

- `validate_audio_path` / `validate_output_directory` → simple Path checks
- `SeparationRunner` abstract base → inlined
- i18n progress reporting → simple (percent, message) callbacks
- Plugin/tier system → removed
- `debleed_piano` → removed (not drum-related)
