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
