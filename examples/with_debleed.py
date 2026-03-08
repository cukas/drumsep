"""Drum separation with bass debleeding for cleaner kick."""
from drumsep import separate

result = separate(
    "drums.wav",
    output_dir="./stems/",
    bass_path="bass.wav",
)

print(f"Done in {result.processing_time:.1f}s")
for name, path in result.stems.items():
    print(f"  {name}: {path}")
