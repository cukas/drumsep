"""Basic drum separation — 5 lines of code."""
from drumsep import separate

result = separate("drums.wav", output_dir="./stems/")

print(f"Done in {result.processing_time:.1f}s")
for name, path in result.stems.items():
    print(f"  {name}: {path}")
