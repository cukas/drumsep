"""Analyze kick drum characteristics."""
from drumsep import analyze_kick
import json

analysis = analyze_kick("kick.wav")

print(f"Fundamental frequency: {analysis.fundamental_freq} Hz")
print(f"Sub-bass energy: {analysis.sub_bass_energy} dB")
print(f"Attack: {analysis.attack_timing_ms} ms")
print(f"Decay: {analysis.decay_time_ms} ms")
print(f"Transient ratio: {analysis.transient_ratio}")
print(f"Spectral centroid: {analysis.spectral_centroid} Hz")
print(f"Kicks per second: {analysis.onsets_per_second}")

print(json.dumps(analysis.to_dict(), indent=2))
