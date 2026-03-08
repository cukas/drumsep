# src/drumsep/cli.py
"""Command-line interface for drumsep."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SUBCOMMANDS = {"analyze", "batch"}


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]

    # Handle --version before anything else
    if "--version" in args:
        print("drumsep 0.1.0")
        return 0

    # Detect if first positional arg is a known subcommand
    positional_args = [a for a in args if not a.startswith("-")]
    if positional_args and positional_args[0] in _SUBCOMMANDS:
        command = positional_args[0]
    else:
        command = None

    if command == "analyze":
        return _parse_and_run_analyze(args)
    elif command == "batch":
        return _parse_and_run_batch(args)
    else:
        return _parse_and_run_separate(args)


def _parse_and_run_separate(args: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="drumsep",
        description="Separate drums into kick, snare, hi-hat, cymbals, and toms",
    )
    parser.add_argument("input", nargs="?", help="Path to drums audio file")
    parser.add_argument("-o", "--output", default="./stems", help="Output directory (default: ./stems)")
    parser.add_argument("--bass", help="Bass stem path for kick debleeding")
    parser.add_argument("--no-enhanced", action="store_true", help="Disable HPSS/transient enhancements")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")

    parsed = parser.parse_args(args)

    if not parsed.input:
        parser.print_usage(sys.stderr)
        return 1

    return _cmd_separate(parsed)


def _parse_and_run_analyze(args: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="drumsep analyze",
        description="Analyze a kick drum stem",
    )
    parser.add_argument("input", help="Path to kick drum audio file")

    # Remove the "analyze" token from args before parsing
    filtered = [a for i, a in enumerate(args) if not (a == "analyze" and i == _first_positional_index(args))]
    parsed = parser.parse_args(filtered)
    return _cmd_analyze(parsed)


def _parse_and_run_batch(args: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="drumsep batch",
        description="Process a folder of drum stems",
    )
    parser.add_argument("input_dir", help="Directory containing drum audio files")
    parser.add_argument("-o", "--output", default="./batch_output", help="Output directory")
    parser.add_argument("--bass", help="Bass stem path for kick debleeding")
    parser.add_argument("--no-enhanced", action="store_true", help="Disable enhancements")

    # Remove the "batch" token from args before parsing
    filtered = [a for i, a in enumerate(args) if not (a == "batch" and i == _first_positional_index(args))]
    parsed = parser.parse_args(filtered)
    return _cmd_batch(parsed)


def _first_positional_index(args: list[str]) -> int:
    """Return the index of the first non-flag argument."""
    for i, a in enumerate(args):
        if not a.startswith("-"):
            return i
    return -1


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
            print()
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
