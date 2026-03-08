# src/drumsep/cli.py
"""Command-line interface for drumsep."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="drumsep",
        description="Separate drums into kick, snare, hi-hat, cymbals, and toms",
    )
    parser.add_argument(
        "--version", action="store_true", help="Show version and exit"
    )

    subparsers = parser.add_subparsers(dest="command")

    # Default: separate
    sep_parser = subparsers.add_parser(
        "separate", help="Separate drums into sub-stems"
    )
    sep_parser.add_argument("input", help="Path to drums audio file")
    sep_parser.add_argument(
        "-o", "--output", default="./stems", help="Output directory (default: ./stems)"
    )
    sep_parser.add_argument("--bass", help="Bass stem path for kick debleeding")
    sep_parser.add_argument(
        "--no-enhanced", action="store_true", help="Disable HPSS/transient enhancements"
    )
    sep_parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress progress output"
    )

    # Analyze subcommand
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze a kick drum stem"
    )
    analyze_parser.add_argument("input", help="Path to kick drum audio file")

    # Batch subcommand
    batch_parser = subparsers.add_parser(
        "batch", help="Process a folder of drum stems"
    )
    batch_parser.add_argument("input_dir", help="Directory containing drum audio files")
    batch_parser.add_argument(
        "-o", "--output", default="./batch_output", help="Output directory"
    )
    batch_parser.add_argument("--bass", help="Bass stem path for kick debleeding")
    batch_parser.add_argument(
        "--no-enhanced", action="store_true", help="Disable enhancements"
    )

    args_list = argv if argv is not None else sys.argv[1:]

    # Handle --version before subparser dispatch
    if "--version" in args_list:
        from drumsep import __version__
        print(f"drumsep {__version__}")
        return 0

    # If first positional arg is not a known subcommand, treat as "separate <file>"
    known_commands = {"separate", "analyze", "batch"}
    positional = [a for a in args_list if not a.startswith("-")]
    if positional and positional[0] not in known_commands:
        args_list = ["separate", *args_list]

    if not args_list:
        parser.print_usage(sys.stderr)
        return 1

    parsed = parser.parse_args(args_list)

    if parsed.command == "analyze":
        return _cmd_analyze(parsed)
    elif parsed.command == "batch":
        return _cmd_batch(parsed)
    elif parsed.command == "separate":
        return _cmd_separate(parsed)
    else:
        parser.print_usage(sys.stderr)
        return 1


def _cmd_separate(args: argparse.Namespace) -> int:
    from drumsep import separate

    def on_progress(pct: int, msg: str) -> None:
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


def _cmd_analyze(args: argparse.Namespace) -> int:
    from drumsep import analyze_kick

    try:
        analysis = analyze_kick(args.input)
        print(json.dumps(analysis.to_dict(), indent=2))
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _cmd_batch(args: argparse.Namespace) -> int:
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
