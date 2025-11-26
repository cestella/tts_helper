"""CLI for TTS Helper - Industrial-grade audiobook text-to-speech processor.

This script provides a complete pipeline for converting text files to audiobooks:
1. Text normalization (numbers, dates, currency → spoken form)
2. Text segmentation (split into TTS-friendly chunks)
3. Speech synthesis (convert text chunks to audio)
4. Audio stitching (combine chunks with silence/crossfade)

Usage:
    python -m tts_helper input.txt --config config.json --output audiobook.mp3
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    with config_path.open("r") as f:
        return json.load(f)


def process_audiobook(
    input_text_path: Path,
    output_path: Path,
    config: Optional[Dict[str, Any]] = None,
    keep_chunks: bool = False,
    verbose: bool = False,
) -> None:
    """Process text file to audiobook using the complete pipeline.

    Args:
        input_text_path: Path to input text file
        output_path: Path to save output audiobook
        config: Configuration dictionary (None = use defaults)
        keep_chunks: Whether to keep individual chunk files
        verbose: Whether to print verbose progress info
    """
    from tts_helper import (
        NemoNormalizer,
        NemoNormalizerConfig,
        SpacySegmenter,
        SpacySegmenterConfig,
        OrpheusTTS,
        OrpheusTTSConfig,
        PydubStitcher,
        PydubStitcherConfig,
    )

    config = config or {}

    # Extract component configs
    normalizer_config_dict = config.get("normalizer", {})
    segmenter_config_dict = config.get("segmenter", {})
    tts_config_dict = config.get("tts", {})
    stitcher_config_dict = config.get("stitcher", {})

    # Step 1: Read input text
    if verbose:
        print(f"Reading input text from: {input_text_path}")

    with input_text_path.open("r", encoding="utf-8") as f:
        input_text = f.read()

    if not input_text.strip():
        print("Error: Input text file is empty", file=sys.stderr)
        sys.exit(1)

    # Step 2: Normalize text (optional, skip if disabled)
    if config.get("skip_normalization", False):
        if verbose:
            print("Skipping normalization (disabled in config)")
        normalized_text = input_text
    else:
        if verbose:
            print("Normalizing text...")

        normalizer_config = NemoNormalizerConfig.from_dict(normalizer_config_dict)
        normalizer = NemoNormalizer(normalizer_config)
        normalized_text = normalizer.normalize(input_text)

        if verbose:
            print(f"  Normalized {len(input_text)} → {len(normalized_text)} characters")

    # Step 3: Segment text
    if verbose:
        print("Segmenting text into chunks...")

    segmenter_config = SpacySegmenterConfig.from_dict(segmenter_config_dict)
    segmenter = SpacySegmenter(segmenter_config)
    chunks = segmenter.segment(normalized_text)

    if verbose:
        print(f"  Created {len(chunks)} chunks")
        total_chars = sum(len(c) for c in chunks)
        avg_chars = total_chars / len(chunks) if chunks else 0
        print(f"  Average chunk size: {avg_chars:.0f} characters")

    # Step 4: Synthesize audio for each chunk
    if verbose:
        print("Synthesizing speech...")

    tts_config = OrpheusTTSConfig.from_dict(tts_config_dict)
    tts = OrpheusTTS(tts_config)

    # Create temp dir for chunks or use output dir if keeping chunks
    if keep_chunks:
        chunks_dir = output_path.parent / f"{output_path.stem}_chunks"
        chunks_dir.mkdir(exist_ok=True, parents=True)
        temp_context = None
    else:
        temp_context = tempfile.TemporaryDirectory()
        chunks_dir = Path(temp_context.__enter__())

    chunk_files: List[Path] = []

    try:
        for i, chunk in enumerate(chunks, 1):
            if verbose:
                print(f"  Chunk {i}/{len(chunks)}: {len(chunk)} chars")

            sample_rate, audio = tts.synthesize(chunk)

            chunk_path = chunks_dir / f"chunk_{i:04d}.wav"
            tts.save_audio(audio, sample_rate, chunk_path)
            chunk_files.append(chunk_path)

        # Step 5: Stitch chunks together
        if verbose:
            print(f"Stitching {len(chunk_files)} chunks together...")

        stitcher_config = PydubStitcherConfig.from_dict(stitcher_config_dict)
        stitcher = PydubStitcher(stitcher_config)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        stitcher.stitch(chunk_files, output_path)

        if verbose:
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"\nAudiobook complete!")
            print(f"  Output: {output_path}")
            print(f"  Size: {file_size_mb:.1f} MB")
            if keep_chunks:
                print(f"  Chunks: {chunks_dir}")

    finally:
        # Clean up temp directory if used
        if temp_context is not None:
            temp_context.__exit__(None, None, None)


def create_default_config() -> Dict[str, Any]:
    """Create default configuration dictionary.

    Returns:
        Default configuration for all pipeline components
    """
    return {
        "normalizer": {
            "language": "en",
            "input_case": "cased",
            "verbose": False,
        },
        "segmenter": {
            "language": "en",
            "strategy": "char_count",
            "max_chars": 300,
            "sentences_per_chunk": 3,
        },
        "tts": {
            "language": "english",
            "voice": "tara",
            "use_gpu": True,
            "n_gpu_layers": -1,
            "verbose": False,
        },
        "stitcher": {
            "silence_duration_ms": 750,
            "crossfade_duration_ms": 0,
            "output_format": "mp3",
            "export_bitrate": "192k",
        },
        "skip_normalization": False,
    }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TTS Helper - Industrial-grade audiobook text-to-speech processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  python -m tts_helper input.txt --output audiobook.mp3

  # With custom configuration
  python -m tts_helper input.txt --config config.json --output audiobook.mp3

  # Generate default config file
  python -m tts_helper --create-config

  # Keep intermediate chunk files
  python -m tts_helper input.txt --output audiobook.mp3 --keep-chunks

  # Verbose output
  python -m tts_helper input.txt --output audiobook.mp3 --verbose

For more information, visit: https://github.com/yourusername/tts_helper
        """,
    )

    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Input text file to convert to audiobook",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output audiobook file (e.g., audiobook.mp3)",
    )

    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="JSON configuration file for pipeline components",
    )

    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create a default configuration file (config.json) and exit",
    )

    parser.add_argument(
        "--keep-chunks",
        action="store_true",
        help="Keep individual audio chunk files",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose progress information",
    )

    args = parser.parse_args()

    # Handle --create-config
    if args.create_config:
        config_path = Path("config.json")
        default_config = create_default_config()

        with config_path.open("w") as f:
            json.dump(default_config, f, indent=2)

        print(f"Created default configuration: {config_path}")
        print("\nEdit this file to customize:")
        print("  - normalizer: Text normalization settings")
        print("  - segmenter: Text chunking settings")
        print("  - tts: Speech synthesis settings")
        print("  - stitcher: Audio stitching settings")
        sys.exit(0)

    # Validate required arguments
    if not args.input:
        parser.error("input file is required (or use --create-config)")

    if not args.output:
        parser.error("--output is required")

    # Validate input file exists
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Load configuration
    if args.config:
        if not args.config.exists():
            print(f"Error: Config file not found: {args.config}", file=sys.stderr)
            sys.exit(1)

        try:
            config = load_config(args.config)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in config file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        config = create_default_config()

    # Process audiobook
    try:
        process_audiobook(
            input_text_path=args.input,
            output_path=args.output,
            config=config,
            keep_chunks=args.keep_chunks,
            verbose=args.verbose,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
