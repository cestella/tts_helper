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

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None


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
    chunk_progress: Optional[Any] = None,
) -> None:
    """Process text file to audiobook using the complete pipeline.

    Args:
        input_text_path: Path to input text file
        output_path: Path to save output audiobook
        config: Configuration dictionary (None = use defaults)
        keep_chunks: Whether to keep individual chunk files
        verbose: Whether to print verbose progress info
        chunk_progress: Optional tqdm progress bar for chunk progress
    """
    from tts_helper import (
        Chunk,
        NemoNormalizer,
        NemoNormalizerConfig,
        SpacySegmenter,
        SpacySegmenterConfig,
        OrpheusTTS,
        OrpheusTTSConfig,
        KokoroTTS,
        KokoroTTSConfig,
        PydubStitcher,
        PydubStitcherConfig,
        TranslationEnhancer,
        TranslationEnhancerConfig,
    )

    config = config or {}

    # Extract component configs
    normalizer_config_dict = config.get("normalizer", {})
    segmenter_config_dict = config.get("segmenter", {})
    tts_config_dict = config.get("tts", {})
    stitcher_config_dict = config.get("stitcher", {})
    tts_engine = config.get("tts_engine", "orpheus").lower()

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
    text_chunks = segmenter.segment(normalized_text)

    # Convert string chunks to Chunk objects
    chunks = [Chunk(text=text) for text in text_chunks]

    if verbose:
        print(f"  Created {len(chunks)} chunks")
        total_chars = sum(len(c.text) for c in chunks)
        avg_chars = total_chars / len(chunks) if chunks else 0
        print(f"  Average chunk size: {avg_chars:.0f} characters")

    # Step 3.5: Enhance chunks (optional)
    enhancer_config_dict = config.get("enhancer", {})
    if enhancer_config_dict:
        enhancer_type = enhancer_config_dict.get("type", "").lower()

        if enhancer_type == "translation":
            if verbose:
                print("Enhancing chunks with translation...")

            # Remove 'type' key before passing to config
            config_without_type = {k: v for k, v in enhancer_config_dict.items() if k != "type"}
            enhancer_config = TranslationEnhancerConfig.from_dict(
                config_without_type
            )
            enhancer = TranslationEnhancer(enhancer_config)
            original_count = len(chunks)
            chunks = enhancer.enhance(chunks)

            if verbose:
                print(f"  Enhanced {original_count} → {len(chunks)} chunks")
        elif enhancer_type:
            print(
                f"Warning: Unknown enhancer type '{enhancer_type}', skipping enhancement",
                file=sys.stderr,
            )

    # Step 4: Synthesize audio for each chunk
    if verbose:
        print(f"Synthesizing speech using {tts_engine.upper()} TTS...")

    # Dynamically select TTS engine based on config
    if tts_engine == "orpheus":
        tts_config = OrpheusTTSConfig.from_dict(tts_config_dict)
        tts = OrpheusTTS(tts_config)
    elif tts_engine == "kokoro":
        tts_config = KokoroTTSConfig.from_dict(tts_config_dict)
        tts = KokoroTTS(tts_config)
    else:
        print(
            f"Error: Unknown TTS engine '{tts_engine}'. Supported engines: 'orpheus', 'kokoro'",
            file=sys.stderr,
        )
        sys.exit(1)

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
            # Handle silence chunks specially
            if chunk.silence_ms is not None:
                if verbose:
                    print(f"  Chunk {i}/{len(chunks)}: Silence ({chunk.silence_ms}ms)")

                # Generate silence audio
                # Use the same sample rate as TTS (typically 24000 for Kokoro, 22050 for Orpheus)
                sample_rate = getattr(tts.config, 'sample_rate', 22050)
                num_samples = int(sample_rate * chunk.silence_ms / 1000.0)
                silence_audio = np.zeros(num_samples, dtype=np.float32)

                chunk_path = chunks_dir / f"chunk_{i:04d}.wav"
                tts.save_audio(silence_audio, sample_rate, chunk_path)
                chunk_files.append(chunk_path)

                # Update chunk progress bar if provided
                if chunk_progress is not None:
                    chunk_progress.update(1)

                continue

            # Normal text chunk - synthesize with TTS
            if verbose:
                chunk_info = f"  Chunk {i}/{len(chunks)}: {len(chunk.text)} chars"
                metadata = []
                if chunk.voice:
                    metadata.append(f"voice={chunk.voice}")
                if chunk.language:
                    metadata.append(f"lang={chunk.language}")
                if chunk.speed is not None:
                    metadata.append(f"speed={chunk.speed}")
                if metadata:
                    chunk_info += f" ({', '.join(metadata)})"
                print(chunk_info)

            # Apply voice/language/speed overrides if specified
            original_voice = None
            original_language = None
            original_speed = None

            if chunk.voice and hasattr(tts.config, 'voice'):
                original_voice = tts.config.voice
                tts.config.voice = chunk.voice

            if chunk.language and hasattr(tts.config, 'language'):
                original_language = tts.config.language
                tts.config.language = chunk.language

            if chunk.speed is not None and hasattr(tts.config, 'speed'):
                original_speed = tts.config.speed
                tts.config.speed = chunk.speed

            try:
                sample_rate, audio = tts.synthesize(chunk.text)

                chunk_path = chunks_dir / f"chunk_{i:04d}.wav"
                tts.save_audio(audio, sample_rate, chunk_path)
                chunk_files.append(chunk_path)

            finally:
                # Restore original config values
                if original_voice is not None:
                    tts.config.voice = original_voice
                if original_language is not None:
                    tts.config.language = original_language
                if original_speed is not None:
                    tts.config.speed = original_speed

            # Update chunk progress bar if provided
            if chunk_progress is not None:
                chunk_progress.update(1)

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


def process_directory(
    input_dir: Path,
    output_dir: Path,
    config: Optional[Dict[str, Any]] = None,
    keep_chunks: bool = False,
    verbose: bool = False,
) -> None:
    """Process all text files in a directory to audiobooks.

    Args:
        input_dir: Path to input directory containing .txt files
        output_dir: Path to output directory for audiobooks
        config: Configuration dictionary (None = use defaults)
        keep_chunks: Whether to keep individual chunk files
        verbose: Whether to print verbose progress info
    """
    # Find all .txt files and sort them
    txt_files = sorted(input_dir.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output format from config
    config = config or {}
    stitcher_config = config.get("stitcher", {})
    output_format = stitcher_config.get("output_format", "mp3")

    # Filter out files that already have outputs
    files_to_process = []
    for txt_file in txt_files:
        output_file = output_dir / f"{txt_file.stem}.{output_format}"
        if output_file.exists():
            if verbose:
                print(f"Skipping {txt_file.name} (output already exists)")
        else:
            files_to_process.append((txt_file, output_file))

    if not files_to_process:
        print("All files already processed!")
        return

    # Process files with progress bars (if not verbose)
    if verbose or not HAS_TQDM:
        # Verbose mode - no progress bars
        for i, (input_file, output_file) in enumerate(files_to_process, 1):
            print(f"\n[{i}/{len(files_to_process)}] Processing: {input_file.name}")
            process_audiobook(
                input_text_path=input_file,
                output_path=output_file,
                config=config,
                keep_chunks=keep_chunks,
                verbose=verbose,
            )
    else:
        # Non-verbose mode - use progress bars
        file_progress = tqdm(
            files_to_process,
            desc="Files",
            unit="file",
            position=0,
            leave=True,
        )

        for input_file, output_file in file_progress:
            file_progress.set_description(f"Files [{input_file.name}]")

            # We need to count chunks first for the progress bar
            # Do a quick segmentation to get chunk count
            from tts_helper import Chunk, SpacySegmenter, SpacySegmenterConfig

            # Read input text
            with input_file.open("r", encoding="utf-8") as f:
                input_text = f.read()

            # Quick segmentation to count chunks
            segmenter_config_dict = config.get("segmenter", {})
            segmenter_config = SpacySegmenterConfig.from_dict(segmenter_config_dict)
            segmenter = SpacySegmenter(segmenter_config)

            # Handle normalization if needed
            if not config.get("skip_normalization", False):
                try:
                    from tts_helper import NemoNormalizer, NemoNormalizerConfig

                    normalizer_config_dict = config.get("normalizer", {})
                    normalizer_config = NemoNormalizerConfig.from_dict(
                        normalizer_config_dict
                    )
                    normalizer = NemoNormalizer(normalizer_config)
                    input_text = normalizer.normalize(input_text)
                except Exception:
                    pass  # Skip normalization if it fails

            text_chunks = segmenter.segment(input_text)

            # Convert to Chunk objects for enhancement counting
            chunks = [Chunk(text=text) for text in text_chunks]

            # Account for enhancement potentially adding more chunks
            enhancer_config_dict = config.get("enhancer", {})
            if enhancer_config_dict.get("type") == "translation":
                # Rough estimate: probability * chunks * 2 additional chunks per translation
                probability = enhancer_config_dict.get("probability", 0.1)
                estimated_additions = int(len(chunks) * probability * 2)
                chunk_count = len(chunks) + estimated_additions
            else:
                chunk_count = len(chunks)

            # Create chunk progress bar
            chunk_progress = tqdm(
                total=chunk_count,
                desc=f"  Chunks",
                unit="chunk",
                position=1,
                leave=False,
            )

            try:
                process_audiobook(
                    input_text_path=input_file,
                    output_path=output_file,
                    config=config,
                    keep_chunks=keep_chunks,
                    verbose=False,
                    chunk_progress=chunk_progress,
                )
            finally:
                chunk_progress.close()


def create_default_config() -> Dict[str, Any]:
    """Create default configuration dictionary.

    Returns:
        Default configuration for all pipeline components
    """
    return {
        "tts_engine": "orpheus",  # Options: "orpheus" or "kokoro"
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
            # Orpheus TTS config (if tts_engine is "orpheus"):
            "language": "english",  # english, french, spanish, italian
            "voice": "tara",  # tara, leah, jess, leo, dan, mia, zac, zoe
            "use_gpu": True,
            "n_gpu_layers": -1,
            "verbose": False,
            # Kokoro TTS config (if tts_engine is "kokoro"):
            # "language": "en-us",  # en-us, en-gb, fr-fr, it, ja, cmn
            # "voice": "af_sarah",  # See README for full voice list
            # "speed": 1.0,  # 0.5 to 2.0
            # "verbose": False,
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
        help="Input text file or directory containing .txt files to convert to audiobook(s)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output audiobook file or directory (for batch processing)",
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
        print("  - tts_engine: Choose 'orpheus' or 'kokoro' TTS engine")
        print("  - normalizer: Text normalization settings")
        print("  - segmenter: Text chunking settings")
        print("  - tts: Speech synthesis settings (depends on tts_engine)")
        print("  - stitcher: Audio stitching settings")
        print("\nNote: See README for TTS engine differences and voice options")
        sys.exit(0)

    # Validate required arguments
    if not args.input:
        parser.error("input file is required (or use --create-config)")

    if not args.output:
        parser.error("--output is required")

    # Validate input exists
    if not args.input.exists():
        print(f"Error: Input path not found: {args.input}", file=sys.stderr)
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

    # Process audiobook or directory
    try:
        if args.input.is_dir():
            # Directory mode - process all .txt files
            process_directory(
                input_dir=args.input,
                output_dir=args.output,
                config=config,
                keep_chunks=args.keep_chunks,
                verbose=args.verbose,
            )
        else:
            # Single file mode
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
