"""Command-line interface for book extractor."""

import argparse
import sys
from pathlib import Path

from .epub_extractor import extract_epub_chapters
from .markdown_extractor import extract_markdown_chapters


def main():
    """Main entry point for book extractor CLI."""
    parser = argparse.ArgumentParser(
        description="Extract chapters from EPUB and Markdown books with lexicographic naming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract EPUB chapters
  python -m book_extractor book.epub --output chapters/

  # Extract Markdown chapters
  python -m book_extractor book.md --output chapters/

  # Verbose output
  python -m book_extractor book.epub --output chapters/ --verbose

Output files will be named with lexicographic ordering:
  0001_chapter_name.txt
  0002_another_chapter.txt
  ...
        """,
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input book file (EPUB or Markdown)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output directory for extracted chapters",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed progress information",
    )

    parser.add_argument(
        "--chapter-pattern",
        type=str,
        default=None,
        help="For markdown: regex pattern used to find chapter boundaries (defaults to '^# .+')",
    )

    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language hint (ISO code, e.g., en, es, fr) used for sentence segmentation and normalization",
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1

    # Determine file type and extract
    suffix = args.input.suffix.lower()

    try:
        if suffix == ".epub":
            files = extract_epub_chapters(
                epub_path=args.input,
                output_dir=args.output,
                verbose=args.verbose,
                language_hint=args.language,
            )
        elif suffix in {".md", ".markdown"}:
            files = extract_markdown_chapters(
                md_path=args.input,
                output_dir=args.output,
                chapter_pattern=args.chapter_pattern,
                verbose=args.verbose,
                language_hint=args.language,
            )
        else:
            print(
                f"Error: Unsupported file type: {suffix}",
                file=sys.stderr,
            )
            print("Supported types: .epub, .md, .markdown", file=sys.stderr)
            return 1

        if not files:
            print("Warning: No chapters were extracted", file=sys.stderr)
            return 1

        if not args.verbose:
            print(f"Successfully extracted {len(files)} chapters to {args.output}")

        return 0

    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nTo install required dependencies:", file=sys.stderr)
        print("  For EPUB: pip install ebooklib trafilatura", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
