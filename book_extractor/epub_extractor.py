"""EPUB chapter extractor with lexicographic naming."""

import re
from pathlib import Path
from typing import List, Optional

try:
    import ebooklib
    from ebooklib import epub
    import trafilatura
except ImportError:
    ebooklib = None
    epub = None
    trafilatura = None

from .text_normalizer import normalize_text_for_tts


def safe_filename(name: str, default: str = "chapter") -> str:
    """Convert a string to a safe filename.

    Args:
        name: The original name
        default: Default name if result is empty

    Returns:
        A safe filename string
    """
    # Replace any non-word/non-dash/non-dot characters with underscore
    slug = re.sub(r"[^\w.-]+", "_", name.strip())
    # Remove leading/trailing underscores and dots
    slug = slug.strip("._")
    return slug or default


def _normalize_language_code(language_hint: Optional[str]) -> str:
    """Deprecated: moved to text_normalizer.normalize_text_for_tts."""
    # Kept for backward compatibility if other modules import it directly
    if not language_hint:
        return "en"
    parts = language_hint.split("-")
    return parts[0].strip().lower() or "en"


def _extract_heading_title(html_content: str) -> Optional[str]:
    """Fallback heading extraction when metadata has no title."""
    match = re.search(
        r"<h([1-3])[^>]*>(.*?)</h\1>",
        html_content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None

    heading_text = re.sub(r"\s+", " ", match.group(2)).strip()
    return heading_text or None


def _normalize_text_for_tts(text: str, language_hint: Optional[str] = None) -> str:
    """Deprecated shim to maintain compatibility."""
    return normalize_text_for_tts(text, language_hint=language_hint)


def extract_text_from_html(html_content: str) -> tuple[Optional[str], str]:
    """Extract title and text from HTML content using trafilatura."""
    if trafilatura is None:
        raise ImportError(
            "trafilatura is required for EPUB extraction. Install with: pip install trafilatura"
        )

    # Extract cleaned text and metadata
    meta = trafilatura.extract_metadata(html_content)
    text = trafilatura.extract(
        html_content,
        include_links=False,
        fast=True,
        favor_recall=True,
        output_format="txt",
    )

    if not text:
        return (None, "")

    title = meta.title if meta else None
    language = meta.language if meta else None
    if not title:
        title = _extract_heading_title(html_content)

    normalized_text = _normalize_text_for_tts(text, language_hint=language)

    return title, normalized_text


def extract_epub_chapters(
    epub_path: Path,
    output_dir: Path,
    verbose: bool = False,
) -> List[Path]:
    """Extract chapters from an EPUB file with lexicographic naming.

    Chapters are saved as text files with names like:
        0001_chapter_title.txt
        0002_another_chapter.txt
        ...

    This ensures proper lexicographic ordering when listing files.

    Args:
        epub_path: Path to the EPUB file
        output_dir: Directory to save extracted chapters
        verbose: Whether to print progress information

    Returns:
        List of paths to extracted chapter files

    Raises:
        ImportError: If required libraries are not installed
        FileNotFoundError: If EPUB file doesn't exist
    """
    if ebooklib is None or epub is None:
        raise ImportError(
            "ebooklib is required for EPUB extraction. "
            "Install with: pip install ebooklib trafilatura"
        )
    if trafilatura is None:
        raise ImportError(
            "trafilatura is required for EPUB extraction. "
            "Install with: pip install trafilatura"
        )

    if not epub_path.exists():
        raise FileNotFoundError(f"EPUB file not found: {epub_path}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Extracting chapters from: {epub_path}")
        print(f"Output directory: {output_dir}")

    # Load the EPUB
    book = epub.read_epub(str(epub_path))

    # Track extracted files and titles for duplicate detection
    extracted_files = []
    seen_titles = {}

    # Get all document items (chapters)
    items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))

    if verbose:
        print(f"Found {len(items)} document items")

    chapter_idx = 1

    for item in items:
        try:
            # Get HTML content
            content = item.get_content()
            if not content:
                continue

            # Extract title and text
            title, text = extract_text_from_html(content.decode("utf-8"))

            # Skip if no meaningful text
            if not text or len(text.strip()) < 50:
                if verbose:
                    print(f"  Skipping item (too short): {item.get_name()}")
                continue

            # Generate filename
            if title:
                # Handle duplicate titles
                base_title = title
                if base_title in seen_titles:
                    seen_titles[base_title] += 1
                    title = f"{base_title}_{seen_titles[base_title]}"
                else:
                    seen_titles[base_title] = 1

                stem = safe_filename(title, default=f"chapter_{chapter_idx:04d}")
            else:
                stem = f"chapter_{chapter_idx:04d}"

            # Create filename with 4-digit prefix for lexicographic ordering
            filename = f"{chapter_idx:04d}_{stem}.txt"
            out_path = output_dir / filename

            # Write the text
            out_path.write_text(text, encoding="utf-8")
            extracted_files.append(out_path)

            if verbose:
                print(f"  [{chapter_idx:04d}] {filename} ({len(text)} chars)")

            chapter_idx += 1

        except Exception as e:
            if verbose:
                print(f"  Error processing item {item.get_name()}: {e}")
            continue

    if verbose:
        print(f"\nExtracted {len(extracted_files)} chapters to {output_dir}")

    return extracted_files
