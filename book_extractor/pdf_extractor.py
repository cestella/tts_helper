"""PDF chapter extractor with lexicographic naming."""

import re
from pathlib import Path
from typing import List, Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pymupdf4llm
except ImportError:
    pymupdf4llm = None


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


def extract_pdf_chapters(
    pdf_path: Path,
    output_dir: Path,
    use_toc: bool = True,
    verbose: bool = False,
) -> List[Path]:
    """Extract chapters from a PDF file with lexicographic naming.

    Chapters are saved as text files with names like:
        0001_chapter_title.txt
        0002_another_chapter.txt
        ...

    This ensures proper lexicographic ordering when listing files.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted chapters
        use_toc: Whether to use table of contents for chapter detection
        verbose: Whether to print progress information

    Returns:
        List of paths to extracted chapter files

    Raises:
        ImportError: If required libraries are not installed
        FileNotFoundError: If PDF file doesn't exist
    """
    if fitz is None:
        raise ImportError(
            "PyMuPDF is required for PDF extraction. "
            "Install with: pip install pymupdf"
        )

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Extracting chapters from: {pdf_path}")
        print(f"Output directory: {output_dir}")

    # Open the PDF
    doc = fitz.open(str(pdf_path))
    extracted_files = []

    try:
        # Try to extract using TOC if available
        if use_toc:
            toc = doc.get_toc()
            if toc and len(toc) > 0:
                if verbose:
                    print(f"Found TOC with {len(toc)} entries")
                extracted_files = _extract_with_toc(doc, toc, output_dir, verbose)
                if extracted_files:
                    return extracted_files

        # Fallback: Extract using markdown (if available) or page-by-page
        if verbose:
            print("No TOC found or TOC extraction failed, using markdown extraction")
        extracted_files = _extract_with_markdown(doc, pdf_path, output_dir, verbose)

    finally:
        doc.close()

    if verbose:
        print(f"\nExtracted {len(extracted_files)} chapters to {output_dir}")

    return extracted_files


def _extract_with_toc(
    doc: "fitz.Document",
    toc: List[tuple],
    output_dir: Path,
    verbose: bool,
) -> List[Path]:
    """Extract chapters using PDF table of contents.

    Args:
        doc: PyMuPDF document object
        toc: Table of contents list
        output_dir: Output directory
        verbose: Whether to print progress

    Returns:
        List of extracted file paths
    """
    extracted_files = []
    seen_titles = {}

    # Process TOC entries
    chapters = []
    for i, (level, title, page_num) in enumerate(toc):
        # Only use level 1 headings as chapter boundaries
        if level == 1:
            start_page = page_num - 1  # PyMuPDF uses 0-based indexing
            # Find end page (next chapter or end of document)
            end_page = doc.page_count
            for j in range(i + 1, len(toc)):
                if toc[j][0] == 1:  # Next level 1 heading
                    end_page = toc[j][2] - 1
                    break
            chapters.append((title, start_page, end_page))

    if not chapters:
        if verbose:
            print("  No level-1 chapters found in TOC")
        return []

    # Extract text for each chapter
    for idx, (title, start_page, end_page) in enumerate(chapters, 1):
        try:
            # Extract text from pages
            text_parts = []
            for page_num in range(start_page, end_page):
                if page_num >= doc.page_count:
                    break
                page = doc[page_num]
                text_parts.append(page.get_text())

            text = "\n".join(text_parts).strip()

            if not text or len(text) < 50:
                if verbose:
                    print(f"  Skipping chapter (too short): {title}")
                continue

            # Handle duplicate titles
            base_title = title
            if base_title in seen_titles:
                seen_titles[base_title] += 1
                title = f"{base_title}_{seen_titles[base_title]}"
            else:
                seen_titles[base_title] = 1

            # Generate filename
            stem = safe_filename(title, default=f"chapter_{idx:04d}")
            filename = f"{idx:04d}_{stem}.txt"
            out_path = output_dir / filename

            # Write the text
            out_path.write_text(text, encoding="utf-8")
            extracted_files.append(out_path)

            if verbose:
                print(f"  [{idx:04d}] {filename} (pages {start_page+1}-{end_page}, {len(text)} chars)")

        except Exception as e:
            if verbose:
                print(f"  Error extracting chapter '{title}': {e}")
            continue

    return extracted_files


def _extract_with_markdown(
    doc: "fitz.Document",
    pdf_path: Path,
    output_dir: Path,
    verbose: bool,
) -> List[Path]:
    """Extract chapters using markdown conversion.

    If pymupdf4llm is available, use it for better text extraction.
    Otherwise, fall back to page-by-page extraction.

    Args:
        doc: PyMuPDF document object
        pdf_path: Path to PDF file
        output_dir: Output directory
        verbose: Whether to print progress

    Returns:
        List of extracted file paths
    """
    if pymupdf4llm is not None:
        return _extract_with_pymupdf4llm(pdf_path, output_dir, verbose)
    else:
        return _extract_page_by_page(doc, output_dir, verbose)


def _extract_with_pymupdf4llm(
    pdf_path: Path,
    output_dir: Path,
    verbose: bool,
) -> List[Path]:
    """Extract using pymupdf4llm for markdown conversion.

    Args:
        pdf_path: Path to PDF file
        output_dir: Output directory
        verbose: Whether to print progress

    Returns:
        List of extracted file paths
    """
    extracted_files = []

    try:
        # Convert PDF to markdown
        md_text = pymupdf4llm.to_markdown(str(pdf_path))

        # Split by markdown headers (# Header)
        chapters = re.split(r'\n# ', md_text)

        seen_titles = {}

        for idx, chapter in enumerate(chapters, 1):
            if not chapter.strip():
                continue

            # Extract title from first line
            lines = chapter.strip().split('\n', 1)
            title = lines[0].strip('#').strip()
            text = lines[1] if len(lines) > 1 else ""

            if not text or len(text) < 50:
                # If no text after header, include full chapter
                text = chapter

            if len(text.strip()) < 50:
                if verbose:
                    print(f"  Skipping chapter (too short): {title}")
                continue

            # Handle duplicate titles
            base_title = title if title else f"chapter_{idx}"
            if base_title in seen_titles:
                seen_titles[base_title] += 1
                display_title = f"{base_title}_{seen_titles[base_title]}"
            else:
                seen_titles[base_title] = 1
                display_title = base_title

            # Generate filename
            stem = safe_filename(display_title, default=f"chapter_{idx:04d}")
            filename = f"{idx:04d}_{stem}.txt"
            out_path = output_dir / filename

            # Write the text
            out_path.write_text(text, encoding="utf-8")
            extracted_files.append(out_path)

            if verbose:
                print(f"  [{idx:04d}] {filename} ({len(text)} chars)")

    except Exception as e:
        if verbose:
            print(f"  Error with pymupdf4llm extraction: {e}")

    return extracted_files


def _extract_page_by_page(
    doc: "fitz.Document",
    output_dir: Path,
    verbose: bool,
) -> List[Path]:
    """Extract PDF page by page as fallback.

    Args:
        doc: PyMuPDF document object
        output_dir: Output directory
        verbose: Whether to print progress

    Returns:
        List of extracted file paths
    """
    extracted_files = []

    # Extract each page as a separate "chapter"
    for page_num in range(doc.page_count):
        try:
            page = doc[page_num]
            text = page.get_text()

            if not text or len(text.strip()) < 50:
                if verbose:
                    print(f"  Skipping page {page_num + 1} (too short)")
                continue

            # Generate filename
            filename = f"{page_num + 1:04d}_page_{page_num + 1}.txt"
            out_path = output_dir / filename

            # Write the text
            out_path.write_text(text, encoding="utf-8")
            extracted_files.append(out_path)

            if verbose:
                print(f"  [{page_num + 1:04d}] {filename} ({len(text)} chars)")

        except Exception as e:
            if verbose:
                print(f"  Error extracting page {page_num + 1}: {e}")
            continue

    return extracted_files
