# Book Extractor

Extract chapters from EPUB and PDF books with lexicographic naming for easy ordering.

## Features

- **EPUB Support**: Extract chapters from EPUB files with automatic title detection
- **PDF Support**: Extract chapters from PDF files using table of contents or markdown conversion
- **Lexicographic Naming**: Chapters are named with 4-digit prefixes for proper ordering (e.g., `0001_chapter_name.txt`)
- **Multiple Extraction Strategies**:
  - EPUB: Uses document structure and heading tags
  - PDF: Uses TOC, markdown conversion, or page-by-page extraction
- **Duplicate Handling**: Automatically handles duplicate chapter titles

## Installation

Install the optional dependencies for book extraction:

```bash
# For EPUB support
pip install ebooklib beautifulsoup4

# For PDF support
pip install pymupdf

# For better PDF extraction (optional)
pip install pymupdf4llm
```

Or install all at once from the main requirements.txt.

## Usage

### Command Line

```bash
# Extract EPUB chapters
python -m book_extractor book.epub --output chapters/

# Extract PDF chapters
python -m book_extractor book.pdf --output chapters/

# Extract PDF without using TOC (page-by-page)
python -m book_extractor book.pdf --output chapters/ --no-toc

# Verbose output
python -m book_extractor book.epub --output chapters/ --verbose
```

### Python API

```python
from pathlib import Path
from book_extractor import extract_epub_chapters, extract_pdf_chapters

# Extract EPUB
files = extract_epub_chapters(
    epub_path=Path("book.epub"),
    output_dir=Path("chapters/"),
    verbose=True
)

# Extract PDF
files = extract_pdf_chapters(
    pdf_path=Path("book.pdf"),
    output_dir=Path("chapters/"),
    use_toc=True,  # Use table of contents
    verbose=True
)

print(f"Extracted {len(files)} chapters")
```

## Output Format

Chapters are saved as text files with lexicographic naming:

```
0001_chapter_one.txt
0002_the_second_chapter.txt
0003_another_chapter.txt
...
0042_the_final_chapter.txt
```

This ensures proper ordering when files are listed alphabetically, making them easy to process sequentially with other tools like `tts_helper`.

## PDF Extraction Strategies

The PDF extractor uses multiple strategies in order of preference:

1. **Table of Contents (TOC)**: If the PDF has a TOC, chapters are extracted based on level-1 headings
2. **Markdown Conversion**: If `pymupdf4llm` is installed, uses markdown conversion for better structure
3. **Page-by-page**: Falls back to extracting each page as a separate file

You can skip TOC extraction with the `--no-toc` flag or `use_toc=False` parameter.

## Examples

### Integration with tts_helper

Extract chapters and convert to audiobook:

```bash
# 1. Extract chapters from EPUB
python -m book_extractor mybook.epub --output chapters/

# 2. Convert each chapter to audio
for chapter in chapters/*.txt; do
    python -m tts_helper "$chapter" --output "audio/$(basename "$chapter" .txt).mp3"
done
```

### Processing with Translation

```bash
# Extract chapters
python -m book_extractor mybook.epub --output chapters/

# Convert with translation enhancement
python -m tts_helper chapters/0001_*.txt --config config.json --output chapter1.mp3
```

## Dependencies

- **EPUB**: `ebooklib`, `beautifulsoup4`
- **PDF**: `pymupdf` (provides `fitz` module)
- **PDF (enhanced)**: `pymupdf4llm` (optional, for better text extraction)
