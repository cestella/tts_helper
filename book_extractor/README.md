# Book Extractor

Extract chapters from EPUB and Markdown books with lexicographic naming for easy ordering.

## Features

- **EPUB Support**: Extract chapters from EPUB files with automatic title detection
- **Markdown Support**: Split markdown files into chapters using a configurable regex pattern
- **Lexicographic Naming**: Chapters are named with 4-digit prefixes for proper ordering (e.g., `0001_chapter_name.txt`)
- **Multiple Extraction Strategies**:
  - EPUB: Uses document structure and heading tags
  - Markdown: Uses regex boundaries (defaults to top-level headings)
- **Duplicate Handling**: Automatically handles duplicate chapter titles
- **Clean Text Output**: Uses `trafilatura` plus spaCy sentence normalization for TTS-ready text (one sentence per line where applicable)

## Installation

Install the optional dependencies for book extraction:

```bash
# For EPUB support
pip install ebooklib trafilatura

# For Markdown support
pip install markdown beautifulsoup4
```

Or install all at once from the main requirements.txt.

## Usage

### Command Line

```bash
# Extract EPUB chapters
python -m book_extractor book.epub --output chapters/

# Extract Markdown chapters with a custom boundary regex
python -m book_extractor book.md --output chapters/ --chapter-pattern "^CHAPTER \\d+"
python -m book_extractor book.md --output chapters/ --language es  # language hint

# Verbose output
python -m book_extractor book.epub --output chapters/ --verbose
```

### Python API

```python
from pathlib import Path
from book_extractor import extract_epub_chapters, extract_markdown_chapters

# Extract EPUB
files = extract_epub_chapters(
    epub_path=Path("book.epub"),
    output_dir=Path("chapters/"),
    verbose=True
)

# Extract Markdown
files = extract_markdown_chapters(
    md_path=Path("book.md"),
    output_dir=Path("chapters/"),
    chapter_pattern=r"^# .+",  # Top-level headings
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

## Markdown Extraction

Markdown extraction splits chapters using a regex boundary pattern:

- Default pattern: `^# .+` (top-level headings)
- Custom pattern: pass `--chapter-pattern "your-regex"`; the matched line is used as the title, and an optional capture group is used if present.
- If no matches are found, the entire document is saved as a single chapter.
- Markdown content is rendered to text (requires `markdown` + `beautifulsoup4`) so headings/links/code fences are stripped before normalization.
- Sentences are re-segmented (spaCy if available, otherwise regex) and written one per line for cleaner downstream processing.

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

- **EPUB**: `ebooklib`, `trafilatura`
- **Markdown**: `markdown`, `beautifulsoup4`
- **Sentence normalization (optional)**: `spacy` (used if installed; falls back to whitespace cleanup)
