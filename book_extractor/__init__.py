"""Book extractor for EPUB and Markdown files.

This package provides utilities to extract text from books in various formats,
with support for chapter detection and lexicographic ordering.
"""

from .epub_extractor import extract_epub_chapters
from .markdown_extractor import extract_markdown_chapters

__all__ = ["extract_epub_chapters", "extract_markdown_chapters"]
