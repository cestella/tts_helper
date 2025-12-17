"""Tests for book extractor utilities."""

import tempfile
from pathlib import Path

import pytest

# Skip all tests if dependencies not available
pytest.importorskip("ebooklib")

from book_extractor import (
    extract_epub_chapters,
    extract_markdown_chapters,
)
from book_extractor.epub_extractor import extract_text_from_html, safe_filename


class TestSafeFilename:
    """Tests for safe_filename function."""

    def test_basic_filename(self) -> None:
        """Test basic filename cleaning."""
        assert safe_filename("Hello World") == "Hello_World"
        assert safe_filename("Chapter 1: The Beginning") == "Chapter_1_The_Beginning"

    def test_special_characters(self) -> None:
        """Test special character removal."""
        assert safe_filename("Hello/World\\Test") == "Hello_World_Test"
        assert safe_filename("Test<>:|?*") == "Test"

    def test_leading_trailing(self) -> None:
        """Test leading/trailing character removal."""
        assert safe_filename("__test__") == "test"
        assert safe_filename("..test..") == "test"
        assert safe_filename("_.test._") == "test"

    def test_empty_result(self) -> None:
        """Test default when result is empty."""
        assert safe_filename("!!!") == "chapter"
        assert safe_filename("", default="empty") == "empty"

    def test_preserve_valid(self) -> None:
        """Test that valid characters are preserved."""
        assert safe_filename("test-name.txt") == "test-name.txt"
        assert safe_filename("my_file_123") == "my_file_123"


class TestExtractTextFromHtml:
    """Tests for HTML text extraction."""

    def test_extract_with_h1(self) -> None:
        """Test extraction with h1 title."""
        html = "<html><body><h1>Chapter Title</h1><p>Content here.</p></body></html>"
        title, text = extract_text_from_html(html, language_hint="en")

        assert title == "Chapter Title."
        assert "Content here" in text

    def test_extract_with_h2(self) -> None:
        """Test extraction with h2 title when no h1."""
        html = "<html><body><h2>Section Title</h2><p>Content here.</p></body></html>"
        title, text = extract_text_from_html(html)

        assert title == "Section Title."
        assert "Content here" in text

    def test_extract_no_title(self) -> None:
        """Test extraction with no title tags."""
        html = "<html><body><p>Just some content.</p></body></html>"
        title, text = extract_text_from_html(html)

        assert title is None
        assert "Just some content" in text

    def test_text_cleanup(self) -> None:
        """Test that text is properly cleaned and normalized."""
        html = """
        <html>
            <body>
                <p>Line 1</p>
                <p>Line 2 with   extra   spaces.</p>
            </body>
        </html>
        """
        title, text = extract_text_from_html(html)

        assert "Line 1" in text
        assert "Line 2 with extra spaces." in text


class TestEpubExtractor:
    """Integration tests for EPUB extraction."""

    @pytest.mark.slow
    def test_import(self) -> None:
        """Test that EPUB extractor can be imported."""
        from book_extractor import extract_epub_chapters

        assert callable(extract_epub_chapters)

    def test_file_not_found(self) -> None:
        """Test error handling for missing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            with pytest.raises(FileNotFoundError):
                extract_epub_chapters(
                    epub_path=Path("nonexistent.epub"),
                    output_dir=output_dir,
                )


class TestBookExtractorModule:
    """Tests for book_extractor module."""

    def test_module_imports(self) -> None:
        """Test that module exports expected functions."""
        import book_extractor

        assert hasattr(book_extractor, "extract_epub_chapters")
        assert hasattr(book_extractor, "extract_markdown_chapters")
        assert callable(book_extractor.extract_epub_chapters)
        assert callable(book_extractor.extract_markdown_chapters)

    def test_module_all(self) -> None:
        """Test __all__ export list."""
        import book_extractor

        assert hasattr(book_extractor, "__all__")
        assert "extract_epub_chapters" in book_extractor.__all__
        assert "extract_markdown_chapters" in book_extractor.__all__


class TestMarkdownExtractor:
    """Tests for Markdown extraction."""

    def test_markdown_extraction_with_pattern(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Split markdown using a custom chapter regex."""
        md_path = tmp_path / "book.md"
        md_path.write_text(
            "CHAPTER 1\nThis is the introduction chapter. It has multiple sentences for testing.\n\n"
            "CHAPTER 2\nThis is the second chapter. It also has multiple sentences.\n",
            encoding="utf-8",
        )
        output_dir = tmp_path / "out"

        files = extract_markdown_chapters(
            md_path=md_path,
            output_dir=output_dir,
            chapter_pattern=r"^CHAPTER \d+",
        )

        assert len(files) == 2
        assert files[0].exists()
        assert files[1].exists()
        # Chapter title should be prepended and sentences split per line
        content = files[0].read_text(encoding="utf-8")
        # Chapter title should be present at the beginning
        assert content.startswith("CHAPTER 1.")
        assert "introduction chapter" in content
        assert "\n" in content  # sentences per line


class TestEpubSentencesPerLine:
    """Ensure EPUB text is segmented per sentence."""

    def test_sentences_per_line(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """EPUB extraction should output one sentence per line."""
        # This is a minimal HTML fragment to exercise extract_text_from_html
        html = """
        <html><body><h1>Title</h1>
        <p>This is sentence one. This is sentence two!</p>
        </body></html>
        """
        title, text = extract_text_from_html(html)
        assert title == "Title."
        assert "This is sentence one." in text
        assert "This is sentence two!" in text
        # Expect newline separation between sentences
        assert "\n" in text

    def test_markdown_file_not_found(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """Ensure missing markdown raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            extract_markdown_chapters(
                md_path=tmp_path / "missing.md",
                output_dir=tmp_path / "out",
            )
