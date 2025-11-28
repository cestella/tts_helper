"""Tests for book extractor utilities."""

import tempfile
from pathlib import Path

import pytest

# Skip all tests if dependencies not available
pytest.importorskip("ebooklib")
pytest.importorskip("bs4")
pytest.importorskip("fitz")

from book_extractor import extract_epub_chapters, extract_pdf_chapters
from book_extractor.epub_extractor import safe_filename, extract_text_from_html
from book_extractor.pdf_extractor import safe_filename as pdf_safe_filename


class TestSafeFilename:
    """Tests for safe_filename function."""

    def test_basic_filename(self):
        """Test basic filename cleaning."""
        assert safe_filename("Hello World") == "Hello_World"
        assert safe_filename("Chapter 1: The Beginning") == "Chapter_1_The_Beginning"

    def test_special_characters(self):
        """Test special character removal."""
        assert safe_filename("Hello/World\\Test") == "Hello_World_Test"
        assert safe_filename("Test<>:|?*") == "Test"

    def test_leading_trailing(self):
        """Test leading/trailing character removal."""
        assert safe_filename("__test__") == "test"
        assert safe_filename("..test..") == "test"
        assert safe_filename("_.test._") == "test"

    def test_empty_result(self):
        """Test default when result is empty."""
        assert safe_filename("!!!") == "chapter"
        assert safe_filename("", default="empty") == "empty"

    def test_preserve_valid(self):
        """Test that valid characters are preserved."""
        assert safe_filename("test-name.txt") == "test-name.txt"
        assert safe_filename("my_file_123") == "my_file_123"


class TestExtractTextFromHtml:
    """Tests for HTML text extraction."""

    def test_extract_with_h1(self):
        """Test extraction with h1 title."""
        html = "<html><body><h1>Chapter Title</h1><p>Content here.</p></body></html>"
        title, text = extract_text_from_html(html)

        assert title == "Chapter Title"
        assert "Content here." in text

    def test_extract_with_h2(self):
        """Test extraction with h2 title when no h1."""
        html = "<html><body><h2>Section Title</h2><p>Content here.</p></body></html>"
        title, text = extract_text_from_html(html)

        assert title == "Section Title"
        assert "Content here." in text

    def test_extract_no_title(self):
        """Test extraction with no title tags."""
        html = "<html><body><p>Just some content.</p></body></html>"
        title, text = extract_text_from_html(html)

        assert title is None
        assert "Just some content." in text

    def test_text_cleanup(self):
        """Test that text is properly cleaned."""
        html = "<html><body><p>Line 1</p><p>Line 2</p></body></html>"
        title, text = extract_text_from_html(html)

        assert "Line 1" in text
        assert "Line 2" in text


class TestEpubExtractor:
    """Integration tests for EPUB extraction."""

    @pytest.mark.slow
    def test_import(self):
        """Test that EPUB extractor can be imported."""
        from book_extractor import extract_epub_chapters
        assert callable(extract_epub_chapters)

    def test_file_not_found(self):
        """Test error handling for missing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            with pytest.raises(FileNotFoundError):
                extract_epub_chapters(
                    epub_path=Path("nonexistent.epub"),
                    output_dir=output_dir,
                )


class TestPdfExtractor:
    """Integration tests for PDF extraction."""

    @pytest.mark.slow
    def test_import(self):
        """Test that PDF extractor can be imported."""
        from book_extractor import extract_pdf_chapters
        assert callable(extract_pdf_chapters)

    def test_file_not_found(self):
        """Test error handling for missing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            with pytest.raises(FileNotFoundError):
                extract_pdf_chapters(
                    pdf_path=Path("nonexistent.pdf"),
                    output_dir=output_dir,
                )

    def test_safe_filename_consistency(self):
        """Test that both modules use consistent filename sanitization."""
        from book_extractor.epub_extractor import safe_filename as epub_safe
        from book_extractor.pdf_extractor import safe_filename as pdf_safe

        test_cases = [
            "Hello World",
            "Chapter 1: The Beginning",
            "Test/File\\Name",
        ]

        for test in test_cases:
            assert epub_safe(test) == pdf_safe(test)


class TestBookExtractorModule:
    """Tests for book_extractor module."""

    def test_module_imports(self):
        """Test that module exports expected functions."""
        import book_extractor

        assert hasattr(book_extractor, "extract_epub_chapters")
        assert hasattr(book_extractor, "extract_pdf_chapters")
        assert callable(book_extractor.extract_epub_chapters)
        assert callable(book_extractor.extract_pdf_chapters)

    def test_module_all(self):
        """Test __all__ export list."""
        import book_extractor

        assert hasattr(book_extractor, "__all__")
        assert "extract_epub_chapters" in book_extractor.__all__
        assert "extract_pdf_chapters" in book_extractor.__all__
