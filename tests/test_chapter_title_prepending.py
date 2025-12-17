"""Tests for chapter title prepending in extractors."""

from book_extractor.epub_extractor import extract_text_from_html


class TestChapterTitlePrepending:
    """Test that chapter titles are prepended to extracted text."""

    def test_epub_title_prepended_to_text(self) -> None:
        """Test that EPUB chapter titles are prepended to the text."""
        html_content = """
        <html>
        <head><title>Chapter One</title></head>
        <body>
        <h1>Chapter One</h1>
        <p>This is the first sentence of the chapter.</p>
        <p>This is the second sentence.</p>
        </body>
        </html>
        """

        title, text = extract_text_from_html(html_content, include_title_in_text=True)

        # Title should be returned (with period added by add_periods_to_headings)
        assert title == "Chapter One."

        # Text should start with the title (one sentence per line after processing)
        lines = [line for line in text.split("\n") if line.strip()]
        assert lines[0].strip() == "Chapter One."
        # Then the actual content sentences
        assert any("first sentence" in line.lower() for line in lines)

    def test_epub_title_not_prepended_when_disabled(self) -> None:
        """Test that title prepending can be disabled."""
        html_content = """
        <html>
        <head><title>Chapter Two</title></head>
        <body>
        <p>This is the content.</p>
        </body>
        </html>
        """

        title, text = extract_text_from_html(html_content, include_title_in_text=False)

        # Title should be returned
        assert title == "Chapter Two"

        # Text should NOT start with the title
        assert not text.startswith("Chapter Two.")
        # But should contain the content
        assert "content" in text.lower()

    def test_epub_handles_missing_title(self) -> None:
        """Test that missing titles don't break extraction."""
        html_content = """
        <html>
        <body>
        <p>Some text without a title.</p>
        </body>
        </html>
        """

        title, text = extract_text_from_html(html_content, include_title_in_text=True)

        # Text should be extracted even without a title
        assert "some text" in text.lower()

    def test_epub_title_with_punctuation(self) -> None:
        """Test that titles ending with punctuation are handled correctly."""
        html_content = """
        <html>
        <head><title>Chapter Three: The Beginning</title></head>
        <body>
        <p>The story continues.</p>
        </body>
        </html>
        """

        title, text = extract_text_from_html(html_content, include_title_in_text=True)

        # Title with punctuation should still get a period
        lines = text.split("\n")
        assert lines[0].strip() == "Chapter Three: The Beginning."
