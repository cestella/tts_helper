from book_extractor.text_normalizer import normalize_text_for_tts


def test_basic_whitespace_cleanup() -> None:
    text = " Line one.  \n\n  Line two with  extra   spaces. "
    normalized = normalize_text_for_tts(text)

    assert "Line one." in normalized
    assert "Line two with extra spaces." in normalized
    # Preserve paragraph break via blank line separator
    assert "\n\n" in normalized


def test_spacy_sentencizer_optional() -> None:
    """Ensure spaCy path does not raise and returns non-empty text."""
    text = "Sentence one! Sentence two? Sentence three."
    normalized = normalize_text_for_tts(text, language_hint="en-US")
    assert normalized.strip() != ""
