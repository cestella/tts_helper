"""Utilities for cleaning and normalizing text for TTS pipelines."""

import re

try:
    import spacy
except ImportError:
    spacy = None  # type: ignore[assignment]


def add_periods_to_headings(html_content: str) -> str:
    """Add periods to heading tags that don't already end with punctuation."""

    def add_period_to_heading(match: re.Match[str]) -> str:
        level = match.group(1)
        attrs = match.group(2)
        text = match.group(3).strip()

        # Only add period if heading doesn't already end with punctuation
        if text and not re.search(r"[.!?]$", text):
            text = text + "."

        return f"<h{level}{attrs}>{text}</h{level}>"

    # Match h1-h6 tags and add periods to their content
    pattern = r"<h([1-6])([^>]*)>(.*?)</h\1>"
    return re.sub(
        pattern, add_period_to_heading, html_content, flags=re.IGNORECASE | re.DOTALL
    )


def _normalize_language_code(language_hint: str | None) -> str:
    """Convert language hints like 'en-US' to spaCy-compatible codes."""
    if not language_hint:
        return "en"
    parts = language_hint.split("-")
    return parts[0].strip().lower() or "en"


def normalize_text_for_tts(text: str, language_hint: str | None = None) -> str:
    """
    Normalize text spacing and sentence boundaries to be TTS-friendly.

    Steps:
      - Trim lines and drop blank ones
      - Preserve paragraph breaks with double newlines
      - If spaCy is available, re-segment sentences for cleaner spacing
    """
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    if not paragraphs:
        return ""

    normalized = "\n\n".join(paragraphs)

    if spacy is None:
        return normalized

    try:
        lang_code = _normalize_language_code(language_hint)
        try:
            nlp = spacy.blank(lang_code)
        except Exception:
            nlp = spacy.blank("xx")

        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

        normalized_paragraphs: list[str] = []
        for para in paragraphs:
            doc = nlp(para)
            sentences = [
                " ".join(sent.text.split()) for sent in doc.sents if sent.text.strip()
            ]
            if sentences:
                normalized_paragraphs.append(" ".join(sentences))

        if normalized_paragraphs:
            return "\n\n".join(normalized_paragraphs)
    except Exception:
        # Fall back to the simple normalization if spaCy fails
        pass

    return normalized
