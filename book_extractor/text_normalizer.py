"""Utilities for cleaning and normalizing text for TTS pipelines."""

from typing import Optional

try:
    import spacy
except ImportError:
    spacy = None


def _normalize_language_code(language_hint: Optional[str]) -> str:
    """Convert language hints like 'en-US' to spaCy-compatible codes."""
    if not language_hint:
        return "en"
    parts = language_hint.split("-")
    return parts[0].strip().lower() or "en"


def normalize_text_for_tts(text: str, language_hint: Optional[str] = None) -> str:
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
                " ".join(sent.text.split())
                for sent in doc.sents
                if sent.text.strip()
            ]
            if sentences:
                normalized_paragraphs.append(" ".join(sentences))

        if normalized_paragraphs:
            return "\n\n".join(normalized_paragraphs)
    except Exception:
        # Fall back to the simple normalization if spaCy fails
        pass

    return normalized
