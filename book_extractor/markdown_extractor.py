"""Markdown chapter extractor with lexicographic naming."""

import re
from html.parser import HTMLParser
from pathlib import Path

import markdown as md_renderer  # type: ignore[import-untyped]
from bs4 import BeautifulSoup

try:
    import spacy
except ImportError:
    spacy = None  # type: ignore[assignment]

from .text_normalizer import add_periods_to_headings, normalize_text_for_tts


def safe_filename(name: str, default: str = "chapter") -> str:
    """Convert a string to a safe filename."""
    slug = re.sub(r"[^\w.-]+", "_", name.strip())
    slug = slug.strip("._")
    return slug or default


def extract_markdown_chapters(
    md_path: Path,
    output_dir: Path,
    chapter_pattern: str | None = None,
    verbose: bool = False,
    language_hint: str = "en",
) -> list[Path]:
    """Extract chapters from a markdown file using a regex boundary pattern.

    Args:
        md_path: Path to the markdown file.
        output_dir: Directory to save extracted chapters.
        chapter_pattern: Regex pattern used to find chapter boundaries. The
            matched line acts as the chapter title. If the pattern includes a
            capture group, that group is used as the title. Defaults to
            headings (`^# .+`).
        verbose: Whether to print progress information.

    Returns:
        List of paths to extracted chapter files.
    """
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    text = md_path.read_text(encoding="utf-8")
    pattern_str = chapter_pattern or r"^# .+"
    pattern = re.compile(pattern_str, re.MULTILINE)
    matches = list(pattern.finditer(text))

    if verbose:
        print(f"Extracting markdown chapters from: {md_path}")
        print(f"Output directory: {output_dir}")
        print(f"Chapter pattern: {pattern.pattern}")
        print(f"Found {len(matches)} matches")

    chapters = []
    if matches:
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            raw_chunk = text[start:end].strip()
            if not raw_chunk:
                continue

            lines = raw_chunk.splitlines()
            heading_line = lines[0].strip() if lines else ""

            # Use captured group as title if present; otherwise strip common heading markers.
            if match.lastindex:
                title = (match.group(1) or heading_line).strip()
            else:
                title = heading_line.lstrip("#").strip()

            body = "\n".join(lines[1:]).strip()
            content = body if body else raw_chunk
            chapters.append((title or f"chapter_{idx+1}", content))
    else:
        # No matches; treat entire document as a single chapter.
        chapters.append((md_path.stem, text))

    extracted_files: list[Path] = []
    seen_titles: dict[str, int] = {}

    for idx, (title, content) in enumerate(chapters, 1):
        rendered = _render_markdown_to_text(content)
        normalized = normalize_text_for_tts(rendered, language_hint=language_hint)
        per_sentence = _sentences_per_line(normalized)
        if not per_sentence or len(per_sentence.strip()) < 50:
            if verbose:
                print(f"  Skipping chapter (too short): {title}")
            continue

        base_title = title
        if base_title in seen_titles:
            seen_titles[base_title] += 1
            title = f"{base_title}_{seen_titles[base_title]}"
        else:
            seen_titles[base_title] = 1

        filename = f"{idx:04d}_{safe_filename(title)}.txt"
        out_path = output_dir / filename
        out_path.write_text(per_sentence, encoding="utf-8")
        extracted_files.append(out_path)

        if verbose:
            print(f"  [{idx:04d}] {filename} ({len(per_sentence)} chars)")

    if verbose:
        print(f"\nExtracted {len(extracted_files)} chapters to {output_dir}")

    return extracted_files


def _strip_markdown(md: str) -> str:
    """Lightweight markdown-to-text cleaner to remove boilerplate syntax."""
    text = md
    # Remove fenced code blocks
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # Remove images: ![alt](url)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    # Replace links [text](url) with text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Drop inline code markers
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # Remove emphasis markers
    text = re.sub(r"[*_]{1,2}([^*_]+)[*_]{1,2}", r"\1", text)
    # Remove heading markers
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # Remove bullet/number list markers
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    return text


def _render_markdown_to_text(md: str) -> str:
    """Render markdown to plain text using python-markdown."""
    html = md_renderer.markdown(md)
    # Add periods to headings for better sentence segmentation
    html = add_periods_to_headings(html)
    return _html_to_text(html)


class _HTMLTextExtractor(HTMLParser):
    """Simple HTML-to-text extractor."""

    def __init__(self) -> None:
        super().__init__()
        self.chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.chunks.append(data)

    def get_text(self) -> str:
        return "\n".join(self.chunks)


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text("\n")


def _sentences_per_line(text: str, language_hint: str = "en") -> str:
    """Segment text into sentences (one per line), preferring spaCy."""
    if not text.strip():
        return ""

    if spacy is None:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        return "\n".join(sentences)

    try:
        try:
            nlp = spacy.blank(language_hint or "en")
        except Exception:
            nlp = spacy.blank("xx")

        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")

        doc = nlp(text)
        sentences = [
            " ".join(sent.text.split()) for sent in doc.sents if sent.text.strip()
        ]
        return "\n".join(sentences)
    except Exception:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        return "\n".join(sentences)
