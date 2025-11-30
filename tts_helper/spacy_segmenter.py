"""
spaCy-based text segmentation for TTS processing.

This module provides a segmenter implementation using spaCy for
sentence boundary detection and intelligent chunking.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import spacy
from spacy.language import Language

from .language import get_iso_code
from .language_models import get_model_for_language, is_language_supported
from .segmenter import Segmenter, SegmenterConfig


@dataclass
class SpacySegmenterConfig(SegmenterConfig):
    """
    Configuration for spaCy-based text segmentation.

    Attributes:
        language: Unified language name (e.g., 'english', 'german', 'french').
        strategy: Chunking strategy. Options:
                 - 'sentence_count': Group by number of sentences
                 - 'char_count': Group by character count (preserving sentence boundaries)
        sentences_per_chunk: Number of sentences per chunk (for 'sentence_count' strategy).
                           Default: 3.
        max_chars: Maximum number of characters per chunk (for 'char_count' strategy).
                  Default: 300.
        min_chars: Minimum number of characters per chunk. Chunks smaller than this
                  will be merged with adjacent chunks. Default: 3.
        model_name: Optional explicit spaCy model name. If not provided,
                   will be determined from the language code.
        disable_pipes: List of spaCy pipeline components to disable for performance.
                      Default: ['ner', 'lemmatizer'].
    """

    language: str = "english"
    strategy: Literal["sentence_count", "char_count"] = "char_count"
    sentences_per_chunk: int = 3
    max_chars: int = 300
    min_chars: int = 3
    model_name: Optional[str] = None
    disable_pipes: List[str] = field(default_factory=lambda: ["ner", "lemmatizer"])

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not is_language_supported(self.language):
            try:
                # Try to get ISO code to show in error message
                iso_code = get_iso_code(self.language)
                error_msg = (
                    f"Language '{self.language}' (ISO code: '{iso_code}') is not supported for spaCy segmentation. "
                    f"Supported languages: english, german, french, spanish, italian, portuguese, dutch, chinese, japanese. "
                    f"You can also specify model_name explicitly."
                )
            except ValueError:
                error_msg = (
                    f"Language '{self.language}' is not recognized. "
                    f"Supported languages: english, german, french, spanish, italian, portuguese, dutch, chinese, japanese. "
                    f"You can also specify model_name explicitly."
                )
            raise ValueError(error_msg)

        if self.strategy not in ["sentence_count", "char_count"]:
            raise ValueError(
                f"strategy must be 'sentence_count' or 'char_count', got '{self.strategy}'"
            )

        if self.sentences_per_chunk <= 0:
            raise ValueError(
                f"sentences_per_chunk must be positive, got {self.sentences_per_chunk}"
            )

        if self.max_chars <= 0:
            raise ValueError(f"max_chars must be positive, got {self.max_chars}")

        if self.min_chars < 0:
            raise ValueError(f"min_chars must be non-negative, got {self.min_chars}")

        if self.min_chars > self.max_chars:
            raise ValueError(
                f"min_chars ({self.min_chars}) cannot be greater than max_chars ({self.max_chars})"
            )


class SpacySegmenter(Segmenter):
    """
    Text segmenter using spaCy for sentence boundary detection.

    This segmenter uses spaCy's sentence segmentation to intelligently
    break text into chunks suitable for TTS processing, respecting
    natural sentence boundaries.
    """

    def __init__(self, config: SpacySegmenterConfig):
        """
        Initialize the spaCy segmenter.

        Args:
            config: Configuration for the segmenter.

        Raises:
            OSError: If the spaCy model cannot be loaded.
        """
        super().__init__(config)
        self.config: SpacySegmenterConfig = config
        self._nlp: Optional[Language] = None

    @property
    def nlp(self) -> Language:
        """
        Lazy-load the spaCy model.

        Returns:
            The loaded spaCy Language model.

        Raises:
            OSError: If the model cannot be loaded.
        """
        if self._nlp is None:
            model_name = self.config.model_name
            if model_name is None:
                model_metadata = get_model_for_language(self.config.language)
                if model_metadata is None:
                    raise ValueError(
                        f"No model found for language '{self.config.language}'"
                    )
                model_name = model_metadata.model_name

            try:
                self._nlp = spacy.load(
                    model_name, disable=self.config.disable_pipes
                )
            except OSError as e:
                raise OSError(
                    f"Failed to load spaCy model '{model_name}'. "
                    f"Make sure it's installed: python -m spacy download {model_name}"
                ) from e

        return self._nlp

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """
        Merge chunks that are too small with adjacent chunks.

        This prevents tiny chunks (e.g., single punctuation marks) that
        can cause TTS failures.

        Args:
            chunks: List of text chunks.

        Returns:
            List of chunks with small chunks merged.
        """
        if not chunks or self.config.min_chars == 0:
            return chunks

        merged: List[str] = []
        i = 0

        while i < len(chunks):
            current = chunks[i]

            # If chunk is too small, try to merge with next or previous
            if len(current) < self.config.min_chars:
                if merged:
                    # Merge with previous chunk
                    merged[-1] = merged[-1] + " " + current
                elif i + 1 < len(chunks):
                    # Merge with next chunk
                    current = current + " " + chunks[i + 1]
                    i += 1
                    merged.append(current)
                else:
                    # Last chunk and nothing to merge with
                    merged.append(current)
            else:
                merged.append(current)

            i += 1

        return merged

    def segment(self, text: str) -> List[str]:
        """
        Segment text into chunks based on the configured strategy.

        This method processes the text with spaCy to detect sentence boundaries,
        then groups sentences using one of two strategies:

        1. sentence_count: Groups a fixed number of sentences per chunk
        2. char_count: Groups sentences while staying under max_chars limit

        Args:
            text: The raw input text to segment.

        Returns:
            List of text chunks, each respecting sentence boundaries and
            minimum/maximum character limits.
        """
        if not text or not text.strip():
            return []

        doc = self.nlp(text)

        if self.config.strategy == "sentence_count":
            chunks = self._segment_by_sentence_count(doc.sents)
        else:  # char_count
            chunks = self._segment_by_char_count(doc.sents)

        # Merge chunks that are too small
        chunks = self._merge_small_chunks(chunks)

        return chunks

    def _segment_by_sentence_count(self, sentences) -> List[str]:
        """
        Segment by grouping a fixed number of sentences per chunk.

        Args:
            sentences: spaCy sentences iterator.

        Returns:
            List of text chunks.
        """
        chunks: List[str] = []
        current_sentences: List[str] = []

        for sent in sentences:
            sentence_text = sent.text.strip()

            # Skip empty sentences
            if not sentence_text:
                continue

            current_sentences.append(sentence_text)

            # Create chunk when we reach the target sentence count
            if len(current_sentences) >= self.config.sentences_per_chunk:
                chunks.append(" ".join(current_sentences))
                current_sentences = []

        # Add remaining sentences as final chunk
        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks

    def _hard_split_text(self, text: str) -> List[str]:
        """
        Hard-split text to guarantee chunks within max_chars.

        This is a fallback for when sentence/punctuation boundaries fail.
        Splits on spaces, then hard-cuts if individual words are too long.

        Args:
            text: Text to split (may be >max_chars).

        Returns:
            List of chunks, each guaranteed to be ≤max_chars.
        """
        if len(text) <= self.config.max_chars:
            return [text]

        import warnings
        warnings.warn(
            f"Hard-splitting {len(text)}-char text (max: {self.config.max_chars}). "
            f"Preview: {text[:100]}..."
        )

        chunks: List[str] = []
        words = text.split()
        current_chunk = ""

        for word in words:
            # If single word is longer than max_chars, hard-cut it
            if len(word) > self.config.max_chars:
                # Save current chunk if any
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Hard-cut the word into max_chars pieces
                for i in range(0, len(word), self.config.max_chars):
                    chunks.append(word[i:i + self.config.max_chars])
                continue

            # Try adding word to current chunk
            test_chunk = (current_chunk + " " + word).strip()
            if len(test_chunk) <= self.config.max_chars:
                current_chunk = test_chunk
            else:
                # Current chunk is full, save it and start new one
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _segment_by_char_count(self, sentences) -> List[str]:
        """
        Segment by character count while preserving sentence boundaries.

        Args:
            sentences: spaCy sentences iterator.

        Returns:
            List of text chunks, guaranteed to be ≤max_chars.
        """
        chunks: List[str] = []
        current_chunk: str = ""

        for sent in sentences:
            sentence_text = sent.text.strip()

            # Skip empty sentences
            if not sentence_text:
                continue

            # If this single sentence is oversized, split it immediately
            if len(sentence_text) > self.config.max_chars:
                # Save current chunk first
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                # Hard-split the oversized sentence
                chunks.extend(self._hard_split_text(sentence_text))
                continue

            # If adding this sentence would exceed the limit, start a new chunk
            if current_chunk and len(current_chunk) + 1 + len(sentence_text) > self.config.max_chars:
                chunks.append(current_chunk)
                current_chunk = sentence_text
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk = current_chunk + " " + sentence_text
                else:
                    current_chunk = sentence_text

        # Add the final chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)

        # GUARANTEE: Validate all chunks are within limit
        # This catches any logic errors in the above code
        oversized = [c for c in chunks if len(c) > self.config.max_chars]
        if oversized:
            # This should never happen, but if it does, fix it
            import warnings
            warnings.warn(
                f"Found {len(oversized)} oversized chunks after segmentation! "
                f"Re-splitting with hard split. This indicates a bug in the segmenter."
            )
            fixed_chunks = []
            for chunk in chunks:
                if len(chunk) > self.config.max_chars:
                    fixed_chunks.extend(self._hard_split_text(chunk))
                else:
                    fixed_chunks.append(chunk)
            chunks = fixed_chunks

        return chunks

    def __repr__(self) -> str:
        """String representation of the segmenter."""
        model_name = self.config.model_name or get_model_for_language(
            self.config.language
        ).model_name

        if self.config.strategy == "sentence_count":
            strategy_info = f"sentences_per_chunk={self.config.sentences_per_chunk}"
        else:
            strategy_info = f"max_chars={self.config.max_chars}"

        return (
            f"SpacySegmenter(language='{self.config.language}', "
            f"model='{model_name}', "
            f"strategy='{self.config.strategy}', "
            f"{strategy_info})"
        )
