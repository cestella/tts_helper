"""
spaCy-based text segmentation for TTS processing.

This module provides a segmenter implementation using spaCy for
sentence boundary detection and intelligent chunking.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional

import spacy
from spacy.language import Language

from .language_models import get_model_for_language, is_language_supported
from .segmenter import Segmenter, SegmenterConfig


@dataclass
class SpacySegmenterConfig(SegmenterConfig):
    """
    Configuration for spaCy-based text segmentation.

    Attributes:
        language: ISO 639-1 language code (e.g., 'en', 'de', 'fr').
        strategy: Chunking strategy. Options:
                 - 'sentence_count': Group by number of sentences
                 - 'char_count': Group by character count (preserving sentence boundaries)
        sentences_per_chunk: Number of sentences per chunk (for 'sentence_count' strategy).
                           Default: 3.
        max_chars: Maximum number of characters per chunk (for 'char_count' strategy).
                  Default: 300.
        model_name: Optional explicit spaCy model name. If not provided,
                   will be determined from the language code.
        disable_pipes: List of spaCy pipeline components to disable for performance.
                      Default: ['ner', 'lemmatizer'].
    """

    language: str = "en"
    strategy: Literal["sentence_count", "char_count"] = "char_count"
    sentences_per_chunk: int = 3
    max_chars: int = 300
    model_name: Optional[str] = None
    disable_pipes: List[str] = field(default_factory=lambda: ["ner", "lemmatizer"])

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not is_language_supported(self.language):
            raise ValueError(
                f"Language '{self.language}' is not supported. "
                f"Please provide a valid language code or specify model_name explicitly."
            )

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
            List of text chunks, each respecting sentence boundaries.
        """
        if not text or not text.strip():
            return []

        doc = self.nlp(text)

        if self.config.strategy == "sentence_count":
            return self._segment_by_sentence_count(doc.sents)
        else:  # char_count
            return self._segment_by_char_count(doc.sents)

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

    def _segment_by_char_count(self, sentences) -> List[str]:
        """
        Segment by character count while preserving sentence boundaries.

        Args:
            sentences: spaCy sentences iterator.

        Returns:
            List of text chunks.
        """
        chunks: List[str] = []
        current_chunk: str = ""

        for sent in sentences:
            sentence_text = sent.text.strip()

            # Skip empty sentences
            if not sentence_text:
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
