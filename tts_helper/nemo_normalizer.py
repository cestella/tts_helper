"""
NeMo-based text normalization for TTS processing.

This module provides a normalizer implementation using NVIDIA's NeMo Text Processing
library for high-quality text normalization.
"""

from dataclasses import dataclass
from typing import List, Optional

from .normalizer import Normalizer, NormalizerConfig


@dataclass
class NemoNormalizerConfig(NormalizerConfig):
    """
    Configuration for NeMo-based text normalization.

    Attributes:
        language: ISO 639-1 language code (e.g., 'en', 'de', 'es').
        input_case: Input text case handling. Options: 'cased', 'lower_cased'.
        cache_dir: Optional directory to cache normalization grammars.
        verbose: Whether to print verbose normalization info.
    """

    language: str = "en"
    input_case: str = "cased"
    cache_dir: Optional[str] = None
    verbose: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        valid_languages = ["en", "de", "es", "pt", "ru", "fr", "vi"]
        if self.language not in valid_languages:
            raise ValueError(
                f"Language '{self.language}' not supported by NeMo. "
                f"Supported languages: {valid_languages}"
            )

        if self.input_case not in ["cased", "lower_cased"]:
            raise ValueError(
                f"input_case must be 'cased' or 'lower_cased', got '{self.input_case}'"
            )


class NemoNormalizer(Normalizer):
    """
    Text normalizer using NeMo Text Processing.

    This normalizer uses NVIDIA's NeMo library to convert written text to
    spoken form, handling numbers, currency, dates, times, and more.

    Examples:
        >>> config = NemoNormalizerConfig(language="en")
        >>> normalizer = NemoNormalizer(config)
        >>> normalizer.normalize("The price is $123.45")
        "The price is one hundred twenty three dollars forty five cents"
    """

    def __init__(self, config: NemoNormalizerConfig):
        """
        Initialize the NeMo normalizer.

        Args:
            config: Configuration for the normalizer.

        Raises:
            ImportError: If nemo_text_processing is not installed.
            RuntimeError: If NeMo normalizer initialization fails.
        """
        super().__init__(config)
        self.config: NemoNormalizerConfig = config
        self._normalizer: Optional[object] = None

    @property
    def normalizer(self) -> object:
        """
        Lazy-load the NeMo normalizer.

        Returns:
            The loaded NeMo Normalizer instance.

        Raises:
            ImportError: If nemo_text_processing is not installed.
            RuntimeError: If normalizer initialization fails.
        """
        if self._normalizer is None:
            try:
                from nemo_text_processing.text_normalization.normalize import (
                    Normalizer as NeMoNormalizer,
                )
            except ImportError as e:
                raise ImportError(
                    "nemo_text_processing is not installed. "
                    "Please install it following the instructions in README.md"
                ) from e

            try:
                self._normalizer = NeMoNormalizer(
                    input_case=self.config.input_case,
                    lang=self.config.language,
                    cache_dir=self.config.cache_dir,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize NeMo normalizer: {e}"
                ) from e

        return self._normalizer

    def normalize(self, text: str) -> str:
        """
        Normalize text using NeMo Text Processing.

        Converts written text to spoken form:
        - Numbers: "123" → "one hundred twenty three"
        - Currency: "$45.50" → "forty five dollars fifty cents"
        - Dates: "Jan 1, 2024" → "january first twenty twenty four"
        - Times: "3:30pm" → "three thirty p m"
        - And more...

        Args:
            text: The raw input text to normalize.

        Returns:
            Normalized text suitable for TTS processing.
        """
        if not text or not text.strip():
            return text

        try:
            normalized = self.normalizer.normalize(
                text, verbose=self.config.verbose, punct_post_process=True
            )
            return normalized
        except Exception as e:
            # If normalization fails, return original text with a warning
            # This prevents the entire pipeline from breaking
            import warnings

            warnings.warn(
                f"NeMo normalization failed for text: '{text}'. "
                f"Error: {e}. Returning original text."
            )
            return text

    def normalize_batch(self, texts: List[str]) -> List[str]:
        """
        Normalize multiple texts efficiently.

        Args:
            texts: List of raw input texts to normalize.

        Returns:
            List of normalized texts.
        """
        # NeMo's normalizer processes texts one at a time
        # We could potentially optimize this in the future
        return [self.normalize(text) for text in texts]

    def __repr__(self) -> str:
        """String representation of the normalizer."""
        return (
            f"NemoNormalizer(language='{self.config.language}', "
            f"input_case='{self.config.input_case}')"
        )
