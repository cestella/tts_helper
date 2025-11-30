"""Translation enhancer using Meta's NLLB model for multilingual audiobooks."""

import random
from dataclasses import dataclass
from typing import List, Optional

from .chunk import Chunk
from .enhancer import Enhancer, EnhancerConfig
from .language import get_flores_code

# Import transformers classes (will be None if not installed)
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None


@dataclass
class TranslationEnhancerConfig(EnhancerConfig):
    """Configuration for translation enhancer using NLLB.

    Args:
        probability: Probability of translating a chunk (0.0 to 1.0)
        source_language: Source language name (e.g., 'english', 'spanish')
        target_language: Target language name (e.g., 'italian', 'french')
        translation_voice: Voice to use for translated chunks (None = use default voice)
        translation_language: TTS language override for translated chunks (None = infer from target_language)
        translation_speed: Speed multiplier for translated chunks (e.g., 0.8 for 80% speed, None = use default)
        pause_before_ms: Duration of silence before translated audio in milliseconds (default: 300)
        pause_after_ms: Duration of silence after translated audio in milliseconds (default: 300)
        model_id: Hugging Face model ID (default: 'facebook/nllb-200-distilled-600M')
        max_length: Maximum length for translation output (default: 400 tokens)
        device: Device to use ('cpu' or 'cuda', default: 'cpu')
        verbose: Whether to print translation info (default: False)
    """

    probability: float = 0.1
    source_language: str = "english"
    target_language: str = "italian"
    translation_voice: Optional[str] = None
    translation_language: Optional[str] = None
    translation_speed: Optional[float] = None
    pause_before_ms: int = 300
    pause_after_ms: int = 300
    model_id: str = "facebook/nllb-200-distilled-600M"
    max_length: int = 400
    device: str = "cpu"
    verbose: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError(
                f"Probability must be between 0.0 and 1.0, got: {self.probability}"
            )

        if self.max_length <= 0:
            raise ValueError(f"max_length must be > 0, got: {self.max_length}")

        if self.device not in ("cpu", "cuda"):
            raise ValueError(f"device must be 'cpu' or 'cuda', got: {self.device}")

        if self.translation_speed is not None and self.translation_speed <= 0:
            raise ValueError(f"translation_speed must be > 0, got: {self.translation_speed}")

        if self.pause_before_ms < 0:
            raise ValueError(f"pause_before_ms must be >= 0, got: {self.pause_before_ms}")

        if self.pause_after_ms < 0:
            raise ValueError(f"pause_after_ms must be >= 0, got: {self.pause_after_ms}")

        # Validate language names (will raise ValueError if unsupported)
        get_flores_code(self.source_language)
        get_flores_code(self.target_language)

    @property
    def source_lang_code(self) -> str:
        """Get FLORES-200 code for source language."""
        return get_flores_code(self.source_language)

    @property
    def target_lang_code(self) -> str:
        """Get FLORES-200 code for target language."""
        return get_flores_code(self.target_language)


class TranslationEnhancer(Enhancer):
    """Translation enhancer using Meta's NLLB model.

    This enhancer randomly translates chunks using Meta's No Language Left Behind
    (NLLB) model and inserts them with announcements into the audiobook processing pipeline.

    Example:
        >>> config = TranslationEnhancerConfig(
        ...     probability=0.2,
        ...     target_language="Italian",
        ...     target_lang_code="ita_Latn",
        ...     source_lang_code="eng_Latn"
        ... )
        >>> enhancer = TranslationEnhancer(config)
        >>> chunks = ["Hello world.", "How are you?"]
        >>> enhanced = enhancer.enhance(chunks)
        # Some chunks may now have translations following them
    """

    def __init__(self, config: TranslationEnhancerConfig):
        """Initialize translation enhancer.

        Args:
            config: Translation enhancer configuration

        Raises:
            ImportError: If transformers or torch is not installed
        """
        super().__init__(config)
        self.config: TranslationEnhancerConfig  # Type hint for IDE support
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        """Lazy-load NLLB model.

        Returns:
            Initialized NLLB model

        Raises:
            ImportError: If transformers or torch is not installed
        """
        if self._model is None:
            if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
                raise ImportError(
                    "transformers is not installed. "
                    "Install it with: pip install transformers torch sentencepiece"
                )

            if self.config.verbose:
                print(f"Loading NLLB model: {self.config.model_id}")

            # Load tokenizer and model from Hugging Face
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_id)

            # Move model to device
            self._model = self._model.to(self.config.device)

            if self.config.verbose:
                print(f"  Model loaded on {self.config.device}")

        return self._model

    @property
    def tokenizer(self):
        """Get tokenizer (loads model if not already loaded)."""
        if self._tokenizer is None:
            _ = self.model  # Trigger model loading which also loads tokenizer
        return self._tokenizer

    def _is_translation_valid(self, original: str, translation: str) -> bool:
        """Validate translation to detect runaway or repetitive translations.

        Args:
            original: Original text
            translation: Translated text

        Returns:
            True if translation is valid, False if it should be discarded
        """
        # Check if translation is empty
        if not translation or not translation.strip():
            return False

        # Check if translation is excessively longer than original (more than 3x)
        # This catches runaway generations
        if len(translation) > len(original) * 3:
            if self.config.verbose:
                print(
                    f"  ⚠️  Translation too long ({len(translation)} chars vs {len(original)} chars), skipping"
                )
            return False

        # Check for repetitive patterns (same 5-word sequence appears 3+ times)
        words = translation.split()
        if len(words) >= 15:  # Only check if enough words
            for i in range(len(words) - 4):
                pattern = " ".join(words[i:i + 5])
                # Count occurrences of this 5-word pattern
                count = translation.count(pattern)
                if count >= 3:
                    if self.config.verbose:
                        print(
                            f"  ⚠️  Detected repetitive pattern '{pattern[:30]}...', skipping translation"
                        )
                    return False

        # Check absolute maximum length (2000 chars to be safe for TTS)
        if len(translation) > 2000:
            if self.config.verbose:
                print(
                    f"  ⚠️  Translation exceeds 2000 chars ({len(translation)} chars), skipping"
                )
            return False

        return True

    def _translate(self, text: str) -> Optional[str]:
        """Translate text using NLLB model.

        Args:
            text: Text to translate

        Returns:
            Translated text, or None if translation failed or invalid
        """
        try:
            # Set source language and tokenize input text
            self.tokenizer.src_lang = self.config.source_lang_code
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True
            )

            # Move inputs to device
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

            # Get target language token ID
            # NLLB uses convert_tokens_to_ids to get language code IDs
            target_lang_id = self.tokenizer.convert_tokens_to_ids(
                self.config.target_lang_code
            )

            # Generate translation
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=target_lang_id,
                max_length=self.config.max_length,
            )

            # Decode translation
            translation = self.tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )[0]

            # Validate translation before returning
            if not self._is_translation_valid(text, translation):
                return None

            if self.config.verbose:
                print(
                    f"  ✓ Translated to {self.config.target_language}: {translation[:50]}..."
                )

            return translation

        except Exception as e:
            if self.config.verbose:
                print(f"  Translation error: {e}")
            return None

    def enhance(self, chunks: List[Chunk]) -> List[Chunk]:
        """Enhance chunks by randomly translating some.

        For each chunk, there's a probability that it will be translated.
        When translated, new chunks are inserted after the original:
        1. Optional pause before translation
        2. The translated chunk (with voice/language/speed overrides)
        3. Optional pause after translation

        Args:
            chunks: List of Chunk objects from segmentation

        Returns:
            Enhanced list of Chunk objects with translations
        """
        if self.config.probability <= 0.0:
            # No translation - return chunks unchanged
            return chunks

        enhanced_chunks = []

        for chunk in chunks:
            # Always keep the original chunk
            enhanced_chunks.append(chunk)

            # Randomly decide whether to translate
            if random.random() < self.config.probability:
                if self.config.verbose:
                    print(f"Translating chunk: {chunk.text[:50]}...")

                # Translate the chunk text
                translation = self._translate(chunk.text)

                if translation:
                    # Add pause before translation (if configured)
                    if self.config.pause_before_ms > 0:
                        enhanced_chunks.append(
                            Chunk(text="", silence_ms=self.config.pause_before_ms)
                        )

                    # Add translated chunk with voice/language/speed overrides
                    enhanced_chunks.append(
                        Chunk(
                            text=translation,
                            voice=self.config.translation_voice,
                            language=self.config.translation_language,
                            speed=self.config.translation_speed,
                        )
                    )

                    # Add pause after translation (if configured)
                    if self.config.pause_after_ms > 0:
                        enhanced_chunks.append(
                            Chunk(text="", silence_ms=self.config.pause_after_ms)
                        )

        return enhanced_chunks

    def __repr__(self) -> str:
        """String representation of enhancer."""
        return (
            f"TranslationEnhancer(probability={self.config.probability}, "
            f"target_language={self.config.target_language})"
        )
