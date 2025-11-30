"""Integration tests for translation enhancer with real NLLB model.

These tests actually call the NLLB model and require:
- transformers library
- torch library
- sentencepiece library
- Internet connection (to download model on first run)

Run with: pytest tests/test_translation_enhancer_integration.py -v -s
"""

import pytest

from tts_helper.chunk import Chunk
from tts_helper.translation_enhancer import (
    TranslationEnhancer,
    TranslationEnhancerConfig,
)

# Skip all tests in this file if transformers is not installed
pytest.importorskip("transformers")
pytest.importorskip("torch")


class TestTranslationEnhancerIntegration:
    """Integration tests with real NLLB model."""

    @pytest.mark.slow
    def test_real_translation_english_to_italian(self) -> None:
        """Test real translation from English to Italian using NLLB.

        This test will download the NLLB model on first run (~1.2GB).
        """
        config = TranslationEnhancerConfig(
            probability=1.0,  # Always translate for testing
            source_language="english",
            target_language="italian",
            model_id="facebook/nllb-200-distilled-600M",
            max_length=400,
            device="cpu",
            verbose=True,
        )

        enhancer = TranslationEnhancer(config)

        # Test with a simple English sentence
        chunks = [Chunk(text="Hello world! How are you today?")]

        enhanced = enhancer.enhance(chunks)

        # Should have 4 chunks: original + pause_before + translation + pause_after
        assert len(enhanced) == 4

        # Check original chunk
        assert enhanced[0].text == "Hello world! How are you today?"

        # Check pause before
        assert enhanced[1].silence_ms == 300

        # Check translation chunk
        translation = enhanced[2].text
        print(f"\nOriginal: {enhanced[0].text}")
        print(f"Translation: {translation}")

        # Check pause after
        assert enhanced[3].silence_ms == 300

        # Basic sanity checks on translation
        assert len(translation) > 0
        assert translation != enhanced[0].text  # Should be different from original

        # Translation should contain Italian words
        # Note: Exact translation may vary, so we check for common Italian words/patterns
        italian_words = ["ciao", "buon", "come", "oggi", "mondo", "stai"]
        has_italian = any(word in translation.lower() for word in italian_words)
        assert (
            has_italian
        ), f"Translation doesn't seem to contain Italian words: {translation}"

    @pytest.mark.slow
    def test_real_translation_english_to_spanish(self) -> None:
        """Test real translation from English to Spanish using NLLB."""
        config = TranslationEnhancerConfig(
            probability=1.0,
            source_language="english",
            target_language="spanish",
            model_id="facebook/nllb-200-distilled-600M",
            device="cpu",
            verbose=True,
        )

        enhancer = TranslationEnhancer(config)

        chunks = [Chunk(text="Good morning! The weather is nice today.")]

        enhanced = enhancer.enhance(chunks)

        assert len(enhanced) == 4

        translation = enhanced[2].text
        print(f"\nOriginal: {enhanced[0].text}")
        print(f"Translation: {translation}")
        assert enhanced[1].silence_ms == 300  # Pause before
        assert enhanced[3].silence_ms == 300  # Pause after

        # Check for Spanish words
        spanish_words = ["buenos", "mañana", "tiempo", "hoy", "buen"]
        has_spanish = any(word in translation.lower() for word in spanish_words)
        assert (
            has_spanish
        ), f"Translation doesn't seem to contain Spanish words: {translation}"

    @pytest.mark.slow
    def test_real_translation_with_voice_override(self) -> None:
        """Test that voice override is set correctly on translated chunks."""
        config = TranslationEnhancerConfig(
            probability=1.0,
            source_language="english",
            target_language="french",
            translation_voice="if_sara",  # Italian voice
            translation_language="it",
            model_id="facebook/nllb-200-distilled-600M",
            device="cpu",
            verbose=True,
        )

        enhancer = TranslationEnhancer(config)

        chunks = [Chunk(text="Thank you very much.")]

        enhanced = enhancer.enhance(chunks)

        assert len(enhanced) == 4

        # Check that voice override is set on translation chunk
        assert enhanced[2].voice == "if_sara"
        assert enhanced[2].language == "it"

        # Check pauses
        assert enhanced[1].silence_ms == 300
        assert enhanced[3].silence_ms == 300

        # Original chunk should have no override
        assert enhanced[0].voice is None

    @pytest.mark.slow
    def test_real_translation_probability(self) -> None:
        """Test that probability affects which chunks get translated."""
        config = TranslationEnhancerConfig(
            probability=0.0,  # Never translate
            source_language="english",
            target_language="italian",
            model_id="facebook/nllb-200-distilled-600M",
            device="cpu",
        )

        enhancer = TranslationEnhancer(config)

        chunks = [
            Chunk(text="Hello world."),
            Chunk(text="How are you?"),
        ]

        enhanced = enhancer.enhance(chunks)

        # With probability=0.0, should return unchanged
        assert len(enhanced) == 2
        assert enhanced[0].text == "Hello world."
        assert enhanced[1].text == "How are you?"
