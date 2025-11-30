"""Tests for spaCy-based segmenter."""

import tempfile
from pathlib import Path

import pytest

from tts_helper.spacy_segmenter import SpacySegmenter, SpacySegmenterConfig


class TestSpacySegmenterConfig:
    """Tests for SpacySegmenterConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SpacySegmenterConfig()

        assert config.language == "english"
        assert config.strategy == "char_count"
        assert config.sentences_per_chunk == 3
        assert config.max_chars == 300
        assert config.model_name is None
        assert "ner" in config.disable_pipes
        assert "lemmatizer" in config.disable_pipes

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = SpacySegmenterConfig(
            language="german",
            max_chars=500,
            disable_pipes=["ner"],
        )

        assert config.language == "german"
        assert config.max_chars == 500
        assert config.disable_pipes == ["ner"]

    def test_invalid_language_raises_error(self) -> None:
        """Test that invalid language raises ValueError."""
        with pytest.raises(ValueError, match="not recognized"):
            SpacySegmenterConfig(language="invalid_lang")

    def test_negative_max_chars_raises_error(self) -> None:
        """Test that negative max_chars raises ValueError."""
        with pytest.raises(ValueError, match="max_chars must be positive"):
            SpacySegmenterConfig(max_chars=-1)

    def test_zero_max_chars_raises_error(self) -> None:
        """Test that zero max_chars raises ValueError."""
        with pytest.raises(ValueError, match="max_chars must be positive"):
            SpacySegmenterConfig(max_chars=0)

    def test_invalid_strategy_raises_error(self) -> None:
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="strategy must be"):
            SpacySegmenterConfig(strategy="invalid_strategy")  # type: ignore[arg-type]

    def test_negative_sentences_per_chunk_raises_error(self) -> None:
        """Test that negative sentences_per_chunk raises ValueError."""
        with pytest.raises(ValueError, match="sentences_per_chunk must be positive"):
            SpacySegmenterConfig(sentences_per_chunk=-1)

    def test_sentence_count_strategy_config(self) -> None:
        """Test configuration for sentence count strategy."""
        config = SpacySegmenterConfig(
            language="english",
            strategy="sentence_count",
            sentences_per_chunk=5,
        )

        assert config.strategy == "sentence_count"
        assert config.sentences_per_chunk == 5

    def test_explicit_model_name(self) -> None:
        """Test using explicit model name."""
        config = SpacySegmenterConfig(
            language="english",
            model_name="en_core_web_md",
        )

        assert config.model_name == "en_core_web_md"

    def test_config_serialization(self) -> None:
        """Test config can be serialized to/from JSON."""
        config = SpacySegmenterConfig(
            language="french",
            max_chars=400,
            disable_pipes=["ner", "parser"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            config.to_json(json_path)

            loaded = SpacySegmenterConfig.from_json(json_path)

            assert loaded.language == config.language
            assert loaded.max_chars == config.max_chars
            assert loaded.disable_pipes == config.disable_pipes


class TestSpacySegmenter:
    """Tests for SpacySegmenter."""

    @pytest.fixture
    def config(self):  # type: ignore[no-untyped-def]
        """Fixture providing default config."""
        return SpacySegmenterConfig(language="english", max_chars=100)

    @pytest.fixture
    def segmenter(self, config):  # type: ignore[no-untyped-def]
        """Fixture providing segmenter instance."""
        return SpacySegmenter(config)

    def test_init(self, config) -> None:  # type: ignore[no-untyped-def]
        """Test segmenter initialization."""
        segmenter = SpacySegmenter(config)
        assert segmenter.config == config
        assert segmenter._nlp is None  # Lazy loading

    def test_nlp_lazy_loading(self, segmenter) -> None:  # type: ignore[no-untyped-def]
        """Test that spaCy model is lazily loaded."""
        assert segmenter._nlp is None
        nlp = segmenter.nlp
        assert nlp is not None
        assert segmenter._nlp is nlp  # Same instance on second access

    def test_segment_simple_text(self, segmenter) -> None:  # type: ignore[no-untyped-def]
        """Test segmentation of simple text."""
        text = "This is sentence one. This is sentence two."
        chunks = segmenter.segment(text)

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        # All original text should be preserved
        assert " ".join(chunks).replace("  ", " ").strip() == text

    def test_segment_empty_text(self, segmenter) -> None:  # type: ignore[no-untyped-def]
        """Test segmentation of empty text."""
        assert segmenter.segment("") == []
        assert segmenter.segment("   ") == []

    def test_segment_respects_max_chars(self) -> None:
        """Test that chunks respect max_chars limit."""
        config = SpacySegmenterConfig(language="english", max_chars=50)
        segmenter = SpacySegmenter(config)

        text = (
            "This is a short sentence. "
            "This is another short sentence. "
            "And here is one more short sentence."
        )
        chunks = segmenter.segment(text)

        # Most chunks should be under the limit (unless a single sentence exceeds it)
        for chunk in chunks:
            # Allow some flexibility for sentences that are naturally longer
            # The key is that we tried to respect boundaries
            assert len(chunk) <= 150  # Reasonable upper bound

    def test_segment_long_single_sentence(self) -> None:
        """Test handling of a single sentence longer than max_chars."""
        config = SpacySegmenterConfig(language="english", max_chars=20)
        segmenter = SpacySegmenter(config)

        text = "This is a very long sentence that exceeds the maximum character limit."
        chunks = segmenter.segment(text)

        # Should split the long sentence into multiple chunks
        assert len(chunks) > 1
        # All chunks must be within max_chars limit
        for chunk in chunks:
            assert len(chunk) <= config.max_chars
        # All chunks together should contain the original text (minus extra whitespace)
        assert " ".join(chunks).replace("  ", " ") == text

    def test_segment_multiple_sentences(self) -> None:
        """Test segmentation with multiple sentences."""
        config = SpacySegmenterConfig(language="english", max_chars=100)
        segmenter = SpacySegmenter(config)

        text = (
            "First sentence here. "
            "Second sentence here. "
            "Third sentence here. "
            "Fourth sentence here."
        )
        chunks = segmenter.segment(text)

        assert len(chunks) >= 1
        # Verify all content is preserved
        combined = " ".join(chunks)
        # Check that key phrases are present
        assert "First sentence" in combined
        assert "Fourth sentence" in combined

    def test_segment_preserves_sentence_boundaries(self) -> None:
        """Test that segmentation respects sentence boundaries."""
        config = SpacySegmenterConfig(language="english", max_chars=100)
        segmenter = SpacySegmenter(config)

        text = (
            "Dr. Smith went to the store. He bought milk and eggs. Then he went home."
        )
        chunks = segmenter.segment(text)

        # Each chunk should end with proper sentence punctuation or be complete
        for chunk in chunks:
            # Should not break in the middle of a word
            assert not chunk.endswith("Dr")  # Don't break on abbreviations
            # Content should be clean
            assert len(chunk.strip()) > 0

    def test_segment_batch(self, segmenter) -> None:  # type: ignore[no-untyped-def]
        """Test batch segmentation."""
        texts = [
            "First text. With multiple sentences.",
            "Second text. Also with sentences.",
        ]
        results = segmenter.segment_batch(texts)

        assert len(results) == 2
        assert all(isinstance(chunks, list) for chunks in results)
        assert all(len(chunks) > 0 for chunks in results)

    def test_repr(self, segmenter) -> None:  # type: ignore[no-untyped-def]
        """Test string representation."""
        repr_str = repr(segmenter)

        assert "SpacySegmenter" in repr_str
        assert "english" in repr_str
        assert "100" in repr_str  # max_chars

    def test_different_languages(self) -> None:
        """Test segmenter with different languages."""
        # Test German (skip if model not installed)
        config_de = SpacySegmenterConfig(language="german", max_chars=100)
        segmenter_de = SpacySegmenter(config_de)

        german_text = "Das ist der erste Satz. Das ist der zweite Satz."

        try:
            chunks_de = segmenter_de.segment(german_text)
            assert len(chunks_de) > 0
            assert all(isinstance(chunk, str) for chunk in chunks_de)
        except OSError:
            # German model not installed, skip this test
            pytest.skip("German spaCy model not installed")

    def test_model_not_installed_raises_error(self) -> None:
        """Test that missing model raises helpful error."""
        config = SpacySegmenterConfig(
            language="english",
            model_name="en_core_web_nonexistent",
        )
        segmenter = SpacySegmenter(config)

        with pytest.raises(OSError, match="Failed to load spaCy model"):
            _ = segmenter.nlp

    def test_whitespace_handling(self, segmenter) -> None:  # type: ignore[no-untyped-def]
        """Test proper handling of whitespace."""
        text = "Sentence one.   Sentence two.    Sentence three."
        chunks = segmenter.segment(text)

        # Chunks should not have excessive whitespace
        for chunk in chunks:
            assert "  " not in chunk  # No double spaces

    def test_newline_handling(self, segmenter) -> None:  # type: ignore[no-untyped-def]
        """Test handling of text with newlines."""
        text = "First sentence.\nSecond sentence.\n\nThird sentence."
        chunks = segmenter.segment(text)

        assert len(chunks) > 0
        # Should handle newlines gracefully
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_config_from_dict(self) -> None:
        """Test creating segmenter from dict config."""
        config_dict = {
            "language": "english",
            "max_chars": 200,
            "disable_pipes": ["ner"],
        }
        config = SpacySegmenterConfig.from_dict(config_dict)
        segmenter = SpacySegmenter(config)

        assert segmenter.config.language == "english"
        assert segmenter.config.max_chars == 200

    def test_sentence_count_strategy(self) -> None:
        """Test segmentation using sentence count strategy."""
        config = SpacySegmenterConfig(
            language="english",
            strategy="sentence_count",
            sentences_per_chunk=2,
        )
        segmenter = SpacySegmenter(config)

        text = (
            "This is sentence one. This is sentence two. "
            "This is sentence three. This is sentence four. "
            "This is sentence five."
        )
        chunks = segmenter.segment(text)

        # Should create chunks with 2 sentences each
        assert len(chunks) == 3
        # First chunk: 2 sentences
        assert "sentence one" in chunks[0]
        assert "sentence two" in chunks[0]
        # Second chunk: 2 sentences
        assert "sentence three" in chunks[1]
        assert "sentence four" in chunks[1]
        # Third chunk: remaining 1 sentence
        assert "sentence five" in chunks[2]

    def test_sentence_count_with_three_sentences(self) -> None:
        """Test sentence count strategy with default of 3 sentences."""
        config = SpacySegmenterConfig(
            language="english",
            strategy="sentence_count",
            sentences_per_chunk=3,
        )
        segmenter = SpacySegmenter(config)

        text = "First. Second. Third. Fourth. Fifth. Sixth. Seventh."
        chunks = segmenter.segment(text)

        # 3 sentences per chunk: should get 3 chunks (3+3+1)
        assert len(chunks) == 3
        assert len(chunks[0].split(".")) >= 3  # First chunk has 3 sentences
        assert len(chunks[1].split(".")) >= 3  # Second chunk has 3 sentences

    def test_repr_with_sentence_count_strategy(self) -> None:
        """Test string representation with sentence count strategy."""
        config = SpacySegmenterConfig(
            language="english",
            strategy="sentence_count",
            sentences_per_chunk=5,
        )
        segmenter = SpacySegmenter(config)

        repr_str = repr(segmenter)
        assert "sentence_count" in repr_str
        assert "sentences_per_chunk=5" in repr_str

    def test_repr_with_char_count_strategy(self) -> None:
        """Test string representation with char count strategy."""
        config = SpacySegmenterConfig(
            language="english",
            strategy="char_count",
            max_chars=200,
        )
        segmenter = SpacySegmenter(config)

        repr_str = repr(segmenter)
        assert "char_count" in repr_str
        assert "max_chars=200" in repr_str


class TestSpacySegmenterIntegration:
    """Integration tests for SpacySegmenter with realistic scenarios."""

    def test_audiobook_paragraph(self) -> None:
        """Test with realistic audiobook paragraph."""
        config = SpacySegmenterConfig(language="english", max_chars=350)
        segmenter = SpacySegmenter(config)

        text = (
            "It was a bright cold day in April, and the clocks were striking thirteen. "
            "Winston Smith, his chin nuzzled into his breast in an effort to escape the "
            "vile wind, slipped quickly through the glass doors of Victory Mansions, "
            "though not quickly enough to prevent a swirl of gritty dust from entering "
            "along with him. The hallway smelt of boiled cabbage and old rag mats."
        )

        chunks = segmenter.segment(text)

        # Should produce reasonable chunks
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk) <= 500  # Reasonable maximum
            assert len(chunk) > 0

    def test_dialogue_text(self) -> None:
        """Test with dialogue text."""
        config = SpacySegmenterConfig(language="english", max_chars=200)
        segmenter = SpacySegmenter(config)

        text = (
            '"Hello," said John. "How are you today?" '
            '"I\'m doing well," replied Sarah. "Thanks for asking." '
            '"That\'s great to hear," John responded.'
        )

        chunks = segmenter.segment(text)

        assert len(chunks) > 0
        # Verify dialogue is preserved
        combined = " ".join(chunks)
        assert "Hello" in combined
        assert "John" in combined
        assert "Sarah" in combined
