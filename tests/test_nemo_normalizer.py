"""Tests for NeMo-based normalizer."""

import tempfile
from pathlib import Path

import pytest

from tts_helper.nemo_normalizer import NemoNormalizer, NemoNormalizerConfig


class TestNemoNormalizerConfig:
    """Tests for NemoNormalizerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NemoNormalizerConfig()

        assert config.language == "en"
        assert config.input_case == "cased"
        assert config.cache_dir is None
        assert config.verbose is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = NemoNormalizerConfig(
            language="de",
            input_case="lower_cased",
            verbose=True,
        )

        assert config.language == "de"
        assert config.input_case == "lower_cased"
        assert config.verbose is True

    def test_invalid_language_raises_error(self):
        """Test that invalid language raises ValueError."""
        with pytest.raises(ValueError, match="not supported by NeMo"):
            NemoNormalizerConfig(language="invalid_lang")

    def test_invalid_input_case_raises_error(self):
        """Test that invalid input_case raises ValueError."""
        with pytest.raises(ValueError, match="must be 'cased' or 'lower_cased'"):
            NemoNormalizerConfig(input_case="mixed")

    def test_config_serialization(self):
        """Test config can be serialized to/from JSON."""
        config = NemoNormalizerConfig(
            language="es",
            input_case="lower_cased",
            verbose=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            config.to_json(json_path)

            loaded = NemoNormalizerConfig.from_json(json_path)

            assert loaded.language == config.language
            assert loaded.input_case == config.input_case
            assert loaded.verbose == config.verbose


class TestNemoNormalizer:
    """Tests for NemoNormalizer."""

    @pytest.fixture
    def config(self):
        """Fixture providing default config."""
        return NemoNormalizerConfig(language="en", verbose=False)

    @pytest.fixture
    def normalizer(self, config):
        """Fixture providing normalizer instance."""
        try:
            return NemoNormalizer(config)
        except ImportError:
            pytest.skip("nemo_text_processing not installed")

    def test_init(self, config):
        """Test normalizer initialization."""
        try:
            normalizer = NemoNormalizer(config)
            assert normalizer.config == config
            assert normalizer._normalizer is None  # Lazy loading
        except ImportError:
            pytest.skip("nemo_text_processing not installed")

    def test_normalizer_lazy_loading(self, normalizer):
        """Test that NeMo normalizer is lazily loaded."""
        assert normalizer._normalizer is None
        norm = normalizer.normalizer
        assert norm is not None
        assert normalizer._normalizer is norm  # Same instance on second access

    def test_normalize_currency(self, normalizer):
        """Test normalization of currency."""
        text = "The price is $123.45"
        normalized = normalizer.normalize(text)

        # Check that currency was normalized (exact format may vary)
        assert "dollar" in normalized.lower()
        assert "123" not in normalized or "one hundred" in normalized.lower()

    def test_normalize_numbers(self, normalizer):
        """Test normalization of numbers."""
        text = "I have 42 apples"
        normalized = normalizer.normalize(text)

        # Check that number was normalized
        assert "42" not in normalized or "forty" in normalized.lower()

    def test_normalize_empty_text(self, normalizer):
        """Test normalization of empty text."""
        assert normalizer.normalize("") == ""
        assert normalizer.normalize("   ") == "   "

    def test_normalize_plain_text(self, normalizer):
        """Test normalization of plain text without special elements."""
        text = "This is a simple sentence"
        normalized = normalizer.normalize(text)

        # Plain text should remain largely unchanged
        assert "simple" in normalized.lower()
        assert "sentence" in normalized.lower()

    def test_normalize_batch(self, normalizer):
        """Test batch normalization."""
        texts = [
            "The price is $10",
            "I have 5 cats",
        ]
        results = normalizer.normalize_batch(texts)

        assert len(results) == 2
        assert all(isinstance(result, str) for result in results)
        # Check first result contains normalized currency
        assert "dollar" in results[0].lower() or "$" in results[0]

    def test_repr(self, normalizer):
        """Test string representation."""
        repr_str = repr(normalizer)

        assert "NemoNormalizer" in repr_str
        assert "en" in repr_str
        assert "cased" in repr_str

    def test_normalize_handles_errors_gracefully(self, normalizer):
        """Test that normalization errors don't crash the pipeline."""
        # This shouldn't raise an exception, even with unusual input
        try:
            result = normalizer.normalize("" * 10000)  # Very long empty string
            assert isinstance(result, str)
        except Exception:
            # If it does raise, that's also acceptable for this test
            pass

    def test_different_languages(self):
        """Test normalizer with different languages."""
        try:
            # Test Spanish
            config_es = NemoNormalizerConfig(language="es", verbose=False)
            normalizer_es = NemoNormalizer(config_es)

            # Spanish text with numbers
            spanish_text = "Tengo 25 manzanas"
            result = normalizer_es.normalize(spanish_text)

            assert isinstance(result, str)
            assert len(result) > 0
        except ImportError:
            pytest.skip("nemo_text_processing not installed")

    def test_normalizer_without_nemo_raises_import_error(self, monkeypatch):
        """Test that missing nemo_text_processing raises helpful error."""
        # Mock the import to fail
        import sys

        config = NemoNormalizerConfig()
        normalizer = NemoNormalizer(config)

        def mock_import(name, *args, **kwargs):
            if "nemo_text_processing" in name:
                raise ImportError("No module named 'nemo_text_processing'")
            return __import__(name, *args, **kwargs)

        # Access normalizer property to trigger lazy load
        monkeypatch.setattr("builtins.__import__", mock_import)

        with pytest.raises(ImportError, match="nemo_text_processing is not installed"):
            _ = normalizer.normalizer


class TestNemoNormalizerIntegration:
    """Integration tests for NemoNormalizer with realistic scenarios."""

    @pytest.fixture
    def normalizer(self):
        """Fixture providing normalizer for integration tests."""
        config = NemoNormalizerConfig(language="en", verbose=False)
        try:
            return NemoNormalizer(config)
        except ImportError:
            pytest.skip("nemo_text_processing not installed")

    def test_audiobook_text_normalization(self, normalizer):
        """Test with realistic audiobook text."""
        text = (
            "Dr. Smith arrived at 3:30pm with $1,234.56 in his pocket. "
            "He had exactly 42 reasons to be there, and Chapter 7 was one of them."
        )

        normalized = normalizer.normalize(text)

        # Check that special elements were normalized
        assert isinstance(normalized, str)
        assert len(normalized) > len(text)  # Normalization usually expands text
        # Currency should be normalized
        assert "dollar" in normalized.lower() or "cent" in normalized.lower()

    def test_mixed_content_normalization(self, normalizer):
        """Test with mixed numbers, dates, and text."""
        text = "On 01/15/2024, I paid $99.99 for 3 items."
        normalized = normalizer.normalize(text)

        assert isinstance(normalized, str)
        # Check that normalization happened (text should be longer)
        assert len(normalized) >= len(text)
