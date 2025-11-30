"""Tests for translation enhancer."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tts_helper.chunk import Chunk
from tts_helper.translation_enhancer import (
    TranslationEnhancer,
    TranslationEnhancerConfig,
)


class TestTranslationEnhancerConfig:
    """Tests for TranslationEnhancerConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = TranslationEnhancerConfig()

        assert config.probability == 0.1
        assert config.source_language == "english"
        assert config.target_language == "italian"
        assert config.target_lang_code == "ita_Latn"  # Property
        assert config.source_lang_code == "eng_Latn"  # Property
        assert config.model_id == "facebook/nllb-200-distilled-600M"
        assert config.max_length == 400
        assert config.device == "cpu"
        assert config.verbose is False

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = TranslationEnhancerConfig(
            probability=0.3,
            source_language="english",
            target_language="spanish",
            model_id="facebook/nllb-200-1.3B",
            max_length=512,
            device="cuda",
            verbose=True,
        )

        assert config.probability == 0.3
        assert config.source_language == "english"
        assert config.target_language == "spanish"
        assert config.target_lang_code == "spa_Latn"  # Property
        assert config.source_lang_code == "eng_Latn"  # Property
        assert config.model_id == "facebook/nllb-200-1.3B"
        assert config.max_length == 512
        assert config.device == "cuda"
        assert config.verbose is True

    def test_probability_validation(self) -> None:
        """Test that probability is validated."""
        # Valid probabilities
        TranslationEnhancerConfig(probability=0.0)
        TranslationEnhancerConfig(probability=0.5)
        TranslationEnhancerConfig(probability=1.0)

        # Invalid probabilities
        with pytest.raises(ValueError, match="Probability must be"):
            TranslationEnhancerConfig(probability=-0.1)

        with pytest.raises(ValueError, match="Probability must be"):
            TranslationEnhancerConfig(probability=1.1)

    def test_max_length_validation(self) -> None:
        """Test that max_length is validated."""
        TranslationEnhancerConfig(max_length=256)
        TranslationEnhancerConfig(max_length=1024)

        with pytest.raises(ValueError, match="max_length must be"):
            TranslationEnhancerConfig(max_length=0)

        with pytest.raises(ValueError, match="max_length must be"):
            TranslationEnhancerConfig(max_length=-1)

    def test_device_validation(self) -> None:
        """Test that device is validated."""
        TranslationEnhancerConfig(device="cpu")
        TranslationEnhancerConfig(device="cuda")

        with pytest.raises(ValueError, match="device must be"):
            TranslationEnhancerConfig(device="tpu")

    def test_config_serialization(self) -> None:
        """Test config can be serialized to/from JSON."""
        config = TranslationEnhancerConfig(
            probability=0.2,
            target_language="french",
            model_id="facebook/nllb-200-distilled-600M",
            device="cpu",
            verbose=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            config.to_json(json_path)

            loaded = TranslationEnhancerConfig.from_json(json_path)

            assert loaded.probability == config.probability  # type: ignore[attr-defined]
            assert loaded.target_language == config.target_language  # type: ignore[attr-defined]
            assert loaded.target_lang_code == config.target_lang_code  # type: ignore[attr-defined]
            assert loaded.model_id == config.model_id  # type: ignore[attr-defined]
            assert loaded.device == config.device  # type: ignore[attr-defined]
            assert loaded.verbose == config.verbose  # type: ignore[attr-defined]

    def test_config_from_dict_with_type_field(self) -> None:
        """Test that config can be loaded from dict with 'type' field filtered out."""
        # Simulate config.json format with 'type' field
        config_dict = {
            "type": "translation",  # This should be filtered out
            "probability": 0.3,
            "target_language": "german",
            "source_language": "english",
        }

        # Filter out 'type' before creating config (as done in __main__.py)
        config_without_type = {k: v for k, v in config_dict.items() if k != "type"}
        config = TranslationEnhancerConfig.from_dict(config_without_type)

        assert config.probability == 0.3  # type: ignore[attr-defined]
        assert config.target_language == "german"  # type: ignore[attr-defined]
        assert config.source_language == "english"  # type: ignore[attr-defined]
        assert (
            config.target_lang_code == "deu_Latn"  # type: ignore[attr-defined]
        )  # Property  # type: ignore[attr-defined]
        assert (
            config.source_lang_code == "eng_Latn"  # type: ignore[attr-defined]
        )  # Property  # type: ignore[attr-defined]


class TestTranslationEnhancer:
    """Tests for TranslationEnhancer."""

    def test_init(self) -> None:
        """Test enhancer initialization."""
        config = TranslationEnhancerConfig()
        enhancer = TranslationEnhancer(config)

        assert enhancer.config == config
        assert enhancer._model is None  # Lazy loading
        assert enhancer._tokenizer is None  # Lazy loading

    def test_repr(self) -> None:
        """Test string representation."""
        config = TranslationEnhancerConfig(probability=0.3, target_language="german")
        enhancer = TranslationEnhancer(config)

        repr_str = repr(enhancer)
        assert "TranslationEnhancer" in repr_str
        assert "0.3" in repr_str
        assert "german" in repr_str

    @patch("tts_helper.translation_enhancer.random.random")
    def test_enhance_no_translation_when_probability_zero(self, mock_random) -> None:  # type: ignore[no-untyped-def]
        """Test that no translation occurs when probability is 0."""
        config = TranslationEnhancerConfig(probability=0.0)
        enhancer = TranslationEnhancer(config)

        chunks = [Chunk(text="Hello world."), Chunk(text="How are you?")]
        enhanced = enhancer.enhance(chunks)

        # Should return chunks unchanged
        assert enhanced == chunks
        # Random should not be called
        mock_random.assert_not_called()

    @patch("tts_helper.translation_enhancer.random.random")
    def test_enhance_with_translation(self, mock_random) -> None:  # type: ignore[no-untyped-def]
        """Test enhancement with translation."""
        # Set up mocks
        mock_random.side_effect = [0.05, 0.95]  # First chunk translates, second doesn't

        # Mock transformers module
        with (
            patch(
                "tts_helper.translation_enhancer.AutoTokenizer"
            ) as mock_tokenizer_class,
            patch(
                "tts_helper.translation_enhancer.AutoModelForSeq2SeqLM"
            ) as mock_model_class,
        ):
            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer.convert_tokens_to_ids.return_value = 12345
            mock_tokenizer.return_value = {
                "input_ids": MagicMock(),
                "attention_mask": MagicMock(),
            }
            mock_tokenizer.batch_decode.return_value = ["Ciao mondo."]
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            # Mock model
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_model.generate.return_value = MagicMock()
            mock_model_class.from_pretrained.return_value = mock_model

            config = TranslationEnhancerConfig(
                probability=0.1,
                target_language="italian",
                pause_before_ms=300,
                pause_after_ms=300,
            )
            enhancer = TranslationEnhancer(config)

            chunks = [Chunk(text="Hello world."), Chunk(text="How are you?")]
            enhanced = enhancer.enhance(chunks)

            # Should have: original + pause_before + translation + pause_after + second chunk
            assert len(enhanced) == 5
            assert enhanced[0].text == "Hello world."
            assert enhanced[1].silence_ms == 300  # Pause before
            assert enhanced[2].text == "Ciao mondo."  # Translation
            assert enhanced[3].silence_ms == 300  # Pause after
            assert enhanced[4].text == "How are you?"

            # Verify model was called correctly
            mock_model.generate.assert_called_once()

    @patch("tts_helper.translation_enhancer.random.random")
    def test_enhance_with_translation_error(self, mock_random) -> None:  # type: ignore[no-untyped-def]
        """Test that translation errors are handled gracefully."""
        mock_random.return_value = 0.05  # Always translate

        # Mock transformers module
        with (
            patch(
                "tts_helper.translation_enhancer.AutoTokenizer"
            ) as mock_tokenizer_class,
            patch(
                "tts_helper.translation_enhancer.AutoModelForSeq2SeqLM"
            ) as mock_model_class,
        ):
            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            # Model raises exception during generate
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_model.generate.side_effect = Exception("Model error")
            mock_model_class.from_pretrained.return_value = mock_model

            config = TranslationEnhancerConfig(
                probability=1.0,
                target_language="spanish",
            )
            enhancer = TranslationEnhancer(config)

            chunks = [Chunk(text="Hello world.")]
            enhanced = enhancer.enhance(chunks)

            # Should only have original chunk (translation failed)
            assert len(enhanced) == 1
            assert enhanced[0].text == "Hello world."

    @patch("tts_helper.translation_enhancer.random.random")
    def test_enhance_multiple_chunks_mixed(self, mock_random) -> None:  # type: ignore[no-untyped-def]
        """Test enhancement with multiple chunks, some translated."""
        # First and third translate, second doesn't
        mock_random.side_effect = [0.05, 0.95, 0.05]

        # Mock transformers module
        with (
            patch(
                "tts_helper.translation_enhancer.AutoTokenizer"
            ) as mock_tokenizer_class,
            patch(
                "tts_helper.translation_enhancer.AutoModelForSeq2SeqLM"
            ) as mock_model_class,
        ):
            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer.convert_tokens_to_ids.return_value = 12345
            mock_tokenizer.return_value = {
                "input_ids": MagicMock(),
                "attention_mask": MagicMock(),
            }
            mock_tokenizer.batch_decode.side_effect = [["Ciao."], ["Arrivederci."]]
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            # Mock model
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_model.generate.return_value = MagicMock()
            mock_model_class.from_pretrained.return_value = mock_model

            config = TranslationEnhancerConfig(
                probability=0.1,
                target_language="italian",
                pause_before_ms=300,
                pause_after_ms=300,
            )
            enhancer = TranslationEnhancer(config)

            chunks = [
                Chunk(text="Hello."),
                Chunk(text="How are you?"),
                Chunk(text="Goodbye."),
            ]
            enhanced = enhancer.enhance(chunks)

            # Should have: chunk1 + pause + translation + pause + chunk2 + chunk3 + pause + translation + pause
            assert len(enhanced) == 9
            assert enhanced[0].text == "Hello."
            assert enhanced[1].silence_ms == 300
            assert enhanced[2].text == "Ciao."
            assert enhanced[3].silence_ms == 300
            assert enhanced[4].text == "How are you?"
            assert enhanced[5].text == "Goodbye."
            assert enhanced[6].silence_ms == 300
            assert enhanced[7].text == "Arrivederci."
            assert enhanced[8].silence_ms == 300

    def test_model_lazy_loading(self) -> None:
        """Test that model is lazy loaded."""
        config = TranslationEnhancerConfig()
        enhancer = TranslationEnhancer(config)

        assert enhancer._model is None
        assert enhancer._tokenizer is None

    def test_model_loading(self) -> None:
        """Test that model and tokenizer are loaded correctly."""
        # Mock transformers module
        with (
            patch(
                "tts_helper.translation_enhancer.AutoTokenizer"
            ) as mock_tokenizer_class,
            patch(
                "tts_helper.translation_enhancer.AutoModelForSeq2SeqLM"
            ) as mock_model_class,
        ):
            # Mock tokenizer
            mock_tokenizer = MagicMock()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

            # Mock model
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            mock_model_class.from_pretrained.return_value = mock_model

            config = TranslationEnhancerConfig(
                model_id="facebook/nllb-200-distilled-600M", device="cpu"
            )
            enhancer = TranslationEnhancer(config)

            # Access model property to trigger loading
            _ = enhancer.model

            # Verify model and tokenizer were loaded
            mock_tokenizer_class.from_pretrained.assert_called_once_with(
                "facebook/nllb-200-distilled-600M"
            )
            mock_model_class.from_pretrained.assert_called_once_with(
                "facebook/nllb-200-distilled-600M"
            )
            mock_model.to.assert_called_once_with("cpu")

    def test_model_without_library_raises_error(self) -> None:
        """Test that accessing model without library raises error."""
        # Mock the module-level imports to be None (simulating missing transformers)
        with (
            patch("tts_helper.translation_enhancer.AutoTokenizer", None),
            patch("tts_helper.translation_enhancer.AutoModelForSeq2SeqLM", None),
        ):
            config = TranslationEnhancerConfig()
            enhancer = TranslationEnhancer(config)

            with pytest.raises(ImportError, match="transformers is not installed"):
                _ = enhancer.model
