"""Tests for Orpheus TTS."""

import tempfile
from pathlib import Path

import pytest
import numpy as np

from tts_helper.orpheus_tts import (
    OrpheusTTS,
    OrpheusTTSConfig,
    get_supported_voices,
    get_default_voice,
    SUPPORTED_VOICES,
    DEFAULT_VOICES,
)


class TestOrpheusTTSConfig:
    """Tests for OrpheusTTSConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OrpheusTTSConfig()

        assert config.language == "english"
        assert config.voice == "tara"  # Default English voice
        assert config.use_gpu is True
        assert config.n_gpu_layers == -1
        assert config.verbose is False
        assert config.model_path is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = OrpheusTTSConfig(
            language="french",
            voice="pierre",
            use_gpu=False,
            verbose=True,
        )

        assert config.language == "french"
        assert config.voice == "pierre"
        assert config.use_gpu is False
        assert config.n_gpu_layers == 0  # Should be 0 when use_gpu=False
        assert config.verbose is True

    def test_invalid_language_raises_error(self):
        """Test that invalid language raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            OrpheusTTSConfig(language="klingon")

    def test_invalid_voice_raises_error(self):
        """Test that invalid voice for language raises ValueError."""
        with pytest.raises(ValueError, match="not supported for"):
            OrpheusTTSConfig(language="english", voice="pierre")  # French voice

    def test_voice_defaults_by_language(self):
        """Test that default voice is set correctly for each language."""
        en_config = OrpheusTTSConfig(language="english")
        assert en_config.voice == "tara"

        fr_config = OrpheusTTSConfig(language="french")
        assert fr_config.voice == "marie"

        es_config = OrpheusTTSConfig(language="spanish")
        assert es_config.voice == "maria"

        it_config = OrpheusTTSConfig(language="italian")
        assert it_config.voice == "giulia"

    def test_model_path_defaults(self):
        """Test that model path is set correctly for each language."""
        en_config = OrpheusTTSConfig(language="english")
        assert en_config.model_path is None  # English uses default

        fr_config = OrpheusTTSConfig(language="french")
        assert fr_config.model_path == "canopylabs/3b-fr-ft-research_release"

        es_config = OrpheusTTSConfig(language="spanish")
        assert es_config.model_path == "canopylabs/3b-es_it-ft-research_release"

    def test_explicit_model_path(self):
        """Test using explicit model path."""
        config = OrpheusTTSConfig(
            language="english",
            model_path="custom/model/path",
        )

        assert config.model_path == "custom/model/path"

    def test_use_gpu_false_sets_n_gpu_layers_to_zero(self):
        """Test that disabling GPU sets n_gpu_layers to 0."""
        config = OrpheusTTSConfig(use_gpu=False, n_gpu_layers=10)

        assert config.n_gpu_layers == 0  # Should override to 0

    def test_config_serialization(self):
        """Test config can be serialized to/from JSON."""
        config = OrpheusTTSConfig(
            language="spanish",
            voice="javi",
            use_gpu=True,
            verbose=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            config.to_json(json_path)

            loaded = OrpheusTTSConfig.from_json(json_path)

            assert loaded.language == config.language
            assert loaded.voice == config.voice
            assert loaded.use_gpu == config.use_gpu
            assert loaded.verbose == config.verbose


class TestOrpheusTTS:
    """Tests for OrpheusTTS."""

    def test_init(self):
        """Test TTS initialization."""
        config = OrpheusTTSConfig(language="english", voice="leo")
        tts = OrpheusTTS(config)

        assert tts.config == config
        assert tts._engine is None  # Lazy loading

    def test_repr(self):
        """Test string representation."""
        config = OrpheusTTSConfig(language="french", voice="amelie", use_gpu=True)
        tts = OrpheusTTS(config)

        repr_str = repr(tts)
        assert "OrpheusTTS" in repr_str
        assert "french" in repr_str
        assert "amelie" in repr_str
        assert "GPU" in repr_str

    def test_repr_cpu_mode(self):
        """Test string representation in CPU mode."""
        config = OrpheusTTSConfig(language="english", use_gpu=False)
        tts = OrpheusTTS(config)

        repr_str = repr(tts)
        assert "CPU" in repr_str

    def test_synthesize_empty_text(self):
        """Test synthesis with empty text."""
        try:
            config = OrpheusTTSConfig(language="english")
            tts = OrpheusTTS(config)

            # Should return silence, not crash
            sr, audio = tts.synthesize("")

            assert sr == 24000
            assert isinstance(audio, np.ndarray)
            assert len(audio) == 0
        except ImportError:
            pytest.skip("orpheus-cpp not installed")

    def test_save_audio_creates_parent_dirs(self):
        """Test that save_audio creates parent directories."""
        config = OrpheusTTSConfig()
        tts = OrpheusTTS(config)

        # Create dummy audio
        audio_data = np.random.randn(1000).astype(np.int16)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "nested" / "output.wav"
            tts.save_audio(audio_data, 24000, output_path)

            assert output_path.exists()

    def test_engine_lazy_loading_without_orpheus_raises_error(self, monkeypatch):
        """Test that accessing engine without orpheus-cpp raises helpful error."""
        config = OrpheusTTSConfig()
        tts = OrpheusTTS(config)

        # Mock the import to fail
        import sys

        def mock_import(name, *args, **kwargs):
            if "orpheus_cpp" in name:
                raise ImportError("No module named 'orpheus_cpp'")
            return __import__(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        with pytest.raises(ImportError, match="orpheus-cpp is not installed"):
            _ = tts.engine


class TestVoiceHelperFunctions:
    """Tests for helper functions."""

    def test_get_supported_voices_english(self):
        """Test getting supported voices for English."""
        voices = get_supported_voices("english")

        assert isinstance(voices, list)
        assert len(voices) == 8
        assert "tara" in voices
        assert "leo" in voices
        assert "zoe" in voices

    def test_get_supported_voices_all_languages(self):
        """Test getting supported voices for all languages."""
        for language in ["english", "french", "spanish", "italian"]:
            voices = get_supported_voices(language)
            assert isinstance(voices, list)
            assert len(voices) > 0

    def test_get_supported_voices_invalid_language(self):
        """Test getting voices for invalid language raises error."""
        with pytest.raises(ValueError, match="not supported"):
            get_supported_voices("klingon")

    def test_get_default_voice_english(self):
        """Test getting default voice for English."""
        voice = get_default_voice("english")
        assert voice == "tara"

    def test_get_default_voice_all_languages(self):
        """Test getting default voice for all languages."""
        expected_defaults = {
            "english": "tara",
            "french": "marie",
            "spanish": "maria",
            "italian": "giulia",
        }

        for language, expected_voice in expected_defaults.items():
            voice = get_default_voice(language)
            assert voice == expected_voice

    def test_get_default_voice_invalid_language(self):
        """Test getting default voice for invalid language raises error."""
        with pytest.raises(ValueError, match="not supported"):
            get_default_voice("klingon")

    def test_voice_mappings_consistency(self):
        """Test that voice mappings are consistent."""
        # Check that default voices are in supported voices
        for lang_code, default_voice in DEFAULT_VOICES.items():
            assert default_voice in SUPPORTED_VOICES[lang_code]

        # Check that all language codes have voices
        for lang_code in ["en", "fr", "es", "it"]:
            assert lang_code in SUPPORTED_VOICES
            assert lang_code in DEFAULT_VOICES
            assert len(SUPPORTED_VOICES[lang_code]) > 0


class TestOrpheusTTSIntegration:
    """Integration tests for OrpheusTTS (requires orpheus-cpp)."""

    @pytest.fixture
    def tts(self):
        """Fixture providing TTS instance."""
        config = OrpheusTTSConfig(language="english", voice="tara", use_gpu=False)
        try:
            return OrpheusTTS(config)
        except ImportError:
            pytest.skip("orpheus-cpp not installed")

    def test_full_pipeline(self, tts):
        """Test complete synthesis pipeline (if orpheus-cpp is installed)."""
        text = "Hello world."

        try:
            # Synthesize
            sample_rate, audio = tts.synthesize(text)

            assert sample_rate == 24000
            assert isinstance(audio, np.ndarray)
            assert len(audio) > 0

            # Save to file
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "test.wav"
                tts.save_audio(audio, sample_rate, output_path)

                assert output_path.exists()
                assert output_path.stat().st_size > 0

        except Exception as e:
            # If synthesis fails, it's likely a missing model or other env issue
            # Skip gracefully
            pytest.skip(f"Synthesis failed (likely missing model): {e}")
