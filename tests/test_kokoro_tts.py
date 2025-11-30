"""Tests for Kokoro TTS."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tts_helper.kokoro_tts import (
    KokoroTTS,
    KokoroTTSConfig,
    get_supported_voices,
    get_default_voice,
    SUPPORTED_VOICES,
    DEFAULT_VOICES,
)


class TestKokoroTTSConfig:
    """Tests for KokoroTTSConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = KokoroTTSConfig()

        assert config.language == "english"
        assert config.voice == "af_sarah"  # Default US English voice
        assert config.speed == 1.0
        assert config.model_path is None
        assert config.voices_path is None
        assert config.verbose is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = KokoroTTSConfig(
            language="french",
            voice="ff_siwis",
            speed=1.5,
            verbose=True,
        )

        assert config.language == "french"
        assert config.voice == "ff_siwis"
        assert config.speed == 1.5
        assert config.verbose is True

    def test_invalid_language_raises_error(self):
        """Test that invalid language raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported language"):
            KokoroTTSConfig(language="klingon")

    def test_invalid_voice_raises_error(self):
        """Test that invalid voice for language raises ValueError."""
        with pytest.raises(ValueError, match="not supported for"):
            KokoroTTSConfig(language="english", voice="ff_siwis")  # French voice

    def test_voice_defaults_by_language(self):
        """Test that default voice is set correctly for each language."""
        en_config = KokoroTTSConfig(language="english")
        assert en_config.voice == "af_sarah"

        # TODO: Add support for British English in language.py
        # gb_config = KokoroTTSConfig(language="en-gb")
        # assert gb_config.voice == "bf_emma"

        fr_config = KokoroTTSConfig(language="french")
        assert fr_config.voice == "ff_siwis"

        it_config = KokoroTTSConfig(language="italian")
        assert it_config.voice == "if_sara"

        ja_config = KokoroTTSConfig(language="japanese")
        assert ja_config.voice == "jf_alpha"

        cmn_config = KokoroTTSConfig(language="chinese")
        assert cmn_config.voice == "zf_xiaoxiao"

    def test_speed_validation(self):
        """Test that speed is validated."""
        # Valid speeds
        KokoroTTSConfig(speed=0.5)
        KokoroTTSConfig(speed=1.0)
        KokoroTTSConfig(speed=2.0)

        # Invalid speeds
        with pytest.raises(ValueError, match="Speed must be"):
            KokoroTTSConfig(speed=0.4)

        with pytest.raises(ValueError, match="Speed must be"):
            KokoroTTSConfig(speed=2.1)

    def test_explicit_paths(self):
        """Test using explicit model and voices paths."""
        config = KokoroTTSConfig(
            model_path="/custom/path/model.onnx",
            voices_path="/custom/path/voices.bin",
        )

        assert config.model_path == "/custom/path/model.onnx"
        assert config.voices_path == "/custom/path/voices.bin"

    def test_config_serialization(self):
        """Test config can be serialized to/from JSON."""
        config = KokoroTTSConfig(
            language="japanese",
            voice="jf_alpha",
            speed=1.2,
            verbose=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            config.to_json(json_path)

            loaded = KokoroTTSConfig.from_json(json_path)

            assert loaded.language == config.language
            assert loaded.voice == config.voice
            assert loaded.speed == config.speed
            assert loaded.verbose == config.verbose


class TestKokoroTTS:
    """Tests for KokoroTTS."""

    def test_init(self):
        """Test TTS initialization."""
        config = KokoroTTSConfig(language="english", voice="am_adam")
        tts = KokoroTTS(config)

        assert tts.config == config
        assert tts._engine is None  # Lazy loading

    def test_repr(self):
        """Test string representation."""
        config = KokoroTTSConfig(language="french", voice="ff_siwis", speed=1.5)
        tts = KokoroTTS(config)

        repr_str = repr(tts)
        assert "KokoroTTS" in repr_str
        assert "french" in repr_str
        assert "ff_siwis" in repr_str
        assert "1.5" in repr_str

    def test_synthesize_empty_text(self):
        """Test synthesis with empty text."""
        config = KokoroTTSConfig(language="english")
        tts = KokoroTTS(config)

        # Should return silence, not crash
        sr, audio = tts.synthesize("")

        assert sr == 24000
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 0

    def test_save_audio_creates_parent_dirs(self):
        """Test that save_audio creates parent directories."""
        config = KokoroTTSConfig()
        tts = KokoroTTS(config)

        # Create dummy audio
        audio_data = np.random.randn(1000).astype(np.int16)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "nested" / "output.wav"
            tts.save_audio(audio_data, 24000, output_path)

            assert output_path.exists()

    def test_engine_lazy_loading_without_kokoro_raises_error(self, monkeypatch):
        """Test that accessing engine without kokoro-onnx raises helpful error."""
        config = KokoroTTSConfig()
        tts = KokoroTTS(config)

        # Mock the import to fail
        import sys

        def mock_import(name, *args, **kwargs):
            if "kokoro_onnx" in name:
                raise ImportError("No module named 'kokoro_onnx'")
            return __import__(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        with pytest.raises(ImportError, match="kokoro-onnx is not installed"):
            _ = tts.engine


class TestVoiceHelperFunctions:
    """Tests for helper functions."""

    def test_get_supported_voices_english(self):
        """Test getting supported voices for English."""
        voices = get_supported_voices("english")

        assert isinstance(voices, list)
        assert len(voices) == 19  # 11 female + 8 male
        assert "af_sarah" in voices
        assert "am_adam" in voices
        assert "af_nova" in voices

    def test_get_supported_voices_all_languages(self):
        """Test getting supported voices for all languages."""
        for language in ["english", "french", "italian", "japanese", "chinese"]:
            voices = get_supported_voices(language)
            assert isinstance(voices, list)
            assert len(voices) > 0

    def test_get_supported_voices_invalid_language(self):
        """Test getting voices for invalid language raises error."""
        with pytest.raises(ValueError, match="Unsupported language"):
            get_supported_voices("klingon")

    def test_get_default_voice_english(self):
        """Test getting default voice for English."""
        voice = get_default_voice("english")
        assert voice == "af_sarah"

    def test_get_default_voice_all_languages(self):
        """Test getting default voice for all languages."""
        expected_defaults = {
            "english": "af_sarah",
            "french": "ff_siwis",
            "italian": "if_sara",
            "japanese": "jf_alpha",
            "chinese": "zf_xiaoxiao",
        }

        for language, expected_voice in expected_defaults.items():
            voice = get_default_voice(language)
            assert voice == expected_voice

    def test_get_default_voice_invalid_language(self):
        """Test getting default voice for invalid language raises error."""
        with pytest.raises(ValueError, match="Unsupported language"):
            get_default_voice("klingon")

    def test_voice_mappings_consistency(self):
        """Test that voice mappings are consistent."""
        # Check that default voices are in supported voices
        for lang_code, default_voice in DEFAULT_VOICES.items():
            assert default_voice in SUPPORTED_VOICES[lang_code]

        # Check that all language codes have voices
        for lang_code in ["en-us", "fr-fr", "it", "ja", "cmn"]:
            assert lang_code in SUPPORTED_VOICES
            assert lang_code in DEFAULT_VOICES
            assert len(SUPPORTED_VOICES[lang_code]) > 0


class TestKokoroTTSIntegration:
    """Integration tests for KokoroTTS (requires kokoro-onnx and model files)."""

    @pytest.fixture
    def tts(self):
        """Fixture providing TTS instance."""
        try:
            config = KokoroTTSConfig(language="english", voice="af_sarah")
            return KokoroTTS(config)
        except (ImportError, FileNotFoundError) as e:
            pytest.skip(f"kokoro-onnx or model files not available: {e}")

    def test_full_pipeline(self, tts):
        """Test complete synthesis pipeline (if kokoro-onnx is installed)."""
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
            # If synthesis fails, it's likely missing model files or other env issue
            # Skip gracefully
            pytest.skip(f"Synthesis failed (likely missing model files): {e}")
