"""Tests for Kokoro TTS v1.1 models and voices."""

import tempfile
from pathlib import Path

import pytest

from tts_helper.kokoro_tts import KokoroTTS, KokoroTTSConfig


@pytest.mark.integration
class TestKokoroV11:
    """Integration tests for Kokoro v1.1 models."""

    def test_bf_vale_voice_synthesis(self):
        """Test synthesis with bf_vale voice (new in v1.1)."""
        # Configure Kokoro to use v1.1 model and voices
        config = KokoroTTSConfig(
            language="english-gb",
            voice="bf_vale",
            speed=1.0,
            model_path="kokoro-v1.1-zh.onnx",
            voices_path="voices-v1.1-zh.bin",
            model_url="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.1-zh.onnx",
            voices_url="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.1-zh.bin",
            verbose=True,
        )

        tts = KokoroTTS(config)

        # Test text
        text = "Hello, this is a test of the bf_vale voice in Kokoro v1.1."

        # Synthesize
        sample_rate, audio = tts.synthesize(text)

        # Verify output
        assert sample_rate == 24000, "Sample rate should be 24kHz"
        assert len(audio) > 0, "Audio should not be empty"
        assert audio.dtype.name == "float32", "Audio should be float32"

        # Test saving to file
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_bf_vale.wav"
            tts.save_audio(audio, sample_rate, output_path)
            assert output_path.exists(), "Output file should exist"
            assert output_path.stat().st_size > 0, "Output file should not be empty"

    def test_v1_1_model_download(self):
        """Test that v1.1 model files are downloaded when specified."""
        config = KokoroTTSConfig(
            language="english-gb",
            voice="bf_vale",
            model_path="kokoro-v1.1-zh.onnx",
            voices_path="voices-v1.1-zh.bin",
            model_url="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.1-zh.onnx",
            voices_url="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.1-zh.bin",
            verbose=False,
        )

        tts = KokoroTTS(config)

        # Access engine property to trigger download
        _ = tts.engine

        # Verify files exist
        model_path = Path("kokoro-v1.1-zh.onnx")
        voices_path = Path("voices-v1.1-zh.bin")

        assert model_path.exists(), "Model file should be downloaded"
        assert voices_path.exists(), "Voices file should be downloaded"
        assert model_path.stat().st_size > 0, "Model file should not be empty"
        assert voices_path.stat().st_size > 0, "Voices file should not be empty"

    def test_config_with_custom_urls(self):
        """Test that custom URLs are properly configured."""
        custom_model_url = "https://example.com/custom-model.onnx"
        custom_voices_url = "https://example.com/custom-voices.bin"

        config = KokoroTTSConfig(
            language="english",
            voice="af_sarah",
            model_url=custom_model_url,
            voices_url=custom_voices_url,
        )

        # Verify config stores URLs
        assert config.model_url == custom_model_url
        assert config.voices_url == custom_voices_url
