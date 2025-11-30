"""Tests for base TTS classes."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tts_helper.tts import TTS, TTSConfig


@dataclass
class MockTTSConfig(TTSConfig):
    """Mock configuration for testing base class."""

    sample_rate: int = 22050
    add_noise: bool = False


class MockTTS(TTS):
    """Mock TTS for testing base class."""

    def synthesize(self, text: str) -> tuple[int, np.ndarray]:
        """Generate mock audio."""
        # Generate simple sine wave
        duration = len(text) * 0.1  # 0.1 second per character
        t = np.linspace(0, duration, int(self.config.sample_rate * duration))  # type: ignore[attr-defined]
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone

        if self.config.add_noise:  # type: ignore[attr-defined]
            audio += np.random.randn(len(audio)) * 0.1

        return self.config.sample_rate, audio.astype(np.float32)  # type: ignore[attr-defined]

    def save_audio(
        self, audio_data: np.ndarray, sample_rate: int, output_path: str | Path
    ) -> None:
        """Save mock audio."""
        from scipy.io.wavfile import write  # type: ignore[import-untyped]

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write(str(path), sample_rate, audio_data)

    def __repr__(self) -> str:
        return f"MockTTS(sample_rate={self.config.sample_rate})"  # type: ignore[attr-defined]


class TestTTSConfig:
    """Tests for TTSConfig base class."""

    def test_to_dict(self) -> None:
        """Test converting config to dictionary."""
        config = MockTTSConfig(sample_rate=48000, add_noise=True)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["sample_rate"] == 48000
        assert config_dict["add_noise"] is True

    def test_from_dict(self) -> None:
        """Test creating config from dictionary."""
        config_dict = {"sample_rate": 48000, "add_noise": True}
        config = MockTTSConfig.from_dict(config_dict)

        assert isinstance(config, MockTTSConfig)
        assert config.sample_rate == 48000
        assert config.add_noise is True

    def test_to_json(self) -> None:
        """Test saving config to JSON file."""
        config = MockTTSConfig(sample_rate=48000)

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            config.to_json(json_path)

            assert json_path.exists()

            with json_path.open("r") as f:
                loaded = json.load(f)

            assert loaded["sample_rate"] == 48000

    def test_from_json(self) -> None:
        """Test loading config from JSON file."""
        config_dict = {"sample_rate": 48000, "add_noise": False}

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"

            with json_path.open("w") as f:
                json.dump(config_dict, f)

            config = MockTTSConfig.from_json(json_path)

            assert isinstance(config, MockTTSConfig)
            assert config.sample_rate == 48000

    def test_json_roundtrip(self) -> None:
        """Test that config survives JSON save/load cycle."""
        original = MockTTSConfig(sample_rate=44100, add_noise=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            original.to_json(json_path)
            loaded = MockTTSConfig.from_json(json_path)

            assert loaded.sample_rate == original.sample_rate
            assert loaded.add_noise == original.add_noise


class TestTTS:
    """Tests for TTS base class."""

    def test_init_with_config(self) -> None:
        """Test TTS initialization with config."""
        config = MockTTSConfig(sample_rate=44100)
        tts = MockTTS(config)

        assert tts.config == config
        assert tts.config.sample_rate == 44100  # type: ignore[attr-defined]

    def test_synthesize(self) -> None:
        """Test basic synthesis."""
        config = MockTTSConfig(sample_rate=22050)
        tts = MockTTS(config)

        text = "hello"
        sample_rate, audio = tts.synthesize(text)

        assert sample_rate == 22050
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    def test_synthesize_batch(self) -> None:
        """Test batch synthesis."""
        config = MockTTSConfig()
        tts = MockTTS(config)

        texts = ["hello", "world"]
        results = tts.synthesize_batch(texts)

        assert len(results) == 2
        for sr, audio in results:
            assert sr == config.sample_rate
            assert isinstance(audio, np.ndarray)
            assert len(audio) > 0

    def test_save_audio(self) -> None:
        """Test saving audio to file."""
        config = MockTTSConfig(sample_rate=22050)
        tts = MockTTS(config)

        sample_rate, audio = tts.synthesize("test")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.wav"
            tts.save_audio(audio, sample_rate, output_path)

            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_save_audio_creates_parent_dirs(self) -> None:
        """Test that save_audio creates parent directories."""
        config = MockTTSConfig()
        tts = MockTTS(config)

        sample_rate, audio = tts.synthesize("test")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "nested" / "output.wav"
            tts.save_audio(audio, sample_rate, output_path)

            assert output_path.exists()
            assert output_path.parent.exists()

    def test_repr(self) -> None:
        """Test string representation."""
        config = MockTTSConfig(sample_rate=48000)
        tts = MockTTS(config)

        repr_str = repr(tts)
        assert "MockTTS" in repr_str
        assert "48000" in repr_str
