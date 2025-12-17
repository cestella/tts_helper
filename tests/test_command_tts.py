"""Tests for command-based TTS engine."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from scipy.io.wavfile import write as wav_write  # type: ignore[import-untyped]

from tts_helper.command_tts import CommandTTS, CommandTTSConfig


class TestCommandTTSConfig:
    """Tests for CommandTTSConfig."""

    def test_default_config(self) -> None:
        """Test that config requires command."""
        with pytest.raises(ValueError, match="command must be specified"):
            CommandTTSConfig()

    def test_valid_config(self) -> None:
        """Test valid configuration."""
        config = CommandTTSConfig(command="my-tts")

        assert config.command == "my-tts"
        assert config.extra_args == []
        assert config.verbose is False

    def test_config_with_extra_args(self) -> None:
        """Test configuration with extra arguments."""
        config = CommandTTSConfig(
            command="my-tts",
            extra_args=["--model", "best", "--speed", "1.5"],
            verbose=True,
        )

        assert config.command == "my-tts"
        assert config.extra_args == ["--model", "best", "--speed", "1.5"]
        assert config.verbose is True

    def test_config_serialization(self) -> None:
        """Test config can be serialized to/from JSON."""
        config = CommandTTSConfig(
            command="my-tts", extra_args=["--model", "fast"], verbose=True
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            config.to_json(json_path)

            loaded = CommandTTSConfig.from_json(json_path)

            assert loaded.command == config.command
            assert loaded.extra_args == config.extra_args
            assert loaded.verbose == config.verbose


class TestCommandTTS:
    """Tests for CommandTTS."""

    def test_init(self) -> None:
        """Test TTS initialization."""
        config = CommandTTSConfig(command="test-tts")
        tts = CommandTTS(config)

        assert tts.config == config
        assert tts.config.command == "test-tts"

    def test_repr(self) -> None:
        """Test string representation."""
        config = CommandTTSConfig(command="my-tts-tool")
        tts = CommandTTS(config)

        repr_str = repr(tts)
        assert "CommandTTS" in repr_str
        assert "my-tts-tool" in repr_str

    def test_synthesize_empty_text(self) -> None:
        """Test that empty text raises ValueError."""
        config = CommandTTSConfig(command="test-tts")
        tts = CommandTTS(config)

        with pytest.raises(ValueError, match="Text cannot be empty"):
            tts.synthesize("")

        with pytest.raises(ValueError, match="Text cannot be empty"):
            tts.synthesize("   ")

    @patch("subprocess.run")
    def test_synthesize_success(self, mock_run: Mock) -> None:
        """Test successful synthesis."""
        config = CommandTTSConfig(command="test-tts")
        tts = CommandTTS(config)

        # Mock successful command execution
        mock_run.return_value = MagicMock(returncode=0, stdout="Success", stderr="")

        # We need to patch the WAV file reading as well
        # Create a mock WAV file
        sample_rate = 24000
        duration = 1.0
        audio_data = np.sin(
            2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration))
        )
        audio_int16 = (audio_data * 32767).astype(np.int16)

        def side_effect_run(cmd: list[str], **kwargs: object) -> MagicMock:
            # Extract output path from command
            output_idx = cmd.index("--output")
            output_path = cmd[output_idx + 1]
            # Write the mock WAV file
            wav_write(output_path, sample_rate, audio_int16)
            return MagicMock(returncode=0, stdout="Success", stderr="")

        mock_run.side_effect = side_effect_run

        # Test synthesis
        sr, audio = tts.synthesize("Hello, world!")

        # Verify results
        assert sr == sample_rate
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0
        assert -1.0 <= audio.min() <= 1.0
        assert -1.0 <= audio.max() <= 1.0

        # Verify command was called correctly
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "test-tts"
        assert "--input" in call_args
        assert "--output" in call_args

    @patch("subprocess.run")
    def test_synthesize_with_extra_args(self, mock_run: Mock) -> None:
        """Test synthesis with extra command arguments."""
        config = CommandTTSConfig(
            command="test-tts",
            extra_args=["--model", "best", "--speed", "1.5"],
        )
        tts = CommandTTS(config)

        sample_rate = 24000
        audio_int16 = (np.random.randn(24000) * 32767).astype(np.int16)

        def side_effect_run(cmd: list[str], **kwargs: object) -> MagicMock:
            output_idx = cmd.index("--output")
            output_path = cmd[output_idx + 1]
            wav_write(output_path, sample_rate, audio_int16)
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = side_effect_run

        sr, audio = tts.synthesize("Test")

        # Verify extra args were included
        call_args = mock_run.call_args[0][0]
        assert "--model" in call_args
        assert "best" in call_args
        assert "--speed" in call_args
        assert "1.5" in call_args

    @patch("subprocess.run")
    def test_synthesize_command_failure(self, mock_run: Mock) -> None:
        """Test handling of command failure."""
        config = CommandTTSConfig(command="test-tts")
        tts = CommandTTS(config)

        # Mock command failure
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["test-tts"],
            output="",
            stderr="Command failed",
        )

        with pytest.raises(RuntimeError, match="TTS command failed"):
            tts.synthesize("Test")

    @patch("subprocess.run")
    def test_synthesize_command_not_found(self, mock_run: Mock) -> None:
        """Test handling of command not found."""
        config = CommandTTSConfig(command="nonexistent-tts")
        tts = CommandTTS(config)

        # Mock FileNotFoundError
        mock_run.side_effect = FileNotFoundError("Command not found")

        with pytest.raises(RuntimeError, match="TTS command not found"):
            tts.synthesize("Test")

    @patch("subprocess.run")
    def test_synthesize_no_output_file(self, mock_run: Mock) -> None:
        """Test handling when command doesn't create output file."""
        config = CommandTTSConfig(command="test-tts")
        tts = CommandTTS(config)

        # Mock successful command but don't create output file
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with pytest.raises(
            RuntimeError, match="TTS command did not create output file"
        ):
            tts.synthesize("Test")

    @patch("subprocess.run")
    def test_synthesize_converts_int16_to_float32(self, mock_run: Mock) -> None:
        """Test that int16 audio is converted to float32."""
        config = CommandTTSConfig(command="test-tts")
        tts = CommandTTS(config)

        sample_rate = 24000
        audio_int16 = np.array([0, 16384, 32767, -16384, -32768], dtype=np.int16)

        def side_effect_run(cmd: list[str], **kwargs: object) -> MagicMock:
            output_idx = cmd.index("--output")
            output_path = cmd[output_idx + 1]
            wav_write(output_path, sample_rate, audio_int16)
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = side_effect_run

        sr, audio = tts.synthesize("Test")

        assert audio.dtype == np.float32
        assert np.allclose(audio[0], 0.0, atol=0.001)
        assert np.allclose(audio[2], 1.0, atol=0.001)  # 32767 / 32768
        assert np.allclose(audio[4], -1.0, atol=0.001)  # -32768 / 32768

    @patch("subprocess.run")
    def test_synthesize_verbose_mode(self, mock_run: Mock, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[type-arg]
        """Test synthesis with verbose logging."""
        config = CommandTTSConfig(command="test-tts", verbose=True)
        tts = CommandTTS(config)

        sample_rate = 24000
        audio_int16 = (np.random.randn(100) * 32767).astype(np.int16)

        def side_effect_run(cmd: list[str], **kwargs: object) -> MagicMock:
            output_idx = cmd.index("--output")
            output_path = cmd[output_idx + 1]
            wav_write(output_path, sample_rate, audio_int16)
            return MagicMock(returncode=0, stdout="TTS output", stderr="")

        mock_run.side_effect = side_effect_run

        tts.synthesize("Test")

        captured = capsys.readouterr()
        assert "Running command:" in captured.out
        assert "test-tts" in captured.out
        assert "Command output:" in captured.out

    def test_save_audio_creates_parent_dirs(self) -> None:
        """Test that save_audio creates parent directories."""
        config = CommandTTSConfig(command="test-tts")
        tts = CommandTTS(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dir" / "output.wav"
            sample_rate = 24000
            audio_data = np.random.randn(24000).astype(np.float32)

            tts.save_audio(audio_data, sample_rate, output_path)

            assert output_path.exists()
            assert output_path.parent.exists()

    def test_save_audio_converts_float_to_int16(self) -> None:
        """Test that save_audio converts float32 to int16."""
        config = CommandTTSConfig(command="test-tts")
        tts = CommandTTS(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.wav"
            sample_rate = 24000
            # Create float32 audio with known values
            audio_data = np.array([0.0, 0.5, 1.0, -0.5, -1.0], dtype=np.float32)

            tts.save_audio(audio_data, sample_rate, output_path)

            # Read back and verify conversion
            from scipy.io.wavfile import (
                read as wav_read,  # type: ignore[import-untyped]
            )

            sr, audio_read = wav_read(str(output_path))
            assert sr == sample_rate
            assert audio_read.dtype == np.int16
            assert len(audio_read) == len(audio_data)

    def test_save_audio_clips_values(self) -> None:
        """Test that save_audio clips values outside [-1, 1]."""
        config = CommandTTSConfig(command="test-tts")
        tts = CommandTTS(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.wav"
            sample_rate = 24000
            # Create audio with values outside valid range
            audio_data = np.array([0.0, 2.0, -3.0, 0.5], dtype=np.float32)

            tts.save_audio(audio_data, sample_rate, output_path)

            # Verify file was created successfully
            assert output_path.exists()
            assert output_path.stat().st_size > 0
