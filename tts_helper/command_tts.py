"""Command-based TTS implementation.

This module provides a TTS engine that delegates synthesis to an external command-line tool.
The command receives text via an input file and produces audio via an output file.
"""

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.io.wavfile import read as wav_read  # type: ignore[import-untyped]
from scipy.io.wavfile import write as wav_write  # type: ignore[import-untyped]

from .tts import TTS, TTSConfig


@dataclass
class CommandTTSConfig(TTSConfig):
    """Configuration for command-based TTS engine.

    Args:
        command: Path to the TTS command to execute
        extra_args: Additional command-line arguments to pass (default: [])
        verbose: Enable verbose logging (default: False)
    """

    command: str = ""
    extra_args: list[str] = field(default_factory=list)
    verbose: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.command:
            raise ValueError("command must be specified")


class CommandTTS(TTS):
    """TTS engine that delegates to an external command.

    This engine:
    - Writes text to a temporary input file
    - Invokes a command with --input <text_file> --output <wav_file>
    - Reads the resulting WAV file
    - Cleans up temporary files automatically

    The command is expected to:
    - Accept --input <path> for the input text file
    - Accept --output <path> for the output WAV file
    - Exit with code 0 on success
    """

    def __init__(self, config: CommandTTSConfig):
        """Initialize command-based TTS engine.

        Args:
            config: Command TTS configuration

        Raises:
            ValueError: If command is not specified
        """
        super().__init__(config)
        self.config: CommandTTSConfig  # Type hint for IDE support

    def synthesize(self, text: str) -> tuple[int, np.ndarray]:
        """Synthesize speech from text using external command.

        Args:
            text: The input text to convert to speech

        Returns:
            Tuple of (sample_rate, audio_data) where:
            - sample_rate: Sample rate in Hz
            - audio_data: NumPy array of audio samples as float32

        Raises:
            RuntimeError: If the command fails or produces invalid output
            ValueError: If text is empty
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Write text to input file
            input_file = tmpdir_path / "input.txt"
            input_file.write_text(text, encoding="utf-8")

            # Define output WAV file
            output_file = tmpdir_path / "output.wav"

            # Build command
            cmd = [
                self.config.command,
                "--input",
                str(input_file),
                "--output",
                str(output_file),
            ]
            cmd.extend(self.config.extra_args)

            if self.config.verbose:
                print(f"Running command: {' '.join(cmd)}")
                print(f"  Input: {len(text)} chars")

            # Execute command
            try:
                if self.config.verbose:
                    # Capture and print output when verbose
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    if result.stdout:
                        print(f"  Command output: {result.stdout}")
                    if result.stderr:
                        print(f"  Command stderr: {result.stderr}")
                else:
                    # Capture output when not verbose (cleaner for batch processing)
                    subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        check=True,
                    )

            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"TTS command failed with exit code {e.returncode}:\n"
                    f"stdout: {e.stdout}\n"
                    f"stderr: {e.stderr}"
                ) from e
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"TTS command not found: {self.config.command}"
                ) from e

            # Verify output file was created
            if not output_file.exists():
                raise RuntimeError(
                    f"TTS command did not create output file: {output_file}"
                )

            # Read WAV file
            try:
                sample_rate, audio_data = wav_read(str(output_file))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read WAV file from {output_file}: {e}"
                ) from e

            # Convert to float32 if needed
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype not in [np.float32, np.float64]:
                audio_data = audio_data.astype(np.float32)

            # Ensure float32
            audio_data = audio_data.astype(np.float32)

            return sample_rate, audio_data

    def save_audio(
        self, audio_data: np.ndarray, sample_rate: int, output_path: str | Path
    ) -> None:
        """Save audio data to a WAV file.

        Args:
            audio_data: NumPy array of audio samples (float32)
            sample_rate: Sample rate in Hz
            output_path: Path where the audio file should be saved

        Raises:
            IOError: If saving fails
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert float32 to int16 for WAV
        if audio_data.dtype in [np.float32, np.float64]:
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * 32767).astype(np.int16)

        try:
            wav_write(str(output_path), sample_rate, audio_data)
        except Exception as e:
            raise OSError(f"Failed to save audio to {output_path}: {e}") from e

    def __repr__(self) -> str:
        """String representation of the TTS engine."""
        return f"CommandTTS(command={self.config.command!r})"
