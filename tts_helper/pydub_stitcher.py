"""Audio stitcher implementation using pydub."""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from .stitcher import Stitcher, StitcherConfig

if TYPE_CHECKING:
    from pydub import AudioSegment  # type: ignore[import-untyped]


@dataclass
class PydubStitcherConfig(StitcherConfig):
    """Configuration for pydub-based audio stitcher.

    Args:
        silence_duration_ms: Milliseconds of silence between segments (default: 500ms)
        crossfade_duration_ms: Milliseconds to crossfade between segments (default: 0, disabled)
        output_format: Output audio format (default: "wav")
        export_bitrate: Bitrate for lossy formats like mp3 (default: "192k")
        sample_rate: Sample rate for output (default: None, uses source rate)
        end_indicator: Add musical chapter end indicator (default: False)
    """

    silence_duration_ms: int = 500
    crossfade_duration_ms: int = 0
    output_format: Literal["wav", "mp3", "ogg", "flac"] = "wav"
    export_bitrate: str = "192k"
    sample_rate: int | None = None
    end_indicator: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.silence_duration_ms < 0:
            raise ValueError("silence_duration_ms must be >= 0")

        if self.crossfade_duration_ms < 0:
            raise ValueError("crossfade_duration_ms must be >= 0")

        if self.output_format not in ["wav", "mp3", "ogg", "flac"]:
            raise ValueError(
                f"output_format must be one of: wav, mp3, ogg, flac. Got: {self.output_format}"
            )

        if self.sample_rate is not None and self.sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")


class PydubStitcher(Stitcher):
    """Audio stitcher using pydub for combining segments.

    This stitcher can:
    - Combine multiple audio files or arrays
    - Add silence between segments
    - Apply crossfades between segments
    - Export to various formats (wav, mp3, ogg, flac)
    """

    def __init__(self, config: PydubStitcherConfig):
        """Initialize pydub stitcher.

        Args:
            config: Pydub stitcher configuration

        Raises:
            ImportError: If pydub is not installed
        """
        super().__init__(config)
        self.config: PydubStitcherConfig  # Type hint for IDE support

        # Lazy import to avoid requiring pydub unless actually used
        try:
            from pydub import AudioSegment

            self._AudioSegment = AudioSegment
        except ImportError as err:
            raise ImportError(
                "pydub is not installed. Install it with: pip install pydub\n"
                "Note: pydub also requires ffmpeg. Install with:\n"
                "  macOS: brew install ffmpeg\n"
                "  Ubuntu: sudo apt-get install ffmpeg\n"
                "  Windows: Download from https://ffmpeg.org/"
            ) from err

    def stitch(self, audio_files: list[str | Path], output_path: str | Path) -> None:
        """Stitch audio files together.

        Args:
            audio_files: List of paths to audio files to stitch
            output_path: Path to save stitched audio file

        Raises:
            FileNotFoundError: If any audio file doesn't exist
            ValueError: If audio_files is empty
        """
        if not audio_files:
            raise ValueError("audio_files cannot be empty")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load all audio segments
        segments = []
        for audio_file in audio_files:
            audio_file = Path(audio_file)
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_file}")

            segment = self._AudioSegment.from_file(str(audio_file))
            segments.append(segment)

        # Combine segments
        combined = self._combine_segments(segments)

        # Export
        self._export(combined, output_path)

    def stitch_from_arrays(
        self,
        audio_arrays: list[tuple[int, np.ndarray]],
        output_path: str | Path,
    ) -> None:
        """Stitch audio arrays together.

        Args:
            audio_arrays: List of (sample_rate, audio_data) tuples
            output_path: Path to save stitched audio file

        Raises:
            ValueError: If audio_arrays is empty or sample rates don't match
        """
        if not audio_arrays:
            raise ValueError("audio_arrays cannot be empty")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check that all sample rates match
        sample_rates = [sr for sr, _ in audio_arrays]
        if len(set(sample_rates)) > 1:
            raise ValueError(
                f"All audio arrays must have the same sample rate. Got: {set(sample_rates)}"
            )

        # Convert arrays to AudioSegment objects via temporary WAV files
        segments = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, (sr, audio_data) in enumerate(audio_arrays):
                # Save array to temporary WAV file
                from scipy.io.wavfile import write  # type: ignore[import-untyped]

                temp_path = Path(tmpdir) / f"temp_{i}.wav"

                # Convert float to int16 if needed
                if audio_data.dtype in [np.float32, np.float64]:
                    # Normalize to [-1, 1] if not already
                    audio_data = np.clip(audio_data, -1.0, 1.0)
                    audio_data = (audio_data * 32767).astype(np.int16)

                write(str(temp_path), sr, audio_data)

                # Load as AudioSegment
                segment = self._AudioSegment.from_wav(str(temp_path))
                segments.append(segment)

            # Combine segments
            combined = self._combine_segments(segments)

            # Export
            self._export(combined, output_path)

    def _combine_segments(self, segments: list) -> "AudioSegment":
        """Combine audio segments with silence and/or crossfade.

        Args:
            segments: List of AudioSegment objects

        Returns:
            AudioSegment: Combined audio
        """
        if not segments:
            return self._AudioSegment.silent(duration=0)

        if len(segments) == 1:
            combined = segments[0]
        else:
            # Start with first segment
            combined = segments[0]

            # Add remaining segments with silence or crossfade
            for segment in segments[1:]:
                if self.config.crossfade_duration_ms > 0:
                    # Crossfade mode: overlapping transition
                    combined = combined.append(
                        segment, crossfade=self.config.crossfade_duration_ms
                    )
                elif self.config.silence_duration_ms > 0:
                    # Silence mode: add gap between segments
                    silence = self._AudioSegment.silent(
                        duration=self.config.silence_duration_ms,
                        frame_rate=combined.frame_rate,
                    )
                    combined = combined + silence + segment
                else:
                    # No gap: directly concatenate
                    combined = combined + segment

        # Add chapter end indicator if enabled
        if self.config.end_indicator:
            from .chapter_indicator import ChapterIndicatorGenerator

            # Generate musical indicator at the same sample rate as the audio
            generator = ChapterIndicatorGenerator(sample_rate=combined.frame_rate)
            indicator_audio = generator.generate(verbose=False)

            # Convert numpy array to AudioSegment via temporary file
            with tempfile.TemporaryDirectory() as tmpdir:
                from scipy.io.wavfile import write

                temp_path = Path(tmpdir) / "indicator.wav"

                # Convert float32 to int16
                indicator_int16 = (np.clip(indicator_audio, -1.0, 1.0) * 32767).astype(
                    np.int16
                )
                write(str(temp_path), combined.frame_rate, indicator_int16)

                # Load as AudioSegment and append
                indicator_segment = self._AudioSegment.from_wav(str(temp_path))

                # Add a small silence before the indicator
                silence = self._AudioSegment.silent(
                    duration=300, frame_rate=combined.frame_rate
                )
                combined = combined + silence + indicator_segment

        return combined

    def _export(self, audio: "AudioSegment", output_path: Path) -> None:
        """Export audio to file.

        Args:
            audio: AudioSegment to export
            output_path: Path to save file
        """
        # Apply sample rate conversion if specified
        if self.config.sample_rate is not None:
            audio = audio.set_frame_rate(self.config.sample_rate)

        # Export based on format
        export_params: dict[str, str] = {"format": self.config.output_format}

        if self.config.output_format == "mp3":
            export_params["bitrate"] = self.config.export_bitrate

        audio.export(str(output_path), **export_params)

    def __repr__(self) -> str:
        """String representation of stitcher."""
        parts = [f"format={self.config.output_format}"]

        if self.config.crossfade_duration_ms > 0:
            parts.append(f"crossfade={self.config.crossfade_duration_ms}ms")
        elif self.config.silence_duration_ms > 0:
            parts.append(f"silence={self.config.silence_duration_ms}ms")
        else:
            parts.append("no-gap")

        return f"PydubStitcher({', '.join(parts)})"
