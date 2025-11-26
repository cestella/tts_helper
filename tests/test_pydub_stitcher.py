"""Tests for pydub stitcher."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io.wavfile import write

from tts_helper.pydub_stitcher import PydubStitcher, PydubStitcherConfig


class TestPydubStitcherConfig:
    """Tests for PydubStitcherConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PydubStitcherConfig()

        assert config.silence_duration_ms == 500
        assert config.crossfade_duration_ms == 0
        assert config.output_format == "wav"
        assert config.export_bitrate == "192k"
        assert config.sample_rate is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PydubStitcherConfig(
            silence_duration_ms=1000,
            crossfade_duration_ms=100,
            output_format="mp3",
            export_bitrate="256k",
            sample_rate=44100,
        )

        assert config.silence_duration_ms == 1000
        assert config.crossfade_duration_ms == 100
        assert config.output_format == "mp3"
        assert config.export_bitrate == "256k"
        assert config.sample_rate == 44100

    def test_negative_silence_raises_error(self):
        """Test that negative silence duration raises ValueError."""
        with pytest.raises(ValueError, match="silence_duration_ms must be"):
            PydubStitcherConfig(silence_duration_ms=-100)

    def test_negative_crossfade_raises_error(self):
        """Test that negative crossfade duration raises ValueError."""
        with pytest.raises(ValueError, match="crossfade_duration_ms must be"):
            PydubStitcherConfig(crossfade_duration_ms=-50)

    def test_invalid_output_format_raises_error(self):
        """Test that invalid output format raises ValueError."""
        with pytest.raises(ValueError, match="output_format must be"):
            PydubStitcherConfig(output_format="invalid")

    def test_zero_sample_rate_raises_error(self):
        """Test that zero sample rate raises ValueError."""
        with pytest.raises(ValueError, match="sample_rate must be"):
            PydubStitcherConfig(sample_rate=0)

    def test_negative_sample_rate_raises_error(self):
        """Test that negative sample rate raises ValueError."""
        with pytest.raises(ValueError, match="sample_rate must be"):
            PydubStitcherConfig(sample_rate=-22050)

    def test_config_serialization(self):
        """Test config can be serialized to/from JSON."""
        config = PydubStitcherConfig(
            silence_duration_ms=750,
            crossfade_duration_ms=50,
            output_format="mp3",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            config.to_json(json_path)

            loaded = PydubStitcherConfig.from_json(json_path)

            assert loaded.silence_duration_ms == config.silence_duration_ms
            assert loaded.crossfade_duration_ms == config.crossfade_duration_ms
            assert loaded.output_format == config.output_format


class TestPydubStitcher:
    """Tests for PydubStitcher."""

    def test_init(self):
        """Test stitcher initialization."""
        try:
            config = PydubStitcherConfig()
            stitcher = PydubStitcher(config)

            assert stitcher.config == config
        except ImportError:
            pytest.skip("pydub not installed")

    def test_repr(self):
        """Test string representation."""
        try:
            config = PydubStitcherConfig(
                silence_duration_ms=500, output_format="mp3"
            )
            stitcher = PydubStitcher(config)

            repr_str = repr(stitcher)
            assert "PydubStitcher" in repr_str
            assert "format=mp3" in repr_str
            assert "silence=500ms" in repr_str
        except ImportError:
            pytest.skip("pydub not installed")

    def test_repr_with_crossfade(self):
        """Test string representation with crossfade."""
        try:
            config = PydubStitcherConfig(crossfade_duration_ms=100)
            stitcher = PydubStitcher(config)

            repr_str = repr(stitcher)
            assert "crossfade=100ms" in repr_str
        except ImportError:
            pytest.skip("pydub not installed")

    def test_repr_with_no_gap(self):
        """Test string representation with no gap."""
        try:
            config = PydubStitcherConfig(silence_duration_ms=0)
            stitcher = PydubStitcher(config)

            repr_str = repr(stitcher)
            assert "no-gap" in repr_str
        except ImportError:
            pytest.skip("pydub not installed")

    def test_stitch_empty_list_raises_error(self):
        """Test that stitching empty list raises ValueError."""
        try:
            config = PydubStitcherConfig()
            stitcher = PydubStitcher(config)

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "output.wav"

                with pytest.raises(ValueError, match="cannot be empty"):
                    stitcher.stitch([], output_path)
        except ImportError:
            pytest.skip("pydub not installed")

    def test_stitch_nonexistent_file_raises_error(self):
        """Test that stitching nonexistent file raises FileNotFoundError."""
        try:
            config = PydubStitcherConfig()
            stitcher = PydubStitcher(config)

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "output.wav"

                with pytest.raises(FileNotFoundError):
                    stitcher.stitch(["/nonexistent/file.wav"], output_path)
        except ImportError:
            pytest.skip("pydub not installed")

    def test_stitch_from_arrays_empty_list_raises_error(self):
        """Test that stitching from empty arrays raises ValueError."""
        try:
            config = PydubStitcherConfig()
            stitcher = PydubStitcher(config)

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "output.wav"

                with pytest.raises(ValueError, match="cannot be empty"):
                    stitcher.stitch_from_arrays([], output_path)
        except ImportError:
            pytest.skip("pydub not installed")

    def test_stitch_from_arrays_mismatched_sample_rates_raises_error(self):
        """Test that mismatched sample rates raise ValueError."""
        try:
            config = PydubStitcherConfig()
            stitcher = PydubStitcher(config)

            arrays = [
                (22050, np.random.randn(1000).astype(np.float32)),
                (44100, np.random.randn(1000).astype(np.float32)),  # Different rate
            ]

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "output.wav"

                with pytest.raises(ValueError, match="same sample rate"):
                    stitcher.stitch_from_arrays(arrays, output_path)
        except ImportError:
            pytest.skip("pydub not installed")

    def test_stitch_creates_parent_dirs(self):
        """Test that stitch creates parent directories."""
        try:
            config = PydubStitcherConfig()
            stitcher = PydubStitcher(config)

            with tempfile.TemporaryDirectory() as tmpdir:
                # Create a test audio file
                audio_path = Path(tmpdir) / "test.wav"
                sample_rate = 22050
                audio_data = (np.random.randn(1000) * 32767).astype(np.int16)
                write(str(audio_path), sample_rate, audio_data)

                # Output to nested path
                output_path = Path(tmpdir) / "nested" / "dir" / "output.wav"
                stitcher.stitch([audio_path], output_path)

                assert output_path.exists()
                assert output_path.parent.exists()
        except ImportError:
            pytest.skip("pydub not installed")

    def test_init_without_pydub_raises_import_error(self, monkeypatch):
        """Test that initializing without pydub raises helpful error."""
        config = PydubStitcherConfig()

        # Mock the import to fail
        import sys

        def mock_import(name, *args, **kwargs):
            if "pydub" in name:
                raise ImportError("No module named 'pydub'")
            return __import__(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        with pytest.raises(ImportError, match="pydub is not installed"):
            PydubStitcher(config)


class TestPydubStitcherIntegration:
    """Integration tests for PydubStitcher (requires pydub and ffmpeg)."""

    @pytest.fixture
    def test_audio_files(self):
        """Create test audio files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create 3 test audio files
            audio_files = []
            for i in range(3):
                audio_path = tmpdir / f"audio_{i}.wav"
                sample_rate = 24000
                duration = 0.5  # 0.5 seconds
                t = np.linspace(0, duration, int(sample_rate * duration))
                # Different frequency for each file
                audio_data = np.sin(2 * np.pi * (440 + i * 100) * t)
                audio_data = (audio_data * 32767).astype(np.int16)
                write(str(audio_path), sample_rate, audio_data)
                audio_files.append(audio_path)

            yield tmpdir, audio_files

    def test_stitch_with_silence(self, test_audio_files):
        """Test stitching files with silence between them."""
        try:
            tmpdir, audio_files = test_audio_files

            config = PydubStitcherConfig(silence_duration_ms=500)
            stitcher = PydubStitcher(config)

            output_path = tmpdir / "output_silence.wav"
            stitcher.stitch(audio_files, output_path)

            assert output_path.exists()
            assert output_path.stat().st_size > 0
        except ImportError:
            pytest.skip("pydub not installed")

    def test_stitch_with_crossfade(self, test_audio_files):
        """Test stitching files with crossfade."""
        try:
            tmpdir, audio_files = test_audio_files

            config = PydubStitcherConfig(crossfade_duration_ms=100)
            stitcher = PydubStitcher(config)

            output_path = tmpdir / "output_crossfade.wav"
            stitcher.stitch(audio_files, output_path)

            assert output_path.exists()
            assert output_path.stat().st_size > 0
        except ImportError:
            pytest.skip("pydub not installed")

    def test_stitch_no_gap(self, test_audio_files):
        """Test stitching files with no gap."""
        try:
            tmpdir, audio_files = test_audio_files

            config = PydubStitcherConfig(silence_duration_ms=0)
            stitcher = PydubStitcher(config)

            output_path = tmpdir / "output_nogap.wav"
            stitcher.stitch(audio_files, output_path)

            assert output_path.exists()
            assert output_path.stat().st_size > 0
        except ImportError:
            pytest.skip("pydub not installed")

    def test_stitch_from_arrays(self, test_audio_files):
        """Test stitching from audio arrays."""
        try:
            tmpdir, _ = test_audio_files

            config = PydubStitcherConfig(silence_duration_ms=300)
            stitcher = PydubStitcher(config)

            # Create test arrays
            sample_rate = 24000
            arrays = []
            for i in range(3):
                duration = 0.5
                t = np.linspace(0, duration, int(sample_rate * duration))
                audio_data = np.sin(2 * np.pi * (440 + i * 100) * t).astype(
                    np.float32
                )
                arrays.append((sample_rate, audio_data))

            output_path = tmpdir / "output_arrays.wav"
            stitcher.stitch_from_arrays(arrays, output_path)

            assert output_path.exists()
            assert output_path.stat().st_size > 0
        except ImportError:
            pytest.skip("pydub not installed")

    def test_stitch_to_mp3(self, test_audio_files):
        """Test stitching to MP3 format."""
        try:
            tmpdir, audio_files = test_audio_files

            config = PydubStitcherConfig(
                output_format="mp3", export_bitrate="128k", silence_duration_ms=200
            )
            stitcher = PydubStitcher(config)

            output_path = tmpdir / "output.mp3"
            stitcher.stitch(audio_files, output_path)

            assert output_path.exists()
            assert output_path.stat().st_size > 0
            assert output_path.suffix == ".mp3"
        except (ImportError, RuntimeError):
            pytest.skip("pydub or ffmpeg not installed")

    def test_stitch_single_file(self, test_audio_files):
        """Test stitching a single file (edge case)."""
        try:
            tmpdir, audio_files = test_audio_files

            config = PydubStitcherConfig()
            stitcher = PydubStitcher(config)

            output_path = tmpdir / "output_single.wav"
            stitcher.stitch([audio_files[0]], output_path)

            assert output_path.exists()
            assert output_path.stat().st_size > 0
        except ImportError:
            pytest.skip("pydub not installed")
