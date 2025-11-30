"""Tests for base stitcher classes."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pytest

from tts_helper.stitcher import Stitcher, StitcherConfig


@dataclass
class MockStitcherConfig(StitcherConfig):
    """Mock configuration for testing base class."""

    join_method: str = "concat"
    add_marker: bool = False


class MockStitcher(Stitcher):
    """Mock stitcher for testing base class."""

    def stitch(
        self, audio_files: List[Union[str, Path]], output_path: Union[str, Path]
    ) -> None:
        """Mock stitch implementation."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Just create a marker file
        with output_path.open("w") as f:
            f.write(f"Stitched {len(audio_files)} files\n")
            f.write(f"Method: {self.config.join_method}\n")

    def stitch_from_arrays(
        self,
        audio_arrays: List[Tuple[int, np.ndarray]],
        output_path: Union[str, Path],
    ) -> None:
        """Mock stitch from arrays implementation."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Just create a marker file
        with output_path.open("w") as f:
            f.write(f"Stitched {len(audio_arrays)} arrays\n")
            f.write(f"Method: {self.config.join_method}\n")

    def __repr__(self) -> str:
        return f"MockStitcher(method={self.config.join_method})"


class TestStitcherConfig:
    """Tests for StitcherConfig base class."""

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = MockStitcherConfig(join_method="blend", add_marker=True)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["join_method"] == "blend"
        assert config_dict["add_marker"] is True

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {"join_method": "blend", "add_marker": True}
        config = MockStitcherConfig.from_dict(config_dict)

        assert isinstance(config, MockStitcherConfig)
        assert config.join_method == "blend"
        assert config.add_marker is True

    def test_to_json(self):
        """Test saving config to JSON file."""
        config = MockStitcherConfig(join_method="blend")

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            config.to_json(json_path)

            assert json_path.exists()

            with json_path.open("r") as f:
                loaded = json.load(f)

            assert loaded["join_method"] == "blend"

    def test_from_json(self):
        """Test loading config from JSON file."""
        config_dict = {"join_method": "blend", "add_marker": False}

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"

            with json_path.open("w") as f:
                json.dump(config_dict, f)

            config = MockStitcherConfig.from_json(json_path)

            assert isinstance(config, MockStitcherConfig)
            assert config.join_method == "blend"

    def test_from_json_file_not_found(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            MockStitcherConfig.from_json("/nonexistent/path.json")

    def test_json_roundtrip(self):
        """Test that config survives JSON save/load cycle."""
        original = MockStitcherConfig(join_method="crossfade", add_marker=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            original.to_json(json_path)
            loaded = MockStitcherConfig.from_json(json_path)

            assert loaded.join_method == original.join_method
            assert loaded.add_marker == original.add_marker

    def test_to_json_creates_parent_directories(self):
        """Test that to_json creates parent directories if needed."""
        config = MockStitcherConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "nested" / "dir" / "config.json"
            config.to_json(json_path)

            assert json_path.exists()
            assert json_path.parent.exists()


class TestStitcher:
    """Tests for Stitcher base class."""

    def test_init_with_config(self):
        """Test stitcher initialization with config."""
        config = MockStitcherConfig(join_method="blend")
        stitcher = MockStitcher(config)

        assert stitcher.config == config
        assert stitcher.config.join_method == "blend"

    def test_stitch(self):
        """Test basic stitching."""
        config = MockStitcherConfig(join_method="concat")
        stitcher = MockStitcher(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.txt"
            stitcher.stitch(["file1.wav", "file2.wav"], output_path)

            assert output_path.exists()
            content = output_path.read_text()
            assert "Stitched 2 files" in content
            assert "Method: concat" in content

    def test_stitch_from_arrays(self):
        """Test stitching from arrays."""
        config = MockStitcherConfig(join_method="concat")
        stitcher = MockStitcher(config)

        arrays = [
            (22050, np.random.randn(1000).astype(np.float32)),
            (22050, np.random.randn(1000).astype(np.float32)),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.txt"
            stitcher.stitch_from_arrays(arrays, output_path)

            assert output_path.exists()
            content = output_path.read_text()
            assert "Stitched 2 arrays" in content

    def test_stitch_creates_parent_dirs(self):
        """Test that stitch creates parent directories."""
        config = MockStitcherConfig()
        stitcher = MockStitcher(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dir" / "output.txt"
            stitcher.stitch(["file1.wav"], output_path)

            assert output_path.exists()
            assert output_path.parent.exists()

    def test_repr(self):
        """Test string representation."""
        config = MockStitcherConfig(join_method="crossfade")
        stitcher = MockStitcher(config)

        repr_str = repr(stitcher)
        assert "MockStitcher" in repr_str
        assert "crossfade" in repr_str
