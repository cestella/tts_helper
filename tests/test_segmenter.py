"""Tests for base segmenter classes."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from tts_helper.segmenter import Segmenter, SegmenterConfig


@dataclass
class MockSegmenterConfig(SegmenterConfig):
    """Mock configuration for testing base class."""

    param1: str = "default"
    param2: int = 42


class MockSegmenter(Segmenter):
    """Mock segmenter for testing base class."""

    def segment(self, text: str) -> list[str]:
        """Simple split by newlines for testing."""
        return [line.strip() for line in text.split("\n") if line.strip()]

    def __repr__(self) -> str:
        return f"MockSegmenter(param1={self.config.param1})"  # type: ignore[attr-defined]


class TestSegmenterConfig:
    """Tests for SegmenterConfig base class."""

    def test_to_dict(self) -> None:
        """Test converting config to dictionary."""
        config = MockSegmenterConfig(param1="test", param2=100)
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["param1"] == "test"
        assert config_dict["param2"] == 100

    def test_from_dict(self) -> None:
        """Test creating config from dictionary."""
        config_dict = {"param1": "test", "param2": 100}
        config = MockSegmenterConfig.from_dict(config_dict)

        assert isinstance(config, MockSegmenterConfig)
        assert config.param1 == "test"
        assert config.param2 == 100

    def test_to_json(self) -> None:
        """Test saving config to JSON file."""
        config = MockSegmenterConfig(param1="test", param2=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            config.to_json(json_path)

            assert json_path.exists()

            with json_path.open("r") as f:
                loaded = json.load(f)

            assert loaded["param1"] == "test"
            assert loaded["param2"] == 100

    def test_from_json(self) -> None:
        """Test loading config from JSON file."""
        config_dict = {"param1": "test", "param2": 100}

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"

            with json_path.open("w") as f:
                json.dump(config_dict, f)

            config = MockSegmenterConfig.from_json(json_path)

            assert isinstance(config, MockSegmenterConfig)
            assert config.param1 == "test"
            assert config.param2 == 100

    def test_from_json_file_not_found(self) -> None:
        """Test loading from non-existent JSON file."""
        with pytest.raises(FileNotFoundError):
            MockSegmenterConfig.from_json("/nonexistent/path/config.json")

    def test_json_roundtrip(self) -> None:
        """Test that config survives JSON save/load cycle."""
        original = MockSegmenterConfig(param1="roundtrip", param2=999)

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            original.to_json(json_path)
            loaded = MockSegmenterConfig.from_json(json_path)

            assert loaded.param1 == original.param1
            assert loaded.param2 == original.param2

    def test_to_json_creates_parent_directories(self) -> None:
        """Test that to_json creates parent directories if needed."""
        config = MockSegmenterConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "subdir" / "nested" / "config.json"
            config.to_json(json_path)

            assert json_path.exists()
            assert json_path.parent.exists()


class TestSegmenter:
    """Tests for Segmenter base class."""

    def test_init_with_config(self) -> None:
        """Test segmenter initialization with config."""
        config = MockSegmenterConfig(param1="test")
        segmenter = MockSegmenter(config)

        assert segmenter.config == config
        assert segmenter.config.param1 == "test"  # type: ignore[attr-defined]

    def test_segment(self) -> None:
        """Test basic segmentation."""
        config = MockSegmenterConfig()
        segmenter = MockSegmenter(config)

        text = "Line 1\nLine 2\nLine 3"
        chunks = segmenter.segment(text)

        assert chunks == ["Line 1", "Line 2", "Line 3"]

    def test_segment_batch(self) -> None:
        """Test batch segmentation."""
        config = MockSegmenterConfig()
        segmenter = MockSegmenter(config)

        texts = ["Text 1\nLine 2", "Text 2\nLine 4"]
        results = segmenter.segment_batch(texts)

        assert len(results) == 2
        assert results[0] == ["Text 1", "Line 2"]
        assert results[1] == ["Text 2", "Line 4"]

    def test_repr(self) -> None:
        """Test string representation."""
        config = MockSegmenterConfig(param1="test_repr")
        segmenter = MockSegmenter(config)

        repr_str = repr(segmenter)
        assert "MockSegmenter" in repr_str
        assert "test_repr" in repr_str
