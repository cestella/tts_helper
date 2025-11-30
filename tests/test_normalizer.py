"""Tests for base normalizer classes."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest

from tts_helper.normalizer import Normalizer, NormalizerConfig


@dataclass
class MockNormalizerConfig(NormalizerConfig):
    """Mock configuration for testing base class."""

    uppercase: bool = False
    prefix: str = ""


class MockNormalizer(Normalizer):
    """Mock normalizer for testing base class."""

    def normalize(self, text: str) -> str:
        """Simple normalization for testing."""
        result = text
        if self.config.uppercase:
            result = result.upper()
        if self.config.prefix:
            result = f"{self.config.prefix}{result}"
        return result

    def __repr__(self) -> str:
        return f"MockNormalizer(uppercase={self.config.uppercase})"


class TestNormalizerConfig:
    """Tests for NormalizerConfig base class."""

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = MockNormalizerConfig(uppercase=True, prefix="TEST:")
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["uppercase"] is True
        assert config_dict["prefix"] == "TEST:"

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {"uppercase": True, "prefix": "TEST:"}
        config = MockNormalizerConfig.from_dict(config_dict)

        assert isinstance(config, MockNormalizerConfig)
        assert config.uppercase is True
        assert config.prefix == "TEST:"

    def test_to_json(self):
        """Test saving config to JSON file."""
        config = MockNormalizerConfig(uppercase=True, prefix="TEST:")

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            config.to_json(json_path)

            assert json_path.exists()

            with json_path.open("r") as f:
                loaded = json.load(f)

            assert loaded["uppercase"] is True
            assert loaded["prefix"] == "TEST:"

    def test_from_json(self):
        """Test loading config from JSON file."""
        config_dict = {"uppercase": True, "prefix": "TEST:"}

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"

            with json_path.open("w") as f:
                json.dump(config_dict, f)

            config = MockNormalizerConfig.from_json(json_path)

            assert isinstance(config, MockNormalizerConfig)
            assert config.uppercase is True
            assert config.prefix == "TEST:"

    def test_from_json_file_not_found(self):
        """Test loading from non-existent JSON file."""
        with pytest.raises(FileNotFoundError):
            MockNormalizerConfig.from_json("/nonexistent/path/config.json")

    def test_json_roundtrip(self):
        """Test that config survives JSON save/load cycle."""
        original = MockNormalizerConfig(uppercase=False, prefix="HELLO:")

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "config.json"
            original.to_json(json_path)
            loaded = MockNormalizerConfig.from_json(json_path)

            assert loaded.uppercase == original.uppercase
            assert loaded.prefix == original.prefix

    def test_to_json_creates_parent_directories(self):
        """Test that to_json creates parent directories if needed."""
        config = MockNormalizerConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "subdir" / "nested" / "config.json"
            config.to_json(json_path)

            assert json_path.exists()
            assert json_path.parent.exists()


class TestNormalizer:
    """Tests for Normalizer base class."""

    def test_init_with_config(self):
        """Test normalizer initialization with config."""
        config = MockNormalizerConfig(uppercase=True)
        normalizer = MockNormalizer(config)

        assert normalizer.config == config
        assert normalizer.config.uppercase is True

    def test_normalize(self):
        """Test basic normalization."""
        config = MockNormalizerConfig(uppercase=True)
        normalizer = MockNormalizer(config)

        text = "hello world"
        normalized = normalizer.normalize(text)

        assert normalized == "HELLO WORLD"

    def test_normalize_with_prefix(self):
        """Test normalization with prefix."""
        config = MockNormalizerConfig(prefix=">>>")
        normalizer = MockNormalizer(config)

        text = "test"
        normalized = normalizer.normalize(text)

        assert normalized == ">>>test"

    def test_normalize_batch(self):
        """Test batch normalization."""
        config = MockNormalizerConfig(uppercase=True)
        normalizer = MockNormalizer(config)

        texts = ["hello", "world"]
        results = normalizer.normalize_batch(texts)

        assert len(results) == 2
        assert results[0] == "HELLO"
        assert results[1] == "WORLD"

    def test_repr(self):
        """Test string representation."""
        config = MockNormalizerConfig(uppercase=True)
        normalizer = MockNormalizer(config)

        repr_str = repr(normalizer)
        assert "MockNormalizer" in repr_str
        assert "True" in repr_str
