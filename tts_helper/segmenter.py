"""
Base classes for text segmentation.

This module defines the abstract base classes for text segmenters and their configurations.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T", bound="SegmenterConfig")


@dataclass
class SegmenterConfig:
    """
    Base configuration class for text segmenters.

    This class provides serialization/deserialization capabilities for configurations
    and can be extended by specific segmenter implementations.
    """

    @classmethod
    def from_dict(cls: type[T], config_dict: dict[str, Any]) -> T:
        """
        Create a configuration instance from a dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters.

        Returns:
            Configuration instance.
        """
        return cls(**config_dict)

    @classmethod
    def from_json(cls: type[T], json_path: str | Path) -> T:
        """
        Load configuration from a JSON file.

        Args:
            json_path: Path to the JSON configuration file.

        Returns:
            Configuration instance.

        Raises:
            FileNotFoundError: If the JSON file doesn't exist.
            json.JSONDecodeError: If the JSON file is invalid.
        """
        path = Path(json_path)
        with path.open("r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return asdict(self)

    def to_json(self, json_path: str | Path, indent: int = 2) -> None:
        """
        Save configuration to a JSON file.

        Args:
            json_path: Path where the JSON file should be saved.
            indent: Number of spaces for JSON indentation (default: 2).
        """
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)


class Segmenter(ABC):
    """
    Abstract base class for text segmenters.

    A segmenter takes raw text and breaks it into clean, logical chunks
    suitable for text-to-speech processing.
    """

    def __init__(self, config: SegmenterConfig):
        """
        Initialize the segmenter with a configuration.

        Args:
            config: Configuration object for this segmenter.
        """
        self.config = config

    @abstractmethod
    def segment(self, text: str) -> list[str]:
        """
        Segment text into chunks.

        Args:
            text: The raw input text to segment.

        Returns:
            List of text chunks, each suitable for TTS processing.
        """
        pass

    def segment_batch(self, texts: list[str]) -> list[list[str]]:
        """
        Segment multiple texts into chunks.

        Args:
            texts: List of raw input texts to segment.

        Returns:
            List of chunk lists, one for each input text.
        """
        return [self.segment(text) for text in texts]

    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the segmenter."""
        pass
