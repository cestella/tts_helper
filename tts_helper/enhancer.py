"""Base classes for text chunk enhancers.

Enhancers sit between text segmentation and TTS, allowing for mutation or
creation of new chunks (e.g., translation, summarization, etc.).
"""

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .chunk import Chunk


@dataclass
class EnhancerConfig:
    """Base configuration for text enhancers."""

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "EnhancerConfig":
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration

        Returns:
            Configuration instance
        """
        return cls(**config_dict)

    def to_json(self, path: Path) -> None:
        """Save configuration to JSON file.

        Args:
            path: Path to save JSON file
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "EnhancerConfig":
        """Load configuration from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Configuration instance

        Raises:
            FileNotFoundError: If path doesn't exist
        """
        with path.open("r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class Enhancer(ABC):
    """Base class for text chunk enhancers.

    Enhancers can mutate existing chunks or create new chunks that are
    inserted into the processing pipeline before TTS synthesis.
    """

    def __init__(self, config: EnhancerConfig):
        """Initialize enhancer with configuration.

        Args:
            config: Enhancer configuration
        """
        self.config = config

    @abstractmethod
    def enhance(self, chunks: list[Chunk]) -> list[Chunk]:
        """Enhance text chunks.

        This method can:
        - Mutate existing chunks
        - Create new chunks with voice/language overrides
        - Return the same chunks unchanged

        Args:
            chunks: List of Chunk objects from segmentation

        Returns:
            Enhanced list of Chunk objects (may be different length)
        """
        pass

    def __repr__(self) -> str:
        """String representation of enhancer."""
        return f"{self.__class__.__name__}(config={self.config})"
