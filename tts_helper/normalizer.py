"""
Base classes for text normalization.

Text normalization converts written text to spoken form, which is essential
for TTS processing. For example: "$123" → "one hundred twenty three dollars"
"""

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T", bound="NormalizerConfig")


@dataclass
class NormalizerConfig:
    """
    Base configuration class for text normalizers.

    This class provides serialization/deserialization capabilities for configurations
    and can be extended by specific normalizer implementations.
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


class Normalizer(ABC):
    """
    Abstract base class for text normalizers.

    A normalizer converts written text to spoken form for TTS processing.
    Examples:
    - "$123.45" → "one hundred twenty three dollars forty five cents"
    - "Dr. Smith" → "Doctor Smith"
    - "3:30pm" → "three thirty p m"
    """

    def __init__(self, config: NormalizerConfig):
        """
        Initialize the normalizer with a configuration.

        Args:
            config: Configuration object for this normalizer.
        """
        self.config = config

    @abstractmethod
    def normalize(self, text: str) -> str:
        """
        Normalize text to spoken form.

        Args:
            text: The raw input text to normalize.

        Returns:
            Normalized text suitable for TTS processing.
        """
        pass

    def normalize_batch(self, texts: list[str]) -> list[str]:
        """
        Normalize multiple texts.

        Args:
            texts: List of raw input texts to normalize.

        Returns:
            List of normalized texts.
        """
        return [self.normalize(text) for text in texts]

    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the normalizer."""
        pass
