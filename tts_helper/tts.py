"""
Base classes for text-to-speech synthesis.

This module defines the abstract base classes for TTS engines and their configurations.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T", bound="TTSConfig")


@dataclass
class TTSConfig:
    """
    Base configuration class for TTS engines.

    This class provides serialization/deserialization capabilities for configurations
    and can be extended by specific TTS engine implementations.
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


class TTS(ABC):
    """
    Abstract base class for text-to-speech engines.

    A TTS engine converts text to audio, supporting various languages and voices.
    """

    def __init__(self, config: TTSConfig):
        """
        Initialize the TTS engine with a configuration.

        Args:
            config: Configuration object for this TTS engine.
        """
        self.config = config

    @abstractmethod
    def synthesize(self, text: str) -> tuple[int, np.ndarray]:
        """
        Synthesize speech from text.

        Args:
            text: The input text to convert to speech.

        Returns:
            Tuple of (sample_rate, audio_data) where:
            - sample_rate: Sample rate in Hz (e.g., 24000)
            - audio_data: NumPy array of audio samples
        """
        pass

    def synthesize_batch(self, texts: list[str]) -> list[tuple[int, np.ndarray]]:
        """
        Synthesize speech from multiple texts.

        Args:
            texts: List of input texts to convert to speech.

        Returns:
            List of (sample_rate, audio_data) tuples.
        """
        return [self.synthesize(text) for text in texts]

    @abstractmethod
    def save_audio(
        self, audio_data: np.ndarray, sample_rate: int, output_path: str | Path
    ) -> None:
        """
        Save audio data to a file.

        Args:
            audio_data: NumPy array of audio samples.
            sample_rate: Sample rate in Hz.
            output_path: Path where the audio file should be saved.

        Raises:
            IOError: If saving fails.
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the TTS engine."""
        pass
