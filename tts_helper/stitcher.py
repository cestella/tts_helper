"""Base classes for audio stitching."""

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np


@dataclass
class StitcherConfig:
    """Base configuration for audio stitchers.

    This class provides JSON serialization capabilities for all stitcher configs.
    """

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            dict: Configuration as dictionary
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "StitcherConfig":
        """Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            StitcherConfig: Configuration instance
        """
        return cls(**config_dict)

    def to_json(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file.

        Args:
            path: Path to save JSON file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "StitcherConfig":
        """Load configuration from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            StitcherConfig: Configuration instance
        """
        path = Path(path)
        with path.open("r") as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)


class Stitcher(ABC):
    """Base class for audio stitchers.

    Stitchers combine multiple audio segments into a single audio file,
    optionally adding silence or crossfades between segments.
    """

    def __init__(self, config: StitcherConfig):
        """Initialize stitcher with configuration.

        Args:
            config: Stitcher configuration
        """
        self.config = config

    @abstractmethod
    def stitch(
        self, audio_files: List[Union[str, Path]], output_path: Union[str, Path]
    ) -> None:
        """Stitch audio files together.

        Args:
            audio_files: List of paths to audio files to stitch
            output_path: Path to save stitched audio file
        """
        pass

    @abstractmethod
    def stitch_from_arrays(
        self,
        audio_arrays: List[Tuple[int, np.ndarray]],
        output_path: Union[str, Path],
    ) -> None:
        """Stitch audio arrays together.

        Args:
            audio_arrays: List of (sample_rate, audio_data) tuples
            output_path: Path to save stitched audio file
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """String representation of stitcher."""
        pass
