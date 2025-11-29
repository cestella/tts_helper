"""
Orpheus TTS implementation using orpheus-cpp and llama-cpp.

This module provides a TTS implementation using NVIDIA's Orpheus models
for high-quality multilingual speech synthesis.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy.io.wavfile import write

from .language import get_orpheus_code
from .tts import TTS, TTSConfig

# Model paths for each language
MODEL_PATHS: Dict[str, Optional[str]] = {
    "en": None,  # Uses default model
    "fr": "canopylabs/3b-fr-ft-research_release",
    "es": "canopylabs/3b-es_it-ft-research_release",
    "it": "canopylabs/3b-es_it-ft-research_release",
}

# Supported voices per language
SUPPORTED_VOICES: Dict[str, List[str]] = {
    "en": ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"],
    "fr": ["pierre", "amelie", "marie"],
    "es": ["javi", "sergio", "maria"],
    "it": ["pietro", "giulia", "carlo"],
}

# Default voices per language
DEFAULT_VOICES: Dict[str, str] = {
    "en": "tara",
    "fr": "marie",
    "es": "maria",
    "it": "giulia",
}


@dataclass
class OrpheusTTSConfig(TTSConfig):
    """
    Configuration for Orpheus TTS.

    Attributes:
        language: Language for synthesis. Options: 'english', 'french', 'spanish', 'italian'.
        voice: Voice ID to use. If None, uses language default.
        use_gpu: Whether to use GPU acceleration (Metal on Mac, CUDA on Linux/Windows).
        n_gpu_layers: Number of layers to offload to GPU. -1 for all, 0 for CPU only.
        verbose: Whether to print verbose output during synthesis.
        model_path: Optional explicit model path. If None, uses language default.
    """

    language: Literal["english", "french", "spanish", "italian"] = "english"
    voice: Optional[str] = None
    use_gpu: bool = True
    n_gpu_layers: int = -1
    verbose: bool = False
    model_path: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        try:
            # Get Orpheus language code (validates the language)
            lang_code = get_orpheus_code(self.language)
        except ValueError:
            raise ValueError(
                f"Language '{self.language}' not supported by Orpheus. "
                f"Orpheus supports: english, french, spanish, italian"
            )

        # Validate or set voice
        if self.voice is None:
            self.voice = DEFAULT_VOICES[lang_code]
        elif self.voice not in SUPPORTED_VOICES[lang_code]:
            raise ValueError(
                f"Voice '{self.voice}' not supported for {self.language}. "
                f"Supported voices: {SUPPORTED_VOICES[lang_code]}"
            )

        # Set model path if not explicitly provided
        if self.model_path is None:
            self.model_path = MODEL_PATHS[lang_code]

        # Validate GPU settings
        if not self.use_gpu:
            self.n_gpu_layers = 0


class OrpheusTTS(TTS):
    """
    Text-to-speech engine using Orpheus models.

    This TTS engine uses NVIDIA's Orpheus models via orpheus-cpp for
    high-quality multilingual speech synthesis with GPU acceleration.

    Supported languages:
    - English: 8 voices (tara, leah, jess, leo, dan, mia, zac, zoe)
    - French: 3 voices (pierre, amelie, marie)
    - Spanish: 3 voices (javi, sergio, maria)
    - Italian: 3 voices (pietro, giulia, carlo)

    Examples:
        >>> config = OrpheusTTSConfig(language="english", voice="tara")
        >>> tts = OrpheusTTS(config)
        >>> sample_rate, audio = tts.synthesize("Hello world")
        >>> tts.save_audio(audio, sample_rate, "output.wav")
    """

    def __init__(self, config: OrpheusTTSConfig):
        """
        Initialize the Orpheus TTS engine.

        Args:
            config: Configuration for the TTS engine.

        Raises:
            ImportError: If orpheus_cpp is not installed.
            RuntimeError: If model initialization fails.
        """
        super().__init__(config)
        self.config: OrpheusTTSConfig = config
        self._engine: Optional[object] = None

    @property
    def engine(self) -> object:
        """
        Lazy-load the Orpheus engine.

        Returns:
            The loaded OrpheusCpp instance.

        Raises:
            ImportError: If orpheus_cpp is not installed.
            RuntimeError: If engine initialization fails.
        """
        if self._engine is None:
            try:
                from orpheus_cpp import OrpheusCpp
            except ImportError as e:
                raise ImportError(
                    "orpheus-cpp is not installed. "
                    "Install it with: pip install orpheus-cpp\n"
                    "Also requires llama-cpp-python. See README for installation."
                ) from e

            try:
                lang_code = get_orpheus_code(self.config.language)

                kwargs = {
                    "verbose": self.config.verbose,
                    "lang": lang_code,
                    "n_gpu_layers": self.config.n_gpu_layers,
                }

                # Add model path if not using default
                if self.config.model_path:
                    kwargs["model_path"] = self.config.model_path

                self._engine = OrpheusCpp(**kwargs)

            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize Orpheus TTS engine: {e}"
                ) from e

        return self._engine

    def synthesize(self, text: str) -> Tuple[int, np.ndarray]:
        """
        Synthesize speech from text.

        Args:
            text: The input text to convert to speech.

        Returns:
            Tuple of (sample_rate, audio_data) where:
            - sample_rate: Sample rate in Hz (24000)
            - audio_data: NumPy array of audio samples (mono, int16)

        Raises:
            RuntimeError: If synthesis fails.
        """
        if not text or not text.strip():
            # Return silence for empty text
            return 24000, np.array([], dtype=np.int16)

        try:
            buffer: List[np.ndarray] = []

            # Use synchronous streaming for simplicity
            for sr, chunk in self.engine.stream_tts_sync(
                text, options={"voice_id": self.config.voice}
            ):
                buffer.append(chunk)

            if not buffer:
                raise RuntimeError("No audio generated")

            # Concatenate all chunks
            audio_data = np.concatenate(buffer, axis=1)

            # Return as 1D array (squeeze to remove channel dimension)
            return sr, audio_data.squeeze().astype(np.int16)

        except Exception as e:
            raise RuntimeError(f"Speech synthesis failed: {e}") from e

    def save_audio(
        self, audio_data: np.ndarray, sample_rate: int, output_path: str | Path
    ) -> None:
        """
        Save audio data to a WAV file.

        Args:
            audio_data: NumPy array of audio samples.
            sample_rate: Sample rate in Hz.
            output_path: Path where the audio file should be saved.

        Raises:
            IOError: If saving fails.
        """
        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Ensure audio is 1D
            if audio_data.ndim > 1:
                audio_data = audio_data.squeeze()

            write(str(path), sample_rate, audio_data)

        except Exception as e:
            raise IOError(f"Failed to save audio to {output_path}: {e}") from e

    def __repr__(self) -> str:
        """String representation of the TTS engine."""
        gpu_status = "GPU" if self.config.use_gpu else "CPU"
        return (
            f"OrpheusTTS(language='{self.config.language}', "
            f"voice='{self.config.voice}', "
            f"mode='{gpu_status}')"
        )


def get_supported_voices(language: str) -> List[str]:
    """
    Get list of supported voices for a language.

    Args:
        language: Unified language name (e.g., 'english', 'french', 'spanish', 'italian').

    Returns:
        List of voice IDs supported for the language.

    Raises:
        ValueError: If language is not supported.
    """
    try:
        lang_code = get_orpheus_code(language)
    except ValueError:
        raise ValueError(
            f"Language '{language}' not supported by Orpheus. "
            f"Orpheus supports: english, french, spanish, italian"
        )

    return SUPPORTED_VOICES[lang_code]


def get_default_voice(language: str) -> str:
    """
    Get the default voice for a language.

    Args:
        language: Unified language name (e.g., 'english', 'french', 'spanish', 'italian').

    Returns:
        Default voice ID for the language.

    Raises:
        ValueError: If language is not supported.
    """
    try:
        lang_code = get_orpheus_code(language)
    except ValueError:
        raise ValueError(
            f"Language '{language}' not supported by Orpheus. "
            f"Orpheus supports: english, french, spanish, italian"
        )

    return DEFAULT_VOICES[lang_code]
