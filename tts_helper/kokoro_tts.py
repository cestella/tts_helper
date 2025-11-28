"""Kokoro TTS implementation using kokoro-onnx."""

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from .tts import TTS, TTSConfig

# Language mappings for Kokoro
LANGUAGE_MAP: Dict[str, str] = {
    "en-us": "en-us",
    "en-gb": "en-gb",
    "fr-fr": "fr-fr",
    "fr": "fr-fr",
    "it": "it",
    "ja": "ja",
    "cmn": "cmn",
    "zh": "cmn",  # Alias for Mandarin Chinese
}

# Voice mappings by language
SUPPORTED_VOICES: Dict[str, List[str]] = {
    "en-us": [
        # Female voices
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        # Male voices
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
        "am_michael", "am_onyx", "am_puck",
    ],
    "en-gb": [
        # Female voices
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        # Male voices
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    ],
    "fr-fr": ["ff_siwis"],
    "it": ["if_sara", "im_nicola"],
    "ja": ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"],
    "cmn": [
        # Female voices
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
        # Male voices
        "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    ],
}

# Default voices for each language
DEFAULT_VOICES: Dict[str, str] = {
    "en-us": "af_sarah",
    "en-gb": "bf_emma",
    "fr-fr": "ff_siwis",
    "it": "if_sara",
    "ja": "jf_alpha",
    "cmn": "zf_xiaoxiao",
}


def get_supported_voices(language: str) -> List[str]:
    """Get supported voices for a language.

    Args:
        language: Language code (e.g., 'en-us', 'fr-fr')

    Returns:
        List of voice names

    Raises:
        ValueError: If language is not supported
    """
    lang = LANGUAGE_MAP.get(language, language)
    if lang not in SUPPORTED_VOICES:
        supported = ", ".join(sorted(LANGUAGE_MAP.keys()))
        raise ValueError(f"Language '{language}' not supported. Supported: {supported}")

    return SUPPORTED_VOICES[lang]


def get_default_voice(language: str) -> str:
    """Get default voice for a language.

    Args:
        language: Language code (e.g., 'en-us', 'fr-fr')

    Returns:
        Default voice name

    Raises:
        ValueError: If language is not supported
    """
    lang = LANGUAGE_MAP.get(language, language)
    if lang not in DEFAULT_VOICES:
        supported = ", ".join(sorted(LANGUAGE_MAP.keys()))
        raise ValueError(f"Language '{language}' not supported. Supported: {supported}")

    return DEFAULT_VOICES[lang]


def download_model_file(url: str, destination: Path, verbose: bool = False) -> None:
    """Download a model file if it doesn't exist.

    Args:
        url: URL to download from
        destination: Path to save the file
        verbose: Whether to print download progress
    """
    if destination.exists():
        return

    if verbose:
        print(f"Downloading {destination.name}...")

    try:
        # Create parent directory if needed
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Download with progress reporting
        def reporthook(block_num, block_size, total_size):
            if verbose and total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                print(f"\r  Progress: {percent:.1f}%", end="", flush=True)

        urllib.request.urlretrieve(url, destination, reporthook if verbose else None)

        if verbose:
            print(f"\n  Downloaded {destination.name} successfully!")

    except Exception as e:
        # Clean up partial download
        if destination.exists():
            destination.unlink()
        raise RuntimeError(f"Failed to download {destination.name}: {e}") from e


@dataclass
class KokoroTTSConfig(TTSConfig):
    """Configuration for Kokoro TTS.

    Args:
        language: Language code (default: "en-us")
        voice: Voice to use (default: auto-selected from language)
        speed: Speech speed multiplier (default: 1.0, range: 0.5-2.0)
        model_path: Path to kokoro ONNX model file (default: None, auto-download)
        voices_path: Path to voices bin file (default: None, auto-download)
        verbose: Whether to print verbose TTS info (default: False)
    """

    language: str = "en-us"
    voice: Optional[str] = None
    speed: float = 1.0
    model_path: Optional[str] = None
    voices_path: Optional[str] = None
    verbose: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Normalize language
        self.language = LANGUAGE_MAP.get(self.language, self.language)

        # Validate language
        if self.language not in SUPPORTED_VOICES:
            supported = ", ".join(sorted(LANGUAGE_MAP.keys()))
            raise ValueError(
                f"Language '{self.language}' not supported. Supported: {supported}"
            )

        # Auto-select voice if not specified
        if self.voice is None:
            self.voice = DEFAULT_VOICES[self.language]

        # Validate voice
        if self.voice not in SUPPORTED_VOICES[self.language]:
            supported = ", ".join(SUPPORTED_VOICES[self.language])
            raise ValueError(
                f"Voice '{self.voice}' not supported for language '{self.language}'. "
                f"Supported voices: {supported}"
            )

        # Validate speed
        if not (0.5 <= self.speed <= 2.0):
            raise ValueError(f"Speed must be between 0.5 and 2.0, got: {self.speed}")


class KokoroTTS(TTS):
    """Kokoro TTS implementation using kokoro-onnx.

    This TTS engine uses the Kokoro model for high-quality speech synthesis
    with support for multiple languages and voices.

    Example:
        >>> config = KokoroTTSConfig(language="en-us", voice="af_sarah")
        >>> tts = KokoroTTS(config)
        >>> sample_rate, audio = tts.synthesize("Hello world!")
        >>> tts.save_audio(audio, sample_rate, "output.wav")
    """

    def __init__(self, config: KokoroTTSConfig):
        """Initialize Kokoro TTS.

        Args:
            config: Kokoro TTS configuration

        Raises:
            ImportError: If kokoro-onnx is not installed
        """
        super().__init__(config)
        self.config: KokoroTTSConfig  # Type hint for IDE support
        self._engine = None

    @property
    def engine(self):
        """Lazy-load Kokoro engine.

        Returns:
            Kokoro: Initialized Kokoro engine

        Raises:
            ImportError: If kokoro-onnx is not installed
        """
        if self._engine is None:
            try:
                from kokoro_onnx import Kokoro
            except ImportError:
                raise ImportError(
                    "kokoro-onnx is not installed. Install it with: pip install kokoro-onnx\n"
                    "Also requires model files. See README for installation."
                )

            # Determine model paths
            model_path = Path(self.config.model_path or "kokoro-v1.0.onnx")
            voices_path = Path(self.config.voices_path or "voices-v1.0.bin")

            # Auto-download model files if they don't exist
            model_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
            voices_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

            try:
                download_model_file(model_url, model_path, verbose=self.config.verbose)
                download_model_file(voices_url, voices_path, verbose=self.config.verbose)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to download Kokoro model files: {e}\n"
                    "You can manually download them:\n"
                    f"  wget {model_url}\n"
                    f"  wget {voices_url}"
                ) from e

            # Initialize Kokoro with model paths
            self._engine = Kokoro(str(model_path), str(voices_path))

            if self.config.verbose:
                print(f"Loaded Kokoro model from: {model_path}")
                print(f"Loaded voices from: {voices_path}")

        return self._engine

    def synthesize(self, text: str) -> Tuple[int, np.ndarray]:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize

        Returns:
            Tuple of (sample_rate, audio_data)
                - sample_rate: Audio sample rate (Hz)
                - audio_data: Audio samples as numpy array (float32)

        Raises:
            ImportError: If kokoro-onnx is not installed
        """
        if not text.strip():
            # Return silence for empty text
            return 24000, np.array([], dtype=np.float32)

        # Synthesize using Kokoro
        samples, sample_rate = self.engine.create(
            text,
            voice=self.config.voice,
            speed=self.config.speed,
            lang=self.config.language,
        )

        # Convert to numpy array if needed
        if not isinstance(samples, np.ndarray):
            samples = np.array(samples, dtype=np.float32)

        return sample_rate, samples

    def save_audio(
        self, audio_data: np.ndarray, sample_rate: int, output_path: Union[str, Path]
    ) -> None:
        """Save audio to WAV file.

        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate in Hz
            output_path: Path to save WAV file
        """
        from scipy.io.wavfile import write

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert float32 to int16 for WAV
        if audio_data.dtype in [np.float32, np.float64]:
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * 32767).astype(np.int16)

        write(str(output_path), sample_rate, audio_data)

    def __repr__(self) -> str:
        """String representation of TTS engine."""
        return (
            f"KokoroTTS(language={self.config.language}, "
            f"voice={self.config.voice}, speed={self.config.speed})"
        )
