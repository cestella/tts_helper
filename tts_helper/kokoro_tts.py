"""Kokoro TTS implementation using kokoro-onnx."""

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .language import get_kokoro_code
from .tts import TTS, TTSConfig

if TYPE_CHECKING:
    from kokoro_onnx import Kokoro

# Voice mappings by language
SUPPORTED_VOICES: dict[str, list[str]] = {
    "en-us": [
        # Female voices
        "af_alloy",
        "af_aoede",
        "af_bella",
        "af_heart",
        "af_jessica",
        "af_kore",
        "af_nicole",
        "af_nova",
        "af_river",
        "af_sarah",
        "af_sky",
        # Male voices
        "am_adam",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_michael",
        "am_onyx",
        "am_puck",
    ],
    "en-gb": [
        # Female voices
        "bf_alice",
        "bf_emma",
        "bf_isabella",
        "bf_lily",
        # Male voices
        "bm_daniel",
        "bm_fable",
        "bm_george",
        "bm_lewis",
    ],
    "fr-fr": ["ff_siwis"],
    "it": ["if_sara", "im_nicola"],
    "ja": ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"],
    "cmn": [
        # Female voices
        "zf_xiaobei",
        "zf_xiaoni",
        "zf_xiaoxiao",
        "zf_xiaoyi",
        # Male voices
        "zm_yunjian",
        "zm_yunxi",
        "zm_yunxia",
        "zm_yunyang",
    ],
}

# Default voices for each language
DEFAULT_VOICES: dict[str, str] = {
    "en-us": "af_sarah",
    "en-gb": "bf_emma",
    "fr-fr": "ff_siwis",
    "it": "if_sara",
    "ja": "jf_alpha",
    "cmn": "zf_xiaoxiao",
}


def get_supported_voices(language: str) -> list[str]:
    """Get supported voices for a language.

    Args:
        language: Unified language name (e.g., 'english', 'french', 'italian')

    Returns:
        List of voice names

    Raises:
        ValueError: If language is not supported
    """
    lang_code = get_kokoro_code(language)
    if lang_code not in SUPPORTED_VOICES:
        raise ValueError(
            f"Language '{language}' (Kokoro code: '{lang_code}') not supported. "
            f"Kokoro supports: english, french, italian, japanese, chinese"
        )

    return SUPPORTED_VOICES[lang_code]


def get_default_voice(language: str) -> str:
    """Get default voice for a language.

    Args:
        language: Unified language name (e.g., 'english', 'french', 'italian')

    Returns:
        Default voice name

    Raises:
        ValueError: If language is not supported
    """
    lang_code = get_kokoro_code(language)
    if lang_code not in DEFAULT_VOICES:
        raise ValueError(
            f"Language '{language}' (Kokoro code: '{lang_code}') not supported. "
            f"Kokoro supports: english, french, italian, japanese, chinese"
        )

    return DEFAULT_VOICES[lang_code]


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
        def reporthook(block_num: int, block_size: int, total_size: int) -> None:
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
        language: Unified language name (default: "english")
        voice: Voice to use (default: auto-selected from language)
        speed: Speech speed multiplier (default: 1.0, range: 0.5-2.0)
        model_path: Path to kokoro ONNX model file (default: None, auto-download)
        voices_path: Path to voices bin file (default: None, auto-download)
        verbose: Whether to print verbose TTS info (default: False)
    """

    language: str = "english"
    voice: str | None = None
    speed: float = 1.0
    model_path: str | None = None
    voices_path: str | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Get Kokoro language code
        lang_code = get_kokoro_code(self.language)

        # Validate language
        if lang_code not in SUPPORTED_VOICES:
            raise ValueError(
                f"Language '{self.language}' (Kokoro code: '{lang_code}') not supported. "
                f"Kokoro supports: english, french, italian, japanese, chinese"
            )

        # Auto-select voice if not specified
        if self.voice is None:
            self.voice = DEFAULT_VOICES[lang_code]

        # Validate voice
        if self.voice not in SUPPORTED_VOICES[lang_code]:
            supported = ", ".join(SUPPORTED_VOICES[lang_code])
            raise ValueError(
                f"Voice '{self.voice}' not supported for language '{self.language}'. "
                f"Supported voices: {supported}"
            )

        # Validate speed
        if not (0.5 <= self.speed <= 2.0):
            raise ValueError(f"Speed must be between 0.5 and 2.0, got: {self.speed}")

    @property
    def kokoro_language_code(self) -> str:
        """Get Kokoro language code."""
        return get_kokoro_code(self.language)


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
        self._engine: "Kokoro | None" = None  # noqa: UP037

    @property
    def engine(self) -> "Kokoro":
        """Lazy-load Kokoro engine.

        Returns:
            Kokoro: Initialized Kokoro engine

        Raises:
            ImportError: If kokoro-onnx is not installed
        """
        if self._engine is None:
            try:
                from kokoro_onnx import Kokoro
            except ImportError as err:
                raise ImportError(
                    "kokoro-onnx is not installed. Install it with: pip install kokoro-onnx\n"
                    "Also requires model files. See README for installation."
                ) from err

            # Determine model paths
            model_path = Path(self.config.model_path or "kokoro-v1.0.onnx")
            voices_path = Path(self.config.voices_path or "voices-v1.0.bin")

            # Auto-download model files if they don't exist
            model_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
            voices_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

            try:
                download_model_file(model_url, model_path, verbose=self.config.verbose)
                download_model_file(
                    voices_url, voices_path, verbose=self.config.verbose
                )
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

    def synthesize(self, text: str) -> tuple[int, np.ndarray]:
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
        try:
            samples, sample_rate = self.engine.create(
                text,
                voice=self.config.voice,  # type: ignore[arg-type]
                speed=self.config.speed,
                lang=self.config.kokoro_language_code,
            )
        except ValueError as e:
            # Handle empty audio (e.g., from non-speech text like separators)
            if "need at least one array to concatenate" in str(e):
                if self.config.verbose:
                    print(
                        f"Warning: Kokoro generated no audio for text (likely non-speech content). "
                        f"Returning silence. Text preview: {text[:100]}..."
                    )
                # Return silence instead of crashing
                return 24000, np.array([], dtype=np.float32)
            # Re-raise other ValueErrors
            raise
        except Exception as e:
            # Provide detailed error information
            import sys
            import traceback

            error_msg = (
                f"\n{'='*80}\n"
                f"ERROR: Kokoro TTS synthesis failed\n"
                f"{'='*80}\n"
                f"Text length: {len(text)} characters\n"
                f"Voice: {self.config.voice}\n"
                f"Language: {self.config.kokoro_language_code}\n"
                f"Speed: {self.config.speed}\n"
                f"\nProblematic text:\n"
                f"{'-'*80}\n"
                f"{text}\n"
                f"{'-'*80}\n"
                f"\nError type: {type(e).__name__}\n"
                f"Error message: {str(e)}\n"
                f"\nFull traceback:\n"
            )
            print(error_msg, file=sys.stderr)
            traceback.print_exc()
            print(f"{'='*80}\n", file=sys.stderr)

            # Re-raise with additional context
            raise RuntimeError(
                f"Kokoro TTS failed on {len(text)}-char text. "
                f"Text preview: {text[:100]}... "
                f"Original error: {str(e)}"
            ) from e

        # Convert to numpy array if needed
        if not isinstance(samples, np.ndarray):
            samples = np.array(samples, dtype=np.float32)

        return sample_rate, samples

    def save_audio(
        self, audio_data: np.ndarray, sample_rate: int, output_path: str | Path
    ) -> None:
        """Save audio to WAV file.

        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate in Hz
            output_path: Path to save WAV file
        """
        from scipy.io.wavfile import write  # type: ignore[import-untyped]

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
