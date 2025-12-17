"""
TTS Helper - Industrial-grade audiobook text-to-speech processor pipeline.

This package provides tools for normalizing, segmenting, and synthesizing text to speech,
with a focus on audiobook production.
"""

from .chunk import Chunk
from .command_tts import CommandTTS, CommandTTSConfig
from .enhancer import Enhancer, EnhancerConfig
from .kokoro_tts import (
    KokoroTTS,
    KokoroTTSConfig,
)
from .kokoro_tts import get_default_voice as get_kokoro_default_voice
from .kokoro_tts import get_supported_voices as get_kokoro_voices
from .nemo_normalizer import NemoNormalizer, NemoNormalizerConfig
from .normalizer import Normalizer, NormalizerConfig
from .per_file_process import PerFileProcessConfig, PerFileProcessManager
from .pydub_stitcher import PydubStitcher, PydubStitcherConfig
from .segmenter import Segmenter, SegmenterConfig
from .spacy_segmenter import SpacySegmenter, SpacySegmenterConfig
from .stitcher import Stitcher, StitcherConfig
from .translation_enhancer import TranslationEnhancer, TranslationEnhancerConfig
from .tts import TTS, TTSConfig

__version__ = "0.1.0"

__all__ = [
    # Core data structures
    "Chunk",
    # Normalization
    "Normalizer",
    "NormalizerConfig",
    "NemoNormalizer",
    "NemoNormalizerConfig",
    # Segmentation
    "Segmenter",
    "SegmenterConfig",
    "SpacySegmenter",
    "SpacySegmenterConfig",
    # Text-to-Speech
    "TTS",
    "TTSConfig",
    # Kokoro TTS
    "KokoroTTS",
    "KokoroTTSConfig",
    "get_kokoro_voices",
    "get_kokoro_default_voice",
    # Command TTS
    "CommandTTS",
    "CommandTTSConfig",
    # Audio stitching
    "Stitcher",
    "StitcherConfig",
    "PydubStitcher",
    "PydubStitcherConfig",
    # Enhancers
    "Enhancer",
    "EnhancerConfig",
    "TranslationEnhancer",
    "TranslationEnhancerConfig",
    # Process Management
    "PerFileProcessManager",
    "PerFileProcessConfig",
]
