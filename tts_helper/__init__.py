"""
TTS Helper - Industrial-grade audiobook text-to-speech processor pipeline.

This package provides tools for normalizing, segmenting, and synthesizing text to speech,
with a focus on audiobook production.
"""

from .normalizer import Normalizer, NormalizerConfig
from .nemo_normalizer import NemoNormalizer, NemoNormalizerConfig
from .segmenter import Segmenter, SegmenterConfig
from .spacy_segmenter import SpacySegmenter, SpacySegmenterConfig
from .tts import TTS, TTSConfig
from .orpheus_tts import OrpheusTTS, OrpheusTTSConfig, get_supported_voices, get_default_voice
from .stitcher import Stitcher, StitcherConfig
from .pydub_stitcher import PydubStitcher, PydubStitcherConfig

__version__ = "0.1.0"

__all__ = [
    "Normalizer",
    "NormalizerConfig",
    "NemoNormalizer",
    "NemoNormalizerConfig",
    "Segmenter",
    "SegmenterConfig",
    "SpacySegmenter",
    "SpacySegmenterConfig",
    "TTS",
    "TTSConfig",
    "OrpheusTTS",
    "OrpheusTTSConfig",
    "get_supported_voices",
    "get_default_voice",
    "Stitcher",
    "StitcherConfig",
    "PydubStitcher",
    "PydubStitcherConfig",
]
