"""
Language to spaCy model mapping and metadata.

This module provides a centralized mapping from language codes to spaCy models
and associated metadata. This mapping can be used throughout the pipeline
for consistent language handling.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class LanguageModelMetadata:
    """Metadata for a spaCy language model."""

    model_name: str
    """The spaCy model name (e.g., 'en_core_web_sm')."""

    language_code: str
    """ISO 639-1 language code (e.g., 'en', 'de', 'fr')."""

    language_name: str
    """Full language name (e.g., 'English', 'German')."""

    description: str
    """Description of the model."""

    pip_package: str
    """Package name for pip installation."""


# Centralized language to model mapping
LANGUAGE_MODEL_MAP: Dict[str, LanguageModelMetadata] = {
    "en": LanguageModelMetadata(
        model_name="en_core_web_sm",
        language_code="en",
        language_name="English",
        description="English small model for core pipeline",
        pip_package="en-core-web-sm",
    ),
    "de": LanguageModelMetadata(
        model_name="de_core_news_sm",
        language_code="de",
        language_name="German",
        description="German small model for core pipeline",
        pip_package="de-core-news-sm",
    ),
    "fr": LanguageModelMetadata(
        model_name="fr_core_news_sm",
        language_code="fr",
        language_name="French",
        description="French small model for core pipeline",
        pip_package="fr-core-news-sm",
    ),
    "es": LanguageModelMetadata(
        model_name="es_core_news_sm",
        language_code="es",
        language_name="Spanish",
        description="Spanish small model for core pipeline",
        pip_package="es-core-news-sm",
    ),
    "it": LanguageModelMetadata(
        model_name="it_core_news_sm",
        language_code="it",
        language_name="Italian",
        description="Italian small model for core pipeline",
        pip_package="it-core-news-sm",
    ),
    "pt": LanguageModelMetadata(
        model_name="pt_core_news_sm",
        language_code="pt",
        language_name="Portuguese",
        description="Portuguese small model for core pipeline",
        pip_package="pt-core-news-sm",
    ),
    "nl": LanguageModelMetadata(
        model_name="nl_core_news_sm",
        language_code="nl",
        language_name="Dutch",
        description="Dutch small model for core pipeline",
        pip_package="nl-core-news-sm",
    ),
    "zh": LanguageModelMetadata(
        model_name="zh_core_web_sm",
        language_code="zh",
        language_name="Chinese",
        description="Chinese small model for core pipeline",
        pip_package="zh-core-web-sm",
    ),
    "ja": LanguageModelMetadata(
        model_name="ja_core_news_sm",
        language_code="ja",
        language_name="Japanese",
        description="Japanese small model for core pipeline",
        pip_package="ja-core-news-sm",
    ),
}


def get_model_for_language(language_code: str) -> Optional[LanguageModelMetadata]:
    """
    Get the spaCy model metadata for a given language code.

    Args:
        language_code: ISO 639-1 language code (e.g., 'en', 'de').

    Returns:
        LanguageModelMetadata if the language is supported, None otherwise.
    """
    return LANGUAGE_MODEL_MAP.get(language_code.lower())


def get_supported_languages() -> list[str]:
    """
    Get a list of supported language codes.

    Returns:
        List of ISO 639-1 language codes.
    """
    return list(LANGUAGE_MODEL_MAP.keys())


def is_language_supported(language_code: str) -> bool:
    """
    Check if a language is supported.

    Args:
        language_code: ISO 639-1 language code (e.g., 'en', 'de').

    Returns:
        True if the language is supported, False otherwise.
    """
    return language_code.lower() in LANGUAGE_MODEL_MAP
