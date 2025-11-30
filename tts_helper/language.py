"""Unified language representation and code mapping.

This module provides a central mapping from user-friendly language names
to the various codes required by different components (normalizer, TTS, translation).
"""


class LanguageCode:
    """Language codes for a specific language in various formats."""

    def __init__(
        self,
        name: str,
        iso_639_1: str,
        flores_200: str,
        nemo: str | None = None,
        kokoro: str | None = None,
    ):
        """Initialize language codes.

        Args:
            name: Human-readable name (e.g., "english", "italian")
            iso_639_1: ISO 639-1 two-letter code (e.g., "en", "it")
            flores_200: FLORES-200 code for NLLB translation (e.g., "eng_Latn", "ita_Latn")
            nemo: NeMo normalizer language code (defaults to iso_639_1)
            kokoro: Kokoro TTS language code (defaults to iso_639_1 with region like "en-us")
        """
        self.name = name.lower()
        self.iso_639_1 = iso_639_1
        self.flores_200 = flores_200
        self.nemo = nemo or iso_639_1
        self.kokoro = kokoro


# Language database
_LANGUAGES: dict[str, LanguageCode] = {
    "english": LanguageCode(
        name="english",
        iso_639_1="en",
        flores_200="eng_Latn",
        nemo="en",
        kokoro="en-us",
    ),
    "italian": LanguageCode(
        name="italian",
        iso_639_1="it",
        flores_200="ita_Latn",
        nemo="it",
        kokoro="it",
    ),
    "spanish": LanguageCode(
        name="spanish",
        iso_639_1="es",
        flores_200="spa_Latn",
        nemo="es",
        kokoro="es",
    ),
    "french": LanguageCode(
        name="french",
        iso_639_1="fr",
        flores_200="fra_Latn",
        nemo="fr",
        kokoro="fr-fr",
    ),
    "german": LanguageCode(
        name="german",
        iso_639_1="de",
        flores_200="deu_Latn",
        nemo="de",
        kokoro="de",
    ),
    "portuguese": LanguageCode(
        name="portuguese",
        iso_639_1="pt",
        flores_200="por_Latn",
        nemo="pt",
        kokoro="pt",
    ),
    "russian": LanguageCode(
        name="russian",
        iso_639_1="ru",
        flores_200="rus_Cyrl",
        nemo="ru",
        kokoro="ru",
    ),
    "chinese": LanguageCode(
        name="chinese",
        iso_639_1="zh",
        flores_200="zho_Hans",
        nemo="zh",
        kokoro="cmn",
    ),
    "japanese": LanguageCode(
        name="japanese",
        iso_639_1="ja",
        flores_200="jpn_Jpan",
        nemo="ja",
        kokoro="ja",
    ),
}


def get_language(name: str) -> LanguageCode:
    """Get language codes for a language name.

    Args:
        name: Language name (case-insensitive, e.g., "english", "Italian")

    Returns:
        LanguageCode object with all code formats

    Raises:
        ValueError: If language is not supported
    """
    name_lower = name.lower()
    if name_lower not in _LANGUAGES:
        supported = ", ".join(sorted(_LANGUAGES.keys()))
        raise ValueError(
            f"Unsupported language: '{name}'. " f"Supported languages: {supported}"
        )
    return _LANGUAGES[name_lower]


def get_nemo_code(language: str) -> str:
    """Get NeMo normalizer language code.

    Args:
        language: Language name (e.g., "english")

    Returns:
        NeMo language code (e.g., "en")
    """
    return get_language(language).nemo


def get_flores_code(language: str) -> str:
    """Get FLORES-200 language code for NLLB translation.

    Args:
        language: Language name (e.g., "english")

    Returns:
        FLORES-200 code (e.g., "eng_Latn")
    """
    return get_language(language).flores_200


def get_iso_code(language: str) -> str:
    """Get ISO 639-1 language code.

    Args:
        language: Language name (e.g., "english")

    Returns:
        ISO 639-1 code (e.g., "en")
    """
    return get_language(language).iso_639_1


def get_kokoro_code(language: str) -> str:
    """Get Kokoro TTS language code.

    Args:
        language: Language name (e.g., "english")

    Returns:
        Kokoro language code (e.g., "en-us")

    Raises:
        ValueError: If language doesn't have a Kokoro code
    """
    lang_code = get_language(language)
    if lang_code.kokoro is None:
        raise ValueError(f"Language '{language}' does not support Kokoro TTS")
    return lang_code.kokoro


def list_supported_languages() -> list[str]:
    """Get list of all supported language names.

    Returns:
        Sorted list of language names
    """
    return sorted(_LANGUAGES.keys())
