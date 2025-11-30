"""Tests for language model mapping."""

from tts_helper.language_models import (
    LANGUAGE_MODEL_MAP,
    LanguageModelMetadata,
    get_model_for_language,
    get_supported_languages,
    is_language_supported,
)


class TestLanguageModelMetadata:
    """Tests for LanguageModelMetadata dataclass."""

    def test_metadata_structure(self) -> None:
        """Test that metadata has expected fields."""
        metadata = LANGUAGE_MODEL_MAP["en"]
        assert isinstance(metadata, LanguageModelMetadata)
        assert metadata.model_name == "en_core_web_sm"
        assert metadata.language_code == "en"
        assert metadata.language_name == "English"
        assert metadata.pip_package == "en-core-web-sm"
        assert len(metadata.description) > 0


class TestLanguageModelMap:
    """Tests for language model mapping functions."""

    def test_get_model_for_supported_language(self) -> None:
        """Test retrieving model for supported language."""
        model = get_model_for_language("english")
        assert model is not None
        assert model.model_name == "en_core_web_sm"
        assert model.language_code == "en"

    def test_get_model_for_unsupported_language(self) -> None:
        """Test retrieving model for unsupported language."""
        model = get_model_for_language("xyz")
        assert model is None

    def test_get_model_case_insensitive(self) -> None:
        """Test that language names are case-insensitive."""
        model_lower = get_model_for_language("english")
        model_upper = get_model_for_language("ENGLISH")
        assert model_lower is not None
        assert model_upper is not None
        assert model_lower.model_name == model_upper.model_name

    def test_get_supported_languages(self) -> None:
        """Test getting list of supported languages."""
        languages = get_supported_languages()
        assert isinstance(languages, list)
        assert len(languages) > 0
        assert "en" in languages
        assert "de" in languages
        assert "fr" in languages

    def test_is_language_supported_true(self) -> None:
        """Test checking if language is supported (positive case)."""
        assert is_language_supported("english") is True
        assert is_language_supported("german") is True
        assert is_language_supported("ENGLISH") is True  # Case insensitive

    def test_is_language_supported_false(self) -> None:
        """Test checking if language is supported (negative case)."""
        assert is_language_supported("xyz") is False
        assert is_language_supported("") is False

    def test_all_mapped_languages_have_required_fields(self) -> None:
        """Test that all language mappings have complete metadata."""
        for lang_code, metadata in LANGUAGE_MODEL_MAP.items():
            assert isinstance(lang_code, str)
            assert len(lang_code) > 0
            assert isinstance(metadata, LanguageModelMetadata)
            assert len(metadata.model_name) > 0
            assert len(metadata.language_code) > 0
            assert len(metadata.language_name) > 0
            assert len(metadata.pip_package) > 0
            assert metadata.language_code == lang_code
