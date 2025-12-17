"""Tests for pySBD integration with SpacySegmenter."""

from tts_helper import SpacySegmenter, SpacySegmenterConfig


class TestPySBDIntegration:
    """Test pySBD sentence boundary detection."""

    def test_pysbd_handles_abbreviations(self) -> None:
        """Test that pySBD correctly handles abbreviations."""
        config = SpacySegmenterConfig(
            language="english",
            strategy="sentence_count",
            sentences_per_chunk=1,
            use_pysbd=True,
        )
        segmenter = SpacySegmenter(config)

        # This text has abbreviations that often confuse simple sentence detectors
        text = "My name is Jonas E. Smith. Please turn to p. 55."

        chunks = segmenter.segment(text)

        # pySBD should correctly identify 2 sentences despite abbreviations
        assert len(chunks) == 2
        assert "Jonas E. Smith" in chunks[0]
        assert "turn to p. 55" in chunks[1]

    def test_pysbd_handles_titles(self) -> None:
        """Test that pySBD correctly handles titles and honorifics."""
        config = SpacySegmenterConfig(
            language="english",
            strategy="sentence_count",
            sentences_per_chunk=1,
            use_pysbd=True,
        )
        segmenter = SpacySegmenter(config)

        text = (
            "Dr. Johnson visited Mr. Smith yesterday. They discussed the new project."
        )

        chunks = segmenter.segment(text)

        # Should correctly identify 2 sentences
        assert len(chunks) == 2
        assert "Dr. Johnson" in chunks[0]
        assert "Mr. Smith" in chunks[0]
        assert "discussed the new project" in chunks[1]

    def test_pysbd_vs_default_sentencizer(self) -> None:
        """Compare pySBD vs default sentencizer on tricky text."""
        text = "The U.S.A. is a country. It has 50 states."

        # With pySBD
        config_pysbd = SpacySegmenterConfig(
            language="english",
            strategy="sentence_count",
            sentences_per_chunk=1,
            use_pysbd=True,
        )
        segmenter_pysbd = SpacySegmenter(config_pysbd)
        chunks_pysbd = segmenter_pysbd.segment(text)

        # With default sentencizer
        config_default = SpacySegmenterConfig(
            language="english",
            strategy="sentence_count",
            sentences_per_chunk=1,
            use_pysbd=False,
        )
        segmenter_default = SpacySegmenter(config_default)
        chunks_default = segmenter_default.segment(text)

        # pySBD should handle U.S.A. better
        # Both should get 2 sentences, but pySBD should keep "U.S.A." intact
        assert len(chunks_pysbd) == 2
        # Default may split differently, but we just ensure it runs
        assert len(chunks_default) >= 1

    def test_pysbd_with_char_count_strategy(self) -> None:
        """Test pySBD works with char_count strategy."""
        config = SpacySegmenterConfig(
            language="english",
            strategy="char_count",
            max_chars=100,
            use_pysbd=True,
        )
        segmenter = SpacySegmenter(config)

        text = "Dr. Smith said hello. Mr. Jones replied. They talked for hours."

        chunks = segmenter.segment(text)

        # Should segment properly respecting sentence boundaries
        assert len(chunks) >= 1
        # All chunks should be within the limit
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_pysbd_config_from_dict(self) -> None:
        """Test that use_pysbd can be set from dict."""
        config = SpacySegmenterConfig.from_dict(
            {
                "language": "english",
                "use_pysbd": True,
                "strategy": "sentence_count",
                "sentences_per_chunk": 1,
            }
        )

        assert config.use_pysbd is True

        segmenter = SpacySegmenter(config)
        text = "Dr. Watson met Mr. Holmes. They solved a case."
        chunks = segmenter.segment(text)

        assert len(chunks) == 2

    def test_pysbd_default_enabled(self) -> None:
        """Test that pySBD is enabled by default."""
        config = SpacySegmenterConfig(language="english")

        assert config.use_pysbd is True

    def test_disable_pysbd(self) -> None:
        """Test that pySBD can be explicitly disabled."""
        config = SpacySegmenterConfig(language="english", use_pysbd=False)

        segmenter = SpacySegmenter(config)
        text = "Hello world. This is a test."
        chunks = segmenter.segment(text)

        # Should still work, just using default sentencizer
        assert len(chunks) >= 1

    def test_pysbd_with_complex_abbreviations(self) -> None:
        """Test pySBD with multiple complex abbreviations."""
        config = SpacySegmenterConfig(
            language="english",
            strategy="sentence_count",
            sentences_per_chunk=1,
            use_pysbd=True,
        )
        segmenter = SpacySegmenter(config)

        # Complex text with multiple abbreviation types
        text = (
            "The meeting is at 3 p.m. in room 42. "
            "Dr. Anderson and Prof. Smith will attend. "
            "Please RSVP to admin@example.com by Jan. 15th."
        )

        chunks = segmenter.segment(text)

        # Should correctly identify 3 sentences
        assert len(chunks) == 3
        assert "3 p.m." in chunks[0]
        assert "Dr. Anderson" in chunks[1]
        assert "Prof. Smith" in chunks[1]
        assert "Jan. 15th" in chunks[2]
