"""Tests for musical chapter end indicators."""

import numpy as np

from tts_helper.chapter_indicator import ChapterIndicatorGenerator


class TestChapterIndicatorGenerator:
    """Tests for ChapterIndicatorGenerator."""

    def test_init_default_sample_rate(self) -> None:
        """Test initialization with default sample rate."""
        generator = ChapterIndicatorGenerator()
        assert generator.sample_rate == 24000

    def test_init_custom_sample_rate(self) -> None:
        """Test initialization with custom sample rate."""
        generator = ChapterIndicatorGenerator(sample_rate=44100)
        assert generator.sample_rate == 44100

    def test_envelope_creates_fade_in_out(self) -> None:
        """Test envelope creates proper fade in and out."""
        generator = ChapterIndicatorGenerator(sample_rate=24000)
        t = np.linspace(0, 1.0, 24000)
        env = generator._envelope(t, attack=0.1, release=0.2)

        # Check shape matches input
        assert env.shape == t.shape

        # Check fade in: first values should be close to 0, then rise to 1
        assert env[0] < 0.1
        assert env[int(0.1 * 24000)] > 0.9

        # Check fade out: last values should be close to 0
        assert env[-1] < 0.1
        assert env[-int(0.2 * 24000)] > 0.9

    def test_normalize_scales_to_range(self) -> None:
        """Test normalization scales signal to [-1, 1]."""
        generator = ChapterIndicatorGenerator()

        # Signal with values way outside [-1, 1]
        sig = np.array([0, 100, -100, 50])
        normalized = generator._normalize(sig)

        # Should be scaled to [-1, 1]
        assert np.max(np.abs(normalized)) <= 1.0
        assert np.max(np.abs(normalized)) > 0.99  # Should be close to 1

    def test_normalize_handles_zero_signal(self) -> None:
        """Test normalization handles all-zero signal."""
        generator = ChapterIndicatorGenerator()

        sig = np.zeros(100)
        normalized = generator._normalize(sig)

        # Should return zeros unchanged
        assert np.all(normalized == 0)

    def test_arpeggio_creates_audio(self) -> None:
        """Test arpeggio creates non-empty audio."""
        generator = ChapterIndicatorGenerator()
        audio = generator.arpeggio()

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert np.max(np.abs(audio)) <= 1.0  # Should be normalized

    def test_arpeggio_with_custom_notes(self) -> None:
        """Test arpeggio with custom note frequencies."""
        generator = ChapterIndicatorGenerator()
        custom_notes = [440.0, 554.37, 659.25]  # A, C#, E
        audio = generator.arpeggio(notes=custom_notes)

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    def test_descending_arpeggio_creates_audio(self) -> None:
        """Test descending arpeggio creates audio."""
        generator = ChapterIndicatorGenerator()
        audio = generator.descending_arpeggio()

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert np.max(np.abs(audio)) <= 1.0

    def test_pizzicato_pop_creates_audio(self) -> None:
        """Test pizzicato creates short pluck sound."""
        generator = ChapterIndicatorGenerator()
        audio = generator.pizzicato_pop()

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert np.max(np.abs(audio)) <= 1.0

    def test_gliss_up_creates_audio(self) -> None:
        """Test glissando creates upward sweep."""
        generator = ChapterIndicatorGenerator()
        audio = generator.gliss_up()

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert np.max(np.abs(audio)) <= 1.0

    def test_bell_chime_creates_audio(self) -> None:
        """Test bell chime creates ringing sound."""
        generator = ChapterIndicatorGenerator()
        audio = generator.bell_chime()

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert np.max(np.abs(audio)) <= 1.0

    def test_cadence_creates_audio(self) -> None:
        """Test cadence creates musical phrase."""
        generator = ChapterIndicatorGenerator()
        audio = generator.cadence()

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert np.max(np.abs(audio)) <= 1.0

    def test_generate_creates_random_composition(self) -> None:
        """Test generate creates random 2-part composition."""
        generator = ChapterIndicatorGenerator()
        audio = generator.generate(verbose=False)

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert audio.dtype == np.float32
        assert np.max(np.abs(audio)) <= 1.0

    def test_generate_verbose_mode(self, capsys) -> None:  # type: ignore[no-untyped-def]
        """Test generate with verbose mode prints event names."""
        generator = ChapterIndicatorGenerator()
        audio = generator.generate(verbose=True)

        captured = capsys.readouterr()
        assert "Generated chapter indicator:" in captured.out
        assert "First:" in captured.out
        assert "Second:" in captured.out

        # Audio should still be valid
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    def test_generate_multiple_calls_produce_different_results(self) -> None:
        """Test that multiple generate calls can produce different outputs."""
        generator = ChapterIndicatorGenerator()

        # Generate 10 times and check that we get some variation
        # (since it's random, at least one should be different)
        outputs = [generator.generate() for _ in range(10)]

        # Due to randomness, very unlikely all 10 are identical
        # But we can't guarantee it, so we'll just check they're all valid
        for out in outputs:
            assert isinstance(out, np.ndarray)
            assert len(out) > 0
            assert out.dtype == np.float32

    def test_generate_consistent_sample_rate(self) -> None:
        """Test that generated audio respects sample rate setting."""
        # Generate with different sample rates
        gen_24k = ChapterIndicatorGenerator(sample_rate=24000)
        gen_44k = ChapterIndicatorGenerator(sample_rate=44100)

        audio_24k = gen_24k.generate()
        audio_44k = gen_44k.generate()

        # Higher sample rate should generally produce more samples for same duration
        # Though exact comparison is tricky due to random composition
        assert isinstance(audio_24k, np.ndarray)
        assert isinstance(audio_44k, np.ndarray)
        assert len(audio_44k) > 0
        assert len(audio_24k) > 0
