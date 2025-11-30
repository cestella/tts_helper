"""
Test that the segmenter GUARANTEES max_chars limit.

This test validates that the segmenter never produces chunks exceeding
max_chars, even with pathological inputs like long vocabulary lists.
"""

from tts_helper.spacy_segmenter import SpacySegmenter, SpacySegmenterConfig


def test_segmenter_guarantee_with_vocabulary_list() -> None:
    """Test that segmenter handles long vocabulary lists without exceeding max_chars."""

    # This is the type of text that was causing 637-char chunks
    problematic_text = """
arancione orange (colour)
inaspettato unexpected
l'insalata salad
il ginocchio knee
la gola throat
lo stomaco stomach
il sangue blood
il dito finger
il dito del piede toe
la schiena back
la faccia face
la testa head
il cuore heart
la mano hand
il collo neck
la gamba leg
il piede foot
il braccio arm
l'orecchio ear
il naso nose
la bocca mouth
il dente tooth
l'occhio eye
"""

    config = SpacySegmenterConfig(
        language="italian",
        strategy="char_count",
        max_chars=150,
    )

    segmenter = SpacySegmenter(config)
    chunks = segmenter.segment(problematic_text)

    print(f"\nGenerated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i}: {len(chunk)} chars - {chunk[:50]}...")

    # GUARANTEE: All chunks must be ≤ max_chars
    oversized = [c for c in chunks if len(c) > config.max_chars]

    if oversized:
        print("\n❌ FAILED: Found oversized chunks!")
        for chunk in oversized:
            print(f"  {len(chunk)} chars: {chunk[:100]}...")
        raise AssertionError(
            f"Found {len(oversized)} chunks exceeding {config.max_chars} chars"
        )
    else:
        print(f"\n✅ SUCCESS: All {len(chunks)} chunks are ≤{config.max_chars} chars")


def test_segmenter_guarantee_with_extreme_text() -> None:
    """Test with various extreme cases."""

    test_cases = [
        # Very long word (should be hard-cut)
        "supercalifragilisticexpialidocious" * 20,
        # No spaces (should be hard-cut)
        "a" * 500,
        # Comma-separated list
        "apple, banana, cherry, date, elderberry, fig, grape, honeydew, " * 10,
        # Normal sentences that happen to be long
        "This is a perfectly normal sentence that just happens to be quite long because it contains many words and clauses that make it exceed the character limit. "
        * 3,
    ]

    config = SpacySegmenterConfig(
        language="english",
        strategy="char_count",
        max_chars=150,
    )

    segmenter = SpacySegmenter(config)

    for i, text in enumerate(test_cases, 1):
        print(f"\n--- Test case {i} ---")
        print(f"Input length: {len(text)} chars")

        chunks = segmenter.segment(text)
        print(f"Generated {len(chunks)} chunks")

        # Validate
        oversized = [c for c in chunks if len(c) > config.max_chars]
        assert not oversized, f"Test case {i}: Found {len(oversized)} oversized chunks"

        max_len = max(len(c) for c in chunks) if chunks else 0
        print(f"✅ All chunks ≤{config.max_chars} (max: {max_len})")


if __name__ == "__main__":
    test_segmenter_guarantee_with_vocabulary_list()
    test_segmenter_guarantee_with_extreme_text()
    print("\n🎉 All guarantee tests passed!")
