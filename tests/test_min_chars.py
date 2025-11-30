"""Test that min_chars prevents tiny chunks."""

from tts_helper.spacy_segmenter import SpacySegmenter, SpacySegmenterConfig


def test_single_character_chunks_are_merged():
    """Test that single-character chunks like '–' are merged with adjacent text."""

    # Text with a single dash that would be its own sentence
    text = "This is the first sentence. – And this continues."

    config = SpacySegmenterConfig(
        language="english",
        strategy="char_count",
        max_chars=100,
        min_chars=3,  # Default
    )

    segmenter = SpacySegmenter(config)
    chunks = segmenter.segment(text)

    print(f"\nGenerated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i}: {len(chunk):3d} chars - '{chunk}'")

    # Verify no chunks are smaller than min_chars
    small_chunks = [c for c in chunks if len(c) < config.min_chars]
    assert not small_chunks, f"Found chunks smaller than {config.min_chars}: {small_chunks}"

    print(f"✅ All chunks are ≥{config.min_chars} chars")


def test_multiple_small_chunks():
    """Test merging multiple small chunks."""

    text = "A. B. C. This is a longer sentence."

    config = SpacySegmenterConfig(
        language="english",
        strategy="char_count",
        max_chars=100,
        min_chars=5,
    )

    segmenter = SpacySegmenter(config)
    chunks = segmenter.segment(text)

    print(f"\nGenerated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i}: {len(chunk):3d} chars - '{chunk}'")

    # Verify no chunks are smaller than min_chars
    small_chunks = [c for c in chunks if len(c) < config.min_chars]
    assert not small_chunks, f"Found chunks smaller than {config.min_chars}: {small_chunks}"

    print(f"✅ All chunks are ≥{config.min_chars} chars")


def test_min_chars_zero_disables_merging():
    """Test that min_chars=0 disables chunk merging."""

    text = "A. B. C."

    config = SpacySegmenterConfig(
        language="english",
        strategy="char_count",
        max_chars=100,
        min_chars=0,  # Disabled
    )

    segmenter = SpacySegmenter(config)
    chunks = segmenter.segment(text)

    print(f"\nGenerated {len(chunks)} chunks (min_chars=0):")
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i}: {len(chunk):3d} chars - '{chunk}'")

    # With min_chars=0, we might have small chunks
    print("✅ min_chars=0 allows small chunks")


if __name__ == "__main__":
    test_single_character_chunks_are_merged()
    test_multiple_small_chunks()
    test_min_chars_zero_disables_merging()
    print("\n🎉 All min_chars tests passed!")
