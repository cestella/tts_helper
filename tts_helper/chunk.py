"""Chunk data structure for enhanced text processing."""

from dataclasses import dataclass


@dataclass
class Chunk:
    """A text chunk with optional voice, language, and speed overrides.

    Args:
        text: The text content of the chunk
        voice: Optional voice override for TTS (None = use default)
        language: Optional language override for TTS (None = use default)
        speed: Optional speed multiplier for TTS (e.g., 0.8 for 80% speed, None = use default)
        silence_ms: If set, this chunk represents silence of this duration in milliseconds
                    (text field is ignored for silence chunks)
    """

    text: str
    voice: str | None = None
    language: str | None = None
    speed: float | None = None
    silence_ms: int | None = None

    def __str__(self) -> str:
        """String representation returns just the text."""
        return self.text

    def __repr__(self) -> str:
        """Detailed representation including metadata."""
        if self.silence_ms is not None:
            return f"Chunk(silence_ms={self.silence_ms})"

        metadata = []
        if self.voice:
            metadata.append(f"voice={self.voice!r}")
        if self.language:
            metadata.append(f"language={self.language!r}")
        if self.speed is not None:
            metadata.append(f"speed={self.speed}")

        if metadata:
            return f"Chunk(text={self.text[:50]!r}..., {', '.join(metadata)})"
        return f"Chunk(text={self.text[:50]!r}...)"
