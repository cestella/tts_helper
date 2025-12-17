"""
spaCy-based text segmentation for TTS processing.

This module provides a segmenter implementation using spaCy for
sentence boundary detection and intelligent chunking.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import spacy
from spacy.language import Language

from .language import get_iso_code
from .language_models import get_model_for_language, is_language_supported
from .segmenter import Segmenter, SegmenterConfig


@dataclass
class SpacySegmenterConfig(SegmenterConfig):
    """
    Configuration for spaCy-based text segmentation.

    Attributes:
        language: Unified language name (e.g., 'english', 'german', 'french').
        strategy: Chunking strategy. Options:
                 - 'sentence_count': Group by number of sentences
                 - 'char_count': Group by character count (preserving sentence boundaries)
                 - 'token_count': Group by BPE token count (for IndexTTS2)
        sentences_per_chunk: Number of sentences per chunk (for 'sentence_count' strategy).
                           Default: 3.
        max_chars: Maximum number of characters per chunk (for 'char_count' strategy).
                  Default: 300.
        min_chars: Minimum number of characters per chunk. Chunks smaller than this
                  will be merged with adjacent chunks. Default: 3.
        token_target: Target number of BPE tokens per chunk (for 'token_count' strategy).
                     If set, this takes priority over sentence_count and char_count.
                     Default: None (use sentence/char-based segmentation).
                     Recommended: 120 for IndexTTS2.
        bpe_model_path: Path to SentencePiece BPE model file. If not provided and
                       token_target is set, will auto-download from HuggingFace.
                       Default: None (auto-download to ~/.cache/tts_helper/bpe.model).
        model_name: Optional explicit spaCy model name. If not provided,
                   will be determined from the language code.
        disable_pipes: List of spaCy pipeline components to disable for performance.
                      Default: ['ner', 'lemmatizer'].
        use_pysbd: Whether to use pySBD for sentence boundary detection instead of
                  spaCy's statistical model. pySBD is rule-based and better handles
                  abbreviations and edge cases. Default: True.
    """

    language: str = "english"
    strategy: Literal["sentence_count", "char_count", "token_count"] = "char_count"
    sentences_per_chunk: int = 3
    max_chars: int = 300
    min_chars: int = 3
    token_target: int | None = None
    bpe_model_path: str | None = None
    model_name: str | None = None
    disable_pipes: list[str] = field(default_factory=lambda: ["ner", "lemmatizer"])
    use_pysbd: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not is_language_supported(self.language):
            try:
                # Try to get ISO code to show in error message
                iso_code = get_iso_code(self.language)
                error_msg = (
                    f"Language '{self.language}' (ISO code: '{iso_code}') is not supported for spaCy segmentation. "
                    f"Supported languages: english, german, french, spanish, italian, portuguese, dutch, chinese, japanese. "
                    f"You can also specify model_name explicitly."
                )
            except ValueError:
                error_msg = (
                    f"Language '{self.language}' is not recognized. "
                    f"Supported languages: english, german, french, spanish, italian, portuguese, dutch, chinese, japanese. "
                    f"You can also specify model_name explicitly."
                )
            raise ValueError(error_msg)

        if self.strategy not in ["sentence_count", "char_count", "token_count"]:
            raise ValueError(
                f"strategy must be 'sentence_count', 'char_count', or 'token_count', got '{self.strategy}'"
            )

        # Auto-set strategy based on token_target if provided
        if self.token_target is not None:
            self.strategy = "token_count"

        if self.strategy == "token_count" and self.token_target is None:
            raise ValueError(
                "token_target must be set when using 'token_count' strategy"
            )

        if self.token_target is not None and self.token_target <= 0:
            raise ValueError(f"token_target must be positive, got {self.token_target}")

        if self.sentences_per_chunk <= 0:
            raise ValueError(
                f"sentences_per_chunk must be positive, got {self.sentences_per_chunk}"
            )

        if self.max_chars <= 0:
            raise ValueError(f"max_chars must be positive, got {self.max_chars}")

        if self.min_chars < 0:
            raise ValueError(f"min_chars must be non-negative, got {self.min_chars}")

        if self.min_chars > self.max_chars:
            raise ValueError(
                f"min_chars ({self.min_chars}) cannot be greater than max_chars ({self.max_chars})"
            )


class SpacySegmenter(Segmenter):
    """
    Text segmenter using spaCy for sentence boundary detection.

    This segmenter uses spaCy's sentence segmentation to intelligently
    break text into chunks suitable for TTS processing, respecting
    natural sentence boundaries.
    """

    def __init__(self, config: SpacySegmenterConfig):
        """
        Initialize the spaCy segmenter.

        Args:
            config: Configuration for the segmenter.

        Raises:
            OSError: If the spaCy model cannot be loaded.
        """
        super().__init__(config)
        self.config: SpacySegmenterConfig = config
        self._nlp: Language | None = None
        self._tokenizer: Any | None = None  # SentencePieceProcessor

    @property
    def nlp(self) -> Language:
        """
        Lazy-load the spaCy model.

        Returns:
            The loaded spaCy Language model.

        Raises:
            OSError: If the model cannot be loaded.
            ImportError: If pySBD is requested but not installed.
        """
        if self._nlp is None:
            model_name = self.config.model_name
            if model_name is None:
                model_metadata = get_model_for_language(self.config.language)
                if model_metadata is None:
                    raise ValueError(
                        f"No model found for language '{self.config.language}'"
                    )
                model_name = model_metadata.model_name

            try:
                self._nlp = spacy.load(model_name, disable=self.config.disable_pipes)
            except OSError as e:
                raise OSError(
                    f"Failed to load spaCy model '{model_name}'. "
                    f"Make sure it's installed: python -m spacy download {model_name}"
                ) from e

            # Add pySBD for better sentence boundary detection if enabled
            if self.config.use_pysbd:
                try:
                    import pysbd
                    from spacy.language import Language
                except ImportError as e:
                    raise ImportError(
                        "pySBD is required for use_pysbd=True. "
                        "Install it with: pip install pysbd"
                    ) from e

                # Register the component only once
                component_name = "pysbd_sentencizer"
                if not Language.has_factory(component_name):

                    @Language.component(component_name)
                    def pysbd_sentencizer(doc):
                        """Custom sentence boundary detection using pySBD."""
                        # Use pySBD to segment the text
                        seg = pysbd.Segmenter(language="en", clean=False)
                        sentences = seg.segment(doc.text)

                        # Set sentence boundaries based on pySBD output
                        char_index = 0
                        sent_starts = []
                        for sent in sentences:
                            # Find where this sentence starts in the original text
                            start = doc.text.find(sent, char_index)
                            if start != -1:
                                sent_starts.append(start)
                                char_index = start + len(sent)

                        # Mark sentence boundaries
                        for token in doc:
                            if token.idx in sent_starts:
                                token.is_sent_start = True
                            else:
                                token.is_sent_start = False

                        return doc

                # Add the custom component to the pipeline
                if component_name not in self._nlp.pipe_names:
                    self._nlp.add_pipe(component_name, first=True)

        return self._nlp

    def _download_bpe_model(self) -> Path:
        """
        Download the BPE model from HuggingFace if not already present.

        Returns:
            Path to the downloaded BPE model file.

        Raises:
            RuntimeError: If download fails.
        """
        # Default cache location
        cache_dir = Path.home() / ".cache" / "tts_helper"
        cache_dir.mkdir(parents=True, exist_ok=True)
        bpe_model_path = cache_dir / "bpe.model"

        # If already downloaded, return it
        if bpe_model_path.exists():
            return bpe_model_path

        # Download from HuggingFace
        hf_url = "https://huggingface.co/IndexTeam/IndexTTS-2/resolve/main/bpe.model"

        try:
            import urllib.request

            print(f"Downloading BPE model from {hf_url}...")
            urllib.request.urlretrieve(hf_url, bpe_model_path)
            print(f"Downloaded BPE model to {bpe_model_path}")
            return bpe_model_path
        except Exception as e:
            raise RuntimeError(
                f"Failed to download BPE model from {hf_url}: {e}"
            ) from e

    @property
    def tokenizer(self) -> Any:
        """
        Lazy-load the SentencePiece tokenizer.

        Returns:
            The loaded SentencePieceProcessor.

        Raises:
            ImportError: If sentencepiece is not installed.
            RuntimeError: If BPE model cannot be loaded.
        """
        if self._tokenizer is None:
            try:
                import sentencepiece as spm
            except ImportError as e:
                raise ImportError(
                    "sentencepiece is required for token-based segmentation. "
                    "Install it with: pip install sentencepiece"
                ) from e

            # Determine BPE model path
            if self.config.bpe_model_path is not None:
                bpe_path = Path(self.config.bpe_model_path)
                if not bpe_path.exists():
                    raise RuntimeError(
                        f"BPE model not found at {bpe_path}. "
                        f"Set bpe_model_path=None to auto-download."
                    )
            else:
                # Auto-download
                bpe_path = self._download_bpe_model()

            # Load tokenizer
            self._tokenizer = spm.SentencePieceProcessor()
            self._tokenizer.Load(str(bpe_path))

        return self._tokenizer

    def _count_tokens(self, text: str) -> int:
        """
        Count the number of BPE tokens in text.

        Args:
            text: Text to tokenize.

        Returns:
            Number of tokens.
        """
        return len(self.tokenizer.Encode(text))

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        """
        Merge chunks that are too small with adjacent chunks.

        This prevents tiny chunks (e.g., single punctuation marks) that
        can cause TTS failures.

        Args:
            chunks: List of text chunks.

        Returns:
            List of chunks with small chunks merged.
        """
        if not chunks or self.config.min_chars == 0:
            return chunks

        merged: list[str] = []
        i = 0

        while i < len(chunks):
            current = chunks[i]

            # If chunk is too small, try to merge with next or previous
            # Use stripped length to avoid counting leading/trailing whitespace
            if len(current.strip()) < self.config.min_chars:
                if merged:
                    # Merge with previous chunk
                    merged[-1] = merged[-1] + " " + current
                elif i + 1 < len(chunks):
                    # Merge with next chunk
                    current = current + " " + chunks[i + 1]
                    i += 1
                    merged.append(current)
                else:
                    # Last chunk and nothing to merge with
                    merged.append(current)
            else:
                merged.append(current)

            i += 1

        return merged

    def segment(self, text: str) -> list[str]:
        """
        Segment text into chunks based on the configured strategy.

        This method processes the text with spaCy to detect sentence boundaries,
        then groups sentences using one of three strategies:

        1. token_count: Groups sentences by BPE token count (IndexTTS2-optimized)
        2. sentence_count: Groups a fixed number of sentences per chunk
        3. char_count: Groups sentences while staying under max_chars limit

        Strategy priority: token_count > sentence_count > char_count

        Args:
            text: The raw input text to segment.

        Returns:
            List of text chunks, each respecting sentence boundaries and
            minimum/maximum character/token limits.
        """
        if not text or not text.strip():
            return []

        doc = self.nlp(text)

        if self.config.strategy == "token_count":
            chunks = self._segment_by_token_count(doc.sents)
        elif self.config.strategy == "sentence_count":
            chunks = self._segment_by_sentence_count(doc.sents)
        else:  # char_count
            chunks = self._segment_by_char_count(doc.sents)

        # Merge chunks that are too small (skip for token_count as it has its own min handling)
        if self.config.strategy != "token_count":
            chunks = self._merge_small_chunks(chunks)

        return chunks

    def _segment_by_token_count(self, sentences: Any) -> list[str]:
        """
        Segment by BPE token count while preserving sentence boundaries.

        Packs as many sentences as possible without exceeding token_target,
        with a minimum of 1 sentence per chunk (even if it exceeds the target).

        This strategy is optimized for IndexTTS2's 120-token context window.

        Args:
            sentences: spaCy sentences iterator.

        Returns:
            List of text chunks, optimized for token count.
        """
        chunks: list[str] = []
        current_sentences: list[str] = []
        current_token_count = 0

        assert self.config.token_target is not None, "token_target must be set"

        for sent in sentences:
            sentence_text = sent.text.strip()

            # Skip empty sentences
            if not sentence_text:
                continue

            sentence_tokens = self._count_tokens(sentence_text)

            # If we have no sentences yet, add this one regardless of size
            # (minimum 1 sentence per chunk)
            if not current_sentences:
                current_sentences.append(sentence_text)
                current_token_count = sentence_tokens
                continue

            # Check if adding this sentence would exceed token_target
            potential_token_count = current_token_count + sentence_tokens

            if potential_token_count <= self.config.token_target:
                # Add sentence to current chunk
                current_sentences.append(sentence_text)
                current_token_count = potential_token_count
            else:
                # Save current chunk and start new one
                chunks.append(" ".join(current_sentences))
                current_sentences = [sentence_text]
                current_token_count = sentence_tokens

        # Add final chunk if not empty
        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks

    def _segment_by_sentence_count(self, sentences: Any) -> list[str]:
        """
        Segment by grouping a fixed number of sentences per chunk.

        Args:
            sentences: spaCy sentences iterator.

        Returns:
            List of text chunks.
        """
        chunks: list[str] = []
        current_sentences: list[str] = []

        for sent in sentences:
            sentence_text = sent.text.strip()

            # Skip empty sentences
            if not sentence_text:
                continue

            current_sentences.append(sentence_text)

            # Create chunk when we reach the target sentence count
            if len(current_sentences) >= self.config.sentences_per_chunk:
                chunks.append(" ".join(current_sentences))
                current_sentences = []

        # Add remaining sentences as final chunk
        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks

    def _hard_split_text(self, text: str) -> list[str]:
        """
        Hard-split text to guarantee chunks within max_chars.

        This is a fallback for when sentence/punctuation boundaries fail.
        Splits on spaces, then hard-cuts if individual words are too long.

        Args:
            text: Text to split (may be >max_chars).

        Returns:
            List of chunks, each guaranteed to be ≤max_chars.
        """
        if len(text) <= self.config.max_chars:
            return [text]

        import warnings

        warnings.warn(
            f"Hard-splitting {len(text)}-char text (max: {self.config.max_chars}). "
            f"Preview: {text[:100]}...",
            stacklevel=2,
        )

        chunks: list[str] = []
        words = text.split()
        current_chunk = ""

        for word in words:
            # If single word is longer than max_chars, hard-cut it
            if len(word) > self.config.max_chars:
                # Save current chunk if any
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Hard-cut the word into max_chars pieces
                for i in range(0, len(word), self.config.max_chars):
                    chunks.append(word[i : i + self.config.max_chars])
                continue

            # Try adding word to current chunk
            test_chunk = (current_chunk + " " + word).strip()
            if len(test_chunk) <= self.config.max_chars:
                current_chunk = test_chunk
            else:
                # Current chunk is full, save it and start new one
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _segment_by_char_count(self, sentences: Any) -> list[str]:
        """
        Segment by character count while preserving sentence boundaries.

        Args:
            sentences: spaCy sentences iterator.

        Returns:
            List of text chunks, guaranteed to be ≤max_chars.
        """
        chunks: list[str] = []
        current_chunk: str = ""

        for sent in sentences:
            sentence_text = sent.text.strip()

            # Skip empty sentences
            if not sentence_text:
                continue

            # If this single sentence is oversized, split it immediately
            if len(sentence_text) > self.config.max_chars:
                # Save current chunk first
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                # Hard-split the oversized sentence
                chunks.extend(self._hard_split_text(sentence_text))
                continue

            # If adding this sentence would exceed the limit, start a new chunk
            if (
                current_chunk
                and len(current_chunk) + 1 + len(sentence_text) > self.config.max_chars
            ):
                chunks.append(current_chunk)
                current_chunk = sentence_text
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk = current_chunk + " " + sentence_text
                else:
                    current_chunk = sentence_text

        # Add the final chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)

        # GUARANTEE: Validate all chunks are within limit
        # This catches any logic errors in the above code
        oversized = [c for c in chunks if len(c) > self.config.max_chars]
        if oversized:
            # This should never happen, but if it does, fix it
            import warnings

            warnings.warn(
                f"Found {len(oversized)} oversized chunks after segmentation! "
                f"Re-splitting with hard split. This indicates a bug in the segmenter.",
                stacklevel=2,
            )
            fixed_chunks = []
            for chunk in chunks:
                if len(chunk) > self.config.max_chars:
                    fixed_chunks.extend(self._hard_split_text(chunk))
                else:
                    fixed_chunks.append(chunk)
            chunks = fixed_chunks

        return chunks

    def __repr__(self) -> str:
        """String representation of the segmenter."""
        model_name = (
            self.config.model_name
            or get_model_for_language(self.config.language).model_name  # type: ignore[union-attr]
        )

        if self.config.strategy == "token_count":
            strategy_info = f"token_target={self.config.token_target}"
        elif self.config.strategy == "sentence_count":
            strategy_info = f"sentences_per_chunk={self.config.sentences_per_chunk}"
        else:
            strategy_info = f"max_chars={self.config.max_chars}"

        return (
            f"SpacySegmenter(language='{self.config.language}', "
            f"model='{model_name}', "
            f"strategy='{self.config.strategy}', "
            f"{strategy_info})"
        )
