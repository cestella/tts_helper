# TTS Helper

An industrial-grade audiobook text-to-speech processor pipeline for intelligent text segmentation and chunking.

## Overview

TTS Helper is a Python library designed to prepare text for text-to-speech (TTS) processing, particularly for audiobook production. It intelligently segments raw text into clean, natural chunks that respect sentence boundaries while staying within configurable character limits.

The library uses spaCy's advanced NLP capabilities for accurate sentence boundary detection across multiple languages, ensuring that TTS systems receive properly formatted text chunks for natural-sounding audio output.

## Features

- **Intelligent Sentence Segmentation**: Uses spaCy for accurate sentence boundary detection
- **Multi-language Support**: Built-in support for English, German, French, Spanish, Italian, Portuguese, Dutch, Chinese, and Japanese
- **Configurable Chunking**: Control maximum chunk size while respecting natural sentence boundaries
- **Type-Safe**: Fully typed Python code for better IDE support and fewer runtime errors
- **JSON Configuration**: Serialize and deserialize configurations from JSON files
- **Extensible Architecture**: Base classes allow easy implementation of custom segmenters
- **Comprehensive Testing**: Full test suite with unit and integration tests
- **Production Ready**: Designed for industrial-grade audiobook processing pipelines

## Installation

### From Source (Development)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cestella/tts_helper.git
   cd tts_helper
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **Note:** On macOS, `nemo_text_processing` requires manual installation (see below). All other dependencies will install automatically.

4. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

   This installs tts-helper in editable mode, allowing you to modify the code and see changes immediately without reinstalling.

### From PyPI (When Published)

```bash
pip install tts-helper
```

### spaCy Language Models

After installation, download the spaCy language model for your target language:

```bash
# English
python -m spacy download en_core_web_sm

# German
python -m spacy download de_core_news_sm

# French
python -m spacy download fr_core_news_sm

# Spanish
python -m spacy download es_core_news_sm

# And so on for other languages...
```

### Installing nemo_text_processing on macOS (Advanced)

If you want to use [NeMo Text Processing](https://github.com/NVIDIA/NeMo-text-processing) for advanced text normalization (e.g., converting "$123.45" to "one hundred twenty three dollars forty five cents"), you'll need to manually install it on macOS due to compilation requirements.

**Prerequisites:**
- Homebrew installed
- macOS with Apple Silicon (M1/M2/M3) or Intel

**Step-by-step installation:**

1. **Install OpenFst via Homebrew:**
   ```bash
   brew install openfst
   ```

2. **Verify OpenFst installation:**
   ```bash
   brew --prefix openfst
   # Should output: /opt/homebrew/opt/openfst
   ```

3. **Set compiler flags and install pynini:**

   The `pynini` library (required by nemo_text_processing) needs to be compiled with proper paths to OpenFst headers and libraries:

   ```bash
   export CFLAGS="-I/opt/homebrew/opt/openfst/include"
   export CXXFLAGS="-I/opt/homebrew/opt/openfst/include"
   export LDFLAGS="-L/opt/homebrew/opt/openfst/lib"

   pip install --no-cache-dir pynini
   ```

4. **Install nemo_text_processing:**

   Once pynini is successfully installed, you can install nemo_text_processing:

   ```bash
   # Install without dependencies (to avoid version conflicts)
   pip install --no-deps nemo_text_processing

   # Install remaining dependencies
   pip install sacremoses cdifflib editdistance inflect joblib pandas regex transformers wget
   ```

5. **Verify installation:**
   ```bash
   python -c "from nemo_text_processing.text_normalization.normalize import Normalizer; print('Success!')"
   ```

**Troubleshooting:**

- **If OpenFst headers not found:** Make sure the symlink is correct with `ls /opt/homebrew/opt/openfst/include/fst/`. If missing, try `brew reinstall openfst`.

- **If pynini build fails:** Ensure you've set all three environment variables (CFLAGS, CXXFLAGS, LDFLAGS) before running pip install.

- **For older Intel Macs:** The paths might be `/usr/local/opt/openfst` instead of `/opt/homebrew/opt/openfst`.

**Note:** This manual installation is only necessary if you need NeMo's text normalization features. The core TTS Helper functionality works without nemo_text_processing.

### Installing Kokoro TTS

If you want to use Kokoro TTS for high-quality, multi-language speech synthesis, you need to install `kokoro-onnx` and download model files.

**Prerequisites:**
- Python 3.10+
- ~200MB disk space for model files

**Step-by-step installation:**

1. **Install kokoro-onnx:**
   ```bash
   pip install kokoro-onnx
   ```

2. **Download model files:**

   Kokoro requires two files: the ONNX model and the voices binary.

   ```bash
   # Download the Kokoro model (v1.0)
   wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v1.0.onnx

   # Download the voices file
   wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices-v1.0.bin
   ```

   Place these files in your working directory or specify custom paths in the config.

3. **Verify installation:**
   ```bash
   python -c "from kokoro_onnx import Kokoro; print('Kokoro TTS ready!')"
   ```

**Supported Languages & Voices:**

Kokoro supports 6 languages with 43+ high-quality voices:

- **US English (en-us)**: 19 voices
  - Female: af_alloy, af_aoede, af_bella, af_heart, af_jessica, af_kore, af_nicole, af_nova, af_river, af_sarah (default), af_sky
  - Male: am_adam, am_echo, am_eric, am_fenrir, am_liam, am_michael, am_onyx, am_puck

- **British English (en-gb)**: 8 voices
  - Female: bf_alice, bf_emma (default), bf_isabella, bf_lily
  - Male: bm_daniel, bm_fable, bm_george, bm_lewis

- **French (fr-fr)**: 1 voice
  - Female: ff_siwis (default)

- **Italian (it)**: 2 voices
  - Female: if_sara (default)
  - Male: im_nicola

- **Japanese (ja)**: 5 voices
  - Female: jf_alpha (default), jf_gongitsune, jf_nezumi, jf_tebukuro
  - Male: jm_kumo

- **Mandarin Chinese (cmn)**: 8 voices
  - Female: zf_xiaobei, zf_xiaoni, zf_xiaoxiao (default), zf_xiaoyi
  - Male: zm_yunjian, zm_yunxi, zm_yunxia, zm_yunyang

**Note:** Model files are ~150MB total. One model supports all languages.

### Installing pydub for Audio Stitching (Advanced)

If you want to stitch multiple audio segments together with silence or crossfades, you need to install `pydub` and `ffmpeg`.

**Prerequisites:**
- Python 3.10+
- ffmpeg installed on your system

**Step-by-step installation:**

1. **Install ffmpeg:**

   **macOS:**
   ```bash
   brew install ffmpeg
   ```

   **Ubuntu/Debian:**
   ```bash
   sudo apt-get update
   sudo apt-get install ffmpeg
   ```

   **Windows:**
   Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) and add to PATH.

2. **Install pydub:**
   ```bash
   pip install pydub
   ```

3. **For Python 3.13+ (audioop compatibility):**
   ```bash
   pip install audioop-lts
   ```

4. **Verify installation:**
   ```bash
   python -c "from pydub import AudioSegment; print('pydub ready!')"
   ```

**Note:** pydub is only required if you want to combine audio segments. The core TTS functionality works without it.

### Installing m4b-tool for M4B Audiobooks (Advanced)

If you want to create M4B audiobooks with chapter markers and metadata from ISBN, you need to install `m4b-tool` and `isbnlib`.

**Prerequisites:**
- Python 3.10+
- m4b-tool installed on your system
- ffmpeg (usually installed with m4b-tool)

**Step-by-step installation:**

1. **Install m4b-tool:**

   **macOS (via Homebrew):**
   ```bash
   brew install m4b-tool
   ```

   **Linux:**
   ```bash
   # Download latest release
   wget https://github.com/sandreas/m4b-tool/releases/latest/download/m4b-tool.tar.gz
   tar -xzf m4b-tool.tar.gz
   sudo mv m4b-tool /usr/local/bin/
   sudo chmod +x /usr/local/bin/m4b-tool
   ```

   **Windows:**
   Download from [m4b-tool releases](https://github.com/sandreas/m4b-tool/releases) and add to PATH.

2. **Install isbnlib:**
   ```bash
   pip install isbnlib
   ```

3. **Verify installation:**
   ```bash
   m4b-tool --version
   python -c "import isbnlib; print('isbnlib ready!')"
   ```

**How it works (Directory Mode Only):**
- When processing a directory with `--isbn`, the tool fetches book metadata (title, author, year)
- After creating individual MP3 files from each chapter, m4b-tool merges them into a single M4B audiobook
- The M4B file includes proper metadata and chapter markers (using filenames as chapter names)
- M4B filename is automatically generated from book title (lowercase, no spaces/punctuation)

**Note:** M4B creation only works in directory mode where you're processing multiple chapter files. For single files, the tool creates a standard audio file without M4B packaging.

## Quick Start

### Basic Usage - Text Segmentation

```python
from tts_helper import SpacySegmenter, SpacySegmenterConfig

# Create a segmenter with default settings (English, 300 char max)
config = SpacySegmenterConfig(language="en", strategy="char_count", max_chars=300)
segmenter = SpacySegmenter(config)

# Segment your text
text = """
It was a bright cold day in April, and the clocks were striking thirteen.
Winston Smith, his chin nuzzled into his breast in an effort to escape the
vile wind, slipped quickly through the glass doors of Victory Mansions.
"""

chunks = segmenter.segment(text)
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: {chunk}")
```

### Sentence-Count Strategy

```python
from tts_helper import SpacySegmenter, SpacySegmenterConfig

# Group by number of sentences instead of characters
config = SpacySegmenterConfig(
    language="en",
    strategy="sentence_count",
    sentences_per_chunk=3  # 3 sentences per chunk
)
segmenter = SpacySegmenter(config)

text = "First sentence. Second sentence. Third sentence. Fourth sentence."
chunks = segmenter.segment(text)
# Result: ["First sentence. Second sentence. Third sentence.", "Fourth sentence."]
```

### Text Normalization for TTS

```python
from tts_helper import NemoNormalizer, NemoNormalizerConfig

# Create a normalizer
config = NemoNormalizerConfig(language="en")
normalizer = NemoNormalizer(config)

# Normalize text (convert to spoken form)
text = "Dr. Smith paid $123.45 at 3:30pm on 01/15/2024."
normalized = normalizer.normalize(text)
print(normalized)
# Output: "Doctor Smith paid one hundred twenty three dollars forty five cents
#          at three thirty p m on january fifteenth twenty twenty four"
```

### Complete Pipeline: Normalize → Segment

```python
from tts_helper import NemoNormalizer, NemoNormalizerConfig
from tts_helper import SpacySegmenter, SpacySegmenterConfig

# Step 1: Normalize text to spoken form
normalizer_config = NemoNormalizerConfig(language="en")
normalizer = NemoNormalizer(normalizer_config)

text = "Chapter 1: Dr. Smith had $1,000 and 3 cats."
normalized_text = normalizer.normalize(text)

# Step 2: Segment normalized text into chunks
segmenter_config = SpacySegmenterConfig(
    language="en",
    strategy="sentence_count",
    sentences_per_chunk=2
)
segmenter = SpacySegmenter(segmenter_config)

chunks = segmenter.segment(normalized_text)
for chunk in chunks:
    print(f"TTS Chunk: {chunk}")
```

### Using Different Languages

```python
from tts_helper import SpacySegmenter, SpacySegmenterConfig

# German segmentation
config_de = SpacySegmenterConfig(language="de", max_chars=400)
segmenter_de = SpacySegmenter(config_de)

german_text = "Das ist der erste Satz. Das ist der zweite Satz."
chunks = segmenter_de.segment(german_text)
```

### Loading Configuration from JSON

```python
from tts_helper import SpacySegmenterConfig, SpacySegmenter

# Save configuration
config = SpacySegmenterConfig(
    language="en",
    max_chars=300,
    disable_pipes=["ner", "lemmatizer"]
)
config.to_json("segmenter_config.json")

# Load configuration
loaded_config = SpacySegmenterConfig.from_json("segmenter_config.json")
segmenter = SpacySegmenter(loaded_config)
```

### Batch Processing

```python
from tts_helper import SpacySegmenter, SpacySegmenterConfig

config = SpacySegmenterConfig(language="en", max_chars=350)
segmenter = SpacySegmenter(config)

# Process multiple texts at once
texts = [
    "First chapter text here...",
    "Second chapter text here...",
    "Third chapter text here..."
]

all_chunks = segmenter.segment_batch(texts)
for i, chunks in enumerate(all_chunks, 1):
    print(f"Chapter {i}: {len(chunks)} chunks")
```

### Complete Pipeline: Normalize → Segment → TTS

```python
from pathlib import Path
from tts_helper import (
    NemoNormalizer, NemoNormalizerConfig,
    SpacySegmenter, SpacySegmenterConfig,
    KokoroTTS, KokoroTTSConfig
)

# Step 1: Normalize text to spoken form
normalizer_config = NemoNormalizerConfig(language="en")
normalizer = NemoNormalizer(normalizer_config)

text = """
Chapter 1: The Discovery

Dr. Smith arrived at 3:30pm on 01/15/2024 with $1,234.56 in funding.
The experiment would change everything. There were 42 test subjects.
"""

normalized_text = normalizer.normalize(text)

# Step 2: Segment into TTS-friendly chunks
segmenter_config = SpacySegmenterConfig(
    language="en",
    strategy="char_count",
    max_chars=300
)
segmenter = SpacySegmenter(segmenter_config)
chunks = segmenter.segment(normalized_text)

# Step 3: Synthesize each chunk to audio
tts_config = KokoroTTSConfig(language="english", voice="am_adam", speed=1.0)
tts = KokoroTTS(tts_config)

output_dir = Path("audiobook_output")
output_dir.mkdir(exist_ok=True)

for i, chunk in enumerate(chunks, 1):
    print(f"Synthesizing chunk {i}/{len(chunks)}...")
    sample_rate, audio = tts.synthesize(chunk)

    output_path = output_dir / f"chunk_{i:03d}.wav"
    tts.save_audio(audio, sample_rate, output_path)

print(f"Generated {len(chunks)} audio files in {output_dir}")
```

### Text-to-Speech with Kokoro TTS

```python
from tts_helper import KokoroTTS, KokoroTTSConfig

# Create TTS with default US English voice (af_sarah)
config = KokoroTTSConfig(language="en-us", voice="af_sarah")
tts = KokoroTTS(config)

# Synthesize speech
text = "Hello world! This is Kokoro text-to-speech."
sample_rate, audio = tts.synthesize(text)

# Save to file
tts.save_audio(audio, sample_rate, "kokoro_output.wav")
```

### Using Different Kokoro Voices and Languages

```python
from tts_helper import KokoroTTS, KokoroTTSConfig, get_kokoro_voices

# List available voices for a language
us_voices = get_kokoro_voices("en-us")
print(f"US English voices: {us_voices}")
# Output: ['af_alloy', 'af_aoede', ..., 'am_adam', 'am_echo', ...]

# British English with male voice
uk_config = KokoroTTSConfig(language="en-gb", voice="bm_george")
uk_tts = KokoroTTS(uk_config)

british_text = "Hello! I'm speaking with a British accent."
sample_rate, audio = uk_tts.synthesize(british_text)
uk_tts.save_audio(audio, sample_rate, "british_output.wav")

# Japanese with female voice
ja_config = KokoroTTSConfig(language="ja", voice="jf_alpha")
ja_tts = KokoroTTS(ja_config)

japanese_text = "こんにちは！元気ですか？"
sample_rate, audio = ja_tts.synthesize(japanese_text)
ja_tts.save_audio(audio, sample_rate, "japanese_output.wav")

# Adjust speech speed (0.5 = slower, 2.0 = faster)
fast_config = KokoroTTSConfig(language="en-us", voice="af_nova", speed=1.5)
fast_tts = KokoroTTS(fast_config)

fast_text = "This will be spoken faster than normal."
sample_rate, audio = fast_tts.synthesize(fast_text)
fast_tts.save_audio(audio, sample_rate, "fast_output.wav")
```

### Using Kokoro for Multi-Language TTS

Kokoro TTS supports 6 languages including Japanese and Chinese, making it ideal for multi-language projects:

```python
# Example: Japanese TTS with Kokoro
from tts_helper import (
    SpacySegmenter, SpacySegmenterConfig,
    KokoroTTS, KokoroTTSConfig,
)

# Segment Japanese text
segmenter = SpacySegmenter(
    SpacySegmenterConfig(language="ja", max_chars=200)
)
japanese_text = "これは日本語のテストです。音声合成が正常に動作しています。"
chunks = segmenter.segment(japanese_text)

# Synthesize with Kokoro
tts = KokoroTTS(KokoroTTSConfig(language="ja", voice="jf_alpha"))

for i, chunk in enumerate(chunks, 1):
    sample_rate, audio = tts.synthesize(chunk)
    tts.save_audio(audio, sample_rate, f"japanese_chunk_{i}.wav")
```

### Audio Stitching with Silence

```python
from pathlib import Path
from tts_helper import PydubStitcher, PydubStitcherConfig

# Create stitcher with 500ms silence between segments
config = PydubStitcherConfig(silence_duration_ms=500, output_format="wav")
stitcher = PydubStitcher(config)

# Stitch multiple audio files together
audio_files = [
    "chunk_001.wav",
    "chunk_002.wav",
    "chunk_003.wav",
]

stitcher.stitch(audio_files, "complete_audiobook.wav")
```

### Audio Stitching with Crossfade

```python
from tts_helper import PydubStitcher, PydubStitcherConfig

# Create stitcher with 100ms crossfade (smooth transitions)
config = PydubStitcherConfig(
    crossfade_duration_ms=100,
    silence_duration_ms=0,  # No silence when using crossfade
    output_format="mp3",
    export_bitrate="192k",
)
stitcher = PydubStitcher(config)

# Stitch and export to MP3
stitcher.stitch(audio_files, "audiobook_crossfade.mp3")
```

### Complete Pipeline: Text → Audio → Stitched Audiobook

```python
from pathlib import Path
from tts_helper import (
    NemoNormalizer, NemoNormalizerConfig,
    SpacySegmenter, SpacySegmenterConfig,
    KokoroTTS, KokoroTTSConfig,
    PydubStitcher, PydubStitcherConfig,
)

# 1. Normalize text
normalizer = NemoNormalizer(NemoNormalizerConfig(language="en"))
text = """
Chapter 1: The Beginning

It was 3:30pm when Dr. Smith discovered the formula.
He had invested $1,234,567 into the research.
There were 42 test subjects waiting.
"""
normalized_text = normalizer.normalize(text)

# 2. Segment into chunks
segmenter = SpacySegmenter(
    SpacySegmenterConfig(language="en", max_chars=300)
)
chunks = segmenter.segment(normalized_text)

# 3. Synthesize each chunk
tts = KokoroTTS(KokoroTTSConfig(language="english", voice="am_adam"))

output_dir = Path("audiobook_chunks")
output_dir.mkdir(exist_ok=True)

chunk_files = []
for i, chunk in enumerate(chunks, 1):
    print(f"Synthesizing chunk {i}/{len(chunks)}...")
    sample_rate, audio = tts.synthesize(chunk)

    chunk_path = output_dir / f"chunk_{i:03d}.wav"
    tts.save_audio(audio, sample_rate, chunk_path)
    chunk_files.append(chunk_path)

# 4. Stitch all chunks together with silence
stitcher = PydubStitcher(
    PydubStitcherConfig(
        silence_duration_ms=750,  # 0.75 second pause between chunks
        output_format="mp3",
        export_bitrate="256k",
    )
)

stitcher.stitch(chunk_files, "complete_audiobook.mp3")
print("Audiobook complete: complete_audiobook.mp3")
```

## Command Line Interface

TTS Helper includes a powerful CLI for converting text files to audiobooks with a single command.

### Quick Start

```bash
# Generate a default configuration file
python -m tts_helper --create-config

# Convert text file to audiobook with defaults
python -m tts_helper input.txt --output audiobook.mp3

# Convert with custom configuration
python -m tts_helper input.txt --config config.json --output audiobook.mp3

# Verbose output to see progress
python -m tts_helper input.txt --output audiobook.mp3 --verbose

# Keep intermediate audio chunks
python -m tts_helper input.txt --output audiobook.mp3 --keep-chunks

# Process directory of chapters to M4B audiobook (directory mode only)
python -m tts_helper chapters_dir/ --output audiobooks/ --isbn 978-0-441-00731-2
```

### Configuration File Format

The CLI uses a JSON configuration file to control all pipeline components. Generate a default configuration with:

```bash
python -m tts_helper --create-config
```

This creates `config.json` with the following structure:

```json
{
  "tts_engine": "kokoro",
  "normalizer": {
    "language": "en",
    "input_case": "cased",
    "verbose": false
  },
  "segmenter": {
    "language": "en",
    "strategy": "char_count",
    "max_chars": 300,
    "sentences_per_chunk": 3
  },
  "tts": {
    "language": "english",
    "voice": "af_sarah",
    "speed": 1.0,
    "verbose": false
  },
  "stitcher": {
    "silence_duration_ms": 750,
    "crossfade_duration_ms": 0,
    "output_format": "mp3",
    "export_bitrate": "192k"
  },
  "m4b_tool_args": {
    "audio-bitrate": "64k",
    "use-filenames-as-chapters": "",
    "jobs": "4"
  },
  "skip_normalization": false
}
```

**Configuration Sections:**

- **tts_engine**: Currently uses `"kokoro"` TTS engine
  - Kokoro: Fast, lightweight, multi-language support, works well on CPU

- **normalizer**: Controls text normalization (numbers, dates, currency → spoken form)
  - See `NemoNormalizerConfig` for available options

- **segmenter**: Controls how text is chunked for TTS
  - See `SpacySegmenterConfig` for available options

- **tts**: Controls Kokoro speech synthesis settings
  - See `KokoroTTSConfig` for available options

- **stitcher**: Controls how audio chunks are combined
  - See `PydubStitcherConfig` for available options

- **m4b_tool_args**: Arguments passed to m4b-tool for M4B creation (optional, only used with `--isbn`)
  - Keys are argument names without `--` prefix
  - Values are argument values, or empty string `""` for flags without values
  - Common arguments:
    - `"audio-bitrate"`: Audio bitrate for M4B (e.g., `"64k"`, `"128k"`)
    - `"use-filenames-as-chapters"`: Use chunk filenames as chapter names (empty string value)
    - `"jobs"`: Number of parallel jobs (e.g., `"4"`)
  - Book metadata (title, author, year) are automatically fetched from ISBN

- **skip_normalization**: Set to `true` to skip the normalization step

### CLI Options

```
usage: python -m tts_helper [-h] [-o OUTPUT] [-c CONFIG] [--create-config]
                            [--keep-chunks] [-v] [input]

positional arguments:
  input                 Input text file to convert to audiobook

options:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   Output audiobook file (e.g., audiobook.mp3)
  -c, --config CONFIG   JSON configuration file for pipeline components
  --create-config       Create a default configuration file (config.json) and exit
  --keep-chunks         Keep individual audio chunk files
  -v, --verbose         Print verbose progress information
```

### Example Workflows

**1. Basic audiobook with defaults:**
```bash
python -m tts_helper story.txt --output story.mp3
```

**2. Custom voice and format:**
Edit `config.json`:
```json
{
  "tts": {
    "language": "english",
    "voice": "leo",
    "use_gpu": true
  },
  "stitcher": {
    "output_format": "wav",
    "silence_duration_ms": 1000
  }
}
```

Run:
```bash
python -m tts_helper story.txt --config config.json --output story.wav
```

**3. French audiobook with crossfade:**
Edit `config.json`:
```json
{
  "normalizer": {
    "language": "fr"
  },
  "segmenter": {
    "language": "fr"
  },
  "tts": {
    "language": "french",
    "voice": "pierre"
  },
  "stitcher": {
    "crossfade_duration_ms": 100,
    "silence_duration_ms": 0,
    "output_format": "mp3"
  }
}
```

Run:
```bash
python -m tts_helper histoire.txt --config config.json --output histoire.mp3 -v
```

**4. Japanese audiobook with Kokoro TTS:**
Edit `config.json`:
```json
{
  "tts_engine": "kokoro",
  "segmenter": {
    "language": "ja",
    "max_chars": 200
  },
  "tts": {
    "language": "ja",
    "voice": "jf_alpha",
    "speed": 1.0
  },
  "stitcher": {
    "output_format": "mp3",
    "silence_duration_ms": 500
  },
  "skip_normalization": true
}
```

Run:
```bash
python -m tts_helper japanese_story.txt --config config.json --output japanese_audiobook.mp3 -v
```

**5. Debug with verbose output and keep chunks:**
```bash
python -m tts_helper input.txt --output output.mp3 --verbose --keep-chunks
```

This will:
- Show detailed progress for each step
- Keep individual chunk files in `output_chunks/` directory
- Useful for debugging or manual inspection

## Configuration Options

### SpacySegmenterConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `language` | `str` | `"en"` | ISO 639-1 language code (e.g., 'en', 'de', 'fr') |
| `strategy` | `Literal["sentence_count", "char_count"]` | `"char_count"` | Chunking strategy: 'sentence_count' or 'char_count' |
| `sentences_per_chunk` | `int` | `3` | Number of sentences per chunk (for 'sentence_count' strategy) |
| `max_chars` | `int` | `300` | Maximum characters per chunk (for 'char_count' strategy) |
| `model_name` | `Optional[str]` | `None` | Explicit spaCy model name (auto-detected if None) |
| `disable_pipes` | `List[str]` | `["ner", "lemmatizer"]` | spaCy pipeline components to disable for performance |

### NemoNormalizerConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `language` | `str` | `"en"` | ISO 639-1 language code (supported: en, de, es, pt, ru, fr, vi) |
| `input_case` | `str` | `"cased"` | Input text case handling: 'cased' or 'lower_cased' |
| `cache_dir` | `Optional[str]` | `None` | Directory to cache normalization grammars |
| `verbose` | `bool` | `False` | Whether to print verbose normalization info |

### KokoroTTSConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `language` | `str` | `"en-us"` | Language code (en-us, en-gb, fr-fr, it, ja, cmn) |
| `voice` | `str` | Language-specific | Voice to use (auto-selected based on language if not specified) |
| `speed` | `float` | `1.0` | Speech speed multiplier (range: 0.5-2.0) |
| `model_path` | `Optional[str]` | `None` | Path to kokoro-v1.0.onnx file (auto-detected if None) |
| `voices_path` | `Optional[str]` | `None` | Path to voices-v1.0.bin file (auto-detected if None) |
| `verbose` | `bool` | `False` | Whether to print verbose TTS info |

**Available Voices by Language:**

- **US English (en-us)**: af_sarah (default), af_alloy, af_aoede, af_bella, af_heart, af_jessica, af_kore, af_nicole, af_nova, af_river, af_sky, am_adam, am_echo, am_eric, am_fenrir, am_liam, am_michael, am_onyx, am_puck
- **British English (en-gb)**: bf_emma (default), bf_alice, bf_isabella, bf_lily, bm_daniel, bm_fable, bm_george, bm_lewis
- **French (fr-fr)**: ff_siwis (default)
- **Italian (it)**: if_sara (default), im_nicola
- **Japanese (ja)**: jf_alpha (default), jf_gongitsune, jf_nezumi, jf_tebukuro, jm_kumo
- **Mandarin Chinese (cmn)**: zf_xiaoxiao (default), zf_xiaobei, zf_xiaoni, zf_xiaoyi, zm_yunjian, zm_yunxi, zm_yunxia, zm_yunyang

**Language Aliases:**
- `"fr"` → `"fr-fr"`
- `"zh"` → `"cmn"`

### PydubStitcherConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `silence_duration_ms` | `int` | `500` | Milliseconds of silence between segments |
| `crossfade_duration_ms` | `int` | `0` | Milliseconds to crossfade between segments (0 = disabled) |
| `output_format` | `Literal["wav", "mp3", "ogg", "flac"]` | `"wav"` | Output audio format |
| `export_bitrate` | `str` | `"192k"` | Bitrate for lossy formats (e.g., "128k", "256k") |
| `sample_rate` | `Optional[int]` | `None` | Sample rate for output (None = use source rate) |

**Notes:**
- When `crossfade_duration_ms` > 0, segments will overlap with a smooth fade transition
- When `silence_duration_ms` > 0, silence is added between segments (ignored if crossfade is enabled)
- Requires ffmpeg to be installed for format conversion

## Supported Languages

TTS Helper includes built-in support for the following languages:

| Language | Code | spaCy Model |
|----------|------|-------------|
| English | `en` | `en_core_web_sm` |
| German | `de` | `de_core_news_sm` |
| French | `fr` | `fr_core_news_sm` |
| Spanish | `es` | `es_core_news_sm` |
| Italian | `it` | `it_core_news_sm` |
| Portuguese | `pt` | `pt_core_news_sm` |
| Dutch | `nl` | `nl_core_news_sm` |
| Chinese | `zh` | `zh_core_web_sm` |
| Japanese | `ja` | `ja_core_news_sm` |

## Advanced Usage

### Creating Custom Segmenters

You can create custom segmenters by extending the base classes:

```python
from typing import List
from tts_helper import Segmenter, SegmenterConfig
from dataclasses import dataclass

@dataclass
class CustomSegmenterConfig(SegmenterConfig):
    delimiter: str = "\n\n"

class CustomSegmenter(Segmenter):
    def segment(self, text: str) -> List[str]:
        # Your custom segmentation logic
        return [chunk.strip() for chunk in text.split(self.config.delimiter)]

    def __repr__(self) -> str:
        return f"CustomSegmenter(delimiter={self.config.delimiter})"

# Use your custom segmenter
config = CustomSegmenterConfig(delimiter="\n\n")
segmenter = CustomSegmenter(config)
```

### Language Model Utilities

```python
from tts_helper.language_models import (
    get_supported_languages,
    get_model_for_language,
    is_language_supported
)

# Get all supported languages
languages = get_supported_languages()
print(f"Supported: {languages}")

# Check if a language is supported
if is_language_supported("de"):
    print("German is supported!")

# Get model metadata
model_info = get_model_for_language("en")
print(f"Model: {model_info.model_name}")
print(f"Language: {model_info.language_name}")
print(f"Install: pip install {model_info.pip_package}")
```

## Architecture

The library is built around a clean, extensible architecture:

- **`Segmenter`**: Abstract base class for all segmenters
- **`SegmenterConfig`**: Base configuration class with JSON serialization
- **`SpacySegmenter`**: spaCy-based implementation for production use
- **Language Models**: Centralized language-to-model mapping for consistency

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/tts_helper.git
cd tts_helper

# Install in development mode
pip install -e ".[dev]"

# Download test language models
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tts_helper --cov-report=html

# Run specific test file
pytest tests/test_spacy_segmenter.py -v
```

## Use Cases

- **Audiobook Production**: Segment book chapters into TTS-friendly chunks
- **Podcast Generation**: Prepare scripts for automated voice generation
- **E-learning Content**: Convert educational materials to audio format
- **Accessibility Tools**: Create audio versions of written content
- **Multi-language Support**: Process content in multiple languages consistently

## Performance Considerations

- The library uses lazy loading for spaCy models to reduce startup time
- Unnecessary pipeline components (NER, lemmatizer) are disabled by default
- Batch processing is available for handling multiple texts efficiently
- spaCy's efficient C extensions provide fast processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of the excellent [spaCy](https://spacy.io/) library
- Inspired by the needs of industrial audiobook production pipelines

## Support

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/yourusername/tts_helper).
