"""M4B audiobook creation using m4b-tool and ISBN metadata."""

import json
import re
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class M4bCreatorConfig:
    """Configuration for M4B audiobook creation.

    Args:
        m4b_tool_args: Dictionary of additional arguments to pass to m4b-tool.
            Keys are argument names (without '--' prefix).
            Values are argument values, or empty string for flags without values.

            Example:
                {
                    "audio-bitrate": "64k",
                    "use-filenames-as-chapters": "",  # Flag with no value
                    "jobs": "4"
                }
    """

    m4b_tool_args: dict[str, str] = field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        """Serialize config to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"m4b_tool_args": self.m4b_tool_args}, f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "M4bCreatorConfig":
        """Load config from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(m4b_tool_args=data.get("m4b_tool_args", {}))


class M4bCreator:
    """Create M4B audiobooks from audio chapters using m4b-tool."""

    def __init__(self, config: M4bCreatorConfig):
        """Initialize M4B creator.

        Args:
            config: M4bCreatorConfig instance
        """
        self.config = config

    def create_m4b(
        self,
        isbn: str,
        chapters_dir: Path,
        output_parent_dir: Path,
        verbose: bool = False,
    ) -> Path:
        """Create M4B audiobook from chapters using ISBN metadata.

        Args:
            isbn: ISBN-10 or ISBN-13 for metadata lookup
            chapters_dir: Directory containing audio chapter files
            output_parent_dir: Parent directory where .m4b file will be created
            verbose: Whether to print verbose output

        Returns:
            Path to created .m4b file

        Raises:
            ValueError: If ISBN metadata cannot be fetched
            RuntimeError: If m4b-tool execution fails
        """
        print(f"Fetching metadata for ISBN: {isbn}")

        # Fetch metadata from ISBN
        metadata = self._fetch_metadata(isbn)

        # Always print metadata
        print("\nBook Metadata:")
        print(f"  Title: {metadata['title']}")
        print(f"  Author(s): {metadata['authors']}")
        print(f"  Year: {metadata['year']}")
        if metadata['publisher']:
            print(f"  Publisher: {metadata['publisher']}")
        if metadata['language']:
            print(f"  Language: {metadata['language']}")
        if metadata['description']:
            desc_preview = metadata['description'][:100] + "..." if len(metadata['description']) > 100 else metadata['description']
            print(f"  Description: {desc_preview}")

        # Sanitize title for filename
        filename = self._sanitize_filename(metadata["title"])

        # Construct output path
        output_file = output_parent_dir / f"{filename}.m4b"

        # Download cover art if available
        cover_path = None
        if metadata.get("cover_url"):
            cover_path = self._download_cover(metadata["cover_url"], verbose)
            if cover_path:
                print(f"  Cover art: Downloaded")

        try:
            # Build m4b-tool command
            cmd = self._build_command(metadata, chapters_dir, output_file, cover_path)

            # Always print command
            print("\nExecuting m4b-tool command:")
            print(f"  {' '.join(cmd)}")
            print()

            # Execute command
            self._execute_command(cmd, verbose)

        finally:
            # Clean up temporary cover file
            if cover_path and cover_path.exists():
                cover_path.unlink()

        return output_file

    def _fetch_metadata(self, isbn: str) -> dict:
        """Fetch book metadata from ISBN using isbnlib.

        Args:
            isbn: ISBN-10 or ISBN-13

        Returns:
            Dictionary with title, authors, year, publisher, language, description, and cover_url

        Raises:
            ValueError: If metadata cannot be fetched
        """
        try:
            import isbnlib  # type: ignore[import-untyped]
        except ImportError as err:
            raise ImportError(
                "isbnlib is not installed. Install with: pip install isbnlib"
            ) from err

        # Clean and validate ISBN
        isbn_clean = isbnlib.canonical(isbn)
        if not isbn_clean:
            raise ValueError(f"Invalid ISBN: {isbn}")

        # Fetch metadata (tries multiple services)
        meta = isbnlib.meta(isbn_clean)
        if not meta:
            raise ValueError(
                f"Could not fetch metadata for ISBN: {isbn}. "
                f"Please check the ISBN is correct and try again."
            )

        # Fetch description (may not be available for all books)
        description = ""
        try:
            desc = isbnlib.desc(isbn_clean)
            if desc:
                description = desc
        except Exception:
            pass  # Description not available, continue without it

        # Fetch cover art URLs (may not be available for all books)
        cover_url = ""
        try:
            cover = isbnlib.cover(isbn_clean)
            if cover:
                # Prefer thumbnail over smallThumbnail for better quality
                cover_url = cover.get("thumbnail", cover.get("smallThumbnail", ""))
        except Exception:
            pass  # Cover not available, continue without it

        return {
            "title": meta.get("Title", "Unknown Title"),
            "authors": ", ".join(meta.get("Authors", ["Unknown Author"])),
            "year": meta.get("Year", ""),
            "publisher": meta.get("Publisher", ""),
            "language": meta.get("Language", ""),
            "description": description,
            "cover_url": cover_url,
        }

    def _download_cover(self, cover_url: str, verbose: bool = False) -> Path | None:
        """Download cover image to temporary file.

        Args:
            cover_url: URL to cover image
            verbose: Whether to print verbose output

        Returns:
            Path to downloaded cover file, or None if download fails
        """
        if not cover_url:
            return None

        try:
            if verbose:
                print(f"Downloading cover art from: {cover_url}")

            # Create temporary file for cover
            # Use jpg extension (most common for book covers)
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False, mode="wb"
            )
            temp_path = Path(temp_file.name)

            # Download cover image
            urllib.request.urlretrieve(cover_url, temp_path)

            if verbose:
                file_size_kb = temp_path.stat().st_size / 1024
                print(f"  Cover downloaded: {file_size_kb:.1f} KB")

            return temp_path

        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to download cover: {e}")
            return None

    def _sanitize_filename(self, title: str) -> str:
        """Sanitize title for filename: lowercase, no spaces/punctuation.

        Args:
            title: Book title

        Returns:
            Sanitized filename (lowercase, underscores, no punctuation)

        Example:
            "Man in the High Castle" -> "man_in_the_high_castle"
        """
        # Remove punctuation (keep alphanumeric, spaces, hyphens)
        sanitized = re.sub(r"[^\w\s-]", "", title)
        # Replace spaces and hyphens with underscores
        sanitized = re.sub(r"[-\s]+", "_", sanitized)
        # Lowercase and strip trailing underscores
        return sanitized.lower().strip("_")

    def _build_command(
        self,
        metadata: dict,
        chapters_dir: Path,
        output_file: Path,
        cover_path: Path | None = None,
    ) -> list:
        """Build m4b-tool merge command with metadata and config args.

        Args:
            metadata: Book metadata (title, authors, year)
            chapters_dir: Directory containing audio chapters
            output_file: Output .m4b file path
            cover_path: Optional path to cover image file

        Returns:
            Command as list of strings
        """
        cmd = ["m4b-tool", "merge", str(chapters_dir)]

        # Add output file
        cmd.extend(["--output-file", str(output_file)])

        # Add metadata args from ISBN
        cmd.extend(["--name", metadata["title"]])
        cmd.extend(["--artist", metadata["authors"]])
        cmd.extend(["--albumartist", metadata["authors"]])
        cmd.extend(["--album", metadata["title"]])

        if metadata["year"]:
            cmd.extend(["--year", str(metadata["year"])])

        if metadata["description"]:
            cmd.extend(["--description", metadata["description"]])
            cmd.extend(["--longdesc", metadata["description"]])

        if metadata["publisher"]:
            # Add publisher to comment field
            cmd.extend(["--comment", f"Publisher: {metadata['publisher']}"])

        # Add cover art if provided
        if cover_path and cover_path.exists():
            cmd.extend(["--cover", str(cover_path)])

        # Add config args
        for key, value in self.config.m4b_tool_args.items():
            if value == "":  # Singleton flag (no value)
                cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", value])

        return cmd

    def _execute_command(self, cmd: list, verbose: bool = False) -> None:
        """Execute m4b-tool command.

        Args:
            cmd: Command as list of strings
            verbose: Whether to print output

        Raises:
            RuntimeError: If command fails
        """
        if verbose:
            print(f"Executing: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"m4b-tool failed with exit code {result.returncode}:\n{result.stderr}"
            )

        if verbose and result.stdout:
            print(result.stdout)

    def __repr__(self) -> str:
        """String representation."""
        return f"M4bCreator(args={self.config.m4b_tool_args})"
