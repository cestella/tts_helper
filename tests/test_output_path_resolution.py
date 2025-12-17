"""Tests for output path resolution and log file placement."""

from pathlib import Path
from unittest.mock import patch

import pytest

from tts_helper.per_file_process import PerFileProcessConfig, PerFileProcessManager


class TestOutputPathResolution:
    """Test that output path resolution works correctly for different input scenarios."""

    @pytest.fixture
    def temp_structure(self, tmp_path: Path):
        """Create a temporary directory structure for testing."""
        # Create directory structure:
        # tmp_path/
        #   audiobooks/
        #     the_book/
        #       chapters_txt/
        #         chapter_01.txt
        #       chapters_mp3/
        base_dir = tmp_path / "audiobooks" / "the_book"
        chapters_txt = base_dir / "chapters_txt"
        chapters_mp3 = base_dir / "chapters_mp3"

        chapters_txt.mkdir(parents=True)
        chapters_mp3.mkdir(parents=True)

        # Create a test text file
        test_file = chapters_txt / "chapter_01.txt"
        test_file.write_text("This is a test chapter.")

        return {
            "base_dir": base_dir,
            "chapters_txt": chapters_txt,
            "chapters_mp3": chapters_mp3,
            "test_file": test_file,
        }

    def test_log_path_with_file_output(self, temp_structure):
        """Test log path when output is a specific file path."""
        # Setup
        input_file = temp_structure["test_file"]
        output_file = temp_structure["chapters_mp3"] / "chapter_01.mp3"

        config = PerFileProcessConfig(
            enabled=True,
            command="echo",
            args=["test"],
        )
        manager = PerFileProcessManager(config)

        # Get log path
        log_path = manager._get_log_path(input_file, output_file)

        # Expected: audiobooks/the_book/logs/chapter_01.log
        expected_log_dir = temp_structure["base_dir"] / "logs"
        expected_log_path = expected_log_dir / "chapter_01.log"

        assert log_path == expected_log_path
        assert log_path.parent == expected_log_dir

    def test_log_path_with_directory_output(self, temp_structure):
        """Test log path when output is a directory (should fail before our fix)."""
        # Setup
        input_file = temp_structure["test_file"]
        output_dir = temp_structure["chapters_mp3"]  # Directory, not file

        # Construct what the actual output file would be
        # This simulates what process_audiobook_with_managed_process does now
        actual_output_file = output_dir / f"{input_file.stem}.mp3"

        config = PerFileProcessConfig(
            enabled=True,
            command="echo",
            args=["test"],
        )
        manager = PerFileProcessManager(config)

        # Get log path with the RESOLVED output file (after our fix)
        log_path = manager._get_log_path(input_file, actual_output_file)

        # Expected: audiobooks/the_book/logs/chapter_01.log
        expected_log_dir = temp_structure["base_dir"] / "logs"
        expected_log_path = expected_log_dir / "chapter_01.log"

        assert log_path == expected_log_path
        assert log_path.parent == expected_log_dir

        # Verify it's NOT going to audiobooks/logs (the bug we fixed)
        wrong_log_dir = temp_structure["base_dir"].parent / "logs"
        assert log_path.parent != wrong_log_dir

    def test_log_path_without_output_path(self, temp_structure):
        """Test log path falls back correctly when no output path provided."""
        # Setup
        input_file = temp_structure["test_file"]

        config = PerFileProcessConfig(
            enabled=True,
            command="echo",
            args=["test"],
        )
        manager = PerFileProcessManager(config)

        # Get log path without output_path
        log_path = manager._get_log_path(input_file, output_path=None)

        # Expected: ./process_logs/chapter_01.log (fallback)
        expected_log_path = Path("./process_logs") / "chapter_01.log"

        assert log_path == expected_log_path

    def test_log_path_with_custom_log_dir(self, temp_structure, tmp_path: Path):
        """Test log path when custom log_dir is specified in config."""
        # Setup
        input_file = temp_structure["test_file"]
        output_file = temp_structure["chapters_mp3"] / "chapter_01.mp3"
        custom_log_dir = tmp_path / "custom_logs"

        config = PerFileProcessConfig(
            enabled=True,
            command="echo",
            args=["test"],
            log_dir=str(custom_log_dir),
        )
        manager = PerFileProcessManager(config)

        # Get log path
        log_path = manager._get_log_path(input_file, output_file)

        # Expected: custom_logs/chapter_01.log
        expected_log_path = custom_log_dir / "chapter_01.log"

        assert log_path == expected_log_path


class TestOutputPathResolutionInMain:
    """Test output path resolution in the main processing flow."""

    def test_output_path_resolution_before_run_for_file(self, tmp_path: Path):
        """Test that output path is resolved BEFORE calling run_for_file."""
        # Create test structure
        input_file = tmp_path / "test.txt"
        input_file.write_text("Test content")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        config = {
            "tts_engine": "kokoro",
            "stitcher": {"output_format": "mp3"},
            "per_file_process": {
                "enabled": False,  # Disabled to avoid actually running process
            },
        }

        # Mock the process_audiobook function to capture what output_path it receives
        with patch("tts_helper.__main__.process_audiobook") as mock_process:
            from tts_helper.__main__ import process_audiobook_with_managed_process

            # Call with directory as output
            process_audiobook_with_managed_process(
                input_text_path=input_file,
                output_path=output_dir,  # Pass directory
                config=config,
                verbose=False,
            )

            # Verify process_audiobook was called with the RESOLVED file path
            assert mock_process.called
            call_kwargs = mock_process.call_args.kwargs

            # Should receive the actual file path, not the directory
            expected_output = output_dir / "test.mp3"
            assert call_kwargs["output_path"] == expected_output
            assert call_kwargs["output_path"].is_absolute()
            assert not call_kwargs["output_path"].is_dir()


class TestPerFileProcessManagerLogCreation:
    """Test that log files are created in the correct location during actual execution."""

    def test_log_file_created_at_correct_location(self, tmp_path: Path):
        """Test that log file is created at the expected location when process runs."""
        # Create test structure
        base_dir = tmp_path / "audiobook"
        chapters_mp3 = base_dir / "chapters_mp3"
        chapters_mp3.mkdir(parents=True)

        input_file = tmp_path / "test.txt"
        input_file.write_text("Test")

        output_file = chapters_mp3 / "test.mp3"

        # Use a simple command that will succeed quickly
        config = PerFileProcessConfig(
            enabled=True,
            command="echo",
            args=["test message"],
        )

        manager = PerFileProcessManager(config)

        # Run the process
        with manager.run_for_file(input_file, output_file):
            # Log file should be created
            expected_log_path = base_dir / "logs" / "test.log"

            # Give it a moment to create the file
            import time

            time.sleep(0.1)

            # Verify log file exists at correct location
            assert expected_log_path.exists()
            assert expected_log_path.is_file()

            # Verify log contains output
            log_content = expected_log_path.read_text()
            assert "test message" in log_content

        # After context exit, log should still exist
        assert expected_log_path.exists()
