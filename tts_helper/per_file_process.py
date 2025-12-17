"""
Per-file process manager for TTS pipeline.

This module provides a process manager that starts and stops a subprocess
for each file being processed, useful for handling server processes that
have memory leaks or degrade over time.
"""

import os
import signal
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO


@dataclass
class PerFileProcessConfig:
    """
    Configuration for per-file process management.

    Attributes:
        enabled: Whether to enable per-file process management.
        command: The command to execute (e.g., 'python', 'node').
        args: Arguments to pass to the command.
        log_dir: Directory to store process logs. If None, auto-determined from output path.
        max_retries: Maximum number of retries if process crashes during TTS.
        shutdown_timeout: Seconds to wait for graceful shutdown before SIGKILL.
    """

    enabled: bool = False
    command: str = ""
    args: list[str] | None = None
    log_dir: str | None = None
    max_retries: int = 5
    shutdown_timeout: float = 5.0

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.args is None:
            self.args = []

        if self.enabled:
            if not self.command:
                raise ValueError("command must be specified when enabled=True")

            if self.max_retries < 0:
                raise ValueError(
                    f"max_retries must be non-negative, got {self.max_retries}"
                )

            if self.shutdown_timeout < 0:
                raise ValueError(
                    f"shutdown_timeout must be non-negative, got {self.shutdown_timeout}"
                )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PerFileProcessConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "enabled": self.enabled,
            "command": self.command,
            "args": self.args or [],
            "log_dir": self.log_dir,
            "max_retries": self.max_retries,
            "shutdown_timeout": self.shutdown_timeout,
        }


class PerFileProcessManager:
    """
    Manages a subprocess that is started for each file and killed afterwards.

    This is useful for server processes that have memory leaks or degrade over
    time. The process is restarted fresh for each file being processed.

    Example:
        >>> config = PerFileProcessConfig(
        ...     enabled=True,
        ...     command="python",
        ...     args=["-m", "indextts.server"],
        ...     log_dir="./logs"
        ... )
        >>> manager = PerFileProcessManager(config)
        >>> with manager.run_for_file(Path("chapter_01.txt")):
        ...     # Process the file while the server is running
        ...     pass
    """

    def __init__(self, config: PerFileProcessConfig):
        """
        Initialize the process manager.

        Args:
            config: Configuration for process management.
        """
        self.config = config
        self._process: subprocess.Popen | None = None
        self._log_file: IO | None = None
        self._current_file: Path | None = None
        self._current_log_path: Path | None = None

    def _get_log_path(self, file_path: Path, output_path: Path | None = None) -> Path:
        """
        Get the log file path for a given input file.

        Args:
            file_path: The input file being processed.
            output_path: The output file path. If provided and log_dir is None,
                        logs will be placed at the same level as the output directory.

        Returns:
            Path to the log file (same name as input, but .log extension).
        """
        # Determine log directory
        if self.config.log_dir is not None:
            log_dir = Path(self.config.log_dir)
        elif output_path is not None:
            # Place logs at the same level as the output directory
            # e.g., output: ~/audiobooks/foo/chapters_mp3/file.mp3
            #       logs:   ~/audiobooks/foo/logs/
            output_dir = output_path.parent
            parent_dir = output_dir.parent

            # Safety check: don't try to create logs in root or other system directories
            # If parent is root (/) or a system directory, fall back to safe default
            if parent_dir == parent_dir.parent or str(parent_dir) in ["/", "/tmp", "/var"]:
                # Fall back to creating logs alongside the output file
                log_dir = output_dir / "process_logs"
            else:
                log_dir = parent_dir / "logs"
        else:
            # Fallback to current directory
            log_dir = Path("./process_logs")

        log_dir.mkdir(parents=True, exist_ok=True)

        # Replace the extension with .log
        log_name = file_path.stem + ".log"
        return log_dir / log_name

    def _start_process(self, file_path: Path, output_path: Path | None = None) -> None:
        """
        Start the configured process.

        Args:
            file_path: The file being processed (used for log naming).
            output_path: The output file path (used for log directory placement).

        Raises:
            RuntimeError: If process fails to start.
        """
        if self._process is not None:
            raise RuntimeError("Process is already running")

        log_path = self._get_log_path(file_path, output_path)
        self._current_log_path = log_path

        # Open log file for writing (truncate if exists)
        # Use line buffering (buffering=1) for real-time tailing
        self._log_file = open(  # noqa: SIM115
            log_path, "w", encoding="utf-8", buffering=1
        )

        # Build the command
        cmd = [self.config.command] + (self.config.args or [])

        try:
            # Start the process in a new process group, redirecting stdout and stderr to log file
            # This allows us to kill the entire process tree (parent + all children)
            self._process = subprocess.Popen(
                cmd,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                bufsize=0,  # Unbuffered for immediate flushing
                start_new_session=True,  # Create new process group
            )
            self._current_file = file_path

        except Exception as e:
            # Clean up log file if process fails to start
            if self._log_file:
                self._log_file.close()
                self._log_file = None
            raise RuntimeError(f"Failed to start process '{' '.join(cmd)}': {e}") from e

    def _stop_process(self) -> None:
        """
        Stop the running process gracefully, then forcefully if needed.

        Sends SIGTERM to the entire process group first, waits for shutdown_timeout,
        then sends SIGKILL to the process group if still running (retrying if needed).
        This ensures all child processes are killed too.
        """
        if self._process is None:
            return

        try:
            # First, try graceful shutdown with SIGTERM to the entire process group
            if self._process.poll() is None:  # Still running
                pgid = None
                try:
                    pgid = os.getpgid(self._process.pid)
                    # Send SIGTERM to the entire process group (graceful shutdown)
                    os.killpg(pgid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    # Process already died or we can't access it, fall back to terminate
                    self._process.terminate()

                # Wait for graceful shutdown
                try:
                    self._process.wait(timeout=self.config.shutdown_timeout)
                except subprocess.TimeoutExpired:
                    # Process didn't stop gracefully, force kill the process group
                    # Retry SIGKILL a few times to ensure processes actually die
                    max_kill_attempts = 3
                    for attempt in range(max_kill_attempts):
                        if self._process.poll() is not None:
                            # Process is dead
                            break

                        try:
                            if pgid is not None:
                                os.killpg(pgid, signal.SIGKILL)
                            else:
                                self._process.kill()
                        except (ProcessLookupError, PermissionError):
                            # Process already died
                            break

                        # Wait a bit for SIGKILL to take effect
                        try:
                            self._process.wait(timeout=0.5)
                            break  # Process died
                        except subprocess.TimeoutExpired:
                            if attempt < max_kill_attempts - 1:
                                continue  # Retry SIGKILL
                            else:
                                # Final wait
                                self._process.wait(timeout=1.0)

        finally:
            # Close log file
            if self._log_file:
                self._log_file.close()
                self._log_file = None

            self._process = None
            self._current_file = None
            self._current_log_path = None

    def is_running(self) -> bool:
        """
        Check if the managed process is currently running.

        Returns:
            True if process is running, False otherwise.
        """
        if self._process is None:
            return False

        return self._process.poll() is None

    def get_log_tail(
        self, num_lines: int | None = None
    ) -> tuple[Path | None, list[str]]:
        """
        Get the last N lines from the current log file, or all lines if num_lines is None.

        Args:
            num_lines: Number of lines to read from the end of the log.
                      If None, returns all lines.

        Returns:
            Tuple of (log_path, lines) where lines is a list of lines.
            Returns (None, []) if no log file exists.
        """
        if self._current_log_path is None or not self._current_log_path.exists():
            return (None, [])

        try:
            with open(self._current_log_path, encoding="utf-8") as f:
                lines = f.readlines()
                # Get last N lines, or all lines if num_lines is None
                if num_lines is None:
                    tail_lines = lines
                else:
                    tail_lines = lines[-num_lines:] if len(lines) > num_lines else lines
                return (self._current_log_path, tail_lines)
        except Exception:
            return (self._current_log_path, [])

    @contextmanager
    def run_for_file(self, file_path: Path, output_path: Path | None = None):
        """
        Context manager to run the process for a specific file.

        The process is started when entering the context and stopped when exiting.

        Args:
            file_path: The file being processed.
            output_path: The output file path (used for log directory placement).

        Yields:
            None

        Example:
            >>> with manager.run_for_file(Path("chapter_01.txt"), Path("output/chapter_01.mp3")):
            ...     # Process the file
            ...     pass
        """
        if not self.config.enabled:
            # If disabled, just yield without starting any process
            yield
            return

        try:
            self._start_process(file_path, output_path)
            yield
        finally:
            self._stop_process()

    def __repr__(self) -> str:
        """String representation of the process manager."""
        if not self.config.enabled:
            return "PerFileProcessManager(disabled)"

        cmd = f"{self.config.command} {' '.join(self.config.args or [])}"
        return f"PerFileProcessManager(command='{cmd}', enabled=True)"
