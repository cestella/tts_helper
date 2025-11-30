.PHONY: help format lint typecheck test check all clean install install-dev

# Use venv executables
VENV_BIN = venv/bin
PYTHON = $(VENV_BIN)/python
BLACK = $(VENV_BIN)/black
ISORT = $(VENV_BIN)/isort
RUFF = $(VENV_BIN)/ruff
MYPY = $(VENV_BIN)/mypy
PYTEST = $(VENV_BIN)/pytest

help:
	@echo "TTS Helper - Development Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  format       Format code with black and isort"
	@echo "  lint         Lint code with ruff"
	@echo "  typecheck    Type check code with mypy (optional)"
	@echo "  test         Run tests with pytest"
	@echo "  check        Run format check, lint, and tests"
	@echo "  all          Format, lint, and test"
	@echo "  clean        Remove build artifacts and cache files"
	@echo "  install      Install package in production mode"
	@echo "  install-dev  Install package with dev dependencies"

format:
	@echo "Formatting code with black..."
	$(BLACK) tts_helper/ tests/ book_extractor/
	@echo "Sorting imports with isort..."
	$(ISORT) tts_helper/ tests/ book_extractor/
	@echo "Formatting complete!"

lint:
	@echo "Linting with ruff..."
	$(RUFF) check tts_helper/ tests/ book_extractor/
	@echo "Type checking with mypy..."
	$(MYPY) tts_helper/ book_extractor/ tests/
	@echo "Linting complete!"

test:
	@echo "Running tests with pytest..."
	$(PYTEST) tests/ -v
	@echo "Tests complete!"

check:
	@echo "Running all checks..."
	@echo ""
	@echo "1. Checking code format..."
	$(BLACK) --check tts_helper/ tests/ book_extractor/
	$(ISORT) --check tts_helper/ tests/ book_extractor/
	@echo ""
	@echo "2. Linting..."
	$(RUFF) check tts_helper/ tests/ book_extractor/
	$(MYPY) tts_helper/ book_extractor/ tests/
	@echo ""
	@echo "3. Running tests..."
	$(PYTEST) tests/ -v
	@echo ""
	@echo "All checks passed!"

all: format lint test

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "Clean complete!"

install:
	@echo "Installing tts-helper..."
	pip install -e .
	@echo "Installation complete!"

install-dev:
	@echo "Installing tts-helper with dev dependencies..."
	pip install -e ".[dev]"
	@echo "Dev installation complete!"
