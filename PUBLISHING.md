# Publishing to PyPI

This guide explains how to build and publish the `tts-helper` package to PyPI.

## Prerequisites

1. Install build tools:
```bash
pip install build twine
```

2. Create accounts on:
   - [PyPI](https://pypi.org/account/register/) (production)
   - [TestPyPI](https://test.pypi.org/account/register/) (testing)

3. Set up API tokens:
   - Generate tokens from PyPI/TestPyPI account settings
   - Store in `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR-TOKEN-HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR-TESTPYPI-TOKEN-HERE
```

## Pre-Publishing Checklist

Before publishing, ensure:

- [ ] All tests pass: `pytest tests/`
- [ ] Code is formatted: `black tts_helper tests/`
- [ ] Imports are sorted: `isort tts_helper tests/`
- [ ] Type checking passes: `mypy tts_helper/`
- [ ] Version number is updated in `pyproject.toml`
- [ ] CHANGELOG.md is updated (if exists)
- [ ] README.md is current and accurate
- [ ] All changes are committed to git
- [ ] Git tag matches version: `git tag v0.1.0`

## Building the Package

1. Clean previous builds:
```bash
rm -rf dist/ build/ *.egg-info
```

2. Build the distribution files:
```bash
python -m build
```

This creates:
- `dist/tts_helper-X.Y.Z-py3-none-any.whl` (wheel)
- `dist/tts_helper-X.Y.Z.tar.gz` (source distribution)

3. Verify the build:
```bash
ls -lh dist/
```

## Testing on TestPyPI

Always test on TestPyPI first:

1. Upload to TestPyPI:
```bash
python -m twine upload --repository testpypi dist/*
```

2. Install from TestPyPI:
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ tts-helper
```

Note: `--extra-index-url` allows installing dependencies from regular PyPI.

3. Test the installed package:
```bash
python -c "from tts_helper import SpacySegmenter; print('Success!')"
```

## Publishing to PyPI

Once testing is successful:

1. Upload to PyPI:
```bash
python -m twine upload dist/*
```

2. Verify the upload:
   - Visit: https://pypi.org/project/tts-helper/
   - Check the version, description, and metadata

3. Test installation from PyPI:
```bash
pip install tts-helper
```

4. Create and push git tag:
```bash
git tag v0.1.0
git push origin v0.1.0
```

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR** version: Incompatible API changes
- **MINOR** version: Add functionality (backwards compatible)
- **PATCH** version: Bug fixes (backwards compatible)

Examples:
- `0.1.0` - Initial development release
- `0.2.0` - Added new features
- `0.2.1` - Bug fixes
- `1.0.0` - First stable release

## Post-Publishing

1. Update version in `pyproject.toml` to next development version
2. Create release notes on GitHub
3. Announce the release (if applicable)
4. Monitor PyPI downloads and issues

## Troubleshooting

### "File already exists" error
You cannot re-upload the same version. Either:
- Increment the version number
- Delete the version on PyPI (for test releases only)

### Import errors after installation
- Check that `MANIFEST.in` includes all necessary files
- Verify `pyproject.toml` has correct package configuration
- Test in a clean virtual environment

### Missing dependencies
- Ensure `requirements.txt` lists all dependencies
- Check that `pyproject.toml` dependencies match

## Development Releases

For pre-releases, use version suffixes:
- `0.1.0a1` - Alpha release
- `0.1.0b1` - Beta release
- `0.1.0rc1` - Release candidate

These won't be installed by default (requires `--pre` flag).

## Automated Publishing (GitHub Actions)

Consider setting up GitHub Actions for automated publishing:

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI

on:
  release:
    types: [created]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install build twine
      - name: Build package
        run: python -m build
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

Add `PYPI_API_TOKEN` to your repository secrets.
