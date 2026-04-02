# Contributing to TTS Arena

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/aynursusuz/tts-arena.git
cd tts-arena

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode with all extras
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Adding a New TTS Engine

This is the most common contribution. Follow these steps:

1. Create a new file in `src/tts_arena/engines/` (e.g., `my_engine.py`)
2. Inherit from `TTSEngine` and implement `load_model()` and `synthesize()`
3. Use the `@register_engine` decorator
4. Add the engine's pip package to `pyproject.toml` optional dependencies
5. Add a test in `tests/unit/test_engines.py`
6. Update the README supported engines table

```python
from tts_arena.engines.base import TTSEngine, TTSResult
from tts_arena.engines.registry import register_engine

@register_engine
class MyEngine(TTSEngine):
    name = "my-engine"
    description = "My awesome TTS engine"
    # ... implement load_model() and synthesize()
```

## Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting. Pre-commit hooks handle this automatically.

## Testing

```bash
pytest                    # Run all tests
pytest tests/unit/        # Run unit tests only
pytest -k "test_kokoro"   # Run specific test
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/add-new-engine`)
3. Make your changes
4. Run tests and linting
5. Submit a PR with a clear description

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
