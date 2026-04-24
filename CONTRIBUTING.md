# Contributing

## Dev setup

```bash
git clone https://github.com/aynursusuz/tts-bench.git
cd tts-bench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Adding a TTS engine

1. Add `src/tts_bench/engines/<name>_engine.py`
2. Subclass `TTSEngine`, implement `load_model()` and `synthesize()`, decorate with `@register_engine`
3. Add the pip package to `pyproject.toml` under `[project.optional-dependencies]`
4. Add a unit test under `tests/unit/`

```python
from tts_bench.engines.base import TTSEngine
from tts_bench.engines.registry import register_engine

@register_engine
class MyEngine(TTSEngine):
    name = "my-engine"
    # implement load_model() and synthesize()
```

## Tests and lint

```bash
pytest
ruff check .
ruff format --check .
```

## PRs

One engine per PR. Include a short description, test output, and any model license notes.
