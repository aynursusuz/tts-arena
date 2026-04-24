# tts-bench

A small Python toolkit for running and comparing open-source TTS models through one interface. Models are added one at a time. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Install

```bash
pip install -e ".[chatterbox]"
```

Drop the extra for the base scaffold; add `dev` for tests and lint.

> Chatterbox currently pulls `perth`, which requires `pkg_resources`. On setuptools ≥ 80 install `"setuptools<80"` in the same env.

## Quickstart: Chatterbox

```python
from tts_bench.engines import get_engine

engine = get_engine("chatterbox")
engine.synthesize_to_file("Hello world!", "out.wav")
```

CLI:

```bash
tts-bench list-engines
tts-bench synthesize "Hello world!" --engine chatterbox --output out.wav
tts-bench benchmark --engine chatterbox
```

Benchmark JSON and WAVs are written to `benchmarks/results/` and `benchmarks/audio_samples/`.

## Results

| Engine | GPU | RTF | Inference (s) | Audio (s) | VRAM (MB) | Sample rate |
|--------|-----|-----|---------------|-----------|-----------|-------------|
| Chatterbox | A100 80GB | 0.437 | 5.90 | 13.52 | 3,107 | 24 000 |

Full JSON: [`benchmarks/results/chatterbox.json`](benchmarks/results/chatterbox.json).  Audio: [`benchmarks/audio_samples/chatterbox.wav`](benchmarks/audio_samples/chatterbox.wav).

## Engines

| Engine | License | Status |
|--------|---------|--------|
| Chatterbox | MIT | Integrated |

## Structure

```
src/tts_bench/
├── engines/        # one file per model, registered via @register_engine
│   ├── base.py
│   ├── registry.py
│   └── chatterbox_engine.py
├── benchmarks/
├── datasets/
└── cli.py
```

## Adding a model

1. Create `src/tts_bench/engines/<name>_engine.py`
2. Subclass `TTSEngine`, implement `load_model()` and `synthesize()`, decorate with `@register_engine`
3. Add the package to `pyproject.toml` under `[project.optional-dependencies]`
4. Add a test under `tests/unit/`

## License

Apache 2.0. See [LICENSE](LICENSE). Individual TTS models carry their own licenses.
