# tts-bench

A single Python interface to open-source text-to-speech models. Install once, call any supported engine the same way.

```python
from tts_bench.engines import get_engine

engine = get_engine("chatterbox")
engine.synthesize_to_file("Hello world!", "out.wav")
```

## Install

```bash
pip install -e ".[chatterbox]"
```

Use `".[dev]"` for tests and linting. Each engine has its own extra; install only the ones you need.

On setuptools 80 or newer, Chatterbox also needs `pip install "setuptools<80"` in the same environment (the upstream `perth` dependency imports `pkg_resources`).

## CLI

```bash
tts-bench list-engines
tts-bench synthesize "Hello world!" --engine chatterbox --output out.wav
tts-bench benchmark --engine chatterbox
```

`benchmark` writes `benchmarks/results/<engine>.json` and `benchmarks/audio_samples/<engine>.wav`.

## Engines

| Engine | Voice cloning | License | Status |
|--------|---------------|---------|--------|
| [Chatterbox](https://github.com/resemble-ai/chatterbox) | yes | MIT | integrated |

More engines will land one at a time. See the list of planned additions in open issues.

## Benchmark

| Engine | RTF | Inference (s) | Audio (s) | VRAM (MB) | Sample rate |
|--------|-----|---------------|-----------|-----------|-------------|
| Chatterbox | 0.437 | 5.90 | 13.52 | 3,107 | 24,000 |

RTF is real-time factor (inference time / audio duration; lower is faster). Full JSON at [`benchmarks/results/chatterbox.json`](benchmarks/results/chatterbox.json), audio at [`benchmarks/audio_samples/chatterbox.wav`](benchmarks/audio_samples/chatterbox.wav).

## Adding an engine

1. Add `src/tts_bench/engines/<name>_engine.py`
2. Subclass `TTSEngine`, implement `load_model()` and `synthesize()`, decorate with `@register_engine`
3. Add the pip package as an optional dependency in `pyproject.toml`
4. Add a unit test under `tests/unit/`

```python
from tts_bench.engines.base import TTSEngine
from tts_bench.engines.registry import register_engine

@register_engine
class MyEngine(TTSEngine):
    name = "my-engine"
    # implement load_model() and synthesize()
```

## License

Apache 2.0. See [LICENSE](LICENSE). Each integrated model keeps its own upstream license.
