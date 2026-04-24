<div align="center">
<h2>tts-bench: One Python interface for open-source TTS models</h2>
<div>
    <a href="https://github.com/aynursusuz/tts-arena/actions/workflows/ci.yml" target="_blank">
        <img src="https://img.shields.io/github/actions/workflow/status/aynursusuz/tts-arena/ci.yml?branch=main&style=for-the-badge&labelColor=2D3748" alt="CI">
    </a>
    <a href="https://github.com/aynursusuz/tts-arena/blob/main/LICENSE" target="_blank">
        <img src="https://img.shields.io/badge/License-Apache_2.0-4ECDC4?style=for-the-badge&labelColor=2D3748" alt="License">
    </a>
    <a href="https://www.python.org/downloads/" target="_blank">
        <img src="https://img.shields.io/badge/Python-3.10+-45B7D1?style=for-the-badge&logo=python&logoColor=white&labelColor=2D3748" alt="Python 3.10+">
    </a>
</div>
</div>

## Overview

tts-bench gathers open-source text-to-speech models behind one Python interface. Install once and call any supported engine the same way; new engines are added one at a time.

## Installation

```bash
git clone https://github.com/aynursusuz/tts-arena.git
cd tts-arena
uv venv --python 3.12 && source .venv/bin/activate

# Base install (no engines)
uv pip install -e .

# With a specific engine
uv pip install -e ".[chatterbox]"
```

Each engine ships as its own extra. Install only the ones you need.

> Chatterbox depends on `perth`, which still imports `pkg_resources`. On setuptools 80 or newer, also run `uv pip install "setuptools<80"` in the same environment.

## Inference

```python
from tts_bench.engines import get_engine

engine = get_engine("chatterbox")
engine.synthesize_to_file("Hello world!", "out.wav")
```

Every engine exposes the same interface. Swap by changing the name:

```python
engine = get_engine("chatterbox")
# engine = get_engine("<next-engine>")
```

### CLI

```bash
tts-bench list-engines
tts-bench synthesize "Hello world!" --engine chatterbox --output out.wav
tts-bench benchmark --engine chatterbox
```

`benchmark` writes one JSON to `benchmarks/results/<engine>.json` and one WAV to `benchmarks/audio_samples/<engine>.wav`.

## Engines

| Engine | Voice cloning | License | Status |
|--------|:-------------:|---------|--------|
| [Chatterbox](https://github.com/resemble-ai/chatterbox) | yes | MIT | integrated |

## Benchmark

| Engine | RTF | Inference (s) | Audio (s) | VRAM (MB) | Sample rate |
|--------|-----|---------------|-----------|-----------|-------------|
| Chatterbox | **0.44** | 5.90 | 13.52 | 3,107 | 24,000 |

*RTF (real-time factor) = inference time / audio duration. Lower is faster.* Full results: [`benchmarks/results/chatterbox.json`](benchmarks/results/chatterbox.json). Audio sample: [`benchmarks/audio_samples/chatterbox.wav`](benchmarks/audio_samples/chatterbox.wav).

## Adding an engine

1. Add `src/tts_bench/engines/<name>_engine.py`
2. Subclass `TTSEngine`, implement `load_model()` and `synthesize()`, decorate with `@register_engine`
3. Add the package to `pyproject.toml` extras
4. Add a unit test under `tests/unit/`

## License

Apache 2.0. See [LICENSE](LICENSE). Each integrated model keeps its own upstream license.
