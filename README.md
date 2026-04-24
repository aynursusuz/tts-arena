<div align="center">
<h2>unitts: One Python interface for open-source TTS models</h2>
<div>
    <a href="https://github.com/aynursusuz/unitts/actions/workflows/ci.yml" target="_blank">
        <img src="https://img.shields.io/github/actions/workflow/status/aynursusuz/unitts/ci.yml?branch=main&style=for-the-badge&labelColor=2D3748" alt="CI">
    </a>
    <a href="https://github.com/aynursusuz/unitts/blob/main/LICENSE" target="_blank">
        <img src="https://img.shields.io/badge/License-Apache_2.0-4ECDC4?style=for-the-badge&labelColor=2D3748" alt="License">
    </a>
    <a href="https://www.python.org/downloads/" target="_blank">
        <img src="https://img.shields.io/badge/Python-3.10+-45B7D1?style=for-the-badge&logo=python&logoColor=white&labelColor=2D3748" alt="Python 3.10+">
    </a>
</div>
</div>

## Overview

unitts gathers open-source text-to-speech models behind one Python interface. Install once and call any supported engine the same way; new engines are added one at a time.

## Installation

```bash
git clone https://github.com/aynursusuz/unitts.git
cd unitts
uv venv --python 3.12 && source .venv/bin/activate

# Base install (no engines)
uv pip install -e .

# Install only the engines you need
uv pip install -e ".[chatterbox]"
uv pip install -e ".[fish-audio]"
```

> Chatterbox depends on `perth`, which still imports `pkg_resources`. On setuptools 80 or newer, also run `uv pip install "setuptools<80"`.
>
> Fish Audio pulls `fish-speech` from its GitHub repo (not on PyPI). `fish-speech` and `chatterbox-tts` currently pin different `torch` versions, so install them in separate environments.

## Inference

```python
from unitts.engines import get_engine

engine = get_engine("chatterbox")
engine.synthesize_to_file("Hello world!", "out.wav")
```

Every engine exposes the same interface. Swap by changing the name:

```python
engine = get_engine("chatterbox")     # local, MIT
engine = get_engine("fish-audio")     # local, s2-pro weights, non-commercial
```

First call to `fish-audio` downloads the 11 GB s2-pro checkpoint from HuggingFace into the default HF cache. Set `FISH_S2_PRO_DIR` to point at an existing local copy.

### CLI

```bash
unitts list-engines
unitts synthesize "Hello world!" --engine chatterbox --output out.wav
unitts benchmark --engine chatterbox
```

`benchmark` writes one JSON to `benchmarks/results/<engine>.json` and one WAV to `benchmarks/audio_samples/<engine>.wav`.

## Engines

| Engine | Type | Voice cloning | License | Status |
|--------|------|:-------------:|---------|--------|
| [Chatterbox](https://github.com/resemble-ai/chatterbox) | local | yes | MIT | integrated |
| [Fish Audio s2-pro](https://huggingface.co/fishaudio/s2-pro) | local | yes | Fish Audio Research License | integrated |

## Benchmark

| Engine | RTF | Inference (s) | Audio (s) | VRAM (MB) | Sample rate |
|--------|-----|---------------|-----------|-----------|-------------|
| Chatterbox | **0.44** | 5.90 | 13.52 | 3,107 | 24,000 |
| Fish Audio s2-pro | 4.00 | 38.29 | 9.57 | 19,105 | 44,100 |

*RTF (real-time factor) = inference time / audio duration. Lower is faster.* Fish Audio measurements are without `--compile`; upstream documents ~5x speedup after kernel fusion. Full results: [`benchmarks/results/`](benchmarks/results/). Audio samples: [`benchmarks/audio_samples/`](benchmarks/audio_samples/).

## Adding an engine

1. Add `src/unitts/engines/<name>_engine.py`
2. Subclass `TTSEngine`, implement `load_model()` and `synthesize()`, decorate with `@register_engine`
3. Add the package to `pyproject.toml` extras
4. Add a unit test under `tests/unit/`

## License

unitts itself is Apache 2.0 (see [LICENSE](LICENSE)). Each integrated model keeps its own upstream license; by invoking an engine you agree to the terms of its model. Third-party model notices are listed in [NOTICE](NOTICE).

**Built with Fish Audio.** The `fish-audio` engine uses Fish Audio s2-pro weights under the Fish Audio Research License (non-commercial). Commercial use of that engine requires a separate license from Fish Audio.
