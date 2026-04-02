<p align="center">
  <h1 align="center">TTS Arena</h1>
  <p align="center">
    A unified toolkit for benchmarking and running inference across <b>all</b> open-source TTS models.
  </p>
</p>

<p align="center">
  <a href="https://github.com/aynursusuz/tts-arena/actions/workflows/ci.yml"><img src="https://github.com/aynursusuz/tts-arena/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
</p>

---

## Why TTS Arena?

The open-source TTS landscape has exploded — there are now **40+ models** across different architectures, languages, and capabilities. TTS Arena provides:

- **One unified interface** to run any TTS model with the same 3 lines of code
- **Fair benchmarks** — same text, same hardware, same metrics
- **Easy comparison** — latency, RTF, VRAM usage, audio quality side-by-side
- **Step-by-step installation** — only install the engines you need

## Benchmark Results

> Results measured on NVIDIA H100 PCIe (81GB) with CUDA 12.2, Python 3.12

| Engine | RTF | Inference (s) | VRAM (MB) | Sample Rate | Voice Clone | Languages | Status |
|--------|-----|---------------|-----------|-------------|-------------|-----------|--------|
| *Testing in progress...* | | | | | | | |

## Supported Engines

| # | Engine | Params | Architecture | License | Status |
|---|--------|--------|-------------|---------|--------|
| 1 | Kokoro | 82M | StyleTTS-inspired | Apache 2.0 | Testing |
| 2 | Dia | 1.6B | DAC + Transformer | Apache 2.0 | Planned |
| 3 | Orpheus | 150M-3B | Llama-based | Apache 2.0 | Planned |
| 4 | F5-TTS | 330M | Flow Matching + DiT | CC-BY-NC / MIT | Planned |
| 5 | Fish Speech | 500M | VQGAN + LLM | Apache 2.0 | Planned |
| 6 | CosyVoice | 0.5B | LLM + Flow Matching | Apache 2.0 | Planned |
| 7 | GPT-SoVITS | 300M+ | GPT + SoVITS | MIT | Planned |
| 8 | ChatTTS | - | Transformer | CC-BY-NC | Planned |
| 9 | Bark | 1B | GPT-style | MIT | Planned |
| 10 | XTTS v2 | 450M | GPT + VITS | MPL 2.0 | Planned |
| 11 | StyleTTS 2 | 150-200M | Style Diffusion | MIT | Planned |
| 12 | Parler-TTS | 880M | DAC + Transformer | Apache 2.0 | Planned |
| 13 | Piper | 15-60M | VITS (ONNX) | GPL 3.0 | Planned |
| 14 | Zonos | 1.6B | SSM Hybrid | Apache 2.0 | Planned |
| 15 | Spark-TTS | 0.5B | Qwen2.5 LLM | Apache 2.0 | Planned |
| 16 | Qwen3-TTS | 0.6-1.7B | LLM | Apache 2.0 | Planned |
| 17 | Sesame CSM | 1B | Llama + Audio | Apache 2.0 | Planned |
| 18 | Chatterbox | 350M-1B | Multi-variant | MIT | Planned |
| 19 | OpenVoice | 50-100M | Voice Cloning | MIT | Planned |
| 20 | OuteTTS | 0.5-1B | Pure LLM | Apache 2.0 | Planned |
| ... | *and 20+ more* | | | | Planned |

## Installation

```bash
# Base installation
pip install tts-arena

# Or install from source
git clone https://github.com/aynursusuz/tts-arena.git
cd tts-arena
pip install -e .

# Install specific engines (only install what you need)
pip install tts-arena[kokoro]         # Kokoro (82M, lightweight)
pip install tts-arena[dia]            # Dia (1.6B, dialogue)
pip install tts-arena[f5tts]          # F5-TTS (flow matching)
pip install tts-arena[all]            # All engines
```

## Quickstart

### Python API

```python
from tts_arena.engines import get_engine

# Load any engine with 2 lines
engine = get_engine("kokoro")
engine.ensure_loaded()

# Synthesize speech
result = engine.synthesize("Hello! Welcome to TTS Arena.")
print(f"Duration: {result.duration_seconds:.2f}s, RTF: {result.real_time_factor:.3f}")

# Save to file
engine.synthesize_to_file("Hello world!", "output.wav")
```

### CLI

```bash
# List all available engines
tts-arena list-engines

# Synthesize with a specific engine
tts-arena synthesize "Hello world!" --engine kokoro --output hello.wav

# Run benchmarks
tts-arena benchmark --engine kokoro --engine dia --engine f5tts
```

## Project Structure

```
tts-arena/
├── src/tts_arena/
│   ├── engines/          # TTS engine adapters (one file per model)
│   │   ├── base.py       # Abstract TTSEngine class
│   │   ├── registry.py   # Engine discovery & instantiation
│   │   ├── kokoro.py     # Kokoro adapter
│   │   └── ...           # More engines added incrementally
│   ├── benchmarks/       # Benchmark runner & metrics
│   ├── datasets/         # Standard test sentences (multi-language)
│   ├── reporting/        # Results formatting
│   └── cli.py            # Typer CLI
├── benchmarks/results/   # Stored benchmark data
├── tests/                # Unit & integration tests
├── examples/             # Usage examples
└── docs/                 # Documentation
```

## Adding a New Engine

```python
from tts_arena.engines.base import TTSEngine, TTSResult
from tts_arena.engines.registry import register_engine

@register_engine
class MyEngine(TTSEngine):
    name = "my-engine"
    description = "My custom TTS engine"
    languages = ["en"]

    def load_model(self):
        # Load your model here
        self.model = ...
        self._loaded = True

    def synthesize(self, text, **kwargs):
        self.ensure_loaded()
        # Run inference and return TTSResult
        ...
```

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.

Individual TTS models may have their own licenses. Check each engine's documentation for details.
