"""Run the Chatterbox engine on a benchmark sentence and save results."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tts_bench.datasets import LONG_TEXT_EN  # noqa: E402
from tts_bench.engines import get_engine  # noqa: E402

TEXT = LONG_TEXT_EN

RESULTS_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "results"
SAMPLES_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "audio_samples"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    engine = get_engine("chatterbox")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    load_start = time.perf_counter()
    engine.ensure_loaded()
    load_time = time.perf_counter() - load_start
    vram_after_load = engine.get_vram_usage_mb()

    result = engine.synthesize(TEXT)
    vram_after_inference = engine.get_vram_usage_mb()

    audio_path = SAMPLES_DIR / "chatterbox.wav"
    sf.write(str(audio_path), result.audio, result.sample_rate)

    entry = {
        "engine": "chatterbox",
        "text": TEXT,
        "duration_seconds": round(result.duration_seconds, 3),
        "inference_time_seconds": round(result.inference_time_seconds, 3),
        "real_time_factor": round(result.real_time_factor, 4),
        "load_time_seconds": round(load_time, 3),
        "sample_rate": result.sample_rate,
        "vram_after_load_mb": round(vram_after_load, 1) if vram_after_load else None,
        "vram_after_inference_mb": round(vram_after_inference, 1) if vram_after_inference else None,
        "gpu": gpu_name,
        "audio_file": str(audio_path.relative_to(Path(__file__).resolve().parents[1])),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    results_file = RESULTS_DIR / "chatterbox.json"
    with results_file.open("w") as f:
        json.dump(entry, f, indent=2)

    print(json.dumps(entry, indent=2))
    print(f"audio saved: {audio_path}")
    print(f"results saved: {results_file}")


if __name__ == "__main__":
    main()
