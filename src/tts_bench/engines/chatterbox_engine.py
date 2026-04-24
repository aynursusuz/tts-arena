"""Chatterbox TTS adapter."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from tts_bench.engines.base import TTSEngine, TTSResult
from tts_bench.engines.registry import register_engine


@register_engine
class ChatterboxEngine(TTSEngine):
    name = "chatterbox"
    description = "Chatterbox TTS"
    url = "https://github.com/resemble-ai/chatterbox"
    license = "MIT"
    languages = ["en"]
    supports_voice_cloning = True
    default_sample_rate = 24000

    def load_model(self) -> None:
        from chatterbox.tts import ChatterboxTTS

        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        self.default_sample_rate = getattr(self.model, "sr", self.default_sample_rate)

    def synthesize(
        self,
        text: str,
        audio_prompt_path: str | None = None,
        **kwargs: Any,
    ) -> TTSResult:
        self.ensure_loaded()

        start = time.perf_counter()
        wav = self.model.generate(text, audio_prompt_path=audio_prompt_path, **kwargs)
        elapsed = time.perf_counter() - start

        audio = wav.detach().cpu().numpy().astype(np.float32).squeeze()
        sample_rate = self.default_sample_rate
        duration = len(audio) / sample_rate

        return TTSResult(
            audio=audio,
            sample_rate=sample_rate,
            duration_seconds=duration,
            inference_time_seconds=elapsed,
            real_time_factor=elapsed / duration if duration else 0.0,
            engine_name=self.name,
            text=text,
        )
