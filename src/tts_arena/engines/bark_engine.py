"""Bark TTS engine adapter — GPT-style generative audio model by Suno."""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from tts_arena.engines.base import TTSEngine, TTSResult
from tts_arena.engines.registry import register_engine

# Bark speaker presets (v2 history prompts).
# Keys are the prompt identifiers used by bark.generate_audio(history_prompt=...).
# See https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683 for full list.
BARK_SPEAKER_PRESETS = {
    "v2/en_speaker_0": "English Speaker 0 (male)",
    "v2/en_speaker_1": "English Speaker 1 (male)",
    "v2/en_speaker_2": "English Speaker 2 (male)",
    "v2/en_speaker_3": "English Speaker 3 (male)",
    "v2/en_speaker_4": "English Speaker 4 (male)",
    "v2/en_speaker_5": "English Speaker 5 (male)",
    "v2/en_speaker_6": "English Speaker 6 (female)",
    "v2/en_speaker_7": "English Speaker 7 (female)",
    "v2/en_speaker_8": "English Speaker 8 (female)",
    "v2/en_speaker_9": "English Speaker 9 (female)",
    "v2/zh_speaker_0": "Chinese Speaker 0",
    "v2/zh_speaker_1": "Chinese Speaker 1",
    "v2/fr_speaker_0": "French Speaker 0",
    "v2/fr_speaker_1": "French Speaker 1",
    "v2/de_speaker_0": "German Speaker 0",
    "v2/de_speaker_1": "German Speaker 1",
    "v2/hi_speaker_0": "Hindi Speaker 0",
    "v2/hi_speaker_1": "Hindi Speaker 1",
    "v2/it_speaker_0": "Italian Speaker 0",
    "v2/it_speaker_1": "Italian Speaker 1",
    "v2/ja_speaker_0": "Japanese Speaker 0",
    "v2/ja_speaker_1": "Japanese Speaker 1",
    "v2/ko_speaker_0": "Korean Speaker 0",
    "v2/ko_speaker_1": "Korean Speaker 1",
    "v2/pl_speaker_0": "Polish Speaker 0",
    "v2/pl_speaker_1": "Polish Speaker 1",
    "v2/pt_speaker_0": "Portuguese Speaker 0",
    "v2/pt_speaker_1": "Portuguese Speaker 1",
    "v2/ru_speaker_0": "Russian Speaker 0",
    "v2/ru_speaker_1": "Russian Speaker 1",
    "v2/es_speaker_0": "Spanish Speaker 0",
    "v2/es_speaker_1": "Spanish Speaker 1",
    "v2/tr_speaker_0": "Turkish Speaker 0",
    "v2/tr_speaker_1": "Turkish Speaker 1",
}


@register_engine
class BarkEngine(TTSEngine):
    name = "bark"
    description = (
        "Bark — GPT-style generative audio model with speech, music, and sound effects"
    )
    url = "https://github.com/suno-ai/bark"
    license = "MIT"
    languages = [
        "en", "zh", "fr", "de", "hi", "it", "ja", "ko", "pl", "pt", "ru", "es", "tr",
    ]
    supports_voice_cloning = True  # via speaker history prompts
    supports_streaming = False
    supports_emotion_control = False
    default_sample_rate = 24_000

    def __init__(self, device: str = "auto", use_small_models: bool = False, **kwargs: Any) -> None:
        super().__init__(device=device, **kwargs)
        self.use_small_models = use_small_models

    def load_model(self) -> None:
        # Ensure cache dirs point to the shared HF cache location.
        os.environ.setdefault("HF_HOME", "/mnt/aynur/hf_cache")
        os.environ.setdefault("XDG_CACHE_HOME", "/mnt/aynur/hf_cache")

        # Bark respects these env vars for GPU / small-model selection.
        if self.device == "cpu":
            os.environ["SUNO_OFFLOAD_CPU"] = "True"
        if self.use_small_models:
            os.environ["SUNO_USE_SMALL_MODELS"] = "True"

        from bark import preload_models  # noqa: E402

        preload_models()
        self._loaded = True

    def synthesize(self, text: str, **kwargs: Any) -> TTSResult:
        self.ensure_loaded()

        from bark import SAMPLE_RATE, generate_audio  # noqa: E402

        history_prompt: str | None = kwargs.get("voice") or kwargs.get("history_prompt")
        text_temp: float = kwargs.get("text_temp", 0.7)
        waveform_temp: float = kwargs.get("waveform_temp", 0.7)

        start = time.perf_counter()

        audio_array = generate_audio(
            text,
            history_prompt=history_prompt,
            text_temp=text_temp,
            waveform_temp=waveform_temp,
        )

        elapsed = time.perf_counter() - start

        # Bark returns a numpy float32 array at SAMPLE_RATE (24 000 Hz).
        audio = np.asarray(audio_array, dtype=np.float32)
        duration = len(audio) / SAMPLE_RATE

        return TTSResult(
            audio=audio,
            sample_rate=SAMPLE_RATE,
            duration_seconds=duration,
            inference_time_seconds=elapsed,
            real_time_factor=elapsed / duration if duration > 0 else float("inf"),
            engine_name=self.name,
            text=text,
            metadata={
                "history_prompt": history_prompt,
                "text_temp": text_temp,
                "waveform_temp": waveform_temp,
                "model": "bark",
                "small_models": self.use_small_models,
            },
        )

    @staticmethod
    def list_voices() -> dict[str, str]:
        """Return available Bark speaker presets."""
        return BARK_SPEAKER_PRESETS.copy()
