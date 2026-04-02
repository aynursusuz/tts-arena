"""Kokoro TTS engine adapter — lightweight 82M param model with excellent quality."""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from tts_arena.engines.base import TTSEngine, TTSResult
from tts_arena.engines.registry import register_engine

# Voices available in Kokoro
KOKORO_VOICES = {
    # American English
    "af_heart": "American Female - Heart (default)",
    "af_bella": "American Female - Bella",
    "af_nicole": "American Female - Nicole",
    "af_aoede": "American Female - Aoede",
    "af_kore": "American Female - Kore",
    "af_sarah": "American Female - Sarah",
    "af_nova": "American Female - Nova",
    "af_sky": "American Female - Sky",
    "af_river": "American Female - River",
    "am_adam": "American Male - Adam",
    "am_michael": "American Male - Michael",
    "am_echo": "American Male - Echo",
    "am_eric": "American Male - Eric",
    "am_liam": "American Male - Liam",
    # British English
    "bf_emma": "British Female - Emma",
    "bf_isabella": "British Female - Isabella",
    "bm_george": "British Male - George",
    "bm_lewis": "British Male - Lewis",
    "bm_daniel": "British Male - Daniel",
}

# Language codes for Kokoro
LANG_CODES = {
    "en-us": "a",
    "en-gb": "b",
    "ja": "j",
    "zh": "z",
    "fr": "f",
    "ko": "k",
    "hi": "h",
    "it": "i",
    "pt-br": "p",
    "es": "e",
}


@register_engine
class KokoroEngine(TTSEngine):
    name = "kokoro"
    description = "Kokoro 82M — lightweight yet high-quality TTS with multiple voices"
    url = "https://github.com/hexgrad/kokoro"
    license = "Apache 2.0"
    languages = ["en-us", "en-gb", "ja", "zh", "fr", "ko", "hi", "it", "pt-br", "es"]
    supports_voice_cloning = False
    supports_streaming = True
    supports_emotion_control = False
    default_sample_rate = 24000

    def __init__(self, device: str = "auto", lang_code: str = "a", **kwargs: Any) -> None:
        super().__init__(device=device, **kwargs)
        self.lang_code = lang_code
        self.pipeline = None

    def load_model(self) -> None:
        os.environ.setdefault("HF_HOME", "/mnt/aynur/hf_cache")
        from kokoro import KPipeline

        self.pipeline = KPipeline(
            lang_code=self.lang_code,
            repo_id="hexgrad/Kokoro-82M",
            device=self.device,
        )
        self._loaded = True

    def synthesize(self, text: str, **kwargs: Any) -> TTSResult:
        self.ensure_loaded()
        voice = kwargs.get("voice", "af_heart")
        speed = kwargs.get("speed", 1.0)

        start = time.perf_counter()

        # Collect all chunks into one audio array
        audio_chunks = []
        for result in self.pipeline(text, voice=voice, speed=speed):
            audio_chunks.append(result.audio)

        if audio_chunks:
            audio = np.concatenate(audio_chunks)
        else:
            audio = np.zeros(0, dtype=np.float32)

        elapsed = time.perf_counter() - start
        duration = len(audio) / self.default_sample_rate

        return TTSResult(
            audio=audio,
            sample_rate=self.default_sample_rate,
            duration_seconds=duration,
            inference_time_seconds=elapsed,
            real_time_factor=elapsed / duration if duration > 0 else float("inf"),
            engine_name=self.name,
            text=text,
            metadata={
                "voice": voice,
                "speed": speed,
                "lang_code": self.lang_code,
                "model": "Kokoro-82M",
                "params": "82M",
            },
        )

    @staticmethod
    def list_voices() -> dict[str, str]:
        return KOKORO_VOICES.copy()
