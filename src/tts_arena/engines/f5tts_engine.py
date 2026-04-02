"""F5-TTS engine adapter — diffusion-based voice cloning TTS."""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from tts_arena.engines.base import TTSEngine, TTSResult
from tts_arena.engines.registry import register_engine


@register_engine
class F5TTSEngine(TTSEngine):
    name = "f5-tts"
    description = "F5-TTS — diffusion transformer for zero-shot voice cloning"
    url = "https://github.com/SWivid/F5-TTS"
    license = "CC-BY-NC / MIT"
    languages = ["en", "zh"]
    supports_voice_cloning = True
    supports_streaming = False
    supports_emotion_control = False
    default_sample_rate = 24000

    def __init__(self, device: str = "auto", **kwargs: Any) -> None:
        super().__init__(device=device, **kwargs)
        self.tts_model = None

    def load_model(self) -> None:
        os.environ.setdefault("HF_HOME", "/mnt/aynur/hf_cache")
        from f5_tts.api import F5TTS

        self.tts_model = F5TTS(device=self.device)
        self._loaded = True

    def synthesize(self, text: str, **kwargs: Any) -> TTSResult:
        """Synthesize speech from text, optionally cloning a reference voice.

        Args:
            text: The text to synthesize.
            **kwargs:
                reference_audio (str): Path to reference audio for voice cloning.
                reference_text (str): Transcript of the reference audio.
                speed (float): Speaking rate multiplier (default 1.0).
        """
        self.ensure_loaded()

        ref_audio = kwargs.get("reference_audio")
        ref_text = kwargs.get("reference_text", "")
        speed = kwargs.get("speed", 1.0)

        start = time.perf_counter()

        wav, sr, _ = self.tts_model.infer(
            ref_file=ref_audio or "",
            ref_text=ref_text,
            gen_text=text,
            speed=speed,
        )

        elapsed = time.perf_counter() - start

        # wav from F5-TTS is a numpy array; ensure float32
        if not isinstance(wav, np.ndarray):
            wav = np.array(wav, dtype=np.float32)
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)

        # Use returned sample rate if provided, otherwise default
        sample_rate = sr if sr else self.default_sample_rate
        duration = len(wav) / sample_rate

        return TTSResult(
            audio=wav,
            sample_rate=sample_rate,
            duration_seconds=duration,
            inference_time_seconds=elapsed,
            real_time_factor=elapsed / duration if duration > 0 else float("inf"),
            engine_name=self.name,
            text=text,
            metadata={
                "model": "F5-TTS",
                "reference_audio": ref_audio,
                "reference_text": ref_text,
                "speed": speed,
            },
        )
