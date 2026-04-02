"""Dia TTS engine adapter — multi-speaker dialogue generation with nonverbal cues."""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from tts_arena.engines.base import TTSEngine, TTSResult
from tts_arena.engines.registry import register_engine

# Nonverbal cues supported by Dia
DIA_NONVERBAL_CUES = [
    "(laughs)", "(clears throat)", "(sighs)", "(gasps)", "(coughs)",
    "(singing)", "(sings)", "(mumbles)", "(beep)", "(groans)",
    "(sniffs)", "(claps)", "(screams)", "(inhales)", "(exhales)",
    "(applause)", "(burps)", "(humming)", "(sneezes)", "(chuckle)",
    "(whistles)",
]


@register_engine
class DiaEngine(TTSEngine):
    name = "dia"
    description = "Dia 1.6B — multi-speaker dialogue generation with nonverbal cues"
    url = "https://github.com/nari-labs/dia"
    license = "Apache 2.0"
    languages = ["en"]
    supports_voice_cloning = True
    supports_streaming = False
    supports_emotion_control = True
    default_sample_rate = 44100

    def __init__(
        self,
        device: str = "auto",
        model_id: str = "nari-labs/Dia-1.6B-0626",
        use_transformers: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, **kwargs)
        self.model_id = model_id
        self.use_transformers = use_transformers
        self.processor = None

    def load_model(self) -> None:
        os.environ.setdefault("HF_HOME", "/mnt/aynur/hf_cache")

        if self.use_transformers:
            self._load_transformers()
        else:
            self._load_native()

        self._loaded = True

    def _load_transformers(self) -> None:
        """Load via Hugging Face transformers (recommended)."""
        from transformers import AutoProcessor, DiaForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = DiaForConditionalGeneration.from_pretrained(
            self.model_id,
        ).to(self.device)

    def _load_native(self) -> None:
        """Load via the dia Python package directly."""
        from dia.model import Dia

        self.model = Dia.from_pretrained(self.model_id, device=self.device)

    def synthesize(self, text: str, **kwargs: Any) -> TTSResult:
        self.ensure_loaded()

        # Generation parameters
        max_new_tokens = kwargs.get("max_new_tokens", 3072)
        guidance_scale = kwargs.get("guidance_scale", 3.0)
        temperature = kwargs.get("temperature", 1.8)
        top_p = kwargs.get("top_p", 0.90)
        top_k = kwargs.get("top_k", 45)

        # Ensure text starts with a speaker tag
        if not text.strip().startswith("[S"):
            text = f"[S1] {text}"

        start = time.perf_counter()

        if self.use_transformers:
            audio = self._synthesize_transformers(
                text,
                audio_prompt=kwargs.get("audio_prompt"),
                max_new_tokens=max_new_tokens,
                guidance_scale=guidance_scale,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        else:
            audio = self._synthesize_native(text)

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
                "model": self.model_id,
                "params": "1.6B",
                "backend": "transformers" if self.use_transformers else "native",
                "guidance_scale": guidance_scale,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            },
        )

    def _synthesize_transformers(
        self,
        text: str,
        *,
        audio_prompt: np.ndarray | None = None,
        max_new_tokens: int = 3072,
        guidance_scale: float = 3.0,
        temperature: float = 1.8,
        top_p: float = 0.90,
        top_k: int = 45,
    ) -> np.ndarray:
        """Run inference via the transformers pipeline."""
        import torch

        text_input = [text]

        if audio_prompt is not None:
            inputs = self.processor(
                text=text_input,
                audio=audio_prompt,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            prompt_len = self.processor.get_audio_prompt_len(
                inputs["decoder_attention_mask"],
            )
        else:
            inputs = self.processor(
                text=text_input,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            prompt_len = None

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                guidance_scale=guidance_scale,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

        if prompt_len is not None:
            decoded = self.processor.batch_decode(outputs, audio_prompt_len=prompt_len)
        else:
            decoded = self.processor.batch_decode(outputs)

        # decoded is a list of numpy arrays (one per batch item)
        audio = decoded[0]
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        return audio

    def _synthesize_native(self, text: str) -> np.ndarray:
        """Run inference via the dia package directly."""
        output = self.model.generate(text)
        if not isinstance(output, np.ndarray):
            output = np.array(output, dtype=np.float32)
        return output

    @staticmethod
    def list_nonverbal_cues() -> list[str]:
        """Return supported nonverbal cue tags."""
        return DIA_NONVERBAL_CUES.copy()
