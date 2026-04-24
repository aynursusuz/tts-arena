"""Engine registry."""

from __future__ import annotations

from typing import Any

from tts_bench.engines.base import TTSEngine

ENGINE_REGISTRY: dict[str, type[TTSEngine]] = {}


def register_engine(cls: type[TTSEngine]) -> type[TTSEngine]:
    """Decorator to register a TTS engine class."""
    ENGINE_REGISTRY[cls.name] = cls
    return cls


def get_engine(name: str, **kwargs: Any) -> TTSEngine:
    """Instantiate a TTS engine by name."""
    if name not in ENGINE_REGISTRY:
        available = ", ".join(sorted(ENGINE_REGISTRY.keys()))
        raise ValueError(f"Unknown engine {name!r}. Available: {available}")
    return ENGINE_REGISTRY[name](**kwargs)


def list_engines() -> list[dict[str, Any]]:
    """List all registered engines with their metadata."""
    engines = []
    for name, cls in sorted(ENGINE_REGISTRY.items()):
        engines.append(
            {
                "name": name,
                "description": cls.description,
                "url": cls.url,
                "license": cls.license,
                "languages": cls.languages,
                "voice_cloning": cls.supports_voice_cloning,
                "streaming": cls.supports_streaming,
                "emotion_control": cls.supports_emotion_control,
            }
        )
    return engines
