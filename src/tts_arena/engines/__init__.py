"""TTS engine adapters — unified interface for all TTS models."""

from tts_arena.engines.base import TTSEngine, TTSResult
from tts_arena.engines.registry import ENGINE_REGISTRY, get_engine, list_engines

# Auto-import all engine modules to trigger @register_engine decorators
import importlib
import pkgutil
import tts_arena.engines as _engines_pkg

for _importer, _modname, _ispkg in pkgutil.iter_modules(_engines_pkg.__path__):
    if _modname not in ("base", "registry"):
        try:
            importlib.import_module(f"tts_arena.engines.{_modname}")
        except ImportError:
            pass  # Engine dependency not installed, skip silently

__all__ = ["TTSEngine", "TTSResult", "ENGINE_REGISTRY", "get_engine", "list_engines"]
