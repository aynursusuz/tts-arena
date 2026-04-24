"""TTS engine adapters."""

import importlib
import pkgutil

from tts_bench.engines.base import TTSEngine, TTSResult
from tts_bench.engines.registry import ENGINE_REGISTRY, get_engine, list_engines
import tts_bench.engines as _engines_pkg

for _importer, _modname, _ispkg in pkgutil.iter_modules(_engines_pkg.__path__):
    if _modname not in ("base", "registry"):
        try:
            importlib.import_module(f"tts_bench.engines.{_modname}")
        except ImportError:
            pass

__all__ = ["TTSEngine", "TTSResult", "ENGINE_REGISTRY", "get_engine", "list_engines"]
