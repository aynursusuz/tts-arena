"""TTS engine adapters."""

import importlib
import pkgutil
from contextlib import suppress

import tts_bench.engines as _engines_pkg
from tts_bench.engines.base import TTSEngine, TTSResult
from tts_bench.engines.registry import ENGINE_REGISTRY, get_engine, list_engines

for _importer, _modname, _ispkg in pkgutil.iter_modules(_engines_pkg.__path__):
    if _modname not in ("base", "registry"):
        with suppress(ImportError):
            importlib.import_module(f"tts_bench.engines.{_modname}")

__all__ = ["ENGINE_REGISTRY", "TTSEngine", "TTSResult", "get_engine", "list_engines"]
