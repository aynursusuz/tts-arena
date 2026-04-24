"""TTS engine adapters."""

import importlib
import pkgutil
from contextlib import suppress

import unitts.engines as _engines_pkg
from unitts.engines.base import TTSEngine, TTSResult
from unitts.engines.registry import ENGINE_REGISTRY, get_engine, list_engines

for _importer, _modname, _ispkg in pkgutil.iter_modules(_engines_pkg.__path__):
    if _modname not in ("base", "registry"):
        with suppress(ImportError):
            importlib.import_module(f"unitts.engines.{_modname}")

__all__ = ["ENGINE_REGISTRY", "TTSEngine", "TTSResult", "get_engine", "list_engines"]
