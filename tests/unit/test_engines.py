"""Tests for engine registry and base class."""

from tts_bench.engines.base import TTSEngine, TTSResult
from tts_bench.engines.registry import ENGINE_REGISTRY, list_engines, register_engine

import numpy as np


def test_tts_result_rtf():
    result = TTSResult(
        audio=np.zeros(24000),
        sample_rate=24000,
        duration_seconds=1.0,
        inference_time_seconds=0.5,
        real_time_factor=0.5,
        engine_name="test",
        text="hello",
    )
    assert result.is_faster_than_realtime is True


def test_tts_result_slow():
    result = TTSResult(
        audio=np.zeros(24000),
        sample_rate=24000,
        duration_seconds=1.0,
        inference_time_seconds=2.0,
        real_time_factor=2.0,
        engine_name="test",
        text="hello",
    )
    assert result.is_faster_than_realtime is False


def test_register_engine():
    @register_engine
    class DummyEngine(TTSEngine):
        name = "dummy"
        description = "A dummy engine for testing"

        def load_model(self):
            self._loaded = True

        def synthesize(self, text, **kwargs):
            audio = np.zeros(24000)
            return TTSResult(
                audio=audio,
                sample_rate=24000,
                duration_seconds=1.0,
                inference_time_seconds=0.01,
                real_time_factor=0.01,
                engine_name=self.name,
                text=text,
            )

    assert "dummy" in ENGINE_REGISTRY
    engine = ENGINE_REGISTRY["dummy"]()
    engine.ensure_loaded()
    result = engine.synthesize("test")
    assert result.engine_name == "dummy"
    assert result.audio.shape == (24000,)

    # Cleanup
    del ENGINE_REGISTRY["dummy"]


def test_list_engines():
    engines = list_engines()
    assert isinstance(engines, list)
