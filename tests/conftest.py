"""Shared test fixtures."""

import pytest
import numpy as np

from tts_bench.engines.base import TTSResult


@pytest.fixture
def sample_tts_result():
    return TTSResult(
        audio=np.random.randn(48000).astype(np.float32),
        sample_rate=24000,
        duration_seconds=2.0,
        inference_time_seconds=0.5,
        real_time_factor=0.25,
        engine_name="test",
        text="This is a test sentence.",
    )
