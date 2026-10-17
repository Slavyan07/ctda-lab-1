from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from PIL import Image

from loliland_agent.memory import Memory
from loliland_agent.models import Observation
from loliland_agent.perception.testpattern import detect_test_frame

SCREENSHOTS = Path(__file__).parent / "fixtures/screenshots"


def frame(name="initial.png", source="mock"):
    path = SCREENSHOTS / name
    with Image.open(path) as image:
        state = detect_test_frame(image)
        width, height = image.size
    assert state is not None
    return Observation(
        source=source,
        window_id=123,
        window_process_id=42,
        window_title="LoliLand Agent Test Window",
        window_bounds=(0, 0, width, height),
        focused=True,
        image_path=path,
        frame_id=hashlib.sha256(path.read_bytes()).hexdigest(),
        visual=state,
        visual_at=time.monotonic(),
        visual_origin="local_test",
    )


@pytest.fixture
def observation():
    return frame()


@pytest.fixture
def memory(tmp_path):
    value = Memory(tmp_path / "memory.sqlite3")
    yield value
    value.close()


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


class FixedSource:
    def __init__(self, observation):
        self.observation = observation
        self.calls = 0

    async def observe(self):
        self.calls += 1
        return self.observation.model_copy(update={"monotonic_at": time.monotonic(), "visual_at": 0.0})
