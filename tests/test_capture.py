from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from loliland_agent.perception.sources import ScreenSource


class WindowProbe:
    def __init__(self):
        self.focus_calls = 0
        self.lose_focus = False
        self.bounds_value = (10, 20, 200, 100)
        self.pid = 42
        self.scale = 1.5

    def focused(self, hwnd):
        self.focus_calls += 1
        return not (self.lose_focus and self.focus_calls > 1)

    def bounds(self, hwnd):
        return self.bounds_value

    def process_id(self, hwnd):
        return self.pid

    def title(self, hwnd):
        return "Minecraft test"

    def dpi_scale(self, hwnd):
        return self.scale

    def thread_dpi_awareness(self):
        pass


class Capture:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def grab(self, bounds):
        assert bounds == {"left": 10, "top": 20, "width": 200, "height": 100}
        return SimpleNamespace(size=(200, 100), rgb=b"\x22\x33\x44" * 200 * 100)


async def test_capture_physical_bounds_and_dpi(tmp_path, monkeypatch):
    monkeypatch.setattr("loliland_agent.perception.sources.mss.mss", Capture)
    observation = await ScreenSource(123, WindowProbe(), tmp_path).observe()
    assert (
        observation.source == "screen"
        and observation.window_process_id == 42
        and observation.dpi_scale == 1.5
    )
    assert observation.visual.scene == "unknown" and observation.focused
    with Image.open(observation.image_path) as image:
        assert image.size == (200, 100) and image.getpixel((0, 0)) == (0x22, 0x33, 0x44)


async def test_focus_loss_discards_frame(tmp_path, monkeypatch):
    monkeypatch.setattr("loliland_agent.perception.sources.mss.mss", Capture)
    api = WindowProbe()
    api.lose_focus = True
    with pytest.raises(RuntimeError):
        await ScreenSource(123, api, tmp_path).observe()
    assert not list(tmp_path.glob("*.png"))


@pytest.mark.parametrize("change", ["pid", "resolution", "dpi"])
async def test_identity_resize_dpi_change_requires_restart(tmp_path, monkeypatch, change):
    monkeypatch.setattr("loliland_agent.perception.sources.mss.mss", Capture)
    api = WindowProbe()
    source = ScreenSource(123, api, tmp_path)
    await source.observe()
    if change == "pid":
        api.pid = 99
    elif change == "dpi":
        api.scale = 2.0
    else:
        api.bounds_value = (10, 20, 400, 200)
    with pytest.raises(RuntimeError):
        await source.observe()
