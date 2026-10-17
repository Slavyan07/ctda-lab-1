from __future__ import annotations

import pytest
from conftest import SCREENSHOTS, frame
from PIL import Image

from loliland_agent.perception.engine import PerceptionEngine, PerceptionError
from loliland_agent.perception.sources import save_frame
from loliland_agent.perception.testpattern import detect_test_frame
from loliland_agent.planning.providers import OfflineProvider


@pytest.mark.parametrize(
    "name,scene,slot",
    [
        ("initial.png", "test_window", 0),
        ("moved.png", "test_window", 0),
        ("inventory.png", "inventory", 0),
        ("hotbar3.png", "test_window", 2),
        ("verification.png", "verification", None),
        ("scaled.png", "test_window", 2),
    ],
)
def test_recorded_screenshots_recognized(name, scene, slot):
    with Image.open(SCREENSHOTS / name) as image:
        state = detect_test_frame(image)
    assert state.scene == scene and state.selected_slot == slot
    if scene == "verification":
        assert state.human_required


@pytest.mark.parametrize("width,height", [(640, 360), (1280, 720), (1920, 1080), (1200, 900)])
def test_normalized_landmarks_support_resolution(width, height):
    from loliland_agent.perception.testpattern import render_test_frame

    state = detect_test_frame(render_test_frame(width, height))
    points = {point.name: point for point in state.landmarks}
    assert points["player"].x == pytest.approx(0.2, abs=0.01)
    assert points["waypoint"].x == pytest.approx(0.8, abs=0.01)


async def test_unknown_image_does_not_invent_game_state(tmp_path):
    path, digest = save_frame(Image.new("RGB", (640, 360), "black"), tmp_path)
    value = frame().model_copy(update={"image_path": path, "frame_id": digest})
    result, _ = await PerceptionEngine().analyze(value, OfflineProvider())
    assert (
        result.visual.scene == "unknown"
        and result.visual.confidence == 0
        and result.visual.human_required is None
    )


async def test_vision_never_trusts_replay_metadata():
    observation = frame("verification.png")
    forged = observation.model_copy(update={"visual": frame().visual})
    result, _ = await PerceptionEngine().analyze(forged, OfflineProvider())
    assert result.visual.scene == "verification"


async def test_cache_exact_image_and_expiry():
    engine = PerceptionEngine(cache_seconds=30)
    first, _ = await engine.analyze(frame(), OfflineProvider())
    second, usage = await engine.analyze(frame(), OfflineProvider())
    assert first.visual_origin == "local_test" and second.visual_origin == "cache" and usage is None
    engine.cache_seconds = 0
    third, _ = await engine.analyze(frame(), OfflineProvider())
    assert third.visual_origin == "local_test"


async def test_frame_hash_rejects_changed_image():
    forged = frame().model_copy(update={"image_path": frame("inventory.png").image_path})
    with pytest.raises(PerceptionError):
        await PerceptionEngine().analyze(forged, OfflineProvider())


def test_frame_cache_is_immutable(tmp_path):
    a, _ = save_frame(Image.new("RGB", (200, 100), "black"), tmp_path)
    old = a.read_bytes()
    b, _ = save_frame(Image.new("RGB", (200, 100), "white"), tmp_path)
    assert a != b and a.read_bytes() == old
