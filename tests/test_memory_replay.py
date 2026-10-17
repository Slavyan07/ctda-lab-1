from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import FixedSource
from PIL import Image

from loliland_agent.memory import Memory
from loliland_agent.perception.recording import RecordingSource
from loliland_agent.perception.sources import ReplayEnded, ReplaySource


def test_memory_survives_restart_and_never_auto_resumes(tmp_path):
    path = tmp_path / "agent.db"
    memory = Memory(path)
    goal = memory.goal("Исследовать базу")
    memory.set_goal_status(goal, "active")
    memory.event("result", {"status": "failed", "reason": "blocked"})
    memory.remember("recipe", "custom:part", {"inputs": {"custom:ore": 3}}, "manual NEI inspection")
    memory.record_usage(100, 20, None)
    memory.close()
    memory = Memory(path)
    assert memory.latest_goal() == "Исследовать базу"
    assert memory.goal("Исследовать базу") == goal
    assert memory.recent()[0]["payload"]["status"] == "failed"
    assert memory.recall("recipe", "custom:part")["value"]["inputs"] == {"custom:ore": 3}
    assert memory.totals() == {"calls": 1, "input_tokens": 100, "output_tokens": 20, "cost_usd": None}
    memory.record_usage(10, 2, 0.01)
    assert memory.totals()["cost_usd"] is None
    memory.close()


def test_knowledge_requires_provenance(tmp_path):
    memory = Memory(tmp_path / "agent.db")
    with pytest.raises(ValueError):
        memory.remember("recipe", "invented", {}, "")
    memory.close()


async def test_record_and_replay_round_trip(tmp_path, observation):
    image = tmp_path / "input.png"
    Image.new("RGB", (20, 20)).save(image)
    observation = observation.model_copy(update={"image_path": image})
    recording = RecordingSource(FixedSource(observation), tmp_path / "recording")
    await recording.observe()
    replay = ReplaySource(recording.path)
    restored = await replay.observe()
    assert restored.source == "replay"
    assert restored.visual == observation.visual
    assert restored.image_path.read_bytes() == image.read_bytes()
    with pytest.raises(ReplayEnded):
        await replay.observe()
    replay.close()
    with pytest.raises(ValueError):
        RecordingSource(FixedSource(observation), tmp_path / "recording")


@pytest.mark.parametrize("image", ["../private.png", "/etc/passwd"])
async def test_replay_rejects_path_escape(tmp_path, image):
    path = tmp_path / "record.jsonl"
    path.write_text(json.dumps({"source": "screen", "image_path": image}), encoding="utf-8")
    replay = ReplaySource(path)
    with pytest.raises(ValueError):
        await replay.observe()
    replay.close()


async def test_example_recording_is_explicitly_non_gameplay():
    replay = ReplaySource(Path("examples/replay/observations.jsonl"))
    try:
        for _ in range(2):
            observation = await replay.observe()
            assert observation.source == "replay"
            assert observation.image_path.is_file()
        with pytest.raises(ReplayEnded):
            await replay.observe()
    finally:
        replay.close()
