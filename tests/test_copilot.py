import asyncio

import pytest
from conftest import FixedSource, frame

from loliland_agent.config import Settings
from loliland_agent.instructions.engine import Copilot, Superseded
from loliland_agent.instructions.verifier import CompletionVerifier
from loliland_agent.models import CompletionReply, HumanInstruction, VisualState
from loliland_agent.planning.providers import DemoProvider, OfflineProvider


def service(tmp_path, memory, provider=None, observation=None):
    events = []
    source = FixedSource(observation or frame())
    settings = Settings(_env_file=None, database_path=tmp_path / "memory.sqlite3", interval_seconds=3)
    value = Copilot(settings, memory, provider or DemoProvider(), source, events.append)
    return value, source, events


async def test_full_human_cycle_with_saved_images(tmp_path, memory):
    engine, source, events = service(tmp_path, memory)
    await engine.start("Построить переработку", 50, 0)
    pending = engine.goals.current.pending
    assert pending.instruction.text == "Открой инвентарь." and pending.before_image.is_file()
    # A human changes the displayed state; Copilot only reads the resulting frame.
    source.observation = frame("inventory.png")
    engine.last_capture = 0
    await engine.check(0)
    goal = engine.goals.current
    assert goal.steps[0].status == "done" and goal.confirmed_count == 1 and goal.manual_count == 0
    assert goal.pending.id != pending.id and goal.pending.instruction.action == "show_recipe"
    assert [e["confirmation"] for e in events if e["kind"] == "confirmed"] == ["visual"]
    assert engine.memory.totals()["calls"] == 0


async def test_unchanged_frame_uncertain_then_explicit_manual(tmp_path, memory):
    engine, _, events = service(tmp_path, memory)
    await engine.start("Цель", 50, 0)
    identifier = engine.goals.current.pending.id
    with pytest.raises(ValueError):
        await engine.manual_confirm(identifier, 0)
    engine.last_capture = 0
    await engine.check(0, manual_requested=True)
    assert engine.goals.current.confirmed_count == 0
    assert (
        engine.goals.current.pending.id == identifier
        and engine.goals.current.pending.verification == "uncertain"
    )
    assert any(e["kind"] == "manual_available" for e in events)
    with pytest.raises(ValueError):
        await engine.manual_confirm("old", 0)
    await engine.manual_confirm(identifier, 0)
    assert engine.goals.current.manual_count == 1
    assert [e["confirmation"] for e in events if e["kind"] == "confirmed"] == ["manual"]


async def test_auto_change_detection_and_cadence(tmp_path, memory):
    engine, source, _ = service(tmp_path, memory)
    await engine.start("Цель", 50, 0)
    await engine.check(0, auto=True)
    assert source.calls == 1
    engine.last_capture = 0
    await engine.check(0, auto=True)
    assert source.calls == 2 and engine.goals.current.pending.verification == "waiting"


async def test_pause_restore_preserves_instruction_without_requery(tmp_path, memory):
    engine, _, _ = service(tmp_path, memory)
    await engine.start("Цель", 50, 0)
    original = engine.goals.current.pending
    restored, source, events = service(tmp_path, memory)
    assert restored.goals.current.status == "paused"
    await restored.start("Цель", 50, 0)
    assert source.calls == 0 and restored.goals.current.pending == original
    assert any(e.get("restored") for e in events)


async def test_staff_screen_pauses_without_instruction(tmp_path, memory):
    engine, _, events = service(tmp_path, memory, observation=frame("verification.png"))
    await engine.start("Цель", 50, 0)
    assert engine.goals.current.status == "paused" and engine.goals.current.pending is None
    assert not any(e["kind"] == "instruction" for e in events)


class Delayed(DemoProvider):
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def plan(self, *args):
        self.started.set()
        await self.release.wait()
        return await super().plan(*args)


async def test_replaced_goal_discards_inflight_model_result(tmp_path, memory):
    provider = Delayed()
    engine, _, events = service(tmp_path, memory, provider)
    job = asyncio.create_task(engine.start("Старая цель", 50, 0))
    await provider.started.wait()
    engine.guard.advance()
    provider.release.set()
    with pytest.raises(Superseded):
        await job
    assert engine.goals.current.pending is None
    assert not any(e["kind"] == "instruction" for e in events)


async def test_guard_can_cancel_capture_wait(tmp_path, memory):
    engine, _, _ = service(tmp_path, memory)
    await engine.start("Цель", 50, 0)
    job = asyncio.create_task(engine.check(0))
    await asyncio.sleep(0.02)
    engine.guard.advance()
    with pytest.raises(Superseded):
        await job
    assert engine.goals.current.confirmed_count == 0


class Incorrect(DemoProvider):
    async def verify(self, *args):
        return CompletionReply(
            outcome="incorrect",
            confidence=1.0,
            evidence="Открыт другой интерфейс",
            correction=HumanInstruction(
                action="open_inventory",
                text="Открой инвентарь персонажа.",
                reason="Это нужный интерфейс",
                expected="Слоты игрока",
            ),
        )


async def test_wrong_action_replaced_with_one_correction(tmp_path, memory):
    engine, source, events = service(tmp_path, memory, Incorrect())
    await engine.start("Цель", 50, 0)
    original = engine.goals.current.pending.id
    source.observation = frame("moved.png")
    engine.last_capture = 0
    await engine.check(0)
    assert engine.goals.current.confirmed_count == 0
    assert engine.goals.current.pending.id != original
    assert engine.goals.current.pending.instruction.task_id == "show-inventory"
    assert any(e["kind"] == "instruction_error" for e in events)


@pytest.mark.parametrize(
    "scene,item,expected",
    [("machine", "печь", True), ("machine", "пресс", False), ("inventory", None, False)],
)
async def test_machine_completion_matches_requested_machine(scene, item, expected):
    after = frame("inventory.png").model_copy(
        update={"visual": VisualState(scene=scene, confidence=1.0, machine_name=item)}
    )
    instruction = HumanInstruction(
        action="open_machine", text="Открой печь.", reason="r", expected="e", item="печь"
    )
    reply = await CompletionVerifier().verify(instruction, frame(), after, OfflineProvider())
    assert (reply.outcome == "confirmed") == expected


async def test_plan_editor_cannot_fabricate_completion(tmp_path, memory):
    engine, _, _ = service(tmp_path, memory)
    await engine.start("Цель", 50, 0)
    goal = engine.goals.current
    import json

    data = [s.model_dump(mode="json") for s in goal.steps]
    data[0]["status"] = "done"
    with pytest.raises(ValueError):
        engine.edit_plan(json.dumps(data))
    assert engine.goals.current.pending is not None
    data[0]["status"] = "active"
    data[0]["priority"] = 99
    engine.edit_plan(json.dumps(data))
    assert engine.goals.current.pending is None and engine.goals.current.steps[0].priority == 99
