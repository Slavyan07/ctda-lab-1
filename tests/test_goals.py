import pytest
from pydantic import ValidationError

from loliland_agent.goals.manager import GoalManager
from loliland_agent.models import HumanInstruction, PendingInstruction, PlanProposal, PlanStep


def instruction(**updates):
    return HumanInstruction(
        action="open_inventory",
        text="Открой инвентарь.",
        reason="Узнать запас",
        expected="Видны слоты",
        **updates,
    )


def proposal(steps):
    return PlanProposal(summary="Собрать факты", steps=steps, instruction=instruction())


def step(name, **kwargs):
    return PlanStep(id=name, title=name, acceptance="Виден результат", **kwargs)


def test_resume_pending_without_automatic_start(memory):
    goals = GoalManager(memory)
    goal = goals.start("Создай печь", "r1")
    goals.set_plan(proposal((step("a"),)))
    goals.issue(PendingInstruction(id="i", instruction=instruction(task_id="a", completes_task=True)))
    restored = GoalManager(memory)
    assert restored.current.status == "paused" and restored.current.pending.id == "i"
    assert restored.start("Создай печь", "r1").id == goal.id
    assert restored.current.pending.id == "i"


@pytest.mark.parametrize("manual", [False, True])
def test_confirmation_requires_exact_instruction_and_finishes_only_intended_task(memory, manual):
    goals = GoalManager(memory)
    goals.start("Цель", "r1")
    goals.set_plan(proposal((step("a"), step("b", depends_on=("a",)))))
    goals.issue(PendingInstruction(id="i", instruction=instruction(task_id="a", completes_task=True)))
    with pytest.raises(ValueError):
        goals.confirm("old", manual=manual, task_completed=True)
    goals.confirm("i", manual=manual, task_completed=True)
    assert [value.id for value in goals.ready()] == ["b"]
    assert goals.current.confirmed_count == 1 and goals.current.manual_count == int(manual)


def test_subaction_does_not_finish_task_and_skip_does_not_unlock(memory):
    goals = GoalManager(memory)
    goals.start("Цель", "r1")
    goals.set_plan(proposal((step("a"), step("b", depends_on=("a",)))))
    goals.issue(PendingInstruction(id="i", instruction=instruction(task_id="a")))
    goals.confirm("i", manual=False, task_completed=True)
    assert goals.current.steps[0].status == "pending"
    goals.issue(PendingInstruction(id="j", instruction=instruction(task_id="a")))
    goals.skip("j")
    assert not goals.ready() and goals.current.steps[0].status == "skipped"
    with pytest.raises(ValueError):
        goals.issue(PendingInstruction(id="k", instruction=instruction(task_id="b")))


@pytest.mark.parametrize(
    "steps",
    [
        (step("a"), step("a")),
        (step("a", depends_on=("missing",)),),
        (step("a", depends_on=("b",)), step("b", depends_on=("a",))),
    ],
)
def test_invalid_dependency_graph(steps):
    with pytest.raises(ValueError):
        GoalManager.validate_steps(steps)


def test_model_cannot_premark_done_and_replan_preserves_confirmation(memory):
    goals = GoalManager(memory)
    goals.start("Цель", "r1")
    with pytest.raises(ValueError):
        goals.set_plan(proposal((step("a", status="done"),)))
    goals.set_plan(proposal((step("a"),)))
    goals.issue(PendingInstruction(id="i", instruction=instruction(task_id="a", completes_task=True)))
    goals.confirm("i", manual=False, task_completed=True)
    goals.set_plan(proposal((step("b", depends_on=("a",)),)))
    assert goals.current.steps[0].status == "done" and goals.ready()[0].id == "b"
    with pytest.raises(ValueError):
        goals.set_plan(proposal((step("a"),)))


def test_goal_queue_versions_and_priorities(memory):
    goals = GoalManager(memory)
    a = goals.start("Цель", "r1", 20)
    b = goals.start("Цель", "r2", 90)
    assert a.id != b.id and goals.workspace.goals[0].status == "paused"
    goals.select(a.id)
    goals.prioritize(60)
    assert goals.current.priority == 60
    goals.complete()
    assert goals.start("Цель", "r1").id != a.id


@pytest.mark.parametrize("text", ["1. Открой инвентарь", "Открой; закрой", "Открой\nЗакрой"])
def test_single_instruction_schema(text):
    with pytest.raises(ValidationError):
        HumanInstruction(action="manual_action", text=text, reason="r", expected="e")


def test_invalid_replan_keeps_current_instruction_and_valid_replan_cancels_it(memory):
    goals = GoalManager(memory)
    goals.start("Цель", "r1")
    goals.set_plan(proposal((step("old"),)))
    original = PendingInstruction(id="i", instruction=instruction(task_id="old"))
    goals.issue(original)
    invalid = PlanProposal(
        summary="Новый план",
        steps=(step("a"), step("b", depends_on=("a",))),
        instruction=instruction(task_id="b"),
    )
    with pytest.raises(ValueError):
        goals.set_plan(invalid)
    assert goals.current.pending == original and goals.current.steps[0].id == "old"
    goals.set_plan(proposal((step("new"),)))
    assert goals.current.pending is None and goals.current.steps[0].id == "new"


def test_deferred_work_and_its_prerequisites_survive_new_plan_stages(memory):
    goals = GoalManager(memory)
    goals.start("Переработка", "r1")
    goals.set_plan(proposal((step("energy"), step("magic", depends_on=("energy",)))))
    current = goals.current
    goals.replace(
        current.model_copy(
            update={"steps": (step("energy"), step("magic", depends_on=("energy",), status="paused"))}
        )
    )
    goals.set_plan(proposal((step("storage"),)))
    assert {s.id for s in goals.current.steps} == {"energy", "magic", "storage"}
    assert next(s for s in goals.current.steps if s.id == "magic").status == "paused"
