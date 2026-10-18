from __future__ import annotations

import json
from uuid import uuid4

from loliland_agent.memory import Memory
from loliland_agent.models import GoalState, PendingInstruction, PlanProposal, PlanStep, Workspace
from loliland_agent.planning.task_graph import TaskGraph, TaskGraphState, TaskNode


class GoalManager:
    def __init__(self, memory: Memory) -> None:
        self.memory = memory
        saved = memory.recall("copilot", "workspace-v1")
        self.workspace = Workspace.model_validate_json(json.dumps(saved["value"])) if saved else Workspace()
        # Nothing launches automatically on restore. Pending instructions remain intact.
        self.workspace = self.workspace.model_copy(
            update={
                "goals": tuple(
                    goal.model_copy(update={"status": "paused"}) if goal.status == "active" else goal
                    for goal in self.workspace.goals
                )
            }
        )
        self.save()

    @property
    def current(self) -> GoalState | None:
        return next((goal for goal in self.workspace.goals if goal.id == self.workspace.current_id), None)

    def save(self) -> None:
        self.memory.remember(
            "copilot",
            "workspace-v1",
            self.workspace.model_dump(mode="json"),
            "local_user_and_verified_copilot_steps",
        )

    def replace(self, goal: GoalState) -> None:
        self.workspace = self.workspace.model_copy(
            update={"goals": tuple(goal if old.id == goal.id else old for old in self.workspace.goals)}
        )
        self.save()

    def start(self, description: str, pack_version: str, priority: int = 50) -> GoalState:
        description = description.strip()
        if not description:
            raise ValueError("Введите главную цель")
        existing = next(
            (
                goal
                for goal in self.workspace.goals
                if goal.description == description
                and goal.pack_version == pack_version
                and goal.status != "completed"
            ),
            None,
        )
        goal = existing or GoalState(
            id=str(uuid4()), description=description, pack_version=pack_version, priority=priority
        )
        goals = tuple(
            old.model_copy(update={"status": "paused"}) if old.status == "active" else old
            for old in self.workspace.goals
        )
        if not existing:
            goals = (*goals, goal)
        if len(goals) > 100:
            raise ValueError("Архив целей заполнен (100)")
        self.workspace = Workspace(current_id=goal.id, goals=goals)
        goal = goal.model_copy(update={"status": "active", "priority": priority})
        self.replace(goal)
        return goal

    def select(self, goal_id: str) -> GoalState:
        if not any(goal.id == goal_id for goal in self.workspace.goals):
            raise ValueError("Неизвестная цель")
        self.workspace = self.workspace.model_copy(update={"current_id": goal_id})
        self.save()
        assert self.current is not None
        return self.current

    def pause(self) -> None:
        if self.current:
            self.replace(self.current.model_copy(update={"status": "paused"}))

    @staticmethod
    def validate_steps(steps: tuple[PlanStep, ...]) -> None:
        ids = {step.id for step in steps}
        if len(ids) != len(steps) or any(
            dependency not in ids for step in steps for dependency in step.depends_on
        ):
            raise ValueError("Дублирующиеся или неизвестные зависимости плана")
        TaskGraph(
            TaskGraphState(
                nodes=tuple(TaskNode(name=step.id, dependencies=step.depends_on) for step in steps)
            )
        )

    def protected_steps(self) -> tuple[PlanStep, ...]:
        goal = self.current
        if goal is None:
            return ()
        ids = {
            step.id for step in goal.steps if step.status in {"done", "paused"} or step.id.startswith("user-")
        }
        by_id = {step.id: step for step in goal.steps}
        pending = list(ids)
        while pending:
            for dependency in by_id[pending.pop()].depends_on:
                if dependency not in ids:
                    ids.add(dependency)
                    pending.append(dependency)
        return tuple(step for step in goal.steps if step.id in ids)

    def set_plan(self, proposal: PlanProposal) -> None:
        goal = self.current
        if goal is None:
            raise ValueError("Нет главной цели")
        if any(step.status != "pending" for step in proposal.steps):
            raise ValueError("Модель не может объявлять задачи уже выполненными")
        protected = self.protected_steps()
        completed = tuple(step for step in protected if step.status == "done")
        completed_ids = {step.id for step in completed}
        if any(step.id in {old.id for old in protected} for step in proposal.steps):
            raise ValueError("Новый план не должен переписывать подтверждённые шаги")
        combined = (*protected, *proposal.steps)
        self.validate_steps(combined)
        task_id = proposal.instruction.task_id
        if task_id is not None and not any(
            step.id == task_id
            and step.status in {"pending", "active"}
            and set(step.depends_on) <= completed_ids
            for step in combined
        ):
            raise ValueError("Первая инструкция нового плана должна относиться к доступной задаче")
        self.replace(
            goal.model_copy(
                update={
                    "steps": combined,
                    "pending": None,
                    "plan_summary": proposal.summary,
                    "target": proposal.target or goal.target,
                }
            )
        )

    def ready(self) -> tuple[PlanStep, ...]:
        goal = self.current
        if goal is None:
            return ()
        done = {step.id for step in goal.steps if step.status == "done"}
        return tuple(
            sorted(
                (
                    step
                    for step in goal.steps
                    if step.status in {"pending", "active"} and set(step.depends_on) <= done
                ),
                key=lambda step: (-step.priority, step.id),
            )
        )

    def issue(self, pending: PendingInstruction) -> None:
        goal = self.current
        if goal is None:
            raise ValueError("Нет главной цели")
        task_id = pending.instruction.task_id
        if task_id is not None and task_id not in {step.id for step in self.ready()}:
            raise ValueError("Инструкция относится к недоступной задаче")
        steps = tuple(
            step.model_copy(update={"status": "active"}) if step.id == task_id else step
            for step in goal.steps
        )
        self.replace(goal.model_copy(update={"pending": pending, "steps": steps}))

    def confirm(self, instruction_id: str, *, manual: bool, task_completed: bool) -> PendingInstruction:
        goal = self.current
        if goal is None or goal.pending is None or goal.pending.id != instruction_id:
            raise ValueError("Подтверждение устарело")
        pending = goal.pending
        done = task_completed and pending.instruction.completes_task
        steps = tuple(
            step.model_copy(update={"status": "done" if done else "pending"})
            if step.id == pending.instruction.task_id
            else step
            for step in goal.steps
        )
        self.replace(
            goal.model_copy(
                update={
                    "pending": None,
                    "steps": steps,
                    "confirmed_count": goal.confirmed_count + 1,
                    "manual_count": goal.manual_count + int(manual),
                }
            )
        )
        return pending

    def skip(self, instruction_id: str) -> None:
        goal = self.current
        if goal is None or goal.pending is None or goal.pending.id != instruction_id:
            raise ValueError("Пропуск устарел")
        task_id = goal.pending.instruction.task_id
        steps = tuple(
            step.model_copy(update={"status": "skipped"}) if step.id == task_id else step
            for step in goal.steps
        )
        self.replace(
            goal.model_copy(update={"pending": None, "steps": steps, "skipped_count": goal.skipped_count + 1})
        )

    def add_task(self, title: str, priority: int = 50) -> PlanStep:
        goal = self.current
        if goal is None:
            raise ValueError("Сначала задайте главную цель")
        task = PlanStep(
            id="user-" + str(uuid4()),
            title=title,
            acceptance="Проверяемый результат задачи; уточняется по изображению",
            priority=priority,
        )
        self.replace(goal.model_copy(update={"steps": (*goal.steps, task)}))
        return task

    def defer(self) -> None:
        goal = self.current
        if goal is None or goal.pending is None:
            raise ValueError("Нет текущей инструкции")
        task_id = goal.pending.instruction.task_id
        steps = tuple(
            step.model_copy(update={"status": "paused"}) if step.id == task_id else step
            for step in goal.steps
        )
        self.replace(goal.model_copy(update={"steps": steps, "pending": None}))
        if not self.ready():
            self.pause()

    def prioritize(self, priority: int) -> None:
        if self.current is None or not 0 <= priority <= 100:
            raise ValueError("Приоритет 0–100")
        self.replace(self.current.model_copy(update={"priority": priority}))

    def complete(self) -> None:
        if self.current:
            self.replace(self.current.model_copy(update={"status": "completed", "pending": None}))
