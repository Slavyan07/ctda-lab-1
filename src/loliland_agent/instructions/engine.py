from __future__ import annotations

import asyncio
import json
import shutil
import threading
import time
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from loliland_agent.config import Settings
from loliland_agent.conversation.manager import conversation_context
from loliland_agent.goals.manager import GoalManager
from loliland_agent.instructions.verifier import CompletionVerifier
from loliland_agent.memory import Memory
from loliland_agent.models import (
    HumanInstruction,
    Observation,
    PendingInstruction,
    RecipeCandidate,
    Usage,
)
from loliland_agent.perception.engine import PerceptionEngine
from loliland_agent.perception.sources import ObservationSource
from loliland_agent.planning.providers import AIProvider
from loliland_agent.recipes.planner import Evidence, RecipeKnowledgeBase, RecipePlanner
from loliland_agent.resources.tracker import ResourceTracker


class Superseded(RuntimeError):
    pass


class RequestGuard:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._revision = 0

    def advance(self) -> int:
        with self._lock:
            self._revision += 1
            return self._revision

    def current(self) -> int:
        with self._lock:
            return self._revision

    def check(self, revision: int) -> None:
        if revision != self.current():
            raise Superseded("Запрос заменён новым действием пользователя")


class Copilot:
    def __init__(
        self,
        settings: Settings,
        memory: Memory,
        provider: AIProvider,
        source: ObservationSource | None,
        emit: Callable[[dict[str, Any]], None],
        guard: RequestGuard | None = None,
    ) -> None:
        self.settings, self.memory, self.provider, self.source = settings, memory, provider, source
        self.goals = GoalManager(memory)
        self.guard = guard or RequestGuard()
        self.emit = emit
        self.perception = PerceptionEngine(settings.analysis_cache_seconds)
        self.verifier = CompletionVerifier()
        self.last_observation: Observation | None = None
        self.last_capture = 0.0
        self.storage_name = ""
        self._failures = 0

    @property
    def knowledge(self) -> RecipeKnowledgeBase:
        version = self.goals.current.pack_version if self.goals.current else self.settings.pack_version
        return RecipeKnowledgeBase(self.memory, version)

    @property
    def resources(self) -> ResourceTracker:
        version = self.goals.current.pack_version if self.goals.current else self.settings.pack_version
        return ResourceTracker(self.memory, version)

    def publish(self, kind: str, payload: dict[str, Any]) -> None:
        if kind not in {"snapshot", "busy"}:
            self.memory.event(kind, payload)
        self.emit({"kind": kind, **payload})

    def usage(self, value: Usage) -> None:
        if self.provider.is_remote:
            self.memory.record_usage(value.input_tokens, value.output_tokens, value.cost_usd)

    def context(self) -> dict[str, Any]:
        goal = self.goals.current
        facts = self.resources.facts()
        material = None
        if goal and goal.target:
            material = (
                RecipePlanner(self.knowledge)
                .plan(
                    goal.target,
                    self.resources.available(),
                    raw_resources=set(facts.get("raw_resources", [])),
                    machines=set(facts.get("machines", [])),
                    technologies=set(facts.get("technologies", [])),
                )
                .model_dump(mode="json")
            )
        return {
            "conversation": conversation_context(self.memory),
            "pack_version": goal.pack_version if goal else self.settings.pack_version,
            "ready_tasks": [step.id for step in self.goals.ready()],
            "protected_task_ids": [step.id for step in self.goals.protected_steps()],
            "recipes": [recipe.model_dump(mode="json") for recipe in self.knowledge.all()],
            "observed_resources": self.resources.available(),
            "resources_are_partial_snapshots": True,
            "confirmed_facts": facts,
            "materials": material,
            "history": self.memory.recent(12),
        }

    def snapshot(self) -> None:
        goal = self.goals.current
        context = self.context()
        self.publish(
            "snapshot",
            {
                "goal": goal.model_dump(mode="json") if goal else None,
                "goals": [
                    item.model_dump(mode="json")
                    for item in sorted(self.goals.workspace.goals, key=lambda value: -value.priority)
                ],
                "context": context,
                "usage": self.memory.totals(),
                "observation": self.last_observation.model_dump(mode="json")
                if self.last_observation
                else None,
            },
        )

    async def _capture(self, revision: int, auto: bool = False) -> Observation | None:
        if self.source is None:
            raise ValueError("Выберите окно игры или загрузите скриншот")
        remaining = self.settings.interval_seconds - (time.monotonic() - self.last_capture)
        if auto and remaining > 0:
            return None
        while remaining > 0:
            self.guard.check(revision)
            await asyncio.sleep(min(0.1, remaining))
            remaining = self.settings.interval_seconds - (time.monotonic() - self.last_capture)
        self.guard.check(revision)
        raw = await self.source.observe()
        self.last_capture = time.monotonic()
        self.guard.check(revision)
        if auto and self.last_observation and raw.frame_id and raw.frame_id == self.last_observation.frame_id:
            return None
        observation, usage = await self.perception.analyze(raw, self.provider)
        self.guard.check(revision)
        if usage is not None:
            self.usage(usage)
        self.last_observation = observation
        if observation.visual.recipes and observation.image_path:
            evidence_dir = self.settings.database_path.parent / "evidence"
            evidence_dir.mkdir(parents=True, exist_ok=True)
            destination = evidence_dir / (observation.frame_id + ".png")
            if not destination.exists():
                shutil.copyfile(observation.image_path, destination)
        self.knowledge.ingest(observation)
        self.resources.ingest(observation, self.storage_name)
        self.publish("observation", observation.model_dump(mode="json"))
        if self.perception.requires_human(observation):
            self.goals.pause()
            self.publish(
                "notice",
                {
                    "text": "На экране проверка персонала/CAPTCHA. Разберитесь с ней вручную; Copilot приостановлен."
                },
            )
            self.snapshot()
            return None
        return observation

    async def start(self, description: str, priority: int, revision: int) -> None:
        self.guard.check(revision)
        goal = self.goals.start(description, self.settings.pack_version, priority)
        self.publish("goal", {"id": goal.id, "description": goal.description})
        self.snapshot()
        if goal.pending:
            self.publish(
                "instruction",
                {
                    "id": goal.pending.id,
                    "text": (goal.pending.clarification or goal.pending.instruction).text,
                    "restored": True,
                },
            )
            self.snapshot()
            return
        observation = await self._capture(revision)
        if observation is None:
            return
        await self._next(observation, revision, make_plan=not bool(goal.steps))

    async def _next(self, observation: Observation, revision: int, make_plan: bool = False) -> None:
        goal = self.goals.current
        if goal is None or goal.status != "active":
            self.snapshot()
            return
        if make_plan or not goal.steps or not self.goals.ready():
            proposal = await self.provider.plan(goal, observation, self.context())
            self.guard.check(revision)
            self.usage(proposal.usage)
            self.goals.set_plan(proposal)
            instruction = proposal.instruction
            self.publish("plan", {"summary": proposal.summary})
        else:
            reply = await self.provider.instruct(goal, observation, self.context())
            self.guard.check(revision)
            self.usage(reply.usage)
            instruction = reply.instruction
            if reply.suggest_goal_complete:
                self.publish(
                    "goal_review",
                    {
                        "text": "ИИ предлагает проверить достижение главной цели. Завершение подтверждает человек."
                    },
                )
        self._issue(instruction, observation)
        self.snapshot()

    def _issue(self, instruction: HumanInstruction, observation: Observation) -> None:
        identifier = str(uuid4())
        before_image = None
        if observation.image_path:
            directory = self.settings.database_path.parent / "instructions"
            directory.mkdir(parents=True, exist_ok=True)
            before_image = directory / (identifier + "-before.png")
            shutil.copyfile(observation.image_path, before_image)
        pending = PendingInstruction(
            id=identifier,
            instruction=instruction,
            before_image=before_image,
            before_frame=observation.frame_id,
        )
        self.goals.issue(pending)
        if instruction.action not in {
            "open_inventory",
            "show_recipe",
            "hover_item",
            "open_storage",
            "open_machine",
            "show_screen",
            "explain",
        }:
            self.resources.invalidate()
        self.publish("instruction", {"id": identifier, **instruction.model_dump(mode="json")})

    async def check(self, revision: int, *, manual_requested: bool = False, auto: bool = False) -> None:
        goal = self.goals.current
        if goal is None:
            raise ValueError("Сначала задайте главную цель")
        if goal.status != "active":
            self.publish("notice", {"text": "Цель на паузе; нажмите Продолжить"})
            return
        observation = await self._capture(revision, auto)
        if observation is None:
            return
        pending = goal.pending
        if pending is None:
            await self._next(observation, revision, make_plan=not bool(goal.steps))
            return
        before = (
            Observation(source="image", image_path=pending.before_image, frame_id=pending.before_frame)
            if pending.before_image and pending.before_image.exists()
            else None
        )
        result = await self.verifier.verify(pending.instruction, before, observation, self.provider)
        self.guard.check(revision)
        # Pure local checks have empty usage and must not inflate API-call counts.
        if result.usage.input_tokens or result.usage.output_tokens or result.usage.cost_usd is not None:
            self.usage(result.usage)
        self.publish("verification", {"instruction_id": pending.id, **result.model_dump(mode="json")})
        if result.requires_human:
            self.goals.pause()
            self.snapshot()
            return
        if result.outcome == "confirmed" and result.confidence >= 0.95:
            self._confirm(
                pending.id, manual=False, task_completed=result.task_completed, evidence=result.evidence
            )
            self._failures = 0
            await self._next(observation, revision)
        elif result.outcome == "incorrect" and result.correction is not None:
            self._failures += 1
            correction = result.correction.model_copy(update={"task_id": pending.instruction.task_id})
            self.publish("instruction_error", {"instruction_id": pending.id, "evidence": result.evidence})
            self._issue(correction, observation)
            if self._failures >= 3:
                self.goals.pause()
                self.publish("notice", {"text": "Повторные ошибки: план приостановлен для пересмотра"})
            self.snapshot()
        else:
            updated = pending.model_copy(
                update={
                    "verification": "uncertain",
                    "evidence": result.evidence,
                    "clarification": result.request_view,
                }
            )
            current = self.goals.current
            assert current is not None
            self.goals.replace(current.model_copy(update={"pending": updated}))
            if result.request_view and (
                pending.clarification is None or pending.clarification.text != result.request_view.text
            ):
                self.publish(
                    "instruction",
                    {"id": pending.id, **result.request_view.model_dump(mode="json"), "clarification": True},
                )
            self.snapshot()
            if manual_requested:
                self.publish(
                    "manual_available",
                    {
                        "instruction_id": pending.id,
                        "text": pending.instruction.text,
                        "evidence": result.evidence,
                    },
                )

    def _confirm(self, identifier: str, *, manual: bool, task_completed: bool, evidence: str) -> None:
        pending = self.goals.confirm(identifier, manual=manual, task_completed=task_completed)
        self.publish(
            "confirmed",
            {
                "instruction_id": identifier,
                "text": pending.instruction.text,
                "confirmation": "manual" if manual else "visual",
                "evidence": evidence,
            },
        )

    async def manual_confirm(self, identifier: str, revision: int) -> None:
        self.guard.check(revision)
        goal = self.goals.current
        if (
            goal is None
            or goal.pending is None
            or goal.pending.id != identifier
            or goal.pending.verification != "uncertain"
        ):
            raise ValueError("Ручное подтверждение недоступно или устарело")
        self._confirm(
            identifier,
            manual=True,
            task_completed=goal.pending.instruction.completes_task,
            evidence="Человек явно подтвердил выполнение; визуального доказательства нет",
        )
        if self.last_observation and self.goals.current and self.goals.current.status == "active":
            await self._next(self.last_observation, revision)
        else:
            self.snapshot()

    async def skip(self, identifier: str, revision: int) -> None:
        self.guard.check(revision)
        self.goals.skip(identifier)
        self.publish("skipped", {"instruction_id": identifier, "text": "Пропуск не подтверждает зависимости"})
        if self.last_observation:
            await self._next(self.last_observation, revision)
        else:
            self.snapshot()

    async def replan(self, revision: int) -> None:
        observation = await self._capture(revision)
        if observation is not None:
            await self._next(observation, revision, make_plan=True)

    async def chat(self, message: str, revision: int) -> None:
        if not message.strip() or len(message) > 8000:
            raise ValueError("Сообщение: 1–8000 символов")
        reply = await self.provider.chat(
            message,
            {
                "goal": self.goals.current.model_dump(mode="json") if self.goals.current else None,
                "last_visual": self.last_observation.visual.model_dump(mode="json")
                if self.last_observation
                else None,
                **self.context(),
            },
        )
        self.guard.check(revision)
        self.usage(reply.usage)
        self.publish("chat", {"role": "user", "text": message})
        self.publish("chat", {"role": "assistant", "text": reply.answer})
        self.snapshot()

    async def import_document(self, text: str, reference: str, revision: int) -> None:
        if not text.strip() or len(text) > 100_000:
            raise ValueError("Документ: 1–100000 символов")
        version = self.goals.current.pack_version if self.goals.current else self.settings.pack_version
        reply = await self.provider.extract_document(text, version)
        self.guard.check(revision)
        self.usage(reply.usage)
        count = self.knowledge.ingest_document(reply.recipes, text, reference)
        self.publish("notice", {"text": f"Из документа сохранено рецептов: {count}"})
        self.snapshot()

    def import_recipe(self, text: str) -> None:
        candidate = RecipeCandidate.model_validate_json(text)
        self.knowledge.add(
            candidate,
            Evidence(
                kind="user",
                reference="explicit_manual_recipe",
                excerpt=candidate.evidence_text,
                confidence=1.0,
            ),
        )
        self.publish("notice", {"text": "Подтверждённый пользователем рецепт сохранён"})
        self.snapshot()

    def import_facts(self, text: str) -> None:
        facts = json.loads(text)
        if not isinstance(facts, dict):
            raise ValueError("Нужен JSON-объект")
        self.resources.set_facts(facts)
        self.snapshot()

    def edit_plan(self, text: str) -> None:
        from pydantic import TypeAdapter

        from loliland_agent.models import PlanStep

        goal = self.goals.current
        if goal is None:
            raise ValueError("Нет цели")
        steps = TypeAdapter(tuple[PlanStep, ...]).validate_json(text, strict=True)
        self.goals.validate_steps(steps)
        old = {step.id: step for step in goal.steps}
        if any(
            step.status != (old[step.id].status if step.id in old else "pending") for step in steps
        ) or any(step.id not in {new.id for new in steps} for step in goal.steps if step.status == "done"):
            raise ValueError("Редактор не может подменять подтверждения или удалять выполненные шаги")
        self.goals.replace(goal.model_copy(update={"steps": steps, "pending": None}))
        self.publish("notice", {"text": "План изменён пользователем; старая инструкция отменена"})
        self.snapshot()
