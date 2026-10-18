from __future__ import annotations

import json
from typing import Any

from loliland_agent.learning.library import Procedure, SkillLibrary
from loliland_agent.memory import Memory
from loliland_agent.models import HumanInstruction
from loliland_agent.recipes.planner import RecipeKnowledgeBase


class ReflectionManager:
    """Verification authority stays outside reflection; failed attempts survive restart."""

    def __init__(self, memory: Memory, version: str, limit: int = 3) -> None:
        self.memory, self.library, self.limit = memory, SkillLibrary(memory, version), limit
        self.knowledge = RecipeKnowledgeBase(memory, version)

    def state(self, goal_id: str) -> dict[str, Any]:
        saved = self.memory.recall("reflection", goal_id)
        return saved["value"] if saved else {"failures": 0, "episode": [], "last": None}

    def blocked(self, goal_id: str) -> bool:
        return bool(self.state(goal_id)["failures"] >= self.limit)

    def record(
        self,
        goal_id: str,
        instruction: HumanInstruction,
        outcome: str,
        evidence: str,
        *,
        manual: bool = False,
        task_completed: bool = False,
    ) -> dict[str, Any]:
        state = self.state(goal_id)
        success = outcome == "confirmed"
        family = instruction.action + ":" + (instruction.item or "")
        families = state.setdefault("families", {})
        families[family] = 0 if success else families.get(family, 0) + 1
        state["failures"] = max(families.values(), default=0)
        reason = (
            "Недостаточно подтверждённых ресурсов"
            if any(word in evidence.casefold() for word in ("не хватает", "недостат", "ресурс"))
            else "Рецепт расходится с наблюдением"
            if "рецепт" in evidence.casefold()
            else "Интерфейс не соответствует ожидаемому"
            if "интерфейс" in evidence.casefold()
            else "Причина пока неизвестна, требуется новое наблюдение"
        )
        report = {
            "expected": instruction.expected,
            "observed": evidence,
            "outcome": outcome,
            "source": "player_report" if manual else "screenshot",
            "cause": "Результат совпал с ожидаемым" if success else reason,
            "repair": "Продолжить" if success else "Запросить недостающие данные и пересчитать зависимости",
        }
        state["last"] = report
        if success:
            state["episode"].append({"instruction": instruction.model_dump(mode="json"), "manual": manual})
            state["episode"] = state["episode"][-30:]
            if task_completed and instruction.action == "manual_action":
                episode = state["episode"]
                self.library.put(
                    Procedure(
                        id=goal_id + ":" + str(instruction.task_id),
                        title=instruction.item or instruction.text,
                        topic=instruction.item or instruction.text,
                        kind="crafting",
                        recipe_ids=tuple(recipe.id for recipe in self.knowledge.all()),
                        prerequisites=tuple(
                            sorted(
                                {r.machine for r in self.knowledge.all() if r.machine}
                                | {tech for r in self.knowledge.all() for tech in r.technologies}
                            )
                        ),
                        steps=tuple(
                            HumanInstruction.model_validate_json(json.dumps(row["instruction"])).model_copy(
                                update={"task_id": None, "completes_task": False}
                            )
                            for row in episode
                        ),
                        explanation=evidence,
                        source="player_report"
                        if any(row["manual"] for row in episode)
                        else "visual_verification",
                        status="user_report" if any(row["manual"] for row in episode) else "verified",
                    )
                )
                state["episode"] = []
        else:
            self.library.note(
                instruction.item or instruction.text, evidence, "verification:" + goal_id, kind="error"
            )
            state["episode"] = []
        self.memory.remember("reflection", goal_id, state, "verified_result_or_explicit_player_report")
        return report
