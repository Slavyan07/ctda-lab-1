"""Deterministic recipe curriculum supplements the local LLM strategic planner."""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any
from uuid import uuid4

from loliland_agent.learning.library import SkillLibrary
from loliland_agent.models import GoalState, HumanInstruction, PlanProposal, PlanStep
from loliland_agent.recipes.planner import MaterialPlan, RecipeKnowledgeBase, RecipePlanner


class AutomaticCurriculum:
    def __init__(self, knowledge: RecipeKnowledgeBase) -> None:
        self.knowledge = knowledge

    def materials(
        self, goal: GoalState, stock: dict[str, int], facts: dict[str, Any]
    ) -> tuple[MaterialPlan, dict[str, str]]:
        assert goal.target is not None
        groups: dict[str, list[str]] = {}
        all_recipes = self.knowledge.all()
        reachable = {goal.target.item}
        frontier = [goal.target.item]
        while frontier:
            current = frontier.pop()
            for recipe in all_recipes:
                if recipe.output.item == current:
                    groups.setdefault(current, [])
                    if recipe.id not in groups[current]:
                        groups[current].append(recipe.id)
                    for dependency in [
                        *(ingredient.item for ingredient in recipe.ingredients),
                        *([recipe.machine] if recipe.machine else []),
                    ]:
                        if dependency not in reachable:
                            reachable.add(dependency)
                            frontier.append(dependency)
        preferences = " ".join(
            str(row["value"].get("quote", ""))
            for row in self.knowledge.memory.list_knowledge("dialogue-notes")
            if row["value"].get("kind") == "preference"
        ).casefold()
        recipes = {recipe.id: recipe for recipe in self.knowledge.all()}
        # Bound alternative search; choose feasibility first, then missing resource amount.
        variants = itertools.islice(itertools.product(*(ids for ids in groups.values())), 128)
        best: tuple[tuple[int, int, int, int], MaterialPlan, dict[str, str]] | None = None
        for variant in variants:
            choices = dict(zip(groups, variant, strict=True))
            plan = RecipePlanner(self.knowledge).plan(
                goal.target,
                stock,
                raw_resources=set(facts.get("raw_resources", [])),
                machines=set(facts.get("machines", [])),
                technologies=set(facts.get("technologies", [])),
                choices=choices,
            )
            preferred = sum(
                bool(recipe.machine and recipe.machine.casefold() in preferences)
                for identifier in choices.values()
                if (recipe := recipes[identifier])
            )
            score = (
                len(plan.blockers),
                sum(plan.raw_missing.values()),
                -preferred,
                sum(plan.crafts.values()),
            )
            if best is None or score < best[0]:
                best = (score, plan, choices)
        assert best is not None
        return best[1], best[2]

    def signature(self, goal: GoalState, stock: dict[str, int], facts: dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(
                [
                    goal.target.model_dump() if goal.target else None,
                    stock,
                    facts,
                    [r.id for r in self.knowledge.all()],
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def propose(self, goal: GoalState, stock: dict[str, int], facts: dict[str, Any]) -> PlanProposal:
        assert goal.target is not None
        material, choices = self.materials(goal, stock, facts)
        prefix = "curr-" + uuid4().hex[:10] + "-"
        steps: list[PlanStep] = []
        instructions: dict[str, HumanInstruction] = {}

        def add(
            key: str,
            title: str,
            action: str,
            text: str,
            expected: str,
            deps: tuple[str, ...] = (),
            item: str | None = None,
            count: int | None = None,
        ) -> str:
            identifier = prefix + key
            steps.append(PlanStep(id=identifier, title=title, acceptance=expected, depends_on=deps))
            instructions[identifier] = HumanInstruction.model_validate(
                {
                    "action": action,
                    "text": text,
                    "reason": "Следующая доступная зависимость подтверждённого рецепта",
                    "expected": expected,
                    "task_id": identifier,
                    "item": item,
                    "count": count,
                    "completes_task": True,
                }
            )
            return identifier

        gates: list[str] = []
        raw_gates: dict[str, str] = {}
        for index, blocker in enumerate(material.blockers):
            gates.append(
                add(
                    "info" + str(index),
                    blocker,
                    "show_recipe",
                    "Покажите рецепт или условия для: " + blocker,
                    "Прочитан полный рецепт или условия доступа",
                )
            )
        for index, (item, count) in enumerate(material.raw_missing.items()):
            raw_gates[item] = add(
                "raw" + str(index),
                f"Подтвердить получение {item} ×{count}",
                "open_inventory",
                f"Покажите запас {item} ×{material.raw_required[item]} в инвентаре",
                "Подтверждено нужное количество ресурса",
                item=item,
                count=material.raw_required[item],
            )
        ids = {item: prefix + "craft" + str(index) for index, item in enumerate(material.crafts)}
        by_id = {r.id: r for r in self.knowledge.all()}
        for item, batches in material.crafts.items() if not material.blockers else []:
            recipe = by_id[choices[item]]
            dependencies = set(gates)
            dependencies.update(
                raw_gates[dependency]
                for parent, dependency in material.edges
                if parent == item and dependency in raw_gates
            )
            dependencies.update(
                ids[dependency]
                for parent, dependency in material.edges
                if parent == item and dependency in ids
            )
            add(
                "craft" + str(list(material.crafts).index(item)),
                f"Изготовить {item} ({batches} циклов)",
                "manual_action",
                f"Изготовьте {item} ({batches} циклов) из "
                + ", ".join(
                    f"{ingredient.item} ×{ingredient.count * batches}" for ingredient in recipe.ingredients
                )
                + (f" в {recipe.machine}" if recipe.machine else ""),
                "Результат крафта виден в инвентаре",
                tuple(sorted(dependencies)),
                item,
                batches * recipe.output.count,
            )
        if not steps:
            add(
                "review",
                "Проверить достижение цели",
                "open_inventory",
                "Покажите результат главной цели в инвентаре",
                "Целевой предмет и количество видны",
                item=goal.target.item,
                count=goal.target.count,
            )
        ready = next(
            (
                step
                for step in steps
                if not step.depends_on and instructions[step.id].action == "manual_action"
            ),
            next(step for step in steps if not step.depends_on),
        )
        selected = instructions[ready.id]
        if selected.item and selected.action == "open_inventory" and selected.item in material.raw_missing:
            library = SkillLibrary(self.knowledge.memory, self.knowledge.pack_version)
            for procedure in library.search(selected.item, reusable=True):
                if (
                    procedure.kind == "acquisition"
                    and procedure.topic == selected.item
                    and len(procedure.steps) == 1
                    and set(procedure.prerequisites)
                    <= set(facts.get("machines", [])) | set(facts.get("technologies", []))
                ):
                    library.use(procedure.id, set(procedure.prerequisites))
                    selected = procedure.steps[0].model_copy(
                        update={"task_id": ready.id, "completes_task": True}
                    )
                    break
        return PlanProposal(
            summary="Программа развития: зависимости рецептов, проверка доступности, затем крафт. "
            + (
                "Есть непроверенные условия."
                if material.blockers or material.raw_missing
                else "Ресурсы подтверждены."
            ),
            steps=tuple(steps),
            instruction=selected,
            target=goal.target,
        )
