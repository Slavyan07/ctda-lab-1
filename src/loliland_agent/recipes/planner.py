from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Literal

from loliland_agent.memory import Memory
from loliland_agent.models import ItemAmount, Observation, Probability, RecipeCandidate, StrictModel


class Evidence(StrictModel):
    kind: Literal["screenshot", "document", "user"]
    reference: str
    excerpt: str
    confidence: Probability


class Recipe(StrictModel):
    id: str
    pack_version: str
    output: ItemAmount
    ingredients: tuple[ItemAmount, ...]
    machine: str | None = None
    technologies: tuple[str, ...] = ()
    evidence: Evidence


class MaterialPlan(StrictModel):
    requested: ItemAmount
    raw_required: dict[str, int] = {}
    raw_missing: dict[str, int] = {}
    crafts: dict[str, int] = {}
    machines: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    edges: tuple[tuple[str, str], ...] = ()


class RecipeKnowledgeBase:
    def __init__(self, memory: Memory, pack_version: str) -> None:
        self.memory, self.pack_version = memory, pack_version
        self.category = "copilot-recipes:" + pack_version

    def all(self) -> tuple[Recipe, ...]:
        return tuple(
            Recipe.model_validate_json(json.dumps(row["value"]))
            for row in self.memory.list_knowledge(self.category)
        )

    def add(self, candidate: RecipeCandidate, evidence: Evidence) -> Recipe:
        if (
            not candidate.complete
            or candidate.confidence < 0.95
            or evidence.confidence < 0.95
            or not candidate.ingredients
        ):
            raise ValueError("Рецепт неполный или недостаточно уверенно прочитан")
        if not evidence.reference.strip() or not evidence.excerpt.strip():
            raise ValueError("Рецепту нужен проверяемый источник")
        identity = {
            "version": self.pack_version,
            "output": candidate.output.model_dump(),
            "ingredients": [item.model_dump() for item in candidate.ingredients],
            "machine": candidate.machine,
            "technologies": candidate.technologies,
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:24]
        recipe = Recipe(
            id=digest,
            pack_version=self.pack_version,
            output=candidate.output,
            ingredients=candidate.ingredients,
            machine=candidate.machine,
            technologies=candidate.technologies,
            evidence=evidence,
        )
        self.memory.remember(
            self.category, digest, recipe.model_dump(mode="json"), evidence.kind + ":" + evidence.reference
        )
        return recipe

    def ingest(self, observation: Observation) -> None:
        if observation.visual.scene != "recipe" or observation.visual_origin == "cache":
            return
        for candidate in observation.visual.recipes:
            if candidate.complete and candidate.confidence >= 0.95:
                self.add(
                    candidate,
                    Evidence(
                        kind="screenshot",
                        reference=observation.frame_id,
                        excerpt=candidate.evidence_text,
                        confidence=candidate.confidence,
                    ),
                )

    def ingest_document(self, candidates: tuple[RecipeCandidate, ...], document: str, reference: str) -> int:
        for candidate in candidates:
            if not candidate.evidence_text.strip() or candidate.evidence_text not in document:
                raise ValueError("Цитата рецепта не найдена в документе")
            if (
                not candidate.complete
                or candidate.confidence < 0.95
                or not candidate.ingredients
                or not reference.strip()
            ):
                raise ValueError("Неполный рецепт или отсутствует источник")
        for candidate in candidates:
            self.add(
                candidate,
                Evidence(
                    kind="document",
                    reference=reference,
                    excerpt=candidate.evidence_text,
                    confidence=candidate.confidence,
                ),
            )
        return len(candidates)


class RecipePlanner:
    def __init__(self, knowledge: RecipeKnowledgeBase) -> None:
        self.knowledge = knowledge

    def plan(
        self,
        target: ItemAmount,
        stock: dict[str, int],
        *,
        raw_resources: set[str] | None = None,
        machines: set[str] | None = None,
        technologies: set[str] | None = None,
        choices: dict[str, str] | None = None,
    ) -> MaterialPlan:
        available = Counter(stock)
        if any(type(count) is not int or count < 0 for count in stock.values()):
            raise ValueError("Отрицательный или неизвестный запас")
        raw, known_machines, known_tech = raw_resources or set(), set(machines or ()), technologies or set()
        recipes = self.knowledge.all()
        raw_required: Counter[str] = Counter()
        raw_missing: Counter[str] = Counter()
        crafts: Counter[str] = Counter()
        blockers: list[str] = []
        needed_machines: set[str] = set()
        edges: set[tuple[str, str]] = set()

        def expand(item: str, quantity: int, path: tuple[str, ...]) -> None:
            if item in path or len(path) > 100:
                blockers.append("Цикл/слишком глубокая зависимость: " + " → ".join((*path, item)))
                return
            if item in raw:
                raw_required[item] += quantity
            used = min(available[item], quantity)
            available[item] -= used
            missing = quantity - used
            if missing <= 0:
                return
            if item in raw:
                raw_missing[item] += missing
                return
            options = [recipe for recipe in recipes if recipe.output.item == item]
            chosen = (choices or {}).get(item)
            if chosen is not None:
                options = [recipe for recipe in options if recipe.id == chosen]
            if not options:
                blockers.append(f"Нужен подтверждённый рецепт или способ добычи: {item} ×{missing}")
                return
            if len(options) > 1:
                blockers.append("Нужно выбрать подтверждённый вариант рецепта: " + item)
                return
            recipe = options[0]
            for technology in recipe.technologies:
                if technology not in known_tech:
                    blockers.append("Не подтверждена технология: " + technology)
            if recipe.machine:
                needed_machines.add(recipe.machine)
                if recipe.machine not in known_machines:
                    known_machines.add(recipe.machine)
                    edges.add((item, recipe.machine))
                    expand(recipe.machine, 1, (*path, item))
            batches = math.ceil(missing / recipe.output.count)
            crafts[item] += batches
            for ingredient in recipe.ingredients:
                edges.add((item, ingredient.item))
                expand(ingredient.item, batches * ingredient.count, (*path, item))
            available[item] += batches * recipe.output.count - missing

        expand(target.item, target.count, ())
        return MaterialPlan(
            requested=target,
            raw_required=dict(raw_required),
            raw_missing=dict(raw_missing),
            crafts=dict(crafts),
            machines=tuple(sorted(needed_machines)),
            blockers=tuple(dict.fromkeys(blockers)),
            edges=tuple(sorted(edges)),
        )
