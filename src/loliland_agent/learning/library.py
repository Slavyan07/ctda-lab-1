"""Searchable, versioned human procedures. No executable game skills."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from loliland_agent.memory import Memory
from loliland_agent.models import HumanInstruction, StrictModel


class Procedure(StrictModel):
    id: str
    title: str
    topic: str
    kind: Literal["recipe", "acquisition", "crafting", "machine", "production", "error", "user_tip"]
    steps: tuple[HumanInstruction, ...] = ()
    explanation: str
    source: str
    status: Literal["verified", "user_report", "hypothesis", "disputed", "superseded"] = "hypothesis"
    prerequisites: tuple[str, ...] = ()
    recipe_ids: tuple[str, ...] = ()
    uses: int = 0


class SkillLibrary:
    def __init__(self, memory: Memory, version: str) -> None:
        self.memory = memory
        self.category = "skills:" + version

    def put(self, procedure: Procedure) -> None:
        if not procedure.source.strip():
            raise ValueError("Процедуре нужен источник")
        self.memory.remember(self.category, procedure.id, procedure.model_dump(mode="json"), procedure.source)

    def all(self) -> list[Procedure]:
        return [
            Procedure.model_validate_json(json.dumps(row["value"]))
            for row in self.memory.list_knowledge(self.category)
        ]

    def search(self, query: str, limit: int = 5, *, reusable: bool = False) -> list[Procedure]:
        terms = set(re.findall(r"\w+", query.casefold()))
        ranked = []
        candidates = [
            Procedure.model_validate_json(json.dumps(row["value"]))
            for row in self.memory.search_knowledge(self.category, query)
        ]
        for item in candidates:
            if reusable and (
                item.status not in {"verified", "user_report"}
                or not item.steps
                or item.kind in {"error", "user_tip"}
            ):
                continue
            words = set(
                re.findall(r"\w+", (item.title + " " + item.topic + " " + item.explanation).casefold())
            )
            score = len(terms & words)
            if score or not terms:
                ranked.append((score, item.uses, item))
        return [
            item
            for _, _, item in sorted(ranked, key=lambda row: (-row[0], -row[1], row[2].id))[
                : max(1, min(limit, 20))
            ]
        ]

    def use(self, identifier: str, facts: set[str]) -> Procedure:
        item = next(value for value in self.all() if value.id == identifier)
        if item.status not in {"verified", "user_report"} or not set(item.prerequisites) <= facts:
            raise ValueError("Стратегия не подтверждена или её условия недоступны")
        updated = item.model_copy(update={"uses": item.uses + 1})
        self.put(updated)
        return updated

    def note(self, topic: str, text: str, source: str, *, kind: str = "user_tip") -> None:
        identifier = hashlib.sha256((topic + text + source).encode()).hexdigest()[:24]
        # Conflicting same-topic reports remain visible; neither becomes an established fact.
        conflicts = [
            item
            for item in self.all()
            if item.topic == topic and item.explanation != text and item.kind == kind
        ]
        for old in conflicts:
            self.put(old.model_copy(update={"status": "disputed"}))
        self.put(
            Procedure.model_validate(
                {
                    "id": identifier,
                    "title": topic,
                    "topic": topic,
                    "kind": kind,
                    "explanation": text,
                    "source": source,
                    "status": "disputed" if conflicts else "user_report",
                }
            )
        )

    def context(self, query: str) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in self.search(query)]
