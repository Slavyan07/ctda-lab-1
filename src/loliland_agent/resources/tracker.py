from __future__ import annotations

import time
from collections import Counter
from typing import Any

from loliland_agent.memory import Memory
from loliland_agent.models import Observation


class ResourceTracker:
    def __init__(self, memory: Memory, pack_version: str) -> None:
        self.memory, self.pack_version = memory, pack_version
        self.category = "copilot-resources:" + pack_version

    def ingest(self, observation: Observation, storage_name: str = "") -> None:
        if observation.visual_origin == "cache":
            return
        groups: dict[str, list[dict[str, Any]]] = {}
        for stack in observation.visual.stacks:
            if stack.confidence < 0.95 or stack.count is None or (stack.item is None and stack.count > 0):
                continue
            if stack.container == "current_container" and (
                not storage_name.strip() or observation.visual.scene != "storage"
            ):
                continue
            scope = "inventory" if stack.container == "inventory" else "storage:" + storage_name.strip()
            groups.setdefault(scope, []).append(stack.model_dump(mode="json"))
        for scope, stacks in groups.items():
            previous = self.memory.recall(self.category, scope)
            # When an old scope became dirty, its unseen slots must not regain validity.
            slots = (
                {}
                if previous is None
                or previous["value"].get("stale", False)
                or previous["value"].get("aggregate", False)
                else previous["value"].get("slots", {})
            )
            for data in stacks:
                slots[data["slot"]] = {
                    "item": data["item"],
                    "count": data["count"],
                    "at": time.time(),
                    "frame": observation.frame_id,
                }
            self.memory.remember(
                self.category,
                scope,
                {"slots": slots, "stale": False, "partial": True, "updated": time.time()},
                "screenshot:" + observation.frame_id,
            )

    def invalidate(self) -> None:
        for row in self.memory.list_knowledge(self.category):
            self.memory.remember(
                self.category, row["name"], {**row["value"], "stale": True}, "human_action_requires_refresh"
            )

    def available(self, max_age: float = 300) -> dict[str, int]:
        result: Counter[str] = Counter()
        now = time.time()
        for row in self.memory.list_knowledge(self.category):
            if row["value"].get("stale", True):
                continue
            for slot in row["value"].get("slots", {}).values():
                if now - slot["at"] <= max_age and slot["item"] is not None:
                    result[slot["item"]] += slot["count"]
        return dict(result)

    def snapshots(self) -> list[dict[str, Any]]:
        return self.memory.list_knowledge(self.category)

    def facts(self) -> dict[str, Any]:
        saved = self.memory.recall("copilot-facts", self.pack_version)
        return (
            saved["value"]
            if saved
            else {"raw_resources": [], "machines": [], "technologies": [], "base": "", "storages": {}}
        )

    def set_facts(self, facts: dict[str, Any]) -> None:
        allowed = {"raw_resources", "machines", "technologies", "base", "storages", "resources"}
        if set(facts) - allowed or any(
            not isinstance(facts.get(name, []), list)
            or not all(isinstance(item, str) for item in facts.get(name, []))
            for name in ("raw_resources", "machines", "technologies")
        ):
            raise ValueError("Неверные подтверждённые факты")
        if (
            not isinstance(facts.get("base", ""), str)
            or not isinstance(facts.get("storages", {}), dict)
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in facts.get("storages", {}).items()
            )
        ):
            raise ValueError("База и координаты хранилищ задаются текстом")
        resources = facts.get("resources", {})
        if not isinstance(resources, dict) or any(
            not isinstance(scope, str)
            or scope != "inventory"
            and not scope.startswith("storage:")
            or not isinstance(items, dict)
            or any(
                not isinstance(item, str) or type(count) is not int or count < 0
                for item, count in items.items()
            )
            for scope, items in resources.items()
        ):
            raise ValueError("Ресурсы: inventory/storage:имя → предмет → неотрицательное целое")
        for scope, items in resources.items():
            slots = {
                "manual:" + item: {"item": item, "count": count, "at": time.time(), "frame": "user"}
                for item, count in items.items()
            }
            self.memory.remember(
                self.category,
                scope,
                {"slots": slots, "stale": False, "partial": True, "aggregate": True, "updated": time.time()},
                "explicit_user_confirmation",
            )
        self.memory.remember(
            "copilot-facts",
            self.pack_version,
            {**self.facts(), **{key: value for key, value in facts.items() if key != "resources"}},
            "explicit_user_confirmation",
        )
