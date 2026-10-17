from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from loliland_agent.models import StrictModel


class TaskNode(StrictModel):
    name: Annotated[str, Field(min_length=1, max_length=1000)]
    dependencies: tuple[str, ...] = ()
    status: Literal["pending", "active", "done", "failed"] = "pending"


class TaskGraphState(StrictModel):
    nodes: tuple[TaskNode, ...] = ()


class DependencyBlocked(RuntimeError):
    pass


class TaskGraph:
    """A persistent DAG of short, verifiable tasks, not an invented recipe graph."""

    def __init__(self, state: TaskGraphState | None = None) -> None:
        self.nodes = {
            node.name: node.model_copy(update={"status": "pending"}) if node.status == "active" else node
            for node in (state or TaskGraphState()).nodes
        }
        self._validate(self.nodes)

    @staticmethod
    def _validate(nodes: dict[str, TaskNode]) -> None:
        incoming = {name: 0 for name in nodes}
        outgoing: dict[str, list[str]] = {name: [] for name in nodes}
        for name, node in nodes.items():
            for dependency in set(node.dependencies):
                if dependency in nodes:
                    incoming[name] += 1
                    outgoing[dependency].append(name)
        ready = [name for name, count in incoming.items() if count == 0]
        processed = 0
        while ready:
            name = ready.pop()
            processed += 1
            for dependent in outgoing[name]:
                incoming[dependent] -= 1
                if incoming[dependent] == 0:
                    ready.append(dependent)
        if processed != len(nodes):
            raise ValueError("Task dependency cycle")

    def register(self, name: str, dependencies: tuple[str, ...]) -> None:
        if len(self.nodes) >= 1000 and name not in self.nodes:
            raise ValueError("Task graph limit reached")
        previous = self.nodes.get(name)
        node = TaskNode(
            name=name, dependencies=dependencies, status=previous.status if previous else "pending"
        )
        updated = {**self.nodes, name: node}
        self._validate(updated)
        self.nodes = updated

    def missing(self, name: str) -> tuple[str, ...]:
        return tuple(
            dependency
            for dependency in self.nodes[name].dependencies
            if dependency not in self.nodes or self.nodes[dependency].status != "done"
        )

    def begin(self, name: str) -> None:
        missing = self.missing(name)
        if missing:
            raise DependencyBlocked("Не подтверждены зависимости: " + ", ".join(missing))
        self.nodes[name] = self.nodes[name].model_copy(update={"status": "active"})

    def finish(self, name: str, verified: bool) -> None:
        self.nodes[name] = self.nodes[name].model_copy(update={"status": "done" if verified else "failed"})

    def ready(self) -> list[str]:
        return [
            name
            for name, node in self.nodes.items()
            if node.status in {"pending", "failed"} and not self.missing(name)
        ]

    def state(self) -> TaskGraphState:
        return TaskGraphState(nodes=tuple(self.nodes.values()))
