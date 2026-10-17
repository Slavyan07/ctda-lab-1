from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any

import httpx
from PySide6.QtCore import QThread, Signal

from loliland_agent.config import Settings
from loliland_agent.instructions.engine import Copilot, RequestGuard, Superseded
from loliland_agent.memory import Memory
from loliland_agent.perception.sources import ObservationSource
from loliland_agent.planning.providers import DemoProvider, ProviderError, make_provider


@dataclass(frozen=True)
class Command:
    kind: str
    data: Any
    revision: int
    delay: float = 0


class CopilotWorker(QThread):
    updated = Signal(dict)

    def __init__(
        self, settings: Settings, memory: Memory, source: ObservationSource | None, demo: bool = False
    ) -> None:
        super().__init__()
        self.settings, self.memory, self.source, self.demo = settings, memory, source, demo
        self.guard = RequestGuard()
        self.commands: queue.Queue[Command] = queue.Queue()
        self.stopping = threading.Event()
        self._revision = 0
        self._failures = 0

    def request(self, kind: str, data: Any = None, *, replace: bool = False, delay: float = 0) -> None:
        revision = self.guard.advance() if replace else self.guard.current()
        self.commands.put(Command(kind, data, revision, delay))

    def stop(self) -> None:
        self.guard.advance()
        self.stopping.set()

    def report(self, event: dict[str, Any]) -> None:
        if event["kind"] not in {"snapshot", "busy"}:
            logging.getLogger("loliland.copilot").info(json.dumps(event, ensure_ascii=False))
            if event["kind"] == "instruction":
                logging.getLogger("loliland.instructions").info(event["text"])
        self.updated.emit({**event, "revision": self._revision})

    def run(self) -> None:
        try:
            asyncio.run(self.work())
        except Exception as error:
            self.report(
                {
                    "kind": "error",
                    "type": type(error).__name__,
                    "text": "Не удалось запустить Copilot; проверьте базу и конфигурацию",
                }
            )

    async def work(self) -> None:
        async with httpx.AsyncClient(trust_env=False) as client:
            provider = DemoProvider() if self.demo else make_provider(self.settings, client)
            service = Copilot(self.settings, self.memory, provider, self.source, self.report, self.guard)
            service.snapshot()
            try:
                while not self.stopping.is_set():
                    try:
                        command = self.commands.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.03)
                        continue
                    if command.revision != self.guard.current():
                        continue
                    self._revision = command.revision
                    self.report({"kind": "busy", "value": True, "operation": command.kind})
                    job = asyncio.create_task(self.dispatch(service, client, command))
                    try:
                        while not job.done():
                            if self.stopping.is_set() or command.revision != self.guard.current():
                                job.cancel()
                                break
                            await asyncio.wait({job}, timeout=0.03)
                        await job
                        self._failures = 0
                    except (asyncio.CancelledError, Superseded):
                        pass
                    except ProviderError as error:
                        self._failures += 1
                        service.publish("error", {"text": str(error), "type": "ProviderError"})
                        if self._failures >= 3:
                            service.goals.pause()
                            service.publish("notice", {"text": "Три ошибки API: цель приостановлена"})
                    except Exception as error:
                        service.publish(
                            "error",
                            {
                                "type": type(error).__name__,
                                "text": "Операция не выполнена. Проверьте источник кадра, цель и формат данных; подтверждения не добавлены.",
                            },
                        )
                    finally:
                        if command.revision == self.guard.current():
                            service.snapshot()
                        self.report({"kind": "busy", "value": False, "operation": command.kind})
            finally:
                service.goals.pause()
                close = getattr(service.source, "close", None)
                if close:
                    close()

    async def dispatch(self, service: Copilot, client: httpx.AsyncClient, command: Command) -> None:
        if command.delay:
            await asyncio.sleep(command.delay)
        self.guard.check(command.revision)
        kind, data = command.kind, command.data
        if kind == "start":
            await service.start(data["goal"], data["priority"], command.revision)
        elif kind in {"check", "auto", "done"}:
            await service.check(command.revision, manual_requested=kind == "done", auto=kind == "auto")
        elif kind == "manual":
            await service.manual_confirm(data, command.revision)
        elif kind == "skip":
            await service.skip(data, command.revision)
        elif kind == "pause":
            service.goals.pause()
            service.snapshot()
        elif kind == "select":
            service.goals.pause()
            service.goals.select(data)
            service.snapshot()
        elif kind == "priority":
            service.goals.prioritize(data)
            service.snapshot()
        elif kind == "complete":
            service.goals.complete()
            service.snapshot()
        elif kind == "replan":
            await service.replan(command.revision)
        elif kind == "chat":
            await service.chat(data, command.revision)
        elif kind == "document":
            await service.import_document(data["text"], data["reference"], command.revision)
        elif kind == "recipe":
            service.import_recipe(data)
        elif kind == "facts":
            service.import_facts(data)
        elif kind == "plan_edit":
            service.edit_plan(data)
        elif kind == "storage":
            service.storage_name = str(data)
        elif kind == "invalidate":
            service.resources.invalidate()
            service.snapshot()
        elif kind == "source":
            close = getattr(service.source, "close", None)
            if close:
                close()
            service.source = data
            service.last_observation = None
            service.last_capture = 0
            service.snapshot()
        elif kind == "configure":
            service.settings = data
            service.provider = DemoProvider() if self.demo else make_provider(data, client)
            from loliland_agent.perception.engine import PerceptionEngine

            service.perception = PerceptionEngine(data.analysis_cache_seconds)
            service.snapshot()
        else:
            raise ValueError("Неизвестная операция")
