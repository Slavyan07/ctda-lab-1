from __future__ import annotations

import asyncio
import contextlib
import queue
import threading
from typing import Any

import httpx
from PySide6.QtCore import QThread, Signal

from loliland_agent.config import Settings
from loliland_agent.conversation.manager import ConversationManager, local_command
from loliland_agent.memory import Memory
from loliland_agent.planning.providers import AIProvider, ProviderError, make_provider


class ConversationWorker(QThread):
    updated = Signal(dict)

    def __init__(self, settings: Settings, memory: Memory, provider: AIProvider | None = None) -> None:
        super().__init__()
        self.settings, self.memory, self.provider = settings, memory, provider
        self.commands: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=12)
        self.stopping = threading.Event()
        self.context: dict[str, Any] = {}
        self.epoch = 0

    def submit(self, text: str, channel: str, context: dict[str, Any]) -> None:
        if local_command(text) is not None:
            self.interrupt()
            while not self.commands.empty():
                try:
                    self.commands.get_nowait()
                except queue.Empty:
                    break
        try:
            self.commands.put_nowait(("message", (text, channel, context, self.epoch)))
        except queue.Full:
            self.updated.emit({"kind": "error", "text": "Очередь диалога заполнена; дождитесь ответа"})

    def configure(self, settings: Settings) -> None:
        self.settings = settings
        self.epoch += 1

    def interrupt(self) -> None:
        self.epoch += 1

    def stop(self) -> None:
        self.stopping.set()
        self.interrupt()

    def run(self) -> None:
        try:
            asyncio.run(self.work())
        except Exception:
            self.updated.emit({"kind": "error", "text": "Служба диалога остановлена; прогресс сохранён"})

    async def work(self) -> None:
        async with httpx.AsyncClient(trust_env=False) as client:
            manager = ConversationManager(self.memory, self.provider or make_provider(self.settings, client))
            active_settings = self.settings
            while not self.stopping.is_set():
                if active_settings is not self.settings:
                    manager = ConversationManager(self.memory, make_provider(self.settings, client))
                    active_settings = self.settings
                try:
                    kind, data = self.commands.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.03)
                    continue
                if kind == "configure":
                    manager = ConversationManager(self.memory, make_provider(data, client))
                    continue
                text, channel, context, epoch = data
                if epoch != self.epoch:
                    continue
                self.updated.emit({"kind": "thinking", "value": True})
                job = asyncio.create_task(manager.respond(text, channel, context))
                try:
                    while not job.done():
                        if self.stopping.is_set() or epoch != self.epoch:
                            job.cancel()
                            break
                        await asyncio.wait({job}, timeout=0.03)
                    reply = await job
                    if epoch == self.epoch:
                        self.updated.emit(
                            {
                                "kind": "reply",
                                "reply": reply.model_dump(mode="json"),
                                "text": text,
                                "channel": channel,
                                "epoch": epoch,
                                "context": context,
                            }
                        )
                    compact = asyncio.create_task(manager.compact())
                    while not compact.done():
                        if self.stopping.is_set() or epoch != self.epoch or not self.commands.empty():
                            compact.cancel()
                            break
                        await asyncio.wait({compact}, timeout=0.03)
                    with contextlib.suppress(asyncio.CancelledError, ProviderError):
                        await compact
                except asyncio.CancelledError:
                    pass
                except ProviderError as error:
                    self.memory.event("dialogue_error", {"text": str(error)})
                    self.updated.emit({"kind": "error", "text": str(error)})
                except Exception:
                    self.memory.event(
                        "dialogue_error", {"text": "Ошибка диалога; пользовательское сообщение сохранено"}
                    )
                    self.updated.emit(
                        {
                            "kind": "error",
                            "text": "Диалог недоступен. Проверьте API; сообщение сохранено, план не изменён",
                        }
                    )
                finally:
                    self.updated.emit({"kind": "thinking", "value": False})
