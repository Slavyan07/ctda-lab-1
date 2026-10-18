from __future__ import annotations

import asyncio
import queue
import threading

import httpx
from PySide6.QtCore import QThread, Signal

from loliland_agent.config import Settings
from loliland_agent.local.scheduler import BudgetError
from loliland_agent.speech.stt import CloudSTT, LocalWhisperSTT, SpeechError, Transcriber


class RecognitionWorker(QThread):
    updated = Signal(dict)

    def __init__(self, settings: Settings, transcriber: Transcriber | None = None) -> None:
        super().__init__()
        self.settings, self.transcriber = settings, transcriber
        self.audio: queue.Queue[bytes] = queue.Queue(maxsize=3)
        self.stopping = threading.Event()

    def submit(self, pcm: bytes) -> None:
        if self.stopping.is_set():
            return
        try:
            self.audio.put_nowait(pcm)
        except queue.Full:
            self.updated.emit(
                {"kind": "error", "text": "Очередь распознавания занята; дождитесь ответа и повторите"}
            )

    def stop(self) -> None:
        self.stopping.set()

    def run(self) -> None:
        try:
            asyncio.run(self.work())
        except Exception:
            self.updated.emit(
                {"kind": "error", "text": "Служба распознавания остановлена; текстовый чат доступен"}
            )

    async def work(self) -> None:
        async with httpx.AsyncClient(trust_env=False) as client:
            transcriber = self.transcriber or (
                LocalWhisperSTT(self.settings)
                if self.settings.stt_provider == "local"
                else CloudSTT(self.settings, client)
            )
            try:
                while not self.stopping.is_set():
                    try:
                        pcm = self.audio.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.03)
                        continue
                    self.updated.emit({"kind": "recognizing", "value": True})
                    job = asyncio.create_task(transcriber.transcribe(pcm))
                    try:
                        while not job.done():
                            if self.stopping.is_set():
                                job.cancel()
                                break
                            await asyncio.wait({job}, timeout=0.05)
                        text = await job
                        if text and not self.stopping.is_set():
                            self.updated.emit({"kind": "transcript", "text": text})
                    except asyncio.CancelledError:
                        pass
                    except (SpeechError, BudgetError) as error:
                        self.updated.emit({"kind": "error", "text": str(error)})
                    except Exception:
                        self.updated.emit(
                            {"kind": "error", "text": "Ошибка STT; проверьте модель, сеть и микрофон"}
                        )
                    finally:
                        self.updated.emit({"kind": "recognizing", "value": False})
            finally:
                close = getattr(transcriber, "close", None)
                if close:
                    await close()
