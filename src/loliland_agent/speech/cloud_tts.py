from __future__ import annotations

import asyncio
import queue
import threading

import httpx
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QObject, QThread, QUrl, Signal
from PySide6.QtMultimedia import QAudioDevice, QAudioOutput, QMediaPlayer

from loliland_agent.config import Settings
from loliland_agent.local.scheduler import BudgetError
from loliland_agent.speech.devices import prepare_audio
from loliland_agent.speech.stt import SpeechError, speech_key

prepare_audio()

VOICES = ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"]


async def synthesize(
    settings: Settings, client: httpx.AsyncClient, text: str, voice: str, speed: float
) -> bytes:
    if settings.free_mode:
        raise SpeechError("Облачный синтез отключён в бесплатном режиме")
    try:
        async with client.stream(
            "POST",
            settings.speech_api_base_url.rstrip("/") + "/audio/speech",
            headers={"Authorization": "Bearer " + speech_key(settings)},
            json={
                "model": settings.tts_model,
                "input": text[:3500],
                "voice": voice,
                "response_format": "wav",
                "speed": speed,
                "instructions": "Говори естественно, ясно и по-русски.",
            },
            timeout=30,
            follow_redirects=False,
        ) as response:
            if response.status_code != 200:
                raise SpeechError(f"Озвучка: API HTTP {response.status_code}; текст остаётся доступен")
            data = bytearray()
            async for chunk in response.aiter_bytes():
                data.extend(chunk)
                if len(data) > 8_000_000:
                    raise SpeechError("Слишком длинная озвучка")
        if not data.startswith(b"RIFF") or data[8:12] != b"WAVE":
            raise SpeechError("Сервис вернул неверный WAV")
        return bytes(data)
    except httpx.HTTPError:
        raise SpeechError("Озвучка недоступна: проверьте сеть; текстовый режим работает") from None


class SynthesisWorker(QThread):
    ready = Signal(bytes, int)
    failed = Signal(str, int)

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self.items: queue.Queue[tuple[str, str, float, int]] = queue.Queue(maxsize=2)
        self.epoch = 0
        self.stopping = threading.Event()

    def cancel(self) -> None:
        self.epoch += 1

    def stop(self) -> None:
        self.stopping.set()
        self.cancel()

    def submit(self, text: str, voice: str, speed: float) -> None:
        self.cancel()
        while not self.items.empty():
            try:
                self.items.get_nowait()
            except queue.Empty:
                break
        self.items.put_nowait((text, voice, speed, self.epoch))

    def run(self) -> None:
        try:
            asyncio.run(self.work())
        except Exception:
            if not self.stopping.is_set():
                self.failed.emit("Служба озвучки остановлена; текст доступен", self.epoch)

    async def work(self) -> None:
        async with httpx.AsyncClient(trust_env=False) as client:
            while not self.stopping.is_set():
                try:
                    text, voice, speed, epoch = self.items.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.03)
                    continue
                if epoch != self.epoch:
                    continue
                if self.settings.tts_provider == "piper":
                    from loliland_agent.speech.piper import synthesize_piper

                    job = asyncio.create_task(synthesize_piper(self.settings, text, speed))
                else:
                    job = asyncio.create_task(synthesize(self.settings, client, text, voice, speed))
                try:
                    while not job.done():
                        if self.stopping.is_set() or epoch != self.epoch:
                            job.cancel()
                            break
                        await asyncio.wait({job}, timeout=0.03)
                    data = await job
                    if epoch == self.epoch and not self.stopping.is_set():
                        self.ready.emit(data, epoch)
                except asyncio.CancelledError:
                    pass
                except (SpeechError, BudgetError) as error:
                    self.failed.emit(str(error), epoch)
                except Exception:
                    self.failed.emit("Ошибка озвучки; текст остаётся доступен", epoch)


class CloudSpeech(QObject):
    available = True
    reason = "Облачный русский TTS; текст отправляется выбранному API"
    error = Signal(str)

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.worker = SynthesisWorker(settings)
        self.worker.setParent(self)
        self.output = QAudioOutput(self)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.output)
        self.player.errorOccurred.connect(self.playback_error)
        self.buffer: QBuffer | None = None
        self.voice = settings.tts_voice
        self.speed = settings.speech_rate
        self.worker.ready.connect(self.play)
        self.worker.failed.connect(self.failed)
        self.worker.start()

    @property
    def playing(self) -> bool:
        return self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    def voice_names(self) -> list[str]:
        return VOICES

    def select_voice(self, name: str) -> None:
        if name in VOICES:
            self.voice = name

    def rate(self, value: float) -> None:
        self.speed = value

    def volume(self, value: float) -> None:
        self.output.setVolume(value)

    def set_device(self, device: QAudioDevice) -> None:
        self.output.setDevice(device)

    def say(self, text: str) -> None:
        self.stop()
        self.worker.submit(text, self.voice, self.speed)

    def stop(self) -> None:
        self.worker.cancel()
        self.player.stop()

    def close(self) -> None:
        self.stop()
        self.worker.stop()

    def failed(self, text: str, epoch: int) -> None:
        if epoch == self.worker.epoch:
            self.reason = text
            self.error.emit(text)

    def playback_error(self, *args: object) -> None:
        self.reason = "Аудиовывод недоступен; текстовый режим работает"
        self.error.emit(self.reason)

    def play(self, data: bytes, epoch: int) -> None:
        if epoch != self.worker.epoch or self.worker.stopping.is_set():
            return
        self.player.setSource(QUrl())
        if self.buffer:
            self.buffer.close()
            self.buffer.deleteLater()
        self.buffer = QBuffer(self)
        self.buffer.setData(QByteArray(data))
        self.buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        self.player.setSourceDevice(self.buffer, QUrl("voice.wav"))
        self.player.play()


class PiperSpeech(CloudSpeech):
    reason = "Локальный Piper TTS: русский голос, без ключей и облачного синтеза"

    def voice_names(self) -> list[str]:
        from loliland_agent.local.models import VOICE

        return [VOICE]

    def select_voice(self, name: str) -> None:
        from loliland_agent.local.models import VOICE

        self.voice = VOICE
