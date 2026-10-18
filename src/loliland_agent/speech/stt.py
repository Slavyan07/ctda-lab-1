from __future__ import annotations

import asyncio
import io
import wave
from typing import Protocol
from urllib.parse import urlsplit

import httpx

from loliland_agent.config import Settings

RATE = 16000


class SpeechError(RuntimeError):
    pass


class Transcriber(Protocol):
    async def transcribe(self, pcm: bytes) -> str: ...


def wav_bytes(pcm: bytes) -> bytes:
    if len(pcm) > RATE * 2 * 20 or len(pcm) % 2:
        raise SpeechError("Длина аудио должна быть не более 20 секунд PCM16 mono")
    stream = io.BytesIO()
    with wave.open(stream, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(pcm)
    return stream.getvalue()


def speech_key(settings: Settings) -> str:
    if settings.free_mode:
        raise SpeechError("Облачная речь отключена: используйте локальные Whisper и Piper")
    key = settings.speech_api_key.get_secret_value()
    if (
        not key
        and urlsplit(settings.speech_api_base_url).hostname == urlsplit(settings.api_base_url).hostname
    ):
        key = settings.api_key.get_secret_value()
    if not key:
        raise SpeechError("Для облачного голоса задайте SPEECH_API_KEY или ключ LLM того же сервиса")
    return key


class CloudSTT:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.settings, self.client = settings, client

    async def transcribe(self, pcm: bytes) -> str:
        data = wav_bytes(pcm)
        try:
            import json

            async with self.client.stream(
                "POST",
                self.settings.speech_api_base_url.rstrip("/") + "/audio/transcriptions",
                headers={"Authorization": "Bearer " + speech_key(self.settings)},
                files={"file": ("speech.wav", data, "audio/wav")},
                data={"model": self.settings.stt_model, "language": "ru", "response_format": "json"},
                timeout=30,
                follow_redirects=False,
            ) as response:
                if response.status_code != 200:
                    raise SpeechError(f"Распознавание: API HTTP {response.status_code}")
                content = bytearray()
                async for chunk in response.aiter_bytes():
                    content.extend(chunk)
                    if len(content) > 100_000:
                        raise SpeechError("Слишком большой ответ распознавания")
            text = json.loads(content)["text"]
            if not isinstance(text, str) or len(text) > 8000:
                raise ValueError("Bad transcript")
            return text.strip()
        except httpx.HTTPError:
            raise SpeechError(
                "Распознавание недоступно: проверьте сеть; текстовый чат продолжает работать"
            ) from None
        except (ValueError, KeyError, TypeError):
            raise SpeechError("Некорректный ответ распознавания") from None


class LocalWhisperSTT:
    def __init__(self, settings: Settings, command: list[str] | None = None) -> None:
        self.settings = settings
        self.command = command
        self.process: asyncio.subprocess.Process | None = None

    async def transcribe(self, pcm: bytes) -> str:
        from loliland_agent.local.scheduler import HEAVY

        async with HEAVY.lease("Whisper CPU", self.settings.min_free_ram_mb):
            try:
                return await self._transcribe(pcm)
            finally:
                if self.settings.unload_after_request:
                    await self.close()

    async def _transcribe(self, pcm: bytes) -> str:
        import base64
        import json
        import subprocess
        import sys

        try:
            if self.process is None or self.process.returncode is not None:
                command = self.command or [
                    sys.executable,
                    "-m",
                    "loliland_agent.speech.local_runner",
                    "--model",
                    self.settings.whisper_model,
                    "--directory",
                    str(self.settings.database_path.parent / "models"),
                    *(["--download"] if self.settings.allow_model_download else []),
                ]
                self.process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    limit=2_000_000,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                    if sys.platform == "win32"
                    else 0,
                )
            process = self.process
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(
                (json.dumps({"wav": base64.b64encode(wav_bytes(pcm)).decode("ascii")}) + "\n").encode()
            )
            await process.stdin.drain()
            line = await asyncio.wait_for(process.stdout.readline(), timeout=120)
            result = json.loads(line)
            if result.get("error"):
                raise SpeechError(str(result["error"])[:1000])
            text = result["text"]
            if not isinstance(text, str) or len(text) > 8000:
                raise ValueError("Bad transcript")
            return text.strip()
        except asyncio.CancelledError:
            await self.close()
            raise
        except SpeechError:
            raise
        except Exception:
            await self.close()
            raise SpeechError("Локальное распознавание недоступно; проверьте установку и модель") from None

    async def close(self) -> None:
        process = self.process
        self.process = None
        if process is not None and process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                return
            try:
                await asyncio.wait_for(process.wait(), 2)
            except TimeoutError:
                process.kill()
                await process.wait()
