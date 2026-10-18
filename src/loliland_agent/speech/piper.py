from __future__ import annotations

import asyncio
import hashlib
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from loliland_agent.config import Settings
from loliland_agent.local.models import VOICE, voice_directory
from loliland_agent.local.scheduler import HEAVY
from loliland_agent.speech.stt import SpeechError


async def synthesize_piper(settings: Settings, text: str, speed: float) -> bytes:
    root = voice_directory(settings)
    model = root / (VOICE + ".onnx")
    checksum = root / (VOICE + ".sha256")
    if not all((root / (VOICE + suffix)).is_file() for suffix in (".onnx", ".onnx.json", ".sha256")):
        raise SpeechError(
            "Голос Piper не установлен. Загрузите его в разделе «Локальные модели» с вашим согласием"
        )
    async with HEAVY.lease("Piper CPU", settings.min_free_ram_mb):
        if hashlib.sha256(model.read_bytes()).hexdigest() != checksum.read_text().strip():
            raise SpeechError("Голос Piper изменён или повреждён; повторите загрузку")
        with TemporaryDirectory(prefix="copilot-piper-") as directory:
            output = Path(directory) / "speech.wav"
            command = [
                sys.executable,
                "-m",
                "loliland_agent.speech.piper_runner",
                "--model",
                str(model.resolve()),
                "--output-file",
                str(output),
                "--length-scale",
                str(1 / speed),
            ]
            environment = {**os.environ, "OMP_NUM_THREADS": "1", "OMP_THREAD_LIMIT": "1", "PYTHONUTF8": "1"}
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=environment,
                creationflags=0x08000000 if sys.platform == "win32" else 0,
            )
            try:
                await asyncio.wait_for(process.communicate(text[:3500].encode("utf-8")), 60)
                if process.returncode != 0 or not output.is_file():
                    raise SpeechError("Piper не создал звук; проверьте пакет piper-tts и голос")
                if output.stat().st_size > 8_000_000:
                    raise SpeechError("Слишком длинная озвучка Piper")
                data = output.read_bytes()
                if not data.startswith(b"RIFF") or data[8:12] != b"WAVE":
                    raise SpeechError("Piper вернул неверный WAV")
                return data
            finally:
                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), 2)
                    except TimeoutError:
                        process.kill()
                        await process.wait()
