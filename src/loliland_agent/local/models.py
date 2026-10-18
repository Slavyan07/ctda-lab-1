from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path

import httpx

from loliland_agent.config import Settings
from loliland_agent.local.scheduler import HEAVY

VOICE = "ru_RU-irina-medium"
VOICE_PATH = "ru/ru_RU/irina/medium"
HF = "https://huggingface.co/rhasspy/piper-voices"


class ModelError(RuntimeError):
    pass


def voice_directory(settings: Settings) -> Path:
    return settings.database_path.parent / "models" / "piper"


class ModelManager:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.settings, self.client = settings, client
        self.base = settings.api_base_url.rstrip("/").removesuffix("/v1")

    async def status(self) -> dict[str, object]:
        try:
            response = await self.client.get(self.base + "/api/tags", timeout=5, follow_redirects=False)
            response.raise_for_status()
            names = [item["name"] for item in response.json().get("models", [])]
            ollama = "Сервер доступен; " + (
                "модель установлена"
                if self.settings.model in names
                else "нужна загрузка " + self.settings.model
            )
        except (httpx.HTTPError, ValueError, KeyError):
            ollama = "Ollama недоступен: установите Ollama и запустите start_ollama.bat"
        root = voice_directory(self.settings)
        return {
            "ollama": ollama,
            "whisper_package": importlib.util.find_spec("faster_whisper") is not None,
            "piper_package": importlib.util.find_spec("piper") is not None,
            "piper_voice": all(
                (root / (VOICE + suffix)).is_file() for suffix in (".onnx", ".onnx.json", ".sha256")
            ),
            "heavy_lane": HEAVY.active,
            "waiting": HEAVY.waiters,
            "free_mode": self.settings.free_mode,
        }

    async def pull_ollama(self, consent: bool, progress: Callable[[str], None]) -> None:
        if not consent:
            raise ModelError("Подтвердите загрузку модели (4b ~3.3 GB, 2b ~2 GB)")
        succeeded = False
        async with (
            HEAVY.lease("Загрузка Ollama"),
            self.client.stream(
                "POST",
                self.base + "/api/pull",
                json={"model": self.settings.model, "stream": True},
                timeout=120,
                follow_redirects=False,
            ) as response,
        ):
            if response.status_code != 200:
                raise ModelError(f"Ollama pull HTTP {response.status_code}")
            async for line in response.aiter_lines():
                if len(line) > 100_000:
                    raise ModelError("Неверный поток загрузки Ollama")
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("error"):
                    raise ModelError("Ollama не загрузил модель; проверьте место на диске и сеть")
                succeeded = succeeded or event.get("status") == "success"
                total = event.get("total", 0)
                done = event.get("completed", 0)
                progress(str(event.get("status", "Загрузка")) + (f" {done / total:.0%}" if total else ""))
        if not succeeded:
            raise ModelError("Поток Ollama завершился без подтверждения успешной загрузки")
        progress("Модель загружена; Ollama проверяет digests при загрузке")

    async def download_piper(self, consent: bool, progress: Callable[[str], None]) -> None:
        if not consent:
            raise ModelError("Нужно согласие на загрузку голоса и его MODEL_CARD")
        async with HEAVY.lease("Загрузка Piper"):
            root = voice_directory(self.settings)
            root.mkdir(parents=True, exist_ok=True)
            tree = await self.client.get(
                f"{HF.replace('https://huggingface.co/', 'https://huggingface.co/api/models/')}/tree/main/{VOICE_PATH}",
                timeout=30,
                follow_redirects=True,
            )
            if tree.status_code != 200:
                raise ModelError(
                    f"Каталог голоса недоступен (HTTP {tree.status_code}). Нужен доступ к huggingface.co; платный резерв не используется"
                )
            (root / (VOICE + ".sha256")).unlink(missing_ok=True)
            entries = tree.json()
            weight = next(
                (entry for entry in entries if entry.get("path") == VOICE_PATH + "/" + VOICE + ".onnx"), None
            )
            expected = (weight or {}).get("lfs", {}).get("oid", "")
            if len(expected) != 64:
                raise ModelError(
                    "Источник не предоставил SHA256 модели; непроверенная модель не устанавливается"
                )
            for suffix in (".onnx", ".onnx.json"):
                name = VOICE + suffix
                destination = root / name
                temporary = root / (name + ".part")
                try:
                    progress("Загрузка " + name)
                    async with self.client.stream(
                        "GET", f"{HF}/resolve/main/{VOICE_PATH}/{name}", timeout=120, follow_redirects=True
                    ) as response:
                        if response.status_code != 200:
                            raise ModelError(f"Голос недоступен: HTTP {response.status_code}")
                        limit = 120_000_000 if suffix == ".onnx" else 1_000_000
                        count = 0
                        digest = hashlib.sha256()
                        with temporary.open("wb") as output:
                            async for chunk in response.aiter_bytes():
                                count += len(chunk)
                                if count > limit:
                                    raise ModelError("Размер голоса превышает лимит")
                                output.write(chunk)
                                digest.update(chunk)
                    if suffix == ".onnx" and digest.hexdigest() != expected:
                        raise ModelError("SHA256 голоса не совпал; файл не установлен")
                    if suffix.endswith("json"):
                        json.loads(temporary.read_text())
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
            card = await self.client.get(
                f"{HF}/resolve/main/{VOICE_PATH}/MODEL_CARD", timeout=30, follow_redirects=True
            )
            if card.status_code != 200:
                raise ModelError("Не удалось получить MODEL_CARD: установка голоса не завершена")
            (root / "MODEL_CARD").write_text(card.text, encoding="utf-8")
            (root / (VOICE + ".sha256")).write_text(expected, encoding="ascii")
            progress(
                "Русский голос Piper установлен с проверкой SHA256; условия — data/models/piper/MODEL_CARD"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Локальные модели Copilot: загрузка только по явному запросу"
    )
    parser.add_argument("action", choices=["status", "ollama", "piper"])
    parser.add_argument(
        "--download", action="store_true", help="Явное согласие на скачивание выбранной модели"
    )
    parser.add_argument("--model", choices=["qwen3-vl:4b", "qwen3-vl:2b"], default="qwen3-vl:4b")
    args = parser.parse_args()

    async def run() -> None:
        async with httpx.AsyncClient(trust_env=False) as client:
            manager = ModelManager(Settings(model=args.model), client)
            if args.action == "status":
                print(json.dumps(await manager.status(), ensure_ascii=False, indent=2))
            elif args.action == "ollama":
                await manager.pull_ollama(args.download, print)
            else:
                await manager.download_piper(args.download, print)

    try:
        asyncio.run(run())
        return 0
    except Exception as error:
        print(
            str(error)
            if isinstance(error, ModelError)
            else "Модели недоступны. Проверьте Ollama/сеть; загрузка не завершена."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
