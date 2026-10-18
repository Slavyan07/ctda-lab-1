from __future__ import annotations

import asyncio
import base64
import io
import json
from contextlib import suppress
from typing import Any

import httpx
from PIL import Image

from loliland_agent.config import Settings
from loliland_agent.local.scheduler import HEAVY, BudgetError, available_ram_mb
from loliland_agent.models import Observation, Usage
from loliland_agent.planning.providers import POLICY, OpenAICompatibleProvider, ProviderError


def compact(value: Any, depth: int = 0) -> Any:
    if isinstance(value, dict):
        return {
            key: compact(item, depth + 1)
            for key, item in value.items()
            if key not in {"image_path", "before_image", "history", "usage"}
        }
    if isinstance(value, (list, tuple)):
        return [compact(item, depth + 1) for item in value[-10:]]
    if isinstance(value, str):
        return value[: (1200 if depth < 3 else 360)]
    return value


class OllamaProvider(OpenAICompatibleProvider):
    is_remote = False
    is_local = True

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        super().__init__(settings, client)
        self.base = settings.api_base_url.removesuffix("/").removesuffix("/v1")

    async def unload(self) -> None:
        response = await self.client.post(
            self.base + "/api/generate",
            json={"model": self.settings.model, "keep_alive": 0},
            timeout=5,
            follow_redirects=False,
        )
        if response.status_code != 200:
            raise ProviderError("Ollama не подтвердил освобождение модели; проверьте ollama ps")

    async def check_budget(self) -> None:
        response = await self.client.get(self.base + "/api/ps", timeout=5, follow_redirects=False)
        if response.status_code != 200:
            raise ProviderError("Не удалось получить потребление памяти Ollama")
        models = response.json().get("models", [])
        vram = sum(int(model.get("size_vram", 0)) for model in models) // (1024 * 1024)
        ram = sum(max(0, int(model.get("size", 0)) - int(model.get("size_vram", 0))) for model in models) // (
            1024 * 1024
        )
        if (
            vram > self.settings.max_model_vram_mb
            or ram > self.settings.max_model_ram_mb
            or available_ram_mb() < self.settings.min_free_ram_mb
        ):
            raise BudgetError(
                "Бюджет памяти превышен. Модель освобождается; выберите 2b, меньше контекст или CPU (num_gpu=0)"
            )

    async def _request(
        self, instruction: str, schema: dict[str, Any], data: dict[str, Any], images: list[Observation]
    ) -> tuple[str, Usage]:
        payload = compact(data)
        prompt = json.dumps(
            {"instruction": instruction, "data": payload, "context_is_partial": True}, ensure_ascii=False
        )
        if len(prompt) > self.settings.ollama_num_ctx * 3:
            raise ProviderError(
                "Локальный контекст слишком велик. Уточните цель/сократите документ; скрытые данные не подставляются"
            )
        image_data = []
        for observation in images:
            if observation.image_path is None:
                raise ProviderError("Для зрения нужен скриншот")
            with Image.open(observation.image_path) as original:
                image = original.convert("RGB")
                image.thumbnail((self.settings.local_image_max_edge,) * 2)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                image_data.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
        body = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": POLICY},
                {"role": "user", "content": prompt, **({"images": image_data} if image_data else {})},
            ],
            "format": schema,
            "stream": False,
            "think": False,
            "keep_alive": 0 if self.settings.unload_after_request else "2m",
            "options": {
                "num_ctx": self.settings.ollama_num_ctx,
                "num_predict": self.settings.ollama_num_predict,
                "num_gpu": self.settings.ollama_num_gpu,
                "num_thread": self.settings.ollama_threads,
                "num_batch": 64,
                "temperature": 0.1,
            },
        }
        try:
            async with HEAVY.lease("Ollama " + self.settings.model, self.settings.min_free_ram_mb):

                async def request() -> dict[str, Any]:
                    async with self.client.stream(
                        "POST",
                        self.base + "/api/chat",
                        json=body,
                        timeout=self.settings.request_timeout,
                        follow_redirects=False,
                    ) as response:
                        if response.status_code != 200:
                            raise ProviderError(
                                f"Ollama HTTP {response.status_code}. Проверьте сервер и загрузите выбранную модель в разделе «Локальные модели»"
                            )
                        raw = bytearray()
                        async for chunk in response.aiter_bytes():
                            raw.extend(chunk)
                            if len(raw) > 2_000_000:
                                raise ProviderError("Слишком большой ответ Ollama")
                    return dict(json.loads(raw))

                job = asyncio.create_task(request())
                try:
                    await self.check_budget()
                    while not job.done():
                        await asyncio.wait({job}, timeout=0.5)
                        if not job.done():
                            await self.check_budget()
                    response = await job
                    await self.check_budget()
                finally:
                    if not job.done():
                        job.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await job
                    if self.settings.unload_after_request:
                        # Cleanup holds the shared lease until the local server acknowledges unload.
                        await asyncio.shield(self.unload())
                content = response["message"]["content"]
                if not isinstance(content, str):
                    raise ValueError("Ollama content must be JSON text")
                return content, Usage(
                    input_tokens=int(response.get("prompt_eval_count", 0)),
                    output_tokens=int(response.get("eval_count", 0)),
                    cost_usd=0.0,
                )
        except BudgetError as error:
            raise ProviderError(str(error)) from None
        except httpx.HTTPError:
            raise ProviderError(
                "Ollama недоступен. Запустите локальный Ollama; облачный резерв отключён"
            ) from None
        except (KeyError, ValueError, TypeError):
            raise ProviderError(
                "Ollama вернул некорректный JSON; попробуйте 2b/другой снимок. Прогресс сохранён"
            ) from None
