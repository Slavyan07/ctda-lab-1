from __future__ import annotations

import hashlib
import re
import time
from collections import OrderedDict
from typing import TYPE_CHECKING

from PIL import Image

from loliland_agent.models import Observation, Usage, VisualState
from loliland_agent.perception.testpattern import detect_test_frame

if TYPE_CHECKING:
    from loliland_agent.planning.providers import AIProvider

INTERVENTION = re.compile(
    r"captcha|капч|проверка персонала|проверка администрац|проверка на бота|staff check|manual verification",
    re.IGNORECASE,
)


class PerceptionError(RuntimeError):
    pass


class PerceptionEngine:
    def __init__(self, cache_seconds: float = 30) -> None:
        self.cache_seconds = cache_seconds
        self._cache: OrderedDict[str, tuple[float, VisualState, str]] = OrderedDict()

    def requires_human(self, observation: Observation) -> bool:
        return (
            observation.visual.human_required is True
            or observation.visual.scene == "verification"
            or any(INTERVENTION.search(text) for text in observation.visual.visible_text)
        )

    async def analyze(
        self, observation: Observation, provider: AIProvider
    ) -> tuple[Observation, Usage | None]:
        if observation.image_path is None:
            raise PerceptionError("Нужен скриншот")
        data = observation.image_path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if observation.frame_id and observation.frame_id != digest:
            raise PerceptionError("Снимок изменился после захвата")
        cached = self._cache.get(digest)
        if cached and time.monotonic() - cached[0] < self.cache_seconds:
            self._cache.move_to_end(digest)
            return observation.model_copy(
                update={
                    "frame_id": digest,
                    "visual": cached[1],
                    "visual_at": observation.monotonic_at,
                    "visual_origin": "cache",
                }
            ), None
        with Image.open(observation.image_path) as image:
            local = detect_test_frame(image)
        usage = None
        origin = "local_test"
        if local is None:
            reply = await provider.perceive(observation)
            local = reply.visual
            usage = reply.usage if provider.is_remote else None
            origin = "llm"
        self._cache[digest] = (time.monotonic(), local, origin)
        while len(self._cache) > 32:
            self._cache.popitem(last=False)
        return observation.model_copy(
            update={
                "frame_id": digest,
                "visual": local,
                "visual_at": observation.monotonic_at,
                "visual_origin": origin,
            }
        ), usage
