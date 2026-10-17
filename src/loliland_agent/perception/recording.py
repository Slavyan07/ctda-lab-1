from __future__ import annotations

import json
import shutil
from pathlib import Path

from loliland_agent.models import Observation
from loliland_agent.perception.sources import ObservationSource


class RecordingSource:
    def __init__(self, source: ObservationSource, directory: Path) -> None:
        self.source, self.directory = source, directory
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "observations.jsonl"
        if self.path.exists():
            raise ValueError("Use a new recording directory; existing recordings are preserved")
        self._index = 0
        self.path.touch(exist_ok=False)

    async def observe(self) -> Observation:
        observation = await self.source.observe()
        payload = observation.model_dump(mode="json")
        if observation.image_path:
            name = f"frame-{self._index:08d}.png"
            shutil.copyfile(observation.image_path, self.directory / name)
            payload["image_path"] = name
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
            stream.flush()
        self._index += 1
        return observation
