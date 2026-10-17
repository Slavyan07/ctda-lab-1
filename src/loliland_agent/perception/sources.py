from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Protocol

import mss
from PIL import Image

from loliland_agent.models import Observation
from loliland_agent.perception.windows import Win32


class ObservationSource(Protocol):
    async def observe(self) -> Observation: ...


class ReplayEnded(RuntimeError):
    pass


def save_frame(image: Image.Image, directory: Path) -> tuple[Path, str]:
    directory.mkdir(parents=True, exist_ok=True)
    import io

    output = io.BytesIO()
    image.save(output, format="PNG")
    data = output.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    path = directory / (digest + ".png")
    if not path.exists():
        path.write_bytes(data)
    path.touch()  # LRU metadata only; image contents remain immutable.
    # Immutable snapshots let the verifier compare the actual before/after images.
    frames = sorted(directory.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in frames[16:]:
        if old != path:
            old.unlink(missing_ok=True)
    return path, digest


class ScreenSource:
    def __init__(self, hwnd: int, api: Win32, directory: Path) -> None:
        self.hwnd, self.api, self.directory = hwnd, api, directory
        self._identity = (api.process_id(hwnd), api.title(hwnd))
        self._bounds: tuple[int, int, int, int] | None = None
        self._dpi: float | None = None

    async def observe(self) -> Observation:
        self.api.thread_dpi_awareness()
        if not self.api.focused(self.hwnd):
            raise RuntimeError("Selected window lost focus")
        bounds = self.api.bounds(self.hwnd)
        identity = (self.api.process_id(self.hwnd), self.api.title(self.hwnd))
        dpi = self.api.dpi_scale(self.hwnd)
        if (
            identity != self._identity
            or self._bounds is not None
            and bounds != self._bounds
            or self._dpi is not None
            and dpi != self._dpi
        ):
            raise RuntimeError("Window identity, resolution or DPI changed; select the window again")
        self._bounds, self._dpi = bounds, dpi
        captured_at, monotonic_at = time.time(), time.monotonic()
        x, y, width, height = bounds
        with mss.mss() as capture:
            shot = capture.grab({"left": x, "top": y, "width": width, "height": height})
        if (
            not self.api.focused(self.hwnd)
            or self.api.bounds(self.hwnd) != bounds
            or (self.api.process_id(self.hwnd), self.api.title(self.hwnd)) != identity
        ):
            raise RuntimeError("Window changed during capture")
        image = Image.frombytes("RGB", shot.size, shot.rgb)
        image.thumbnail((1280, 720))
        path, digest = save_frame(image, self.directory)
        return Observation(
            source="screen",
            captured_at=captured_at,
            monotonic_at=monotonic_at,
            window_id=self.hwnd,
            window_process_id=identity[0],
            window_title=identity[1],
            window_bounds=bounds,
            dpi_scale=dpi,
            focused=True,
            image_path=path,
            frame_id=digest,
        )


class ReplaySource:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self._file = self.path.open(encoding="utf-8")

    async def observe(self) -> Observation:
        line = self._file.readline(1_000_001)
        if not line:
            raise ReplayEnded("Recording finished")
        if len(line) > 1_000_000:
            raise ValueError("Replay line exceeds size limit")
        observation = Observation.model_validate_json(line)
        path = observation.image_path
        if path is not None:
            path = (self.path.parent / path).resolve()
            if not path.is_relative_to(self.path.parent) or not path.is_file():
                raise ValueError("Replay image escapes recording directory")
        return observation.model_copy(
            update={
                "source": "replay",
                "image_path": path,
                "monotonic_at": time.monotonic(),
                "captured_at": time.time(),
                "frame_id": "",
                "visual_at": 0.0,
            }
        )

    def close(self) -> None:
        self._file.close()


class ImageSource:
    def __init__(self, path: Path, directory: Path) -> None:
        self.path, self.directory = path, directory

    async def observe(self) -> Observation:
        if self.path.stat().st_size > 20_000_000:
            raise ValueError("Screenshot file too large")
        with Image.open(self.path) as original:
            if original.width * original.height > 20_000_000:
                raise ValueError("Screenshot dimensions too large")
            image = original.convert("RGB")
            image.thumbnail((1280, 720))
            path, digest = save_frame(image, self.directory)
        return Observation(source="image", image_path=path, frame_id=digest, window_title=self.path.name)
