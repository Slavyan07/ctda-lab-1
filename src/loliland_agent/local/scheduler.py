from __future__ import annotations

import asyncio
import ctypes
import sys
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path


class BudgetError(RuntimeError):
    pass


def available_ram_mb() -> int:
    if sys.platform == "win32":

        class Status(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("load", ctypes.c_ulong),
                ("total", ctypes.c_ulonglong),
                ("available", ctypes.c_ulonglong),
                ("page_total", ctypes.c_ulonglong),
                ("page_available", ctypes.c_ulonglong),
                ("virtual_total", ctypes.c_ulonglong),
                ("virtual_available", ctypes.c_ulonglong),
                ("extended", ctypes.c_ulonglong),
            ]

        status = Status()
        status.length = ctypes.sizeof(status)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise BudgetError("Не удалось проверить свободную RAM")
        return int(status.available // (1024 * 1024))
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise BudgetError("Не удалось проверить свободную RAM")


class HeavyScheduler:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = ""
        self.waiters = 0

    @asynccontextmanager
    async def lease(self, name: str, min_free_mb: int = 0) -> AsyncIterator[None]:
        self.waiters += 1
        try:
            while not self.lock.acquire(blocking=False):
                await asyncio.sleep(0.03)
        finally:
            self.waiters -= 1
        try:
            self.active = name
            if min_free_mb and available_ram_mb() < min_free_mb:
                raise BudgetError("Мало свободной RAM: закройте лишние программы или выберите qwen3-vl:2b")
            yield
        finally:
            self.active = ""
            self.lock.release()


HEAVY = HeavyScheduler()
