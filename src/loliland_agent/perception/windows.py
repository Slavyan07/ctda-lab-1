from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Window:
    hwnd: int
    title: str


class Win32:
    def __init__(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("Window capture requires Windows; use a screenshot or replay here")
        self.user32: Any = ctypes.WinDLL("user32", use_last_error=True)
        self.thread_dpi_awareness()
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.user32.IsWindow.argtypes = [wintypes.HWND]
        self.user32.IsWindowVisible.argtypes = [wintypes.HWND]
        self.user32.IsIconic.argtypes = [wintypes.HWND]
        self.user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        self.user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        self.user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        self.user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]

    def thread_dpi_awareness(self) -> None:
        # Per-monitor physical coordinates, including capture worker threads. Qt may already own process DPI.
        fn = getattr(self.user32, "SetThreadDpiAwarenessContext", None)
        if fn is not None:
            fn.argtypes = [ctypes.c_void_p]
            fn.restype = ctypes.c_void_p
            fn(ctypes.c_void_p(-4))

    def dpi_scale(self, hwnd: int) -> float:
        fn = getattr(self.user32, "GetDpiForWindow", None)
        if fn is None:
            return 1.0
        fn.argtypes = [wintypes.HWND]
        return float(fn(hwnd)) / 96.0

    def title(self, hwnd: int) -> str:
        length = int(self.user32.GetWindowTextLengthW(hwnd))
        text = ctypes.create_unicode_buffer(length + 1)
        self.user32.GetWindowTextW(hwnd, text, len(text))
        return str(text.value)

    def process_id(self, hwnd: int) -> int:
        pid = wintypes.DWORD()
        if not self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid)) or not pid.value:
            raise RuntimeError("Cannot identify selected window process")
        return int(pid.value)

    def windows(self) -> list[Window]:
        result: list[Window] = []
        callback_type = ctypes.__dict__["WINFUNCTYPE"](wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def visit(hwnd: int, _: int) -> bool:
            if self.user32.IsWindowVisible(hwnd):
                length = int(self.user32.GetWindowTextLengthW(hwnd))
                title = ctypes.create_unicode_buffer(length + 1)
                self.user32.GetWindowTextW(hwnd, title, len(title))
                if title.value:
                    result.append(Window(int(hwnd), title.value))
            return True

        self.user32.EnumWindows(callback_type(visit), 0)
        return result

    def bounds(self, hwnd: int) -> tuple[int, int, int, int]:
        if not self.user32.IsWindow(hwnd) or self.user32.IsIconic(hwnd):
            raise RuntimeError("Selected game window is closed or minimized")
        rect, origin = wintypes.RECT(), wintypes.POINT(0, 0)
        if not self.user32.GetClientRect(hwnd, ctypes.byref(rect)):
            raise RuntimeError("Cannot read selected client rectangle")
        if not self.user32.ClientToScreen(hwnd, ctypes.byref(origin)):
            raise RuntimeError("Cannot locate selected client rectangle")
        width, height = rect.right - rect.left, rect.bottom - rect.top
        if width < 100 or height < 100:
            raise RuntimeError("Selected window is too small")
        return origin.x, origin.y, width, height

    def focused(self, hwnd: int) -> bool:
        return bool(self.user32.IsWindow(hwnd)) and self.user32.GetForegroundWindow() == hwnd

    def key_pressed(self, virtual_key: int) -> bool:
        # Read-only notification hotkeys. No synthetic input or focus changes.
        return bool(self.user32.GetAsyncKeyState(virtual_key) & 0x8000)
