"""Passive physical key state only. No hooks, suppression or synthetic game input."""

KEYS = {**{f"F{i}": 0x6F + i for i in range(1, 13)}, "SPACE": 0x20, "CTRL": 0x11, "ALT": 0x12, "SHIFT": 0x10}


def parse_hotkey(text: str) -> tuple[int, ...]:
    parts = text.upper().replace(" ", "").split("+")
    if len(parts) < 2 or len(parts) > 4 or len(set(parts)) != len(parts) or any(p not in KEYS for p in parts):
        raise ValueError("PTT: комбинация CTRL/ALT/SHIFT и SPACE/F1..F12, например CTRL+ALT+SPACE")
    if (
        not any(p in {"CTRL", "ALT", "SHIFT"} for p in parts)
        or sum(p not in {"CTRL", "ALT", "SHIFT"} for p in parts) != 1
    ):
        raise ValueError("PTT требует модификатор и одну основную клавишу")
    if {"F8", "F9", "F10"}.intersection(parts):
        raise ValueError("F8/F9/F10 заняты панелью Copilot")
    return tuple(KEYS[p] for p in parts)
