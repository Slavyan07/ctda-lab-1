import ast
from pathlib import Path


def test_program_has_no_game_input_bridge_or_process_memory_capability():
    forbidden = {
        "SendInput",
        "keybd_event",
        "mouse_event",
        "SetCursorPos",
        "SetForegroundWindow",
        "PostMessageW",
        "SendMessageW",
        "ReadProcessMemory",
        "WriteProcessMemory",
        "OpenProcess",
        "CreateRemoteThread",
        "NativeInput",
        "ActionExecutor",
        "pyautogui",
        "pynput",
        "keyboard",
        "pydirectinput",
    }
    for path in Path("src/loliland_agent").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden, (path, node.attr)
            elif isinstance(node, ast.Name):
                assert node.id not in forbidden, (path, node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = (
                    [item.name.split(".")[0] for item in node.names]
                    if isinstance(node, ast.Import)
                    else [(node.module or "").split(".")[0]]
                )
                assert not forbidden.intersection(modules), path
        assert "forge" not in path.name.lower() and "bridge" not in path.name.lower()
