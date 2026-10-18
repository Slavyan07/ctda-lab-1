# LoliLand AI Copilot 0.4

Голосовой AI-напарник для Windows 11: свободный русский диалог, визуальное планирование,
общая память и пошаговые инструкции. Все действия в Minecraft выполняет человек.

Доставка — **LoliLand_AI_Copilot_Windows.zip**, Git для установки и обновления не нужен.

Начните с [README_RU.md](README_RU.md) и [WINDOWS_SETUP.md](WINDOWS_SETUP.md).

Разработка: Python 3.11, `uv sync --locked --extra dev`, `uv run ruff check .`,
`uv run ruff format --check .`, `uv run mypy src`, `QT_QPA_PLATFORM=offscreen uv run pytest -q`.
Сборка без Git: `python tools/build_zip.py LoliLand_AI_Copilot_Windows.zip`.
