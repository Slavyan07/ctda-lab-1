# Replace MNIST project with LoliLand AI Copilot

Replace the selected repository with a standalone Windows 11 Copilot for LoliLand TechnoMagicRPG.
A Russian goal now drives screenshot analysis, a dependency plan, one instruction, human action,
and before/after verification. All gameplay remains human; the package contains no input executor,
Minecraft bridge, mod integration, game process access or packet handling.

Includes PySide6 desktop UI, OpenAI/OpenAI-compatible vision providers, persisted goal queue,
recipe provenance and pack versioning, material calculation, partial inventory/storage tracking,
SQLite history, Russian system TTS, notification hotkeys, capture/replay, Windows installers and setup docs.
Uncertain results remain pending or receive explicit manual confirmation. Pause/goal changes cancel
in-flight requests and discard stale model responses.

Validation: 93 passing tests, strict mypy, ruff, dependency checks, sdist/wheel build,
installed-wheel Qt human cycle and headless replay. Tests use saved synthetic screenshots and mocked
API/capture/human actions. Actual LoliLand, Windows screen/audio devices and a live vision API were
not exercised here; manual acceptance steps are documented in docs/acceptance.md.
