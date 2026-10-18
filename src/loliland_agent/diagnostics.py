"""Installer health check: device enumeration only, never microphone recording."""

from __future__ import annotations

import argparse
import importlib
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка Copilot и аудиоустройств")
    parser.add_argument(
        "--no-audio", action="store_true", help="Явно пропустить физические аудиоустройства в CI"
    )
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11) or sys.maxsize <= 2**32:
        print("Нужен Python 3.11 x64.")
        return 1
    for name in (
        "pydantic",
        "httpx",
        "PIL",
        "mss",
        "PySide6.QtWidgets",
        "keyring",
        "faster_whisper",
        "piper",
        "onnxruntime",
    ):
        try:
            importlib.import_module(name)
        except Exception:
            print(f"Не удалось загрузить зависимость {name}; повторите install.bat.")
            return 1
    from loliland_agent.speech.credentials import load_settings

    settings = load_settings()
    if not settings.free_mode:
        print("Внимание: FREE_MODE выключен. Для бесплатного режима включите его.")
        return 1
    print("Локальный режим без API-ключей: Ollama / Whisper / Piper. Зависимости доступны.")
    print("Модели проверяются отдельно: python -m loliland_agent.local.models status")
    if args.no_audio:
        print("Физические аудиоустройства не проверялись (--no-audio).")
        return 0
    from loliland_agent.speech.devices import prepare_audio

    prepare_audio()
    from PySide6.QtMultimedia import QMediaDevices
    from PySide6.QtWidgets import QApplication

    from loliland_agent.speech.audio import microphones

    app = QApplication.instance() or QApplication([])
    failures = 0
    try:
        devices = microphones()
        print(f"Устройств ввода: {len(devices)}. Запись не включалась.")
        if not devices:
            failures += 1
    except Exception:
        print("Микрофон/PortAudio недоступен. Проверьте устройство и разрешения Windows.")
        failures += 1
    outputs = QMediaDevices.audioOutputs()
    print(f"Устройств вывода: {len(outputs)}.")
    if not outputs:
        failures += 1
    from loliland_agent.speech.tts import RussianQtSpeech

    speech = RussianQtSpeech()
    print(speech.reason)
    print(
        "Текстовый режим доступен."
        if failures
        else "Базовая проверка устройств пройдена. Проверьте запись и звук в панели."
    )
    app.processEvents()
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
