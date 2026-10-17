from __future__ import annotations

from typing import Protocol


class SpeechBackend(Protocol):
    available: bool
    reason: str

    def say(self, text: str) -> None: ...
    def stop(self) -> None: ...
    def volume(self, value: float) -> None: ...


class RussianQtSpeech:
    def __init__(self) -> None:
        from PySide6.QtCore import QLocale
        from PySide6.QtTextToSpeech import QTextToSpeech

        self.engine = QTextToSpeech()
        voices = [
            voice
            for voice in self.engine.availableVoices()
            if voice.locale().language() == QLocale.Language.Russian
        ]
        self.available = bool(voices)
        self.reason = (
            "Русский голос готов"
            if voices
            else "Русский голос TTS не найден. Установите русский пакет речи Windows; текстовый режим работает."
        )
        if voices:
            self.engine.setVoice(voices[0])
        self.engine.errorOccurred.connect(self._error)

    def _error(self, *args: object) -> None:
        self.available = False
        self.reason = "Ошибка системного TTS; используйте текст или проверьте пакет русской речи"

    def say(self, text: str) -> None:
        if not self.available:
            raise RuntimeError(self.reason)
        self.engine.say(text)

    def stop(self) -> None:
        self.engine.stop()

    def volume(self, value: float) -> None:
        self.engine.setVolume(value)


class SpeechController:
    def __init__(self, backend: SpeechBackend) -> None:
        self.backend = backend
        self.enabled = False
        self.last = ""
        self._volume = 80
        self.backend.volume(0.8)

    def enable(self, value: bool) -> bool:
        self.enabled = value and self.backend.available
        if not self.enabled:
            self.backend.stop()
        return self.enabled

    def set_volume(self, value: int) -> None:
        if not 0 <= value <= 100:
            raise ValueError("Громкость: 0–100")
        self._volume = value
        self.backend.volume(value / 100)

    def instruction(self, text: str) -> None:
        self.last = text
        if self.enabled:
            self.backend.stop()
            self.backend.say(text)

    def repeat(self) -> str:
        if self.last and self.enabled:
            self.backend.stop()
            self.backend.say(self.last)
        return self.last

    def cancel(self) -> None:
        self.backend.stop()


class SilentSpeech:
    available = False
    reason = "Системный TTS недоступен; инструкции доступны текстом"

    def say(self, text: str) -> None:
        raise RuntimeError(self.reason)

    def stop(self) -> None:
        pass

    def volume(self, value: float) -> None:
        pass
