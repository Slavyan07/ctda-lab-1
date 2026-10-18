from __future__ import annotations

from typing import Any

from pydantic import ValidationError
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtMultimedia import QMediaDevices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from loliland_agent.config import Settings
from loliland_agent.speech.audio import MicrophoneCapture, Segmenter, microphones
from loliland_agent.speech.stt import SpeechError
from loliland_agent.speech.tts import SpeechController
from loliland_agent.speech.worker import RecognitionWorker


class VoicePanel(QWidget):
    transcript = Signal(str)
    settings_changed = Signal(object)
    interrupt_requested = Signal()
    repeat_requested = Signal()
    service_finished = Signal()
    audio_event = Signal(dict)

    def __init__(self, settings: Settings, speech: SpeechController) -> None:
        super().__init__()
        self.settings, self.speech = settings, speech
        self.capture: MicrophoneCapture | None = None
        self.recognizers: list[RecognitionWorker] = []
        self.recognizer: RecognitionWorker | None = None
        self._last_speaking = False
        w: QWidget
        layout = QVBoxLayout(self)
        self.privacy = QLabel(
            "Микрофон выключен по умолчанию. Cloud STT отправляет запись API. PTT/кнопка «Перебить» прерывают голос; Hands-Free не слушает во время озвучки ИИ."
        )
        self.privacy.setWordWrap(True)
        layout.addWidget(self.privacy)
        form = QFormLayout()
        self.stt = QComboBox()
        self.stt.addItems(["disabled", "cloud", "local"])
        self.stt.setCurrentText(settings.stt_provider)
        self.mode = QComboBox()
        self.mode.addItems(["ptt", "hands_free"])
        self.mode.setCurrentText(settings.voice_mode)
        self.microphone = QComboBox()
        self.output = QComboBox()
        self.tts = QComboBox()
        self.tts.addItems(["system", "cloud"])
        self.tts.setCurrentText(settings.tts_provider)
        self.voice = QComboBox()
        self.hotkey = QLineEdit(settings.ptt_hotkey)
        self.url = QLineEdit(settings.speech_api_base_url)
        self.key = QLineEdit(settings.speech_api_key.get_secret_value())
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.stt_model = QLineEdit(settings.stt_model)
        self.tts_model = QLineEdit(settings.tts_model)
        self.local_model = QComboBox()
        self.local_model.addItems(["tiny", "base"])
        self.local_model.setCurrentText(settings.whisper_model)
        self.download = QCheckBox(
            "Разрешаю загрузку выбранной Whisper-модели (~75/145 MB) при первом распознавании"
        )
        self.download.setChecked(settings.allow_model_download)
        self.speed = QSlider(Qt.Orientation.Horizontal)
        self.speed.setRange(50, 200)
        self.speed.setValue(int(settings.speech_rate * 100))
        for label, widget in [
            ("STT: disabled / cloud / local", self.stt),
            ("Режим микрофона", self.mode),
            ("Микрофон", self.microphone),
            ("TTS: system / cloud", self.tts),
            ("Выход звука (cloud TTS)", self.output),
            ("Голос ИИ", self.voice),
            ("Скорость речи: 0.5–2", self.speed),
            ("PTT, без перехвата клавиш", self.hotkey),
            ("Speech API URL", self.url),
            ("Ключ API речи (скрыт)", self.key),
            ("Облачная STT-модель", self.stt_model),
            ("Облачная TTS-модель", self.tts_model),
            ("Whisper CPU int8, 2 потока", self.local_model),
        ]:
            form.addRow(label, widget)
        layout.addLayout(form)
        layout.addWidget(self.download)
        row = QHBoxLayout()
        self.refresh = QPushButton("Обновить устройства")
        self.apply = QPushButton("Применить голосовые настройки")
        self.mute = QCheckBox("Включить микрофон")
        self.refresh.clicked.connect(self.refresh_devices)
        self.apply.clicked.connect(self.apply_settings)
        self.mute.toggled.connect(self.microphone_changed)
        for w in (self.refresh, self.apply, self.mute):
            row.addWidget(w)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.ptt = QPushButton("Удерживай для разговора")
        self.interrupt = QPushButton("Перебить")
        self.repeat = QPushButton("Повтори")
        self.ptt.pressed.connect(lambda: self.hold(True))
        self.ptt.released.connect(lambda: self.hold(False))
        self.interrupt.clicked.connect(self.interrupt_requested.emit)
        self.repeat.clicked.connect(self.repeat_requested.emit)
        for w in (self.ptt, self.interrupt, self.repeat):
            row.addWidget(w)
        layout.addLayout(row)
        self.listening = QLabel("Микрофон выключен")
        self.thinking = QLabel("ИИ ожидает сообщения")
        self.speaking = QLabel("ИИ молчит")
        self.status = QLabel()
        self.status.setWordWrap(True)
        for w in (self.listening, self.thinking, self.speaking, self.status):
            layout.addWidget(w)
        self.recognized = QLineEdit()
        self.recognized.setReadOnly(True)
        self.recognized.setPlaceholderText("Текст распознанной речи")
        layout.addWidget(self.recognized)
        self.history = QPlainTextEdit()
        self.history.setReadOnly(True)
        self.history.document().setMaximumBlockCount(500)
        layout.addWidget(self.history, 1)
        self.audio_event.connect(self.on_audio_event)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(50)
        self.refresh_devices()
        self.update_voices()

    def update_voices(self) -> None:
        self.voice.clear()
        names: list[str] = getattr(self.speech.backend, "voice_names", lambda: [])()
        self.voice.addItems(names)
        if self.settings.tts_voice in names:
            self.voice.setCurrentText(self.settings.tts_voice)
        self.status.setText(self.speech.backend.reason)

    def refresh_devices(self) -> None:
        self.microphone.clear()
        self.microphone.addItem("Устройство ввода Windows по умолчанию", None)
        try:
            for index, name in microphones():
                self.microphone.addItem(name, index)
        except Exception:
            self.status.setText("Аудиоустройства недоступны; текстовый чат работает")
        selected = self.microphone.findData(self.settings.microphone_device)
        if selected >= 0:
            self.microphone.setCurrentIndex(selected)
        self.output.clear()
        self.output.addItem("Устройство вывода по умолчанию", "")
        for device in QMediaDevices.audioOutputs():
            self.output.addItem(device.description(), bytes(device.id().toHex().data()).decode("ascii"))
        selected = self.output.findData(self.settings.audio_output_device)
        if selected >= 0:
            self.output.setCurrentIndex(selected)

    def apply_settings(self) -> None:
        from pydantic import SecretStr

        try:
            settings = Settings(
                **{
                    **self.settings.model_dump(),
                    "stt_provider": self.stt.currentText(),
                    "voice_mode": self.mode.currentText(),
                    "microphone_device": self.microphone.currentData(),
                    "audio_output_device": self.output.currentData(),
                    "tts_provider": self.tts.currentText(),
                    "tts_voice": self.voice.currentText() or "coral",
                    "speech_rate": self.speed.value() / 100,
                    "ptt_hotkey": self.hotkey.text(),
                    "speech_api_base_url": self.url.text(),
                    "speech_api_key": SecretStr(self.key.text()),
                    "stt_model": self.stt_model.text(),
                    "tts_model": self.tts_model.text(),
                    "whisper_model": self.local_model.currentText(),
                    "allow_model_download": self.download.isChecked(),
                }
            )
        except ValidationError:
            QMessageBox.warning(
                self,
                "Настройки не применены",
                "Проверьте HTTPS URL, скорость и комбинацию PTT с CTRL/ALT/SHIFT; F8/F9/F10 заняты панелью.",
            )
            return
        self.mute.setChecked(False)
        self.settings = settings
        self.settings_changed.emit(settings)

    def microphone_changed(self, enabled: bool) -> None:
        if not enabled:
            if self.capture:
                self.capture.close()
                self.capture = None
            if self.recognizer:
                self.recognizer.stop()
                self.recognizer = None
            self.listening.setText("Микрофон выключен")
            return
        if self.settings.stt_provider == "disabled":
            self.status.setText("Выберите STT cloud/local и нажмите «Применить»")
            self.mute.setChecked(False)
            return
        recognizer = RecognitionWorker(self.settings)
        recognizer.setParent(self)
        recognizer.updated.connect(self.on_audio_event)
        recognizer.finished.connect(self.service_finished.emit)
        capture = MicrophoneCapture(
            Segmenter(self.settings.voice_mode, self.settings.vad_threshold, self.settings.vad_silence),
            recognizer.submit,
            lambda text: self.audio_event.emit({"kind": "status", "text": text}),
        )
        self.recognizer = recognizer
        self.recognizers.append(recognizer)
        self.capture = capture
        try:
            capture.start(self.settings.microphone_device)
        except SpeechError as error:
            self.status.setText(str(error))
            self.mute.setChecked(False)
            return
        recognizer.start()

    def hold(self, value: bool) -> None:
        if self.capture is None:
            return
        if value:
            self.interrupt_requested.emit()
        if self.capture:
            if value:
                self.capture.ptt.set()
            else:
                self.capture.ptt.clear()

    def on_audio_event(self, event: dict[str, Any]) -> None:
        if event["kind"] == "transcript":
            self.recognized.setText(event["text"])
            self.transcript.emit(event["text"])
        elif event["kind"] == "recognizing":
            self.status.setText("Распознаю речь…" if event["value"] else "Распознавание завершено")
        else:
            self.status.setText(event["text"])

    def poll(self) -> None:
        speaking = self.speech.speaking
        self.speaking.setText("ИИ говорит" if speaking else "ИИ молчит")
        if self.capture:
            self.capture.set_speaking(speaking)
            active = self.capture.segmenter.active and not speaking
            self.listening.setText("ИИ слушает" if active else "Микрофон включён: ожидание / защита от эха")
        self.ptt.setEnabled(self.capture is not None)

    def shutdown(self) -> None:
        self.timer.stop()
        self.mute.setChecked(False)
        for worker in self.recognizers:
            worker.stop()

    def running(self) -> bool:
        return any(worker.isRunning() for worker in self.recognizers)
