from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
import threading
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from loliland_agent.config import Settings
from loliland_agent.memory import Memory
from loliland_agent.models import Observation
from loliland_agent.perception.recording import RecordingSource
from loliland_agent.perception.sources import ImageSource, ObservationSource, ReplaySource, ScreenSource
from loliland_agent.perception.windows import Win32
from loliland_agent.speech.tts import RussianQtSpeech, SilentSpeech, SpeechBackend, SpeechController
from loliland_agent.ui.worker import CopilotWorker


class GuardedWindowCapture:
    def __init__(self, source: ScreenSource, phase: Any) -> None:
        self.source, self.phase = source, phase

    async def observe(self) -> Observation:
        self.phase.emit(True)
        try:
            await asyncio.sleep(0.2)
            return await self.source.observe()
        finally:
            self.phase.emit(False)


class MainWindow(QMainWindow):
    hotkey = Signal(str)
    capture_phase = Signal(bool)

    def __init__(
        self,
        settings: Settings,
        replay: Path | None = None,
        image: Path | None = None,
        demo: bool = False,
        record: Path | None = None,
        speech_backend: SpeechBackend | None = None,
    ) -> None:
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.settings, self.demo, self.record = settings, demo, record
        self.worker: CopilotWorker
        widget: QWidget
        self.memory = Memory(settings.database_path)
        self._state: dict[str, Any] = {}
        self._busy = False
        self._closing = False
        self._compact = False
        self._capture_hidden = False
        self._source_native = False
        self._selected_hwnd: int | None = None
        self._hotkey_stop = threading.Event()
        self._hotkey_thread: threading.Thread | None = None
        self.api = Win32() if sys.platform == "win32" and not (image or replay or demo) else None
        if speech_backend is None:
            try:
                speech_backend = RussianQtSpeech()
            except Exception:
                speech_backend = SilentSpeech()
        self.speech = SpeechController(speech_backend)
        self.speech.set_volume(settings.speech_volume)
        self.setWindowTitle("LoliLand AI Copilot — действия выполняет человек")
        self.resize(1180, 850)
        root = QWidget()
        layout = QVBoxLayout(root)
        self.setCentralWidget(root)
        self.connection = QLabel("Источник не выбран")
        self.status = QLabel("Copilot готов. Игровой ввод отсутствует.")
        for label in (self.connection, self.status):
            label.setTextFormat(Qt.TextFormat.PlainText)
            layout.addWidget(label)
        goal_row = QHBoxLayout()
        self.goal_input = QLineEdit()
        self.goal_input.setPlaceholderText("Главная цель обычным русским языком")
        self.goal_input.setMaxLength(2000)
        self.priority = QSpinBox()
        self.priority.setRange(0, 100)
        self.priority.setValue(50)
        self.priority.setPrefix("Приоритет ")
        self.start_button = QPushButton("Начать / Продолжить")
        self.start_button.clicked.connect(self.start_goal)
        goal_row.addWidget(self.goal_input, 1)
        goal_row.addWidget(self.priority)
        goal_row.addWidget(self.start_button)
        layout.addLayout(goal_row)
        self.goal_queue = QComboBox()
        self.goal_queue.activated.connect(self.select_goal)
        layout.addWidget(self.goal_queue)
        source_row = QHBoxLayout()
        self.windows = QComboBox()
        self.refresh_button = QPushButton("Обновить окна")
        self.connect_button = QPushButton("Наблюдать окно")
        self.image_button = QPushButton("Скриншот…")
        self.refresh_button.clicked.connect(self.refresh_windows)
        self.connect_button.clicked.connect(self.connect_window)
        self.image_button.clicked.connect(self.open_image)
        for widget in (self.windows, self.refresh_button, self.connect_button, self.image_button):
            source_row.addWidget(widget)
        self.source_row_widget = QWidget()
        self.source_row_widget.setLayout(source_row)
        layout.addWidget(self.source_row_widget)
        instruction_box = QWidget()
        instructions = QVBoxLayout(instruction_box)
        self.task = QLabel("Текущая задача: —")
        self.next_action = QLabel("Следующее действие появится после задания цели и анализа кадра.")
        self.next_action.setStyleSheet("font-size: 18px; font-weight: 600;")
        self.next_action.setWordWrap(True)
        self.reason = QLabel()
        self.reason.setWordWrap(True)
        for label in (self.task, self.next_action, self.reason):
            label.setTextFormat(Qt.TextFormat.PlainText)
            instructions.addWidget(label)
        layout.addWidget(instruction_box)
        buttons = QHBoxLayout()
        self.check_button = QPushButton("Проверить / Что дальше? [F9]")
        self.done_button = QPushButton("Готово")
        self.repeat_button = QPushButton("Повтори [F10]")
        self.skip_button = QPushButton("Пропустить")
        self.pause_button = QPushButton("Пауза [F8]")
        self.check_button.clicked.connect(lambda: self.request_check(False))
        self.done_button.clicked.connect(self.done)
        self.repeat_button.clicked.connect(self.repeat)
        self.skip_button.clicked.connect(self.skip)
        self.pause_button.clicked.connect(self.pause)
        for button in (
            self.check_button,
            self.done_button,
            self.repeat_button,
            self.skip_button,
            self.pause_button,
        ):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        options = QHBoxLayout()
        self.auto = QCheckBox("Проверять изменившийся кадр автоматически")
        self.voice = QCheckBox("Русская озвучка")
        self.voice.toggled.connect(self.voice_changed)
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(settings.speech_volume)
        self.volume.setMaximumWidth(120)
        self.volume.valueChanged.connect(self.volume_changed)
        self.cancel_voice = QPushButton("Остановить голос")
        self.cancel_voice.clicked.connect(self.speech.cancel)
        self.compact = QCheckBox("Компактно поверх игры")
        self.compact.toggled.connect(self.toggle_compact)
        self.api_button = QPushButton("Настройки API")
        self.api_button.clicked.connect(self.configure_api)
        for widget in (self.auto, self.voice, self.volume, self.cancel_voice, self.compact, self.api_button):
            options.addWidget(widget)
        layout.addLayout(options)
        self.voice_status = QLabel(self.speech.backend.reason)
        self.voice_status.setWordWrap(True)
        layout.addWidget(self.voice_status)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        observe = QWidget()
        observe_layout = QVBoxLayout(observe)
        self.perception_label = QLabel("Кадр ещё не проанализирован")
        self.perception_label.setWordWrap(True)
        self.perception_label.setTextFormat(Qt.TextFormat.PlainText)
        self.screenshot = QLabel("Скриншот выбранного окна")
        self.screenshot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.screenshot.setMinimumHeight(200)
        observe_layout.addWidget(self.perception_label)
        observe_layout.addWidget(self.screenshot, 1)
        self.storage_name = QLineEdit()
        self.storage_name.setPlaceholderText(
            "Имя текущего хранилища (неизвестные контейнеры в запас не засчитываются)"
        )
        self.storage_name.editingFinished.connect(
            lambda: self.worker.request("storage", self.storage_name.text())
        )
        observe_layout.addWidget(self.storage_name)
        if demo:
            self.demo_next = QPushButton("Человек: показать следующий демонстрационный кадр")
            self.demo_next.clicked.connect(self.demo_advance)
            observe_layout.addWidget(self.demo_next)
        self.tabs.addTab(observe, "Наблюдение")
        plan_widget = QWidget()
        plan_layout = QVBoxLayout(plan_widget)
        self.progress = QProgressBar()
        self.plan_view = QTreeWidget()
        self.plan_view.setHeaderLabels(["Задача", "Состояние", "Приоритет", "Зависимости"])
        plan_layout.addWidget(self.progress)
        plan_layout.addWidget(self.plan_view)
        plan_buttons = QHBoxLayout()
        self.replan_button = QPushButton("Перепланировать")
        self.edit_plan_button = QPushButton("Изменить план")
        self.complete_button = QPushButton("Главная цель достигнута")
        self.replan_button.clicked.connect(
            lambda: self.worker.request("replan", replace=True, delay=self.capture_delay())
        )
        self.edit_plan_button.clicked.connect(self.edit_plan)
        self.complete_button.clicked.connect(self.complete_goal)
        for widget in (self.replan_button, self.edit_plan_button, self.complete_button):
            plan_buttons.addWidget(widget)
        plan_layout.addLayout(plan_buttons)
        self.tabs.addTab(plan_widget, "План развития")
        resources = QWidget()
        resource_layout = QVBoxLayout(resources)
        self.resource_view = QPlainTextEdit()
        self.resource_view.setReadOnly(True)
        resource_layout.addWidget(self.resource_view)
        facts_button = QPushButton("Подтверждённые факты / ресурсы / база…")
        facts_button.clicked.connect(self.edit_facts)
        stale_button = QPushButton("Пометить известные ресурсы устаревшими")
        stale_button.clicked.connect(lambda: self.worker.request("invalidate", replace=True))
        resource_layout.addWidget(facts_button)
        resource_layout.addWidget(stale_button)
        self.tabs.addTab(resources, "Ресурсы и рецепты")
        recipe_buttons = QHBoxLayout()
        doc_button = QPushButton("Изучить документ сборки…")
        recipe_button = QPushButton("Подтвердить рецепт…")
        doc_button.clicked.connect(self.import_document)
        recipe_button.clicked.connect(self.import_recipe)
        recipe_buttons.addWidget(doc_button)
        recipe_buttons.addWidget(recipe_button)
        resource_layout.addLayout(recipe_buttons)
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        self.chat_view = QPlainTextEdit()
        self.chat_view.setReadOnly(True)
        chat_layout.addWidget(self.chat_view)
        chat_row = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Вопрос ИИ о цели, рецептах или следующем шаге")
        self.chat_send = QPushButton("Отправить")
        self.chat_send.clicked.connect(self.send_chat)
        self.chat_input.returnPressed.connect(self.send_chat)
        chat_row.addWidget(self.chat_input)
        chat_row.addWidget(self.chat_send)
        chat_layout.addLayout(chat_row)
        self.tabs.addTab(chat_widget, "Чат с ИИ")
        self.journal = QPlainTextEdit()
        self.journal.setReadOnly(True)
        self.journal.document().setMaximumBlockCount(500)
        self.tabs.addTab(self.journal, "История инструкций")
        self.usage_label = QLabel()
        layout.addWidget(self.usage_label)
        self.refresh_windows()
        source: ObservationSource | None = None
        self._demo_index = 0
        self._demo_paths = [
            Path(__file__).parent.parent / "assets/demo-initial.png",
            Path(__file__).parent.parent / "assets/demo-inventory.png",
        ]
        if demo:
            source = ImageSource(self._demo_paths[0], settings.database_path.parent / "capture")
            self.connection.setText("DEMO: искусственные снимки, без Minecraft и без нейросети")
            self.goal_input.setText("Демонстрация: показать инвентарь")
        elif replay:
            source = ReplaySource(replay)
            self.connection.setText("Запись: " + replay.name)
        elif image:
            source = ImageSource(image, settings.database_path.parent / "capture")
            self.connection.setText("Изображение: " + image.name)
        if source and record:
            source = RecordingSource(source, record)
        self.worker = CopilotWorker(settings, self.memory, source, demo)
        self.worker.setParent(self)
        self.worker.updated.connect(self.on_event)
        self.worker.finished.connect(self.worker_finished)
        self.capture_phase.connect(self.on_capture_phase)
        self.hotkey.connect(self.on_hotkey)
        self.auto_timer = QTimer(self)
        self.auto_timer.setInterval(int(settings.interval_seconds * 1000))
        self.auto_timer.timeout.connect(self.auto_check)
        self.auto_timer.start()
        self.worker.start()
        if settings.speech_enabled:
            self.voice.setChecked(True)
        if self.api:

            def watch() -> None:
                assert self.api is not None
                previous = {"pause": False, "check": False, "repeat": False}
                while not self._hotkey_stop.wait(0.04):
                    hwnd = self._selected_hwnd
                    for kind, key in [("pause", 0x77), ("check", 0x78), ("repeat", 0x79)]:
                        pressed = self.api.key_pressed(key)
                        if pressed and not previous[kind] and hwnd is not None and self.api.focused(hwnd):
                            self.hotkey.emit(kind)
                        previous[kind] = pressed

            self._hotkey_thread = threading.Thread(
                target=watch, daemon=True, name="copilot-notification-hotkeys"
            )
            self._hotkey_thread.start()

    def refresh_windows(self) -> None:
        self.windows.clear()
        if self.api:
            for window in self.api.windows():
                if re.search(self.settings.window_title_pattern, window.title) and not re.search(
                    r"(?i)launcher|лаунчер|copilot", window.title
                ):
                    self.windows.addItem(window.title, window.hwnd)
        else:
            self.windows.addItem("Захват окна доступен в Windows; можно загрузить снимок", None)

    def capture_delay(self) -> float:
        if self._source_native:
            self.status.setText("Через 3 секунды: вернитесь в игровое окно; Copilot сам фокус не меняет")
            return 3
        return 0

    def connect_window(self) -> None:
        hwnd = self.windows.currentData()
        if self.api is None or hwnd is None:
            return
        try:
            source: ObservationSource = GuardedWindowCapture(
                ScreenSource(int(hwnd), self.api, self.settings.database_path.parent / "capture"),
                self.capture_phase,
            )
            if self.record:
                source = RecordingSource(source, self.record)
        except (RuntimeError, ValueError):
            QMessageBox.warning(
                self, "Источник недоступен", "Выберите существующее окно и новый каталог записи"
            )
            return
        self._selected_hwnd = int(hwnd)
        self._source_native = True
        self.worker.request("source", source, replace=True)
        self.connection.setText("Выбрано окно: " + self.windows.currentText())

    def open_image(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Открыть скриншот", "", "Изображения (*.png *.jpg *.jpeg)"
        )
        if filename:
            self._source_native = False
            self._selected_hwnd = None
            self.worker.request(
                "source",
                ImageSource(Path(filename), self.settings.database_path.parent / "capture"),
                replace=True,
            )
            self.connection.setText("Изображение: " + Path(filename).name)

    def demo_advance(self) -> None:
        self._demo_index = min(self._demo_index + 1, len(self._demo_paths) - 1)
        self.worker.request(
            "source",
            ImageSource(self._demo_paths[self._demo_index], self.settings.database_path.parent / "capture"),
            replace=True,
        )
        self.status.setText("Человек показал новый демонстрационный снимок; нажмите Проверить или Готово")

    def start_goal(self) -> None:
        if not self.goal_input.text().strip():
            return
        self.speech.cancel()
        self.worker.request(
            "start",
            {"goal": self.goal_input.text(), "priority": self.priority.value()},
            replace=True,
            delay=self.capture_delay(),
        )

    def select_goal(self, index: int) -> None:
        identifier = self.goal_queue.itemData(index)
        if identifier:
            self.worker.request("select", identifier, replace=True)

    def request_check(self, manual: bool, hotkey: bool = False) -> None:
        self.worker.request("done" if manual else "check", delay=0 if hotkey else self.capture_delay())

    def done(self) -> None:
        pending = (self._state.get("goal") or {}).get("pending")
        if pending and pending["verification"] == "uncertain":
            self.manual_question(
                {
                    "instruction_id": pending["id"],
                    "text": pending["instruction"]["text"],
                    "evidence": pending["evidence"],
                }
            )
        else:
            self.request_check(True)

    def manual_question(self, event: dict[str, Any]) -> None:
        question = (
            "Визуального подтверждения нет.\nИнструкция: "
            + event["text"]
            + "\nПричина: "
            + event["evidence"]
            + "\nПодтвердить выполнение со слов пользователя? Это не будет записано как визуальная проверка."
        )
        if (
            QMessageBox.question(
                self,
                "Ручное подтверждение",
                question,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            == QMessageBox.StandardButton.Yes
        ):
            self.worker.request("manual", event["instruction_id"], replace=True)

    def skip(self) -> None:
        pending = (self._state.get("goal") or {}).get("pending")
        if pending:
            self.worker.request("skip", pending["id"], replace=True)

    def pause(self) -> None:
        self.auto.setChecked(False)
        self.speech.cancel()
        self.worker.request("pause", replace=True)
        self.status.setText("Пауза; новые инструкции отменены")

    def repeat(self) -> None:
        try:
            self.speech.repeat()
        except RuntimeError:
            self.voice.setChecked(False)
        self.status.setText(self.speech.last or self.next_action.text())

    def voice_changed(self, value: bool) -> None:
        enabled = self.speech.enable(value)
        if value and not enabled:
            self.voice.blockSignals(True)
            self.voice.setChecked(False)
            self.voice.blockSignals(False)
        self.voice_status.setText(self.speech.backend.reason)

    def volume_changed(self, value: int) -> None:
        self.speech.set_volume(value)

    def toggle_compact(self, value: bool) -> None:
        self._compact = value
        self.tabs.setVisible(not value)
        self.source_row_widget.setVisible(not value)
        self.goal_queue.setVisible(not value)
        self.goal_input.setVisible(not value)
        self.priority.setVisible(not value)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, value)
        self.resize(600, 300) if value else self.resize(1180, 850)
        self.show()

    def on_capture_phase(self, capturing: bool) -> None:
        if capturing and self._compact and self.isVisible():
            self._capture_hidden = True
            self.hide()
        elif not capturing and self._capture_hidden:
            self._capture_hidden = False
            if not self._closing:
                self.show()

    def auto_check(self) -> None:
        goal = self._state.get("goal")
        if self.auto.isChecked() and not self._busy and goal and goal["status"] == "active":
            if self._source_native and (
                self.api is None or self._selected_hwnd is None or not self.api.focused(self._selected_hwnd)
            ):
                return
            self.worker.request("auto")

    def on_hotkey(self, kind: str) -> None:
        if kind == "pause":
            self.pause()
        elif kind == "repeat":
            self.repeat()
        elif kind == "check" and not self._busy:
            self.request_check(False, True)

    def send_chat(self) -> None:
        text = self.chat_input.text().strip()
        if text:
            self.worker.request("chat", text)
            self.chat_input.clear()

    def complete_goal(self) -> None:
        if (
            QMessageBox.question(
                self,
                "Главная цель",
                "Подтверждаете достижение главной цели?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            == QMessageBox.StandardButton.Yes
        ):
            self.worker.request("complete", replace=True)

    def json_dialog(self, title: str, value: Any, operation: str) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        layout = QVBoxLayout(dialog)
        text = QPlainTextEdit(json.dumps(value, ensure_ascii=False, indent=2))
        layout.addWidget(text)
        save = QPushButton("Сохранить подтверждённые данные")
        layout.addWidget(save)

        def accept() -> None:
            self.worker.request(operation, text.toPlainText(), replace=True)
            dialog.accept()

        save.clicked.connect(accept)
        dialog.resize(750, 600)
        dialog.exec()

    def edit_plan(self) -> None:
        goal = self._state.get("goal")
        if goal:
            self.json_dialog(
                "Изменить задачи, приоритеты и зависимости; подтверждения неизменяемы",
                goal["steps"],
                "plan_edit",
            )

    def edit_facts(self) -> None:
        facts = (self._state.get("context") or {}).get("confirmed_facts", {})
        self.json_dialog(
            "Подтверждённые пользователем факты (только ваша информация)", {**facts, "resources": {}}, "facts"
        )

    def import_recipe(self) -> None:
        self.json_dialog(
            "Подтверждённый рецепт конкретной версии сборки",
            {
                "output": {"item": "УКАЖИТЕ ПРЕДМЕТ", "count": 1},
                "ingredients": [{"item": "УКАЖИТЕ КОМПОНЕНТ", "count": 1}],
                "machine": None,
                "technologies": [],
                "complete": False,
                "confidence": 0.0,
                "evidence_text": "Источник и проверенные количества",
            },
            "recipe",
        )

    def import_document(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Документ конкретной сборки", "", "Текст (*.txt *.md)"
        )
        if not filename:
            return
        try:
            path = Path(filename)
            if path.stat().st_size > 500_000:
                raise ValueError("Document too large")
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, ValueError):
            QMessageBox.warning(self, "Документ недоступен", "Нужен UTF-8 текст до 100000 символов")
            return
        reference = path.name + ":" + hashlib.sha256(text.encode()).hexdigest()
        self.worker.request("document", {"text": text, "reference": reference})

    def configure_api(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("API: изменения действуют в текущем сеансе")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        provider = QComboBox()
        provider.addItems(["offline", "openai", "compatible", "ollama"])
        provider.setCurrentText(self.settings.provider)
        model = QLineEdit(self.settings.model)
        url = QLineEdit(self.settings.api_base_url)
        key = QLineEdit(self.settings.api_key.get_secret_value())
        key.setEchoMode(QLineEdit.EchoMode.Password)
        pack = QLineEdit(self.settings.pack_version)
        interval = QSpinBox()
        interval.setRange(3, 3600)
        interval.setValue(int(self.settings.interval_seconds))
        for label, widget in [
            ("Провайдер", provider),
            ("Vision-модель", model),
            ("API URL", url),
            ("Ключ LLM (не Minecraft)", key),
            ("Версия сборки", pack),
            ("Интервал наблюдения, сек", interval),
        ]:
            form.addRow(label, widget)
        layout.addLayout(form)
        layout.addWidget(
            QLabel(
                "Ключ не записывается на диск этой формой. Для постоянных настроек используйте .env вне Git. Новая версия сборки применяется при старте новой цели."
            )
        )
        apply = QPushButton("Применить")
        layout.addWidget(apply)

        def accept() -> None:
            from pydantic import SecretStr, ValidationError

            try:
                updated = Settings(  # type: ignore[call-arg]
                    _env_file=None,
                    **{
                        **self.settings.model_dump(),
                        "provider": provider.currentText(),
                        "model": model.text(),
                        "api_base_url": url.text(),
                        "api_key": SecretStr(key.text()),
                        "pack_version": pack.text(),
                        "interval_seconds": interval.value(),
                    },
                )
            except ValidationError:
                QMessageBox.warning(
                    dialog,
                    "Неверные настройки",
                    "Проверьте провайдера, HTTPS URL и наличие ключа; значения скрыты",
                )
                return
            self.settings = updated
            self.worker.request("configure", updated, replace=True)
            self.auto_timer.setInterval(int(updated.interval_seconds * 1000))
            dialog.accept()

        apply.clicked.connect(accept)
        dialog.resize(600, 400)
        dialog.exec()

    def on_event(self, event: dict[str, Any]) -> None:
        if event.get("revision", 0) != self.worker.guard.current():
            return
        kind = event["kind"]
        if kind == "busy":
            self._busy = event["value"]
            if self._busy:
                self.status.setText("Анализ: " + event["operation"])
            elif self.status.text().startswith("Анализ: "):
                self.status.setText("Ожидаю действия человека")
            for widget in (self.check_button, self.done_button, self.skip_button, self.chat_send):
                widget.setEnabled(not self._busy)
            return
        if kind == "snapshot":
            self.render_snapshot(event)
            return
        if kind == "observation":
            self.render_observation(event)
        elif kind == "instruction":
            self.next_action.setText(event["text"])
            self.reason.setText(event.get("reason", ""))
            try:
                self.speech.instruction(event["text"])
            except RuntimeError:
                self.voice.setChecked(False)
                self.voice_status.setText(self.speech.backend.reason)
        elif kind == "manual_available":
            self.manual_question(event)
        elif kind == "chat":
            self.chat_view.appendPlainText(("Вы: " if event["role"] == "user" else "ИИ: ") + event["text"])
        elif kind in {"error", "notice", "goal_review"}:
            self.status.setText(event["text"])
        if kind not in {"observation"}:
            self.journal.appendPlainText(
                json.dumps(
                    {key: value for key, value in event.items() if key != "revision"}, ensure_ascii=False
                )
            )

    def render_observation(self, value: dict[str, Any]) -> None:
        visual = value["visual"]
        self.perception_label.setText(
            f"{visual['scene']} · уверенность {visual['confidence']:.0%} · {visual['summary']}"
        )
        if value.get("image_path"):
            self.screenshot.setPixmap(
                QPixmap(value["image_path"]).scaled(
                    900, 380, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
            )
        self.connection.setText(
            "Окно наблюдается" if value["source"] == "screen" else "Снимок / запись — не подключение к игре"
        )

    def render_snapshot(self, event: dict[str, Any]) -> None:
        self._state = event
        goal = event["goal"]
        self.goal_queue.blockSignals(True)
        self.goal_queue.clear()
        for candidate in event["goals"]:
            self.goal_queue.addItem(
                f"[{candidate['priority']}] {candidate['status']}: {candidate['description']}",
                candidate["id"],
            )
        if goal:
            self.goal_queue.setCurrentIndex(self.goal_queue.findData(goal["id"]))
        self.goal_queue.blockSignals(False)
        if goal:
            if not self.goal_input.hasFocus():
                self.goal_input.setText(goal["description"])
            self.priority.setValue(goal["priority"])
            self.plan_view.clear()
            for step in goal["steps"]:
                QTreeWidgetItem(
                    self.plan_view,
                    [step["title"], step["status"], str(step["priority"]), ", ".join(step["depends_on"])],
                )
            self.progress.setRange(0, max(1, len(goal["steps"])))
            self.progress.setValue(sum(step["status"] == "done" for step in goal["steps"]))
            pending = goal["pending"]
            if pending:
                instruction = pending["clarification"] or pending["instruction"]
                self.next_action.setText(instruction["text"])
                self.reason.setText(
                    instruction["reason"] + ("\n" + pending["evidence"] if pending["evidence"] else "")
                )
                self.speech.last = instruction["text"]
                task_id = pending["instruction"]["task_id"]
                self.task.setText(
                    "Текущая задача: "
                    + next(
                        (step["title"] for step in goal["steps"] if step["id"] == task_id), "Сбор информации"
                    )
                )
        if event.get("observation"):
            self.render_observation(event["observation"])
        context = event["context"]
        self.resource_view.setPlainText(
            "Известны только наблюдавшиеся/подтверждённые ресурсы; скрытое содержимое не учтено.\n"
            + json.dumps(
                {
                    "materials": context["materials"],
                    "known_resources": context["observed_resources"],
                    "recipes": context["recipes"],
                    "facts": context["confirmed_facts"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        usage = event["usage"]
        cost = "неизвестна" if usage["cost_usd"] is None else f"${usage['cost_usd']:.6f}"
        self.usage_label.setText(
            f"API: {usage['calls']} · токены {usage['input_tokens']} / {usage['output_tokens']} · стоимость {cost}"
        )
        if not self.journal.toPlainText():
            for row in reversed(context["history"]):
                if row["kind"] in {"instruction", "confirmed", "verification", "skipped"}:
                    self.journal.appendPlainText(json.dumps(row["payload"], ensure_ascii=False))

    def worker_finished(self) -> None:
        if self._closing:
            self.close()
        else:
            self.status.setText("Фоновый процесс завершён")

    def closeEvent(self, event: QCloseEvent) -> None:
        self._closing = True
        self.auto_timer.stop()
        self.speech.cancel()
        self._hotkey_stop.set()
        if self._hotkey_thread:
            self._hotkey_thread.join(timeout=0.5)
        self.worker.stop()
        if self.worker.isRunning():
            event.ignore()
            return
        self.worker.wait()
        self.worker.updated.disconnect(self.on_event)
        self.worker.finished.disconnect(self.worker_finished)
        self.memory.close()
        event.accept()


def launch(
    settings: Settings,
    replay: Path | None = None,
    image: Path | None = None,
    demo: bool = False,
    record: Path | None = None,
) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(settings, replay, image, demo, record)
    window.show()
    return app.exec()
