import time

from PySide6.QtCore import QCoreApplication, QEvent

from loliland_agent.config import Settings
from loliland_agent.speech.tts import SilentSpeech, SpeechController
from loliland_agent.ui.app import MainWindow


def until(qapp, predicate, timeout=6):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    assert predicate(), "Qt event did not arrive"


def close(qapp, window):
    worker = window.worker
    window.close()
    until(qapp, lambda: not worker.isRunning())
    qapp.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_desktop_human_cycle_pause_and_resume(qapp, tmp_path):
    window = MainWindow(
        Settings(_env_file=None, database_path=tmp_path / "copilot.db", interval_seconds=3),
        demo=True,
        speech_backend=SilentSpeech(),
    )
    window.show()
    try:
        until(qapp, lambda: bool(window._state))
        window.goal_input.setText("Создай переработку руд")
        window.start_button.click()
        until(qapp, lambda: window.next_action.text() == "Открой инвентарь." and not window._busy)
        assert window._state["goal"]["confirmed_count"] == 0
        window.demo_next.click()  # A human supplies the changed observation.
        until(qapp, lambda: not window._busy)
        window.check_button.click()
        until(qapp, lambda: window._state["goal"]["confirmed_count"] == 1 and not window._busy)
        assert window._state["goal"]["manual_count"] == 0 and window.progress.value() == 1
        assert "NEI" in window.next_action.text()
        window.repeat_button.click()
        assert "NEI" in window.status.text()
        window.pause_button.click()
        until(qapp, lambda: window._state["goal"]["status"] == "paused" and not window._busy)
        window.compact.setChecked(True)
        assert not window.tabs.isVisible()
    finally:
        close(qapp, window)


def test_window_closes_during_pending_work(qapp, tmp_path):
    window = MainWindow(
        Settings(_env_file=None, database_path=tmp_path / "close.db"),
        demo=True,
        speech_backend=SilentSpeech(),
    )
    window.show()
    window.start_button.click()
    close(qapp, window)


class FakeSpeech:
    available = True
    reason = "Русский тестовый голос"

    def __init__(self):
        self.spoken = []
        self.stops = 0
        self.levels = []

    def say(self, text):
        self.spoken.append(text)

    def stop(self):
        self.stops += 1

    def volume(self, value):
        self.levels.append(value)


def test_speech_controls_repeat_cancel_and_volume():
    backend = FakeSpeech()
    controller = SpeechController(backend)
    controller.instruction("Открой инвентарь.")
    assert not backend.spoken
    assert controller.enable(True)
    controller.repeat()
    controller.instruction("Покажи рецепт.")
    controller.set_volume(25)
    controller.cancel()
    controller.enable(False)
    controller.repeat()
    assert backend.spoken == ["Открой инвентарь.", "Покажи рецепт."]
    assert backend.levels[-1] == 0.25 and backend.stops >= 4
    assert SpeechController(SilentSpeech()).enable(True) is False
