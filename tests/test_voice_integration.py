import asyncio
import threading

from conftest import frame
from test_ui import close, until

from loliland_agent.config import Settings
from loliland_agent.conversation.models import DialogueReply
from loliland_agent.planning.providers import DemoProvider, OfflineProvider
from loliland_agent.speech.tts import SilentSpeech
from loliland_agent.ui.app import MainWindow


def window(qapp, tmp_path):
    value = MainWindow(
        Settings(
            _env_file=None,
            provider="offline",
            tts_provider="system",
            database_path=tmp_path / "voice.db",
            interval_seconds=3,
        ),
        demo=True,
        speech_backend=SilentSpeech(),
    )
    value.show()
    until(qapp, lambda: bool(value._state))
    value.start_goal()
    until(qapp, lambda: bool((value._state.get("goal") or {}).get("pending")) and not value._busy)
    return value


def test_spoken_commands_update_goal_panel_and_persist_dialogue(qapp, tmp_path):
    view = window(qapp, tmp_path)
    try:
        view.submit_dialogue("Добавь в план строительство склада", "voice")
        until(
            qapp, lambda: any("склада" in s["title"] for s in view._state["goal"]["steps"]) and not view._busy
        )
        assert view.plan_view.topLevelItemCount() == 2
        view.submit_dialogue("Поставь приоритет 80", "voice")
        until(qapp, lambda: view._state["goal"]["priority"] == 80 and not view._busy)
        view.submit_dialogue("Отложи текущую задачу", "voice")
        until(qapp, lambda: view._state["goal"]["steps"][0]["status"] == "paused" and not view._busy)
        view.submit_dialogue("Пауза", "voice")
        until(qapp, lambda: view._state["goal"]["status"] == "paused" and not view._busy)
        assert any(row["channel"] == "voice" for row in view.memory.messages())
        assert view._state["goal"]["confirmed_count"] == 0
    finally:
        close(qapp, view)


def test_goal_change_previous_and_resume_without_camera(qapp, tmp_path):
    view = window(qapp, tmp_path)
    try:
        original = view._state["goal"]["id"]
        view.submit_dialogue("Смени цель на магию", "voice")
        until(qapp, lambda: view._state["goal"]["description"] == "магию" and not view._busy)
        assert view._state["goal"]["id"] != original
        view.submit_dialogue("Вернись к предыдущей цели", "voice")
        until(qapp, lambda: view._state["goal"]["id"] == original and not view._busy)
        view.submit_dialogue("Пауза", "voice")
        until(qapp, lambda: view._state["goal"]["status"] == "paused" and not view._busy)
        view.submit_dialogue("Продолжай", "voice")
        until(qapp, lambda: view._state["goal"]["status"] == "active" and not view._busy)
        assert len(view._state["goals"]) == 2
    finally:
        close(qapp, view)


def test_dialogue_works_while_screen_plan_waits_and_pause_cancels_it(qapp, tmp_path, monkeypatch):
    entered = threading.Event()

    class SlowVisionPlanner(DemoProvider):
        async def plan(self, *args):
            entered.set()
            while True:
                await asyncio.sleep(0.05)

    class FastDialogue(OfflineProvider):
        async def converse(self, text, context):
            return DialogueReply(
                answer="Обсудим солнечные панели, но сначала нужен их рецепт.", intent="suggest_plan"
            )

    monkeypatch.setattr("loliland_agent.ui.worker.make_provider", lambda *args: SlowVisionPlanner())
    monkeypatch.setattr("loliland_agent.conversation.worker.make_provider", lambda *args: FastDialogue())
    view = MainWindow(
        Settings(
            _env_file=None, provider="offline", tts_provider="system", database_path=tmp_path / "parallel.db"
        ),
        image=frame().image_path,
        speech_backend=SilentSpeech(),
    )
    view.show()
    try:
        until(qapp, lambda: bool(view._state))
        view.goal_input.setText("Переработка руд")
        view.start_goal()
        until(qapp, lambda: entered.is_set())
        view.submit_dialogue("Может, сделаем солнечные панели?", "voice")
        until(qapp, lambda: "Обсудим солнечные панели" in view.voice_panel.history.toPlainText())
        assert view._busy and view._state["goal"]["pending"] is None
        view.submit_dialogue("Пауза", "voice")
        until(qapp, lambda: view._state["goal"]["status"] == "paused" and not view._busy)
        assert view._state["goal"]["pending"] is None
    finally:
        close(qapp, view)


def test_switch_ptt_hands_free_and_microphone_error_leave_text_usable(qapp, tmp_path, monkeypatch):
    from loliland_agent.speech.stt import SpeechError

    view = window(qapp, tmp_path)
    try:
        panel = view.voice_panel
        panel.mode.setCurrentText("hands_free")
        panel.stt.setCurrentText("local")
        # Avoid changing the test's silent backend; test settings signal is validated independently.
        values = []
        panel.settings_changed.disconnect(view.apply_voice_settings)
        panel.settings_changed.connect(values.append)
        panel.apply_settings()
        assert values[-1].voice_mode == "hands_free"
        monkeypatch.setattr(
            "loliland_agent.ui.voice_panel.MicrophoneCapture.start",
            lambda *args: (_ for _ in ()).throw(SpeechError("Микрофон недоступен")),
        )
        panel.mute.setChecked(True)
        assert not panel.mute.isChecked() and panel.capture is None and "Микрофон" in panel.status.text()
        view.submit_dialogue("Какая у нас цель?", "text")
        until(qapp, lambda: "Для свободного разговора" in view.chat_view.toPlainText())
    finally:
        close(qapp, view)
