import json

import httpx
import pytest
from pydantic import SecretStr

from loliland_agent.config import Settings
from loliland_agent.conversation.manager import (
    ConversationManager,
    conversation_context,
    is_directive,
    local_command,
)
from loliland_agent.conversation.models import DialogueNote, DialogueReply, DialogueSummary
from loliland_agent.planning.providers import OfflineProvider, OpenAICompatibleProvider


@pytest.mark.parametrize(
    "text",
    [
        "Может, сделаем реактор?",
        "А может, добавь реактор в план?",
        "Мне кажется, нам нужно сначала заняться энергией",
        "Не добавляй реактор",
        "Что если сменить цель?",
        "Не надо менять план",
    ],
)
def test_suggestions_are_not_plan_authorization(text):
    assert not is_directive(text)
    assert local_command(text) is None


@pytest.mark.parametrize(
    "text,intent",
    [
        ("Пауза", "pause"),
        ("Продолжай", "resume"),
        ("Возобнови работу", "resume"),
        ("Добавь в план строительство склада", "add_task"),
        ("Смени цель на изучение магии", "change_goal"),
        ("Вернись к предыдущей цели", "previous_goal"),
        ("Отложи текущую задачу", "defer_task"),
        ("Я закончил этот крафт", "done"),
        ("Проверь, всё ли я сделал правильно", "check"),
        ("Повтори последнюю инструкцию", "repeat"),
        ("Поставь приоритет 80", "prioritize"),
    ],
)
def test_natural_direct_commands(text, intent):
    assert local_command(text).intent == intent


class DialogueProvider(OfflineProvider):
    def __init__(self, reply):
        self.reply = reply
        self.contexts = []

    async def converse(self, text, context):
        self.contexts.append(context)
        return self.reply


async def test_model_cannot_turn_suggestion_into_plan_change(memory):
    text = "Может, сделаем реактор?"
    manager = ConversationManager(
        memory,
        DialogueProvider(
            DialogueReply(answer="Обсудим варианты", intent="add_task", task="Реактор", directive_quote=text)
        ),
    )
    reply = await manager.respond(text, "voice", {})
    assert reply.intent == "suggest_plan" and "не изменён" in reply.answer
    assert memory.recall("copilot", "workspace-v1") is None


async def test_quoted_user_facts_and_assumptions_have_separate_provenance(memory):
    text = "Запасы: медь находится в сундуке справа"
    reply = DialogueReply(
        answer="Запомнил твоё сообщение о меди",
        intent="memorize",
        notes=(
            DialogueNote(
                kind="user_report",
                key="copper_storage",
                text="Медь справа, 999 штук",
                quote="медь находится в сундуке справа",
            ),
            DialogueNote(kind="assumption", key="efficiency", text="Солнечные панели могут быть выгоднее"),
            DialogueNote(kind="user_report", key="invented", text="Невидимое золото", quote="золото справа"),
        ),
    )
    await ConversationManager(memory, DialogueProvider(reply)).respond(text, "voice", {})
    report = memory.recall("dialogue-notes", "copper_storage")["value"]
    assert report["provenance"] == "user_report" and report["text"] == "медь находится в сундуке справа"
    assert "999" not in report["text"] and report["interpretation"].endswith("999 штук")
    assert memory.recall("dialogue-notes", "efficiency")["value"]["provenance"] == "ai_assumption"
    assert memory.recall("dialogue-notes", "invented") is None
    assert memory.list_knowledge("copilot-resources:unspecified") == []


async def test_corrections_replace_note_and_restart_restores_context(memory):
    first = DialogueProvider(
        DialogueReply(
            answer="Запомнил",
            notes=(DialogueNote(kind="preference", key="branch", text="Магия", quote="Мне нравится магия"),),
        )
    )
    await ConversationManager(memory, first).respond("Мне нравится магия", "text", {})
    second = DialogueProvider(
        DialogueReply(
            answer="Исправил",
            intent="correct_fact",
            notes=(
                DialogueNote(kind="preference", key="branch", text="Техника", quote="Мне нравится техника"),
            ),
        )
    )
    await ConversationManager(memory, second).respond("Мне нравится техника", "voice", {})
    context = conversation_context(memory)
    assert len(context["notes"]) == 1 and context["notes"][0]["value"]["text"] == "Мне нравится техника"
    assert [r["channel"] for r in memory.messages()] == ["text", "text", "voice", "voice"]
    manager = ConversationManager(memory, first)
    await manager.respond("Почему?", "text", {"goal": {"description": "Печь"}})
    assert first.contexts[-1]["conversation"]["recent_messages"][0]["text"] == "Мне нравится магия"
    assert first.contexts[-1]["goal"]["description"] == "Печь"


async def test_bounded_context_and_summary(memory):
    class Summarizer(OfflineProvider):
        async def summarize(self, context):
            assert len(context["messages"]) <= 32
            return DialogueSummary(text="Пользователь сообщил о сундуке; рецепты ещё не проверены")

    for i in range(30):
        memory.message("user" if i % 2 == 0 else "assistant", "text", f"Сообщение {i}")
    await ConversationManager(memory, Summarizer()).compact()
    summary = memory.recall("conversation", "summary")["value"]
    assert summary["through"] == 22 and summary["provenance"] == "ai_summary_not_verified_facts"
    assert len(conversation_context(memory)["recent_messages"]) == 24
    assert len(memory.messages(after=22)) == 8


async def test_dialogue_cloud_contract_and_api_failure(memory):
    response = DialogueReply(
        answer="У солнечных панелей пока неизвестна цена: покажи рецепт NEI", intent="suggest_plan"
    )
    requests = []

    def transport(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": response.model_dump_json()}}]})

    settings = Settings(
        _env_file=None,
        provider="compatible",
        free_mode=False,
        api_key=SecretStr("test-key"),
        api_base_url="https://vision.example/v1",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        result = await ConversationManager(memory, OpenAICompatibleProvider(settings, client)).respond(
            "А может, лучше солнечные панели?", "voice", {"visual": {"scene": "inventory"}}
        )
    assert result.intent == "suggest_plan" and "NEI" in result.answer
    data = json.loads(requests[0]["messages"][1]["content"][0]["text"])["data"]
    assert data["context"]["visual"]["scene"] == "inventory"
    assert len(requests[0]["messages"][1]["content"]) == 1


async def test_offline_explicit_memory_and_dialogue_summary_never_skip_oldest_messages(memory):
    manager = ConversationManager(memory, OfflineProvider())
    await manager.respond("Запомни: медь находится в сундуке справа", "voice", {})
    assert memory.list_knowledge("dialogue-notes")[0]["value"]["provenance"] == "user_report"
    for i in range(80):
        memory.message("user", "text", str(i))
    await manager.compact()
    assert memory.recall("conversation", "summary")["value"]["through"] == 32


def test_voice_stop_interrupts_inflight_dialogue(qapp, memory):
    import asyncio
    import threading

    from test_ui import until

    from loliland_agent.config import Settings
    from loliland_agent.conversation.worker import ConversationWorker

    entered = threading.Event()

    class Slow(OfflineProvider):
        async def converse(self, *args):
            entered.set()
            while True:
                await asyncio.sleep(0.03)

    worker = ConversationWorker(Settings(_env_file=None), memory, Slow())
    replies = []
    worker.updated.connect(replies.append)
    worker.start()
    try:
        worker.submit("Обсудим генераторы", "voice", {})
        until(qapp, entered.is_set)
        worker.submit("Пауза", "voice", {})
        until(
            qapp, lambda: any(row["kind"] == "reply" and row["reply"]["intent"] == "pause" for row in replies)
        )
        assert not any(row["kind"] == "reply" and row["reply"]["intent"] == "conversation" for row in replies)
    finally:
        worker.stop()
        assert worker.wait(2000)
