from __future__ import annotations

import re
from typing import Any

from loliland_agent.conversation.models import DialogueNote, DialogueReply
from loliland_agent.memory import Memory
from loliland_agent.planning.providers import AIProvider
from loliland_agent.recipes.planner import RecipeKnowledgeBase
from loliland_agent.resources.tracker import ResourceTracker

SUGGESTION = re.compile(
    r"^(?:слушай[, ]+|а\s+)?(?:может|как насч[её]т|что если|стоит ли|не лучше ли|предлагаю|мне кажется)", re.I
)
NEGATION = re.compile(r"\b(?:не (?:добавляй|меняй|изменяй|надо|нужно|хочу)|пока не)\b", re.I)
DIRECTIVE = re.compile(
    r"\b(?:добавь|измени|смени|замени|поставь|отложи|перенеси|вернись|продолж(?:и|ай|им)|возобнови|остановись|пауза|приостанови|запомни|исправь|обнови|перепланируй|предпочитаю|давай (?:займ[её]мся|перейд[её]м|отложим|начн[её]м))\b",
    re.I,
)
MUTATING = {
    "change_goal",
    "add_task",
    "replan",
    "pause",
    "resume",
    "previous_goal",
    "defer_task",
    "prioritize",
}


def is_directive(text: str) -> bool:
    return bool(DIRECTIVE.search(text)) and not SUGGESTION.search(text.strip()) and not NEGATION.search(text)


def local_command(text: str) -> DialogueReply | None:
    normalized = text.strip().lower().rstrip(".!")
    commands = {
        "pause": r"(?:пауза|остановись|приостанови(?: задачу| план| работу)?)",
        "resume": r"(?:продолжи|продолжай|продолжим|возобнови(?: работу| задачу)?)",
        "repeat": r"(?:повтори(?: (?:последнюю )?(?:инструкцию|сообщение))?)",
        "check": r"(?:проверь(?:,? вс[её] ли я сделал правильно| результат)?|что дальше)",
        "done": r"(?:готово|я (?:закончил|сделал)(?: этот крафт| крафт| это)?)",
        "previous_goal": r"вернись к предыдущей цели",
        "defer_task": r"отложи текущую задачу",
    }
    answers = {
        "pause": "Поставлю текущую цель на паузу.",
        "resume": "Продолжу текущую цель с сохранённого шага.",
        "repeat": "Повторяю последнее сообщение.",
        "check": "Проверю новый кадр.",
        "done": "Проверю результат по изображению. Само сообщение не подтверждает крафт.",
        "previous_goal": "Вернусь к предыдущей цели.",
        "defer_task": "Отложу текущую задачу, сохранив её в плане.",
    }
    for intent, pattern in commands.items():
        if re.fullmatch(pattern, normalized):
            return DialogueReply.model_validate(
                {"answer": answers[intent], "intent": intent, "directive_quote": text}
            )
    if not is_directive(text):
        return None
    match = re.search(r"добавь\s+(?:в (?:наш )?план\s+)?(.+)", text, re.I)
    if match:
        return DialogueReply(
            intent="add_task",
            task=match[1].strip()[:800],
            directive_quote=text,
            answer="Передам новую задачу в план. Её выполнение ещё не подтверждено.",
        )
    match = re.search(
        r"(?:смени|измени|замени)\s+(?:главную )?цель\s+(?:на\s+)?(.+)|давай\s+займ[её]мся\s+(.+)", text, re.I
    )
    if match:
        return DialogueReply(
            intent="change_goal",
            goal=(match[1] or match[2]).strip()[:2000],
            directive_quote=text,
            answer="Сменю цель; предыдущая останется в очереди на паузе.",
        )
    match = re.search(r"(?:поставь|измени)\s+приоритет\s+(?:на\s+)?(\d{1,3})", text, re.I)
    if match and int(match[1]) <= 100:
        return DialogueReply(
            intent="prioritize",
            priority=int(match[1]),
            directive_quote=text,
            answer="Передам новый приоритет текущей цели.",
        )
    if re.match(r"(?:запомни|исправь|предпочитаю)\b", normalized):
        import hashlib

        return DialogueReply(
            answer="Сохраню это как информацию от тебя. Изображением она пока не подтверждена.",
            intent="memorize",
            notes=(
                DialogueNote(
                    kind="preference" if normalized.startswith("предпочитаю") else "user_report",
                    key="user-" + hashlib.sha256(text.encode()).hexdigest()[:20],
                    text=text[:1500],
                    quote=text[:1500],
                ),
            ),
        )
    return None


def conversation_context(memory: Memory) -> dict[str, Any]:
    summary = memory.recall("conversation", "summary")
    return {
        "summary": summary["value"] if summary else None,
        "recent_messages": memory.messages(24),
        "notes": sorted(memory.list_knowledge("dialogue-notes"), key=lambda row: -row["updated"])[:40],
        "provenance_rules": "Image observations, user reports, and AI assumptions are distinct. A summary is lossy context, never independent evidence.",
    }


class ConversationManager:
    def __init__(self, memory: Memory, provider: AIProvider) -> None:
        self.memory, self.provider = memory, provider

    async def respond(self, text: str, channel: str, context: dict[str, Any]) -> DialogueReply:
        if not text.strip() or len(text) > 8000:
            raise ValueError("Сообщение: 1–8000 символов")
        self.memory.message("user", channel, text)
        saved = self.memory.recall("copilot", "workspace-v1")
        if saved:
            workspace = saved["value"]
            goal = next(
                (value for value in workspace["goals"] if value["id"] == workspace["current_id"]), None
            )
            if goal:
                tracker = ResourceTracker(self.memory, goal["pack_version"])
                context = {
                    **context,
                    "goal": goal,
                    "observed_resources": tracker.available(),
                    "confirmed_facts": tracker.facts(),
                    "recipes": [
                        recipe.model_dump(mode="json")
                        for recipe in RecipeKnowledgeBase(self.memory, goal["pack_version"]).all()
                    ],
                }
        local = local_command(text)
        reply = local or await self.provider.converse(
            text, {**context, "conversation": conversation_context(self.memory)}
        )
        if local is None and self.provider.is_remote:
            self.memory.record_usage(
                reply.usage.input_tokens, reply.usage.output_tokens, reply.usage.cost_usd
            )
        if reply.intent in MUTATING and (
            not is_directive(text) or not reply.directive_quote or reply.directive_quote not in text
        ):
            reply = reply.model_copy(
                update={
                    "intent": "suggest_plan",
                    "answer": reply.answer + "\nПлан не изменён: предложение оставлено для обсуждения.",
                }
            )
        if reply.intent == "change_goal" and not reply.goal or reply.intent == "add_task" and not reply.task:
            reply = reply.model_copy(
                update={"intent": "conversation", "answer": "Уточни, какую цель или задачу нужно добавить."}
            )
        for note in reply.notes:
            if note.kind != "assumption" and (not note.quote.strip() or note.quote not in text):
                continue
            self.memory.remember(
                "dialogue-notes",
                note.key,
                {
                    "kind": note.kind,
                    "text": note.text if note.kind == "assumption" else note.quote,
                    "interpretation": note.text,
                    "quote": note.quote,
                    "provenance": "ai_assumption" if note.kind == "assumption" else "user_report",
                    "channel": channel,
                },
                "conversation:" + ("ai_assumption" if note.kind == "assumption" else "explicit_user_quote"),
            )
        self.memory.message("assistant", channel, reply.answer)
        self.memory.event(
            "conversation_intent",
            {"intent": reply.intent, "channel": channel, "directive": reply.directive_quote},
        )
        return reply

    async def compact(self) -> None:
        previous = self.memory.recall("conversation", "summary")
        value: dict[str, Any] = previous["value"] if previous else {"through": 0, "text": ""}
        rows = self.memory.messages(40, after=int(value["through"]), oldest=True)
        if len(rows) < 24:
            return
        older = rows[:-8]
        reply = await self.provider.summarize(
            {
                "previous_summary": value["text"],
                "messages": older,
                "notes": conversation_context(self.memory)["notes"],
            }
        )
        if self.provider.is_remote:
            self.memory.record_usage(
                reply.usage.input_tokens, reply.usage.output_tokens, reply.usage.cost_usd
            )
        self.memory.remember(
            "conversation",
            "summary",
            {"through": older[-1]["id"], "text": reply.text, "provenance": "ai_summary_not_verified_facts"},
            "ai_summary",
        )
