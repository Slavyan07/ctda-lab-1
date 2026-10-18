from typing import Annotated, Literal

from pydantic import Field

from loliland_agent.models import StrictModel, Usage

Intent = Literal[
    "conversation",
    "minecraft_question",
    "task_question",
    "suggest_plan",
    "change_goal",
    "add_task",
    "replan",
    "pause",
    "resume",
    "previous_goal",
    "defer_task",
    "report_resource",
    "correct_fact",
    "memorize",
    "preference",
    "explain",
    "repeat",
    "check",
    "done",
    "prioritize",
]


class DialogueNote(StrictModel):
    kind: Literal["user_report", "preference", "decision", "assumption"]
    key: Annotated[str, Field(min_length=1, max_length=150)]
    text: Annotated[str, Field(min_length=1, max_length=1500)]
    quote: Annotated[str, Field(max_length=2000)] = ""


class DialogueReply(StrictModel):
    answer: Annotated[str, Field(min_length=1, max_length=6000)]
    intent: Intent = "conversation"
    directive_quote: Annotated[str, Field(max_length=2000)] = ""
    goal: Annotated[str, Field(max_length=2000)] = ""
    task: Annotated[str, Field(max_length=800)] = ""
    priority: Annotated[int, Field(ge=0, le=100)] = 50
    notes: Annotated[tuple[DialogueNote, ...], Field(max_length=10)] = ()
    usage: Usage = Field(default_factory=Usage)


class DialogueSummary(StrictModel):
    text: Annotated[str, Field(min_length=1, max_length=4000)]
    usage: Usage = Field(default_factory=Usage)
