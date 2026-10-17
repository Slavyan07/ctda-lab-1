from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Probability = Annotated[float, Field(ge=0, le=1)]
Name = Annotated[str, Field(min_length=1, max_length=250)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False, strict=True)


class ItemAmount(StrictModel):
    item: Name
    count: Annotated[int, Field(ge=1, le=1_000_000)]


class RecipeCandidate(StrictModel):
    output: ItemAmount
    ingredients: tuple[ItemAmount, ...]
    machine: Name | None = None
    technologies: tuple[Name, ...] = ()
    complete: bool = False
    confidence: Probability
    evidence_text: Annotated[str, Field(min_length=1, max_length=4000)]


class VisibleStack(StrictModel):
    container: Literal["inventory", "current_container"]
    slot: Name
    item: Name | None = None
    count: Annotated[int, Field(ge=0, le=1_000_000)] | None = None
    confidence: Probability


class Tooltip(StrictModel):
    item: Name | None = None
    text: Annotated[str, Field(max_length=4000)] = ""
    confidence: Probability = 0.0


class Landmark(StrictModel):
    name: Name
    kind: Literal["waypoint", "gui_button", "inventory_slot", "player", "hazard"]
    x: Probability
    y: Probability
    confidence: Probability


class VisualState(StrictModel):
    scene: Literal[
        "gameplay",
        "inventory",
        "recipe",
        "machine",
        "storage",
        "menu",
        "disconnected",
        "dead",
        "verification",
        "unknown",
        "test_window",
    ] = "unknown"
    confidence: Probability = 0.0
    human_required: bool | None = None
    selected_slot: Annotated[int, Field(ge=0, le=8)] | None = None
    visible_text: tuple[Annotated[str, Field(max_length=2000)], ...] = ()
    landmarks: tuple[Landmark, ...] = ()
    stacks: tuple[VisibleStack, ...] = ()
    recipes: tuple[RecipeCandidate, ...] = ()
    tooltip: Tooltip | None = None
    machine_name: Name | None = None
    summary: Annotated[str, Field(max_length=3000)] = "Состояние не распознано"


class Observation(StrictModel):
    source: Literal["screen", "mock", "replay", "image"]
    captured_at: float = Field(default_factory=time.time)
    monotonic_at: float = Field(default_factory=time.monotonic)
    window_id: int | None = None
    window_process_id: Annotated[int, Field(ge=1)] | None = None
    window_title: str = ""
    window_bounds: tuple[int, int, int, int] | None = None
    dpi_scale: Annotated[float, Field(gt=0, le=8)] = 1.0
    focused: bool = False
    image_path: Path | None = None
    frame_id: str = ""
    visual: VisualState = Field(default_factory=VisualState)
    visual_at: float = 0.0
    visual_origin: Literal["none", "local_test", "llm", "cache"] = "none"


class Usage(StrictModel):
    input_tokens: Annotated[int, Field(ge=0)] = 0
    output_tokens: Annotated[int, Field(ge=0)] = 0
    cost_usd: Annotated[float, Field(ge=0)] | None = None


class VisionReply(StrictModel):
    visual: VisualState
    usage: Usage = Field(default_factory=Usage)


class HumanInstruction(StrictModel):
    action: Literal[
        "open_inventory",
        "show_recipe",
        "hover_item",
        "open_storage",
        "take_items",
        "place_items",
        "open_machine",
        "show_screen",
        "manual_action",
        "explain",
    ]
    text: Annotated[str, Field(min_length=1, max_length=800)]
    reason: Annotated[str, Field(min_length=1, max_length=3000)]
    expected: Annotated[str, Field(min_length=1, max_length=2000)]
    task_id: str | None = None
    item: Name | None = None
    count: Annotated[int, Field(ge=1, le=1_000_000)] | None = None
    completes_task: bool = False

    @model_validator(mode="after")
    def one_instruction(self) -> HumanInstruction:
        if "\n" in self.text or ";" in self.text or self.text.lstrip().startswith(("1.", "1)")):
            raise ValueError("Give one short action, not a list")
        if self.completes_task and self.task_id is None:
            raise ValueError("Task completion needs a task id")
        return self


class PlanStep(StrictModel):
    id: Name
    title: Annotated[str, Field(min_length=1, max_length=800)]
    acceptance: Annotated[str, Field(min_length=1, max_length=1200)]
    depends_on: tuple[Name, ...] = ()
    priority: Annotated[int, Field(ge=0, le=100)] = 50
    status: Literal["pending", "active", "done", "skipped", "paused"] = "pending"


class PlanProposal(StrictModel):
    summary: Annotated[str, Field(min_length=1, max_length=4000)]
    steps: Annotated[tuple[PlanStep, ...], Field(max_length=100)] = ()
    instruction: HumanInstruction
    target: ItemAmount | None = None
    usage: Usage = Field(default_factory=Usage)


class InstructionReply(StrictModel):
    instruction: HumanInstruction
    suggest_goal_complete: bool = False
    usage: Usage = Field(default_factory=Usage)


class CompletionReply(StrictModel):
    outcome: Literal["confirmed", "incorrect", "uncertain"]
    confidence: Probability
    evidence: Annotated[str, Field(min_length=1, max_length=3000)]
    task_completed: bool = False
    correction: HumanInstruction | None = None
    request_view: HumanInstruction | None = None
    requires_human: bool = False
    usage: Usage = Field(default_factory=Usage)


class PendingInstruction(StrictModel):
    id: str
    instruction: HumanInstruction
    before_image: Path | None = None
    before_frame: str = ""
    created_at: float = Field(default_factory=time.time)
    verification: Literal["waiting", "uncertain", "incorrect"] = "waiting"
    evidence: str = ""
    clarification: HumanInstruction | None = None


class GoalState(StrictModel):
    id: str
    description: Annotated[str, Field(min_length=1, max_length=2000)]
    pack_version: Name
    priority: Annotated[int, Field(ge=0, le=100)] = 50
    status: Literal["active", "paused", "completed"] = "active"
    plan_summary: str = ""
    steps: tuple[PlanStep, ...] = ()
    target: ItemAmount | None = None
    pending: PendingInstruction | None = None
    confirmed_count: int = 0
    manual_count: int = 0
    skipped_count: int = 0


class Workspace(StrictModel):
    current_id: str | None = None
    goals: tuple[GoalState, ...] = ()


class KnowledgeReply(StrictModel):
    recipes: tuple[RecipeCandidate, ...] = ()
    usage: Usage = Field(default_factory=Usage)


class ChatReply(StrictModel):
    answer: Annotated[str, Field(min_length=1, max_length=8000)]
    usage: Usage = Field(default_factory=Usage)
