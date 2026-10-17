from __future__ import annotations

from loliland_agent.models import CompletionReply, HumanInstruction, Observation
from loliland_agent.perception.engine import PerceptionEngine
from loliland_agent.planning.providers import AIProvider


class CompletionVerifier:
    async def verify(
        self,
        instruction: HumanInstruction,
        before: Observation | None,
        after: Observation,
        provider: AIProvider,
    ) -> CompletionReply:
        if PerceptionEngine().requires_human(after):
            return CompletionReply(
                outcome="uncertain",
                confidence=1.0,
                evidence="На экране ручная проверка; Copilot приостановлен",
                requires_human=True,
            )
        if before and before.frame_id == after.frame_id:
            return CompletionReply(
                outcome="uncertain",
                confidence=1.0,
                evidence="Изображение не изменилось; выполнение не подтверждено",
                request_view=HumanInstruction(
                    action="show_screen",
                    text="Покажи результат предыдущего действия.",
                    reason="Текущий кадр совпадает с предыдущим.",
                    expected="Видимое подтверждение изменения",
                ),
            )
        visual = after.visual
        confirmed = False
        if visual.confidence >= 0.95:
            if instruction.action == "open_inventory":
                confirmed = visual.scene == "inventory"
            elif instruction.action == "open_storage":
                confirmed = visual.scene == "storage"
            elif instruction.action == "open_machine":
                confirmed = visual.scene == "machine" and (
                    instruction.item is None or visual.machine_name == instruction.item
                )
            elif instruction.action == "show_recipe":
                confirmed = visual.scene == "recipe" and any(
                    recipe.complete
                    and recipe.confidence >= 0.95
                    and (instruction.item is None or recipe.output.item == instruction.item)
                    for recipe in visual.recipes
                )
            elif instruction.action == "hover_item":
                confirmed = (
                    visual.tooltip is not None
                    and visual.tooltip.confidence >= 0.95
                    and bool(visual.tooltip.text)
                    and (instruction.item is None or visual.tooltip.item == instruction.item)
                )
        if confirmed:
            return CompletionReply(
                outcome="confirmed",
                confidence=visual.confidence,
                evidence="На новом изображении виден запрошенный интерфейс/предмет",
                task_completed=instruction.completes_task,
            )
        return await provider.verify(instruction, before, after)
