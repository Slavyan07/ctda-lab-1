import json

import httpx
import pytest
from conftest import frame
from pydantic import SecretStr, ValidationError

from loliland_agent.config import Settings
from loliland_agent.models import (
    ChatReply,
    CompletionReply,
    GoalState,
    HumanInstruction,
    InstructionReply,
    KnowledgeReply,
    PlanProposal,
    VisualState,
)
from loliland_agent.planning.providers import (
    OpenAICompatibleProvider,
    OpenAIProvider,
    ProviderError,
    make_provider,
)


def config(**updates):
    return Settings(
        _env_file=None,
        provider="compatible",
        api_base_url="https://vision.example/v1",
        api_key=SecretStr("not-a-real-key"),
        input_price_per_million=1.0,
        output_price_per_million=2.0,
        **updates,
    )


def instruction():
    return HumanInstruction(
        action="open_inventory", text="Открой инвентарь.", reason="Нужны предметы", expected="Видны слоты"
    )


@pytest.mark.parametrize("method", ["perceive", "plan", "instruct", "verify", "chat", "extract_document"])
async def test_multimodal_provider_validated_contracts(method):
    result = {
        "perceive": VisualState(scene="inventory", confidence=0.99),
        "plan": PlanProposal(summary="Проверить запас", instruction=instruction()),
        "instruct": InstructionReply(instruction=instruction()),
        "verify": CompletionReply(outcome="uncertain", confidence=0.5, evidence="Не видно"),
        "chat": ChatReply(answer="Нужен рецепт сборки."),
        "extract_document": KnowledgeReply(),
    }[method]

    def response(request):
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        content = body["messages"][1]["content"]
        assert "HUMAN" in body["messages"][0]["content"]
        if method in {"perceive", "plan", "instruct", "verify"}:
            assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        else:
            assert len(content) == 1
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": result.model_dump_json()}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        provider = OpenAICompatibleProvider(config(), client)
        goal = GoalState(id="g", description="Цель", pack_version="r1")
        args = {
            "perceive": (frame(),),
            "plan": (goal, frame(), {}),
            "instruct": (goal, frame(), {}),
            "verify": (instruction(), frame(), frame("inventory.png")),
            "chat": ("Почему?", {}),
            "extract_document": ("Рецепт", "r1"),
        }[method]
        reply = await getattr(provider, method)(*args)
    assert reply.usage.input_tokens == 100 and reply.usage.cost_usd == 0.0002


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": []},
        {"choices": [{"message": {"content": "not json"}}]},
        {"choices": [{"message": {"content": '{"scene":"invented"}'}}]},
    ],
)
async def test_bad_response_no_secret_leak(payload):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    ) as client:
        with pytest.raises(ProviderError) as error:
            await OpenAICompatibleProvider(config(), client).perceive(frame())
    assert "not-a-real-key" not in str(error.value)


@pytest.mark.parametrize("code", [302, 401, 429, 500])
async def test_http_failures_are_sanitized_and_redirects_refused(code):
    calls = []

    def response(request):
        calls.append(request)
        return httpx.Response(code, headers={"Location": "https://other.example/leak"}, text="not-a-real-key")

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        with pytest.raises(ProviderError) as error:
            await OpenAICompatibleProvider(config(), client).chat("Привет", {})
    assert len(calls) == 1 and str(code) in str(error.value) and "not-a-real-key" not in str(error.value)


@pytest.mark.parametrize(
    "updates",
    [
        {"api_base_url": "http://remote.example/v1"},
        {"api_base_url": "https://name:secret@example.com/v1"},
        {"provider": "ollama", "api_base_url": "http://localhost:11434/v1", "vision": False},
        {"interval_seconds": 0.1},
        {"provider": "bogus"},
    ],
)
def test_invalid_settings(updates):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **updates)


def test_offline_default_and_separate_openai_interface():
    settings = Settings(_env_file=None)
    assert settings.provider == "offline"
    with httpx.Client():
        pass
    assert "not-a-real-key" not in repr(config())
    with pytest.raises(ValueError):
        OpenAIProvider(config(), None)
    assert make_provider(settings, None).is_remote is False


async def test_multimodal_copilot_human_loop_with_mock_api(tmp_path, memory):
    from conftest import FixedSource
    from PIL import Image

    from loliland_agent.instructions.engine import Copilot
    from loliland_agent.models import Observation, PlanStep
    from loliland_agent.perception.sources import save_frame

    first, digest1 = save_frame(Image.new("RGB", (640, 360), "navy"), tmp_path / "frames")
    second, digest2 = save_frame(Image.new("RGB", (640, 360), "gray"), tmp_path / "frames")
    calls = []

    def response(request):
        body = json.loads(request.content)
        title = json.loads(body["messages"][1]["content"][0]["text"])["schema"]["title"]
        calls.append(title)
        if title == "VisualState":
            result = VisualState(
                scene="gameplay" if len(calls) == 1 else "inventory", confidence=0.99, summary="Mock vision"
            )
        elif title == "PlanProposal":
            result = PlanProposal(
                summary="Посмотреть запас",
                steps=(
                    PlanStep(id="a", title="Показать инвентарь", acceptance="Видны слоты"),
                    PlanStep(
                        id="b",
                        title="Изучить рецепт",
                        acceptance="Виден полный NEI-рецепт",
                        depends_on=("a",),
                    ),
                ),
                instruction=instruction().model_copy(update={"task_id": "a", "completes_task": True}),
            )
        else:
            result = InstructionReply(
                instruction=HumanInstruction(
                    action="show_recipe",
                    text="Покажи рецепт печи в NEI.",
                    reason="Нужен точный рецепт сборки",
                    expected="Полный рецепт",
                )
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": result.model_dump_json()}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        )

    source = FixedSource(Observation(source="image", image_path=first, frame_id=digest1))
    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        engine = Copilot(
            config(database_path=tmp_path / "memory.sqlite3"),
            memory,
            OpenAICompatibleProvider(config(), client),
            source,
            lambda _: None,
        )
        await engine.start("Построить печь", 50, 0)
        source.observation = Observation(source="image", image_path=second, frame_id=digest2)
        engine.last_capture = 0
        await engine.check(0)
    assert calls == ["VisualState", "PlanProposal", "VisualState", "InstructionReply"]
    assert (
        engine.goals.current.confirmed_count == 1
        and engine.goals.current.pending.instruction.action == "show_recipe"
    )
    assert memory.totals()["calls"] == 4
