import asyncio
import hashlib
import json
import sys
from pathlib import Path

import httpx
import pytest
from conftest import FixedSource, frame
from pydantic import ValidationError

from loliland_agent.config import Settings
from loliland_agent.conversation.manager import ConversationManager
from loliland_agent.conversation.models import DialogueReply
from loliland_agent.goals.manager import GoalManager
from loliland_agent.instructions.engine import Copilot
from loliland_agent.instructions.verifier import CompletionVerifier
from loliland_agent.learning.curriculum import AutomaticCurriculum
from loliland_agent.learning.library import Procedure, SkillLibrary
from loliland_agent.learning.reflection import ReflectionManager
from loliland_agent.local.models import HF, VOICE, VOICE_PATH, ModelError, ModelManager
from loliland_agent.local.scheduler import BudgetError, HeavyScheduler
from loliland_agent.memory import Memory
from loliland_agent.models import (
    HumanInstruction,
    ItemAmount,
    RecipeCandidate,
    VisualState,
)
from loliland_agent.planning.providers import DemoProvider, OfflineProvider, ProviderError, make_provider
from loliland_agent.recipes.planner import Evidence, RecipeKnowledgeBase
from loliland_agent.speech.cloud_tts import synthesize
from loliland_agent.speech.piper import synthesize_piper
from loliland_agent.speech.stt import LocalWhisperSTT, SpeechError


def instruction(**changes):
    return HumanInstruction(
        action="manual_action",
        text="Изготовьте плату.",
        reason="Нужна для цели",
        expected="Плата в инвентаре",
        item="плата",
        count=1,
        **changes,
    )


def recipe(kb, item, ingredients, output=1, machine=None):
    return kb.add(
        RecipeCandidate(
            output=ItemAmount(item=item, count=output),
            ingredients=tuple(ItemAmount(item=name, count=count) for name, count in ingredients.items()),
            machine=machine,
            complete=True,
            confidence=1.0,
            evidence_text="Подтверждаю рецепт",
        ),
        Evidence(kind="user", reference="player", excerpt="Подтверждаю рецепт", confidence=1.0),
    )


def curriculum_setup(memory):
    goals = GoalManager(memory)
    goal = goals.start("Сделать процессор", "r1")
    goal = goal.model_copy(update={"target": ItemAmount(item="процессор", count=1)})
    goals.replace(goal)
    kb = RecipeKnowledgeBase(memory, "r1")
    recipe(kb, "процессор", {"плата": 2})
    recipe(kb, "плата", {"медь": 3}, output=2)
    return goals, goal, kb, AutomaticCurriculum(kb)


def test_multistage_curriculum_dag_and_resource_shortage(memory):
    goals, goal, _, curr = curriculum_setup(memory)
    facts = {"raw_resources": ["медь"], "machines": [], "technologies": []}
    full = curr.propose(goal, {"медь": 3}, facts)
    goals.validate_steps(full.steps)
    assert len(full.steps) == 2
    first = next(step for step in full.steps if step.id == full.instruction.task_id)
    assert "плата" in first.title and not first.depends_on
    assert "медь ×3" in full.instruction.text
    last = next(step for step in full.steps if "процессор" in step.title)
    assert first.id in last.depends_on
    short = curr.propose(goal, {"медь": 1}, facts)
    assert short.instruction.action == "open_inventory" and short.instruction.count == 3
    assert short.instruction.item == "медь"
    assert all(step.depends_on for step in short.steps if "Изготовить" in step.title)


def test_unknown_recipe_and_unavailable_technology_are_information_tasks(memory):
    _, goal, kb, curr = curriculum_setup(memory)
    unknown = goal.model_copy(update={"target": ItemAmount(item="реактор", count=1)})
    proposal = curr.propose(unknown, {}, {})
    assert proposal.instruction.action == "show_recipe"
    assert not any("Изготовить" in task.title for task in proposal.steps)
    recipe(kb, "реактор", {"медь": 1}, machine="пресс")
    proposal = curr.propose(unknown, {"медь": 99}, {"raw_resources": ["медь"]})
    assert proposal.instruction.action == "show_recipe" and "пресс" in proposal.instruction.text


def test_agent_chooses_feasible_recipe_without_asking_player(memory):
    _, goal, kb, curr = curriculum_setup(memory)
    easy = recipe(kb, "процессор", {"камень": 1})
    plan, choices = curr.materials(goal, {"камень": 1}, {"raw_resources": ["камень", "медь"]})
    assert choices["процессор"] == easy.id and not plan.raw_missing
    assert (
        curr.propose(goal, {"камень": 1}, {"raw_resources": ["камень", "медь"]}).instruction.action
        == "manual_action"
    )


def test_recipe_conflict_and_explicit_correction_are_versioned(memory):
    kb = RecipeKnowledgeBase(memory, "r1")
    old = recipe(kb, "плата", {"медь": 3})
    corrected = RecipeCandidate(
        output=ItemAmount(item="плата", count=1),
        ingredients=(ItemAmount(item="медь", count=2),),
        complete=True,
        confidence=1.0,
        evidence_text="Исправь рецепт: 2 меди на плату",
    )
    new = kb.add(corrected, Evidence(kind="screenshot", reference="screen", excerpt="2 меди", confidence=1.0))
    assert not kb.all() and kb.status(old.id) == kb.status(new.id) == "disputed"
    kb.correct(
        (old.id,),
        corrected,
        Evidence(
            kind="user", reference="explicit correction", excerpt=corrected.evidence_text, confidence=1.0
        ),
    )
    assert [r.id for r in kb.all()] == [new.id]
    assert kb.status(old.id) == "superseded"
    assert len(memory.list_knowledge(kb.category)) == 2
    assert RecipeKnowledgeBase(memory, "other-pack").all() == ()


def test_reflection_persists_stops_repetition_and_stores_verified_skill(tmp_path):
    path = tmp_path / "restart.sqlite3"
    memory = Memory(path)
    reflection = ReflectionManager(memory, "r1", 3)
    for _ in range(3):
        report = reflection.record("goal", instruction(), "incorrect", "Не хватает ресурса медь")
    assert "ресурс" in report["cause"].lower() and reflection.blocked("goal")
    memory.close()
    memory = Memory(path)
    try:
        restored = ReflectionManager(memory, "r1", 3)
        assert restored.blocked("goal")
        restored.record("other", instruction(), "confirmed", "Плата видна", task_completed=True)
        skills = restored.library.search("плата", reusable=True)
        assert skills[0].status == "verified" and len(skills[0].steps) == 1
        used = restored.library.use(skills[0].id, set())
        assert used.uses == 1
        restored.record(
            "manual", instruction(), "confirmed", "Сообщение игрока", manual=True, task_completed=True
        )
        assert any(item.status == "user_report" for item in restored.library.all())
    finally:
        memory.close()


def test_library_preserves_contradicting_reports_and_filters_prerequisites(memory):
    lib = SkillLibrary(memory, "r1")
    lib.note("Источник меди", "Медь есть в шахте", "user1")
    lib.note("Источник меди", "Меди в шахте нет", "user2")
    assert len(lib.search("меди")) == 2
    assert all(item.status == "disputed" for item in lib.all())
    assert lib.search("медь", reusable=True) == []
    procedure = Procedure(
        id="p",
        title="Плата",
        topic="плата",
        kind="machine",
        steps=(instruction(),),
        explanation="Проверено",
        source="image:1",
        status="verified",
        prerequisites=("пресс",),
    )
    lib.put(procedure)
    with pytest.raises(ValueError):
        lib.use("p", set())
    assert lib.use("p", {"пресс"}).uses == 1
    with pytest.raises(ValidationError):
        Procedure.model_validate({**procedure.model_dump(), "javascript": "bot.craft()"})


async def test_curriculum_engine_reuses_strategy_and_restores_progress(memory, tmp_path):
    settings = Settings(_env_file=None, database_path=tmp_path / "engine.db")
    engine = Copilot(settings, memory, DemoProvider(), FixedSource(frame()), lambda _: None)
    _, goal, kb, _ = curriculum_setup(memory)
    engine.goals = GoalManager(memory)
    engine.goals.start(goal.description, "r1")
    engine.resources.set_facts({"raw_resources": ["медь"], "resources": {"inventory": {"медь": 3}}})
    skill = Procedure(
        id="learned",
        title="плата",
        topic="плата",
        kind="crafting",
        steps=(instruction().model_copy(update={"text": "Изготовьте плату в верстаке.", "count": 2}),),
        explanation="Результат проверен",
        source="image:old",
        status="verified",
        recipe_ids=tuple(r.id for r in kb.all()),
    )
    engine.library.put(skill)
    await engine._next(frame(), 0)
    assert engine.goals.current.pending.instruction.text == "Изготовьте плату в верстаке."
    assert next(item for item in engine.library.all() if item.id == "learned").uses == 1
    original = engine.goals.current.pending
    restored = Copilot(settings, memory, DemoProvider(), None, lambda _: None)
    assert restored.goals.current.pending == original and restored.goals.current.status == "paused"
    newer = engine.goals.start("Другая цель", "r1")
    assert newer.id != goal.id
    assert next(g for g in engine.goals.workspace.goals if g.id == goal.id).status == "paused"


async def test_inventory_window_alone_does_not_prove_resource_count():
    request = HumanInstruction(
        action="open_inventory",
        text="Покажите медь.",
        reason="Проверка запаса",
        expected="10 меди",
        item="медь",
        count=10,
        task_id="g",
        completes_task=True,
    )
    reply = await CompletionVerifier().verify(
        request,
        frame(),
        frame("inventory.png").model_copy(update={"visual": VisualState(scene="inventory", confidence=1.0)}),
        OfflineProvider(),
    )
    assert reply.outcome != "confirmed"


async def test_voice_recipe_correction_is_structured_and_suggestions_do_not_mutate(memory):
    corrected = RecipeCandidate(
        output=ItemAmount(item="плата", count=1),
        ingredients=(ItemAmount(item="медь", count=2),),
        complete=True,
        confidence=1.0,
        evidence_text="2 меди на плату",
    )

    class Provider(OfflineProvider):
        async def converse(self, text, context):
            return DialogueReply(
                answer="Исправлю рецепт по вашим данным",
                intent="correct_fact",
                directive_quote=text,
                recipe_correction=corrected,
                replace_recipe_ids=("old",),
            )

    manager = ConversationManager(memory, Provider())
    good = await manager.respond("Исправь рецепт: 2 меди на плату", "voice", {})
    assert good.intent == "correct_fact" and good.recipe_correction == corrected
    bad = await manager.respond("Может, исправь рецепт: 2 меди на плату?", "voice", {})
    assert bad.intent == "suggest_plan" and bad.recipe_correction is None


async def test_heavy_scheduler_serialization_cancellation_and_low_memory(monkeypatch):
    scheduler = HeavyScheduler()
    entered = asyncio.Event()
    release = asyncio.Event()
    order = []

    async def first():
        async with scheduler.lease("vision"):
            entered.set()
            order.append("vision")
            await release.wait()

    async def next_job():
        async with scheduler.lease("speech"):
            order.append("speech")

    one = asyncio.create_task(first())
    await entered.wait()
    two = asyncio.create_task(next_job())
    await asyncio.sleep(0.05)
    assert order == ["vision"] and scheduler.waiters == 1
    two.cancel()
    with pytest.raises(asyncio.CancelledError):
        await two
    release.set()
    await one
    assert not scheduler.active and not scheduler.waiters
    monkeypatch.setattr("loliland_agent.local.scheduler.available_ram_mb", lambda: 100)
    with pytest.raises(BudgetError):
        async with scheduler.lease("piper", 2048):
            pass
    assert not scheduler.lock.locked()


async def test_native_ollama_keyless_vision_json_and_unload():
    requests = []

    def server(request):
        requests.append(request)
        assert request.url.host == "127.0.0.1"
        assert "authorization" not in request.headers
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"done": True})
        body = json.loads(request.content)
        assert request.url.path == "/api/chat" and body["model"] == "qwen3-vl:4b"
        assert body["messages"][1]["images"][0]
        assert body["format"]["title"] == "VisualState"
        assert body["keep_alive"] == 0 and body["options"]["num_gpu"] == 0
        return httpx.Response(
            200,
            json={
                "message": {"content": VisualState(scene="inventory", confidence=0.99).model_dump_json()},
                "prompt_eval_count": 20,
                "eval_count": 10,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        settings = Settings(_env_file=None)
        result = await make_provider(settings, client).perceive(frame())
    assert result.visual.scene == "inventory" and result.usage.cost_usd == 0
    assert requests[-1].url.path == "/api/generate"


async def test_ollama_memory_budget_unloads_and_has_no_cloud_fallback():
    paths = []

    async def server(request):
        paths.append(request.url.path)
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"size": 9 * 1024**3, "size_vram": 0}]})
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"done": True})
        await asyncio.sleep(5)
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        with pytest.raises(ProviderError, match="Бюджет"):
            await make_provider(Settings(_env_file=None), client).chat("Привет", {})
    assert paths[-1] == "/api/generate" and all(path.startswith("/api/") for path in paths)


@pytest.mark.parametrize(
    "update",
    [
        {"provider": "openai"},
        {"tts_provider": "cloud"},
        {"stt_provider": "cloud"},
        {"api_base_url": "https://remote.example/v1"},
        {"model": "paid-model"},
    ],
)
def test_free_mode_rejects_paid_or_remote_inference(update):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **update)


async def test_cloud_speech_cannot_run_in_free_mode():
    with pytest.raises(SpeechError, match="Free|Бесплат|бесплат|free|отключ"):
        await synthesize(Settings(_env_file=None), None, "Привет", "ru_RU-irina-medium", 1.0)


async def test_piper_download_integrity_and_no_silent_download(tmp_path):
    weights = b"verified-test-model"
    digest = hashlib.sha256(weights).hexdigest()
    calls = []

    def server(request):
        calls.append(str(request.url))
        if "/api/models/" in request.url.path:
            return httpx.Response(
                200, json=[{"path": VOICE_PATH + "/" + VOICE + ".onnx", "lfs": {"oid": digest}}]
            )
        if request.url.path.endswith(".onnx"):
            return httpx.Response(200, content=weights)
        if request.url.path.endswith(".json"):
            return httpx.Response(200, json={"audio": {"sample_rate": 22050}})
        return httpx.Response(200, text="Test voice license / MODEL_CARD")

    settings = Settings(_env_file=None, database_path=tmp_path / "data/copilot.db")
    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        manager = ModelManager(settings, client)
        with pytest.raises(ModelError):
            await manager.download_piper(False, lambda _: None)
        assert not calls
        await manager.download_piper(True, lambda _: None)
    root = tmp_path / "data/models/piper"
    assert (root / (VOICE + ".sha256")).read_text() == digest and (root / "MODEL_CARD").is_file()
    # Tampering is caught before any native synthesis process starts.
    (root / (VOICE + ".onnx")).write_bytes(b"tampered")
    with pytest.raises(SpeechError, match="повреждён"):
        await synthesize_piper(settings, "Привет", 1.0)
    assert all(url.startswith(HF) or url.startswith("https://huggingface.co/api/models/") for url in calls)


async def test_piper_local_subprocess_wire_and_wav(monkeypatch, tmp_path):
    root = tmp_path / "models/piper"
    root.mkdir(parents=True)
    for suffix, data in [
        (".onnx", b"model"),
        (".onnx.json", b"{}"),
        (".sha256", hashlib.sha256(b"model").hexdigest().encode()),
    ]:
        (root / (VOICE + suffix)).write_bytes(data)
    real = asyncio.create_subprocess_exec
    commands = []

    async def substitute(*args, **kwargs):
        commands.append(args)
        output = args[args.index("--output-file") + 1]
        program = "import pathlib,sys; data=sys.stdin.buffer.read(); assert data.decode('utf-8')=='Привет'; pathlib.Path(sys.argv[1]).write_bytes(b'RIFF'+b'0'*4+b'WAVE'+b'0'*16)"
        return await real(sys.executable, "-c", program, output, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", substitute)
    data = await synthesize_piper(
        Settings(_env_file=None, database_path=tmp_path / "copilot.db"), "Привет", 1.0
    )
    assert data.startswith(b"RIFF") and commands[0][1:3] == ("-m", "loliland_agent.speech.piper_runner")
    assert "--cuda" not in commands[0]


async def test_default_whisper_unloads_owned_process_after_utterance():
    program = "import sys,json; [(print(json.dumps({'text':'Пауза'}),flush=True)) for line in sys.stdin]"
    engine = LocalWhisperSTT(Settings(_env_file=None), [sys.executable, "-u", "-c", program])
    assert await engine.transcribe(bytes(16000 * 2)) == "Пауза"
    assert engine.process is None


async def test_truncated_ollama_pull_is_not_success():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, content=b'{"status":"pulling manifest"}\n')
        )
    ) as client:
        with pytest.raises(ModelError, match="без подтверждения"):
            await ModelManager(Settings(_env_file=None), client).pull_ollama(True, lambda _: None)


def test_curriculum_respects_machine_preference_when_paths_are_feasible(memory):
    goals = GoalManager(memory)
    goal = goals.start("Плата", "r1").model_copy(update={"target": ItemAmount(item="плата", count=1)})
    kb = RecipeKnowledgeBase(memory, "r1")
    recipe(kb, "плата", {"медь": 1}, machine="верстак")
    preferred = recipe(kb, "плата", {"олово": 1}, machine="пресс")
    memory.remember(
        "dialogue-notes", "machine", {"kind": "preference", "quote": "Предпочитаю пресс"}, "player"
    )
    result, choices = AutomaticCurriculum(kb).materials(
        goal, {"медь": 1, "олово": 1}, {"raw_resources": ["медь", "олово"], "machines": ["пресс", "верстак"]}
    )
    assert choices["плата"] == preferred.id and not result.blockers


async def test_voice_recipe_correction_applies_through_worker_and_replans(memory, tmp_path):
    from loliland_agent.ui.worker import Command, CopilotWorker

    settings = Settings(_env_file=None, database_path=tmp_path / "voice-correction.db")
    engine = Copilot(settings, memory, DemoProvider(), FixedSource(frame()), lambda _: None)
    goal = engine.goals.start("Плата", "r1").model_copy(update={"target": ItemAmount(item="плата", count=1)})
    engine.goals.replace(goal)
    old = recipe(engine.knowledge, "плата", {"медь": 3})
    engine.resources.set_facts({"raw_resources": ["медь"], "resources": {"inventory": {"медь": 2}}})
    corrected = RecipeCandidate(
        output=ItemAmount(item="плата", count=1),
        ingredients=(ItemAmount(item="медь", count=2),),
        complete=True,
        confidence=1.0,
        evidence_text="Исправь рецепт: 2 меди на плату",
    )
    reply = DialogueReply(
        answer="Исправлю",
        intent="correct_fact",
        directive_quote=corrected.evidence_text,
        recipe_correction=corrected,
        replace_recipe_ids=(old.id,),
    )
    worker = CopilotWorker(settings, memory, None)
    async with httpx.AsyncClient() as client:
        await worker.dispatch(
            engine,
            client,
            Command(
                kind="dialogue_action",
                data={"reply": reply.model_dump_json(), "expected_goal": goal.id},
                revision=0,
            ),
        )
    assert engine.knowledge.all()[0].ingredients[0].count == 2
    assert engine.goals.current.pending.instruction.action == "manual_action"
    assert engine.knowledge.status(old.id) == "superseded"


async def test_failed_verification_limit_survives_copilot_restart(tmp_path, memory):
    from test_copilot import Incorrect

    settings = Settings(_env_file=None, database_path=tmp_path / "failure.db", interval_seconds=3)
    source = FixedSource(frame())
    engine = Copilot(settings, memory, Incorrect(), source, lambda _: None)
    await engine.start("Цель", 50, 0)
    for image in ["moved.png", "initial.png", "moved.png"]:
        source.observation = frame(image)
        engine.last_capture = 0
        await engine.check(0)
    goal = engine.goals.current
    assert goal.status == "paused" and engine.reflection.blocked(goal.id)
    restarted = Copilot(settings, memory, Incorrect(), source, lambda _: None)
    await restarted.start("Цель", 50, 0)
    restarted.goals.replace(restarted.goals.current.model_copy(update={"pending": None}))
    await restarted._next(frame(), 0, make_plan=True)
    assert restarted.goals.current.status == "paused" and restarted.goals.current.pending is None


async def test_real_piper_runner_uses_offline_cpu_single_thread(monkeypatch, tmp_path):
    import io
    import types

    from loliland_agent.speech import piper_runner

    options = []
    sessions = []
    calls = []

    class SessionOptions:
        def __init__(self):
            options.append(self)

    class Voice:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def synthesize_wav(self, text, output, config):
            calls.append((text, config))
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(22050)
            output.writeframes(b"\0" * 100)

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        types.SimpleNamespace(
            disable_telemetry_events=lambda: None,
            SessionOptions=SessionOptions,
            InferenceSession=lambda *a, **kw: sessions.append(kw),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "piper", types.SimpleNamespace(PiperVoice=Voice, SynthesisConfig=lambda **kw: kw)
    )
    monkeypatch.setitem(
        sys.modules,
        "piper.config",
        types.SimpleNamespace(PiperConfig=types.SimpleNamespace(from_dict=lambda d: d)),
    )
    model = tmp_path / "voice.onnx"
    Path(str(model) + ".json").write_text("{}")
    output = tmp_path / "voice.wav"
    monkeypatch.setattr(sys, "argv", ["piper_runner", "--model", str(model), "--output-file", str(output)])
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(buffer=io.BytesIO("Привет".encode())))
    piper_runner.main()
    assert sessions[0]["providers"] == ["CPUExecutionProvider"]
    assert options[0].intra_op_num_threads == options[0].inter_op_num_threads == 1
    assert calls[0][0] == "Привет" and output.read_bytes()[8:12] == b"WAVE"


async def test_direct_legacy_adapter_cannot_bypass_free_mode():
    from loliland_agent.planning.providers import OpenAICompatibleProvider

    calls = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: calls.append(request))
    ) as client:
        with pytest.raises(ProviderError, match="отключён"):
            await OpenAICompatibleProvider(Settings(_env_file=None), client).chat("Привет", {})
    assert not calls


async def test_target_already_in_stock_requests_human_goal_review(memory, tmp_path):
    events = []
    engine = Copilot(
        Settings(_env_file=None, database_path=tmp_path / "review.db"),
        memory,
        DemoProvider(),
        None,
        events.append,
    )
    goal = engine.goals.start("Плата", "r1").model_copy(update={"target": ItemAmount(item="плата", count=1)})
    engine.goals.replace(goal)
    engine.resources.set_facts({"resources": {"inventory": {"плата": 1}}})
    await engine._next(frame(), 0)
    assert engine.goals.current.status == "paused" and engine.goals.current.pending is None
    assert any(event["kind"] == "goal_review" for event in events)


def test_success_of_unrelated_instruction_does_not_reset_failure_family(memory):
    reflection = ReflectionManager(memory, "r1", 3)
    reflection.record("g", instruction(), "incorrect", "Нет платы")
    reflection.record(
        "g",
        HumanInstruction(
            action="open_inventory", text="Откройте инвентарь.", reason="Проверка", expected="Инвентарь"
        ),
        "confirmed",
        "Открыт инвентарь",
    )
    reflection.record("g", instruction(), "incorrect", "Нет платы")
    reflection.record("g", instruction(), "incorrect", "Нет платы")
    assert reflection.blocked("g")


def test_skill_sqlite_search_survives_reopen_and_fts_fallback(tmp_path):
    path = tmp_path / "search.db"
    memory = Memory(path)
    library = SkillLibrary(memory, "r1")
    library.note("Подсказка", "Пресс питается энергией", "player")
    assert library.search("ПРЕСС")[0].explanation == "Пресс питается энергией"
    memory.close()
    memory = Memory(path)
    try:
        library = SkillLibrary(memory, "r1")
        assert library.search("пресс")
        memory._fts = False
        assert library.search("пресс")
        assert not library.search('SQL injection " OR NOT *')
    finally:
        memory.close()


async def test_voice_reports_unlock_technology_and_invalidate_unknown_stock(memory, tmp_path):
    from loliland_agent.ui.worker import Command, CopilotWorker

    settings = Settings(_env_file=None, database_path=tmp_path / "reports.db")
    engine = Copilot(settings, memory, DemoProvider(), None, lambda _: None)
    goal = engine.goals.start("Цель", "r1")
    engine.resources.set_facts({"machines": ["верстак"], "resources": {"inventory": {"медь": 3}}})
    worker = CopilotWorker(settings, memory, None)
    async with httpx.AsyncClient() as client:
        reply = DialogueReply(
            answer="Учту ваше сообщение",
            intent="report_resource",
            directive_quote="Открыта технология пара",
            facts_patch={"technologies": ["пар"], "machines": ["пресс"]},
        )
        await worker.dispatch(
            engine,
            client,
            Command(
                kind="dialogue_action",
                data={"reply": reply.model_dump_json(), "expected_goal": goal.id},
                revision=0,
            ),
        )
        assert engine.resources.facts()["technologies"] == ["пар"]
        assert engine.resources.facts()["machines"] == ["верстак", "пресс"]
        reply = DialogueReply(
            answer="Пересчитаю после нового наблюдения",
            intent="report_resource",
            directive_quote="Не хватает меди",
        )
        await worker.dispatch(
            engine,
            client,
            Command(
                kind="dialogue_action",
                data={"reply": reply.model_dump_json(), "expected_goal": goal.id},
                revision=0,
            ),
        )
        assert engine.resources.available() == {}


async def test_suggested_facts_cannot_become_confirmed_player_report(memory):
    class Provider(OfflineProvider):
        async def converse(self, text, context):
            return DialogueReply(
                answer="Предложение",
                intent="report_resource",
                directive_quote=text,
                facts_patch={"technologies": ["пар"]},
            )

    reply = await ConversationManager(memory, Provider()).respond(
        "Может, откроем технологию пара?", "voice", {}
    )
    assert reply.facts_patch == {}


async def test_user_added_task_takes_priority_over_automatic_curriculum(memory, tmp_path):
    engine = Copilot(
        Settings(_env_file=None, database_path=tmp_path / "priority.db"),
        memory,
        DemoProvider(),
        None,
        lambda _: None,
    )
    goal = engine.goals.start("Цель", "r1").model_copy(update={"target": ItemAmount(item="плата", count=1)})
    engine.goals.replace(goal)
    task = engine.goals.add_task("Проверить склад", priority=100)
    await engine._next(frame(), 0)
    assert engine.goals.current.pending.instruction.task_id == task.id
    assert not engine.goals.current.pending.instruction.completes_task


async def test_ollama_cancellation_waits_for_unload_and_releases_lane():
    from loliland_agent.local.scheduler import HEAVY

    entered = asyncio.Event()
    unloaded = []

    async def server(request):
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/api/generate":
            unloaded.append(True)
            return httpx.Response(200, json={"done": True})
        entered.set()
        await asyncio.Event().wait()

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        task = asyncio.create_task(make_provider(Settings(_env_file=None), client).chat("Привет", {}))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert unloaded and not HEAVY.active and not HEAVY.lock.locked()


def test_whisper_cached_model_never_checks_network_even_with_download_permission(monkeypatch):
    import types

    from loliland_agent.speech.local_runner import create_model

    calls = []

    def model(name, **kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=model))
    create_model("tiny", "cache", True)
    assert len(calls) == 1 and calls[0]["local_files_only"] is True
    assert calls[0]["device"] == "cpu" and calls[0]["compute_type"] == "int8"


def test_free_preferences_migrate_legacy_cloud_settings_without_loading_keys(tmp_path, monkeypatch):
    from loliland_agent.speech.credentials import load_settings

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "provider": "openai",
                "model": "gpt-old",
                "api_base_url": "https://api.openai.com/v1",
                "tts_provider": "cloud",
                "stt_provider": "cloud",
                "api_key": "discard-me",
            }
        )
    )
    monkeypatch.setattr(
        "loliland_agent.speech.credentials.vault",
        lambda: (_ for _ in ()).throw(AssertionError("Vault must not be accessed")),
    )
    settings = load_settings(path)
    assert settings.free_mode and settings.provider == "ollama" and settings.api_key.get_secret_value() == ""
    assert settings.tts_provider == "piper" and settings.stt_provider == "local"


async def test_unconfirmed_ollama_unload_is_not_reported_as_success():
    def server(request):
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        if request.url.path == "/api/generate":
            return httpx.Response(500, json={"error": "cannot unload"})
        return httpx.Response(200, json={"message": {"content": '{"answer":"Привет"}'}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        with pytest.raises(ProviderError, match="освобождение"):
            await make_provider(Settings(_env_file=None), client).chat("Привет", {})


async def test_headless_replay_stops_when_reflection_pauses_goal(tmp_path):
    from loliland_agent.__main__ import headless
    from loliland_agent.models import Observation

    screenshot = frame().image_path
    (tmp_path / "frame.png").write_bytes(screenshot.read_bytes())
    recording = tmp_path / "observations.jsonl"
    recording.write_text(
        (Observation(source="replay", image_path=Path("frame.png")).model_dump_json() + "\n") * 6
    )
    # Disable cadence only for this regression: the replay intentionally has frames beyond the pause.
    settings = Settings(
        _env_file=None, provider="offline", database_path=tmp_path / "headless.db"
    ).model_copy(update={"interval_seconds": 0})
    await asyncio.wait_for(headless(settings, recording, "Цель"), timeout=2)
    memory = Memory(settings.database_path)
    try:
        restored = GoalManager(memory)
        assert restored.current.status == "paused"
        assert ReflectionManager(memory, "unspecified").blocked(restored.current.id)
    finally:
        memory.close()
