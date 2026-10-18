import asyncio
import io
import wave
from array import array
from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr

from loliland_agent.config import Settings
from loliland_agent.speech.audio import MicrophoneCapture, Segmenter
from loliland_agent.speech.cloud_tts import SynthesisWorker, synthesize
from loliland_agent.speech.hotkeys import parse_hotkey
from loliland_agent.speech.stt import CloudSTT, LocalWhisperSTT, SpeechError, speech_key, wav_bytes
from loliland_agent.speech.worker import RecognitionWorker


def pcm(amplitude=2000, seconds=0.02):
    return array("h", [amplitude] * int(16000 * seconds)).tobytes()


def test_ptt_ends_on_release_and_filters_clicks():
    segmenter = Segmenter()
    for _ in range(20):
        assert segmenter.feed(pcm(), pressed=True) is None
    result = segmenter.feed(pcm(0), pressed=False)
    assert len(result) == 20 * 640 and segmenter.active is False
    segmenter.feed(pcm(), pressed=True)
    assert segmenter.feed(pcm(0), pressed=False) is None


def test_hands_free_vad_noise_silence_max_duration_and_echo():
    segmenter = Segmenter("hands_free", silence=0.3)
    for _ in range(50):
        assert segmenter.feed(pcm(20)) is None
    for _ in range(20):
        assert segmenter.feed(pcm()) is None
    results = [segmenter.feed(pcm(0)) for _ in range(20)]
    assert sum(value is not None for value in results) == 1
    for _ in range(200):
        assert segmenter.feed(pcm(), suppressed=True) is None
    results = [segmenter.feed(pcm()) for _ in range(1100)]
    assert sum(value is not None for value in results) == 1
    assert max(len(value) for value in results if value) <= 16000 * 2 * 20


def test_wav_format_and_duration_limit():
    data = wav_bytes(pcm(seconds=0.4))
    with wave.open(io.BytesIO(data)) as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)
    with pytest.raises(SpeechError):
        wav_bytes(b"0" * (16000 * 40 + 2))


@pytest.mark.parametrize(
    "key", ["W", "SPACE", "F8", "CTRL+F8", "CTRL+ALT+F9", "CTRL+SHIFT", "CTRL+CTRL+SPACE"]
)
def test_ptt_rejects_plain_or_reserved_hotkeys(key):
    with pytest.raises(ValueError):
        parse_hotkey(key)


def test_configurable_modifier_hotkey():
    assert parse_hotkey("CTRL+ALT+SPACE") == (0x11, 0x12, 0x20)
    assert parse_hotkey("SHIFT+F6") == (0x10, 0x75)


async def test_cloud_stt_real_wav_wire_contract():
    def response(request):
        assert request.url.path == "/v1/audio/transcriptions"
        assert (
            b"speech.wav" in request.content
            and b"RIFF" in request.content
            and b'name="language"' in request.content
        )
        return httpx.Response(200, json={"text": "Может, сначала займёмся энергией?"})

    settings = Settings(_env_file=None, speech_api_key=SecretStr("test-speech-key"))
    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        assert "энергией" in await CloudSTT(settings, client).transcribe(pcm(seconds=0.4))


@pytest.mark.parametrize("code", [401, 429, 500, 302])
async def test_speech_api_failures_are_safe(code):
    settings = Settings(_env_file=None, speech_api_key=SecretStr("test-speech-key"))
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(code, text="test-speech-key"))
    ) as client:
        with pytest.raises(SpeechError) as stt_error:
            await CloudSTT(settings, client).transcribe(pcm())
        with pytest.raises(SpeechError) as tts_error:
            await synthesize(settings, client, "Привет", "coral", 1.0)
    assert "test-speech-key" not in str(stt_error.value) + str(tts_error.value)


async def test_cloud_tts_wav_contract():
    data = wav_bytes(pcm(seconds=0.4))

    def response(request):
        import json

        body = json.loads(request.content)
        assert body["voice"] == "coral" and body["speed"] == 1.25 and body["response_format"] == "wav"
        return httpx.Response(200, content=data)

    settings = Settings(_env_file=None, speech_api_key=SecretStr("test-key"))
    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        assert await synthesize(settings, client, "Открой рецепт", "coral", 1.25) == data


def test_api_key_not_reused_for_another_speech_host():
    with pytest.raises(SpeechError):
        speech_key(
            Settings(
                _env_file=None, api_key=SecretStr("key"), speech_api_base_url="https://another.example/v1"
            )
        )


def test_local_whisper_cpu_and_explicit_download_policy(monkeypatch, tmp_path):
    import sys

    from loliland_agent.speech.local_runner import create_model

    calls = []

    class Model:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Model))
    create_model("tiny", str(tmp_path), False)
    assert (
        calls[0]["local_files_only"]
        and calls[0]["device"] == "cpu"
        and calls[0]["compute_type"] == "int8"
        and calls[0]["cpu_threads"] == 2
    )


async def test_local_subprocess_transcribes_and_is_closed(tmp_path):
    import sys

    program = (
        "import sys,json; [(print(json.dumps({'text':'Добавь склад'}),flush=True)) for line in sys.stdin]"
    )
    engine = LocalWhisperSTT(Settings(_env_file=None), [sys.executable, "-u", "-c", program])
    try:
        assert await engine.transcribe(pcm()) == "Добавь склад"
        process = engine.process
    finally:
        await engine.close()
    assert process.returncode is not None and engine.process is None


async def test_local_model_cancel_terminates_only_owned_subprocess():
    import sys

    program = "import sys,time; sys.stdin.readline(); time.sleep(60)"
    engine = LocalWhisperSTT(Settings(_env_file=None), [sys.executable, "-u", "-c", program])
    job = asyncio.create_task(engine.transcribe(pcm()))
    for _ in range(100):
        if engine.process is not None:
            break
        await asyncio.sleep(0.01)
    process = engine.process
    assert process is not None
    job.cancel()
    with pytest.raises(asyncio.CancelledError):
        await job
    assert process.returncode is not None and engine.process is None


def test_unavailable_microphone_does_not_start_services(monkeypatch):
    import sys

    adapter = SimpleNamespace(
        check_input_settings=lambda **_: None,
        RawInputStream=lambda **_: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )
    monkeypatch.setitem(sys.modules, "sounddevice", adapter)
    capture = MicrophoneCapture(Segmenter(), lambda _: None, lambda _: None)
    with pytest.raises(SpeechError, match="Микрофон недоступен"):
        capture.start()
    assert capture.thread is None and capture.stream is None


def test_recognition_queue_delivers_russian_and_stops(qapp):
    from test_ui import until

    class Transcriber:
        async def transcribe(self, data):
            return "Добавь в план склад"

    worker = RecognitionWorker(Settings(_env_file=None), Transcriber())
    events = []
    worker.updated.connect(events.append)
    worker.start()
    worker.submit(pcm(seconds=0.4))
    try:
        until(qapp, lambda: any(e["kind"] == "transcript" for e in events))
    finally:
        worker.stop()
        assert worker.wait(2000)
    assert next(e["text"] for e in events if e["kind"] == "transcript") == "Добавь в план склад"


def test_synthesis_interrupt_discards_old_audio_and_stops(qapp, monkeypatch):
    from test_ui import until

    async def slow(*args):
        await asyncio.sleep(5)
        return b"old audio"

    monkeypatch.setattr("loliland_agent.speech.cloud_tts.synthesize", slow)
    worker = SynthesisWorker(Settings(_env_file=None))
    results = []
    worker.ready.connect(lambda *args: results.append(args))
    worker.start()
    worker.submit("Старая инструкция", "coral", 1.0)
    until(qapp, lambda: worker.items.empty())
    worker.cancel()
    worker.stop()
    assert worker.wait(2000) and not results
