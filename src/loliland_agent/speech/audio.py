from __future__ import annotations

import math
import queue
import threading
import time
from array import array
from collections import deque
from collections.abc import Callable
from typing import Any, Literal

from loliland_agent.speech.stt import RATE, SpeechError


class Segmenter:
    """PCM16 VAD/PTT; min speech 300 ms, silence endpoint, max 20 seconds."""

    def __init__(
        self, mode: Literal["ptt", "hands_free"] = "ptt", threshold: float = 0.015, silence: float = 0.7
    ) -> None:
        self.mode, self.threshold, self.silence = mode, threshold, silence
        self.reset()

    def reset(self) -> None:
        self.buffer = bytearray()
        self.voiced = 0.0
        self.quiet = 0.0
        self.preroll: deque[bytes] = deque(maxlen=10)
        self.active = False

    def feed(self, pcm: bytes, *, pressed: bool = False, suppressed: bool = False) -> bytes | None:
        if suppressed:
            self.reset()
            return None
        samples = array("h", pcm)
        if not samples:
            return None
        duration = len(samples) / RATE
        energy = math.sqrt(sum(int(x) * int(x) for x in samples) / len(samples)) / 32768
        voiced = energy >= self.threshold
        if self.mode == "ptt":
            if pressed:
                self.active = True
                self.buffer.extend(pcm)
                if voiced:
                    self.voiced += duration
            elif self.active:
                return self.finish()
        else:
            if not self.active:
                self.preroll.append(pcm)
                if not voiced:
                    return None
                self.active = True
                self.buffer.extend(b"".join(self.preroll))
                self.preroll.clear()
            else:
                self.buffer.extend(pcm)
            if voiced:
                self.voiced += duration
                self.quiet = 0
            else:
                self.quiet += duration
            if self.quiet >= self.silence:
                return self.finish()
        if len(self.buffer) >= RATE * 2 * 20:
            return self.finish()
        return None

    def finish(self) -> bytes | None:
        result = bytes(self.buffer[: RATE * 2 * 20]) if self.voiced >= 0.3 else None
        self.reset()
        return result


class MicrophoneCapture:
    def __init__(
        self, segmenter: Segmenter, utterance: Callable[[bytes], None], status: Callable[[str], None]
    ) -> None:
        self.segmenter, self.utterance, self.status = segmenter, utterance, status
        self.frames: queue.Queue[bytes] = queue.Queue(maxsize=250)
        self.stopping = threading.Event()
        self.ptt = threading.Event()
        self.speaking = threading.Event()
        self.echo_until = 0.0
        self.stream: Any = None
        self.thread: threading.Thread | None = None
        self.rate = RATE
        self.overflow = threading.Event()

    def start(self, device: int | None = None) -> None:
        self.stopping.clear()
        try:
            import sounddevice as sd

            try:
                sd.check_input_settings(device=device, channels=1, dtype="int16", samplerate=RATE)
            except Exception:
                self.rate = int(sd.query_devices(device, "input")["default_samplerate"])

            def callback(data: Any, _frames: int, _time: Any, status: Any) -> None:
                if status:
                    self.overflow.set()
                try:
                    self.frames.put_nowait(bytes(data))
                except queue.Full:
                    self.overflow.set()

            self.stream = sd.RawInputStream(
                device=device,
                samplerate=self.rate,
                channels=1,
                dtype="int16",
                blocksize=int(self.rate * 0.02),
                callback=callback,
            )
            self.stream.start()
        except Exception:
            self.close()
            raise SpeechError(
                "Микрофон недоступен. Проверьте устройство и разрешение Windows: Конфиденциальность → Микрофон → классические приложения"
            ) from None
        self.thread = threading.Thread(target=self.run, daemon=True, name="copilot-audio-vad")
        self.thread.start()
        self.status("Микрофон включён; PTT или VAD ожидает речь")

    def set_speaking(self, value: bool) -> None:
        if value:
            self.speaking.set()
        else:
            if self.speaking.is_set():
                self.echo_until = time.monotonic() + 0.4
            self.speaking.clear()

    def run(self) -> None:
        # resampling is inexpensive and separate from PortAudio's callback.
        import audioop

        state = None
        while not self.stopping.is_set():
            try:
                pcm = self.frames.get(timeout=0.05)
            except queue.Empty:
                continue
            if self.overflow.is_set():
                self.overflow.clear()
                self.segmenter.reset()
                while not self.frames.empty():
                    try:
                        self.frames.get_nowait()
                    except queue.Empty:
                        break
                self.status("Аудиобуфер переполнен; фраза отброшена, повторите")
                continue
            if self.rate != RATE:
                pcm, state = audioop.ratecv(pcm, 2, 1, self.rate, RATE, state)
            result = self.segmenter.feed(
                pcm,
                pressed=self.ptt.is_set(),
                suppressed=self.speaking.is_set() or time.monotonic() < self.echo_until,
            )
            if result:
                self.utterance(result)

    def close(self) -> None:
        self.stopping.set()
        self.ptt.clear()
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        if self.thread:
            self.thread.join(timeout=2)
        self.segmenter.reset()


def microphones() -> list[tuple[int, str]]:
    import sounddevice as sd

    return [
        (index, str(value["name"]))
        for index, value in enumerate(sd.query_devices())
        if value["max_input_channels"] > 0
    ]
