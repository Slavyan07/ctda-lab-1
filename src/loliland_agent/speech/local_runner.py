"""Private CPU STT subprocess. Its lifetime never affects the Minecraft process."""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from typing import Any


def create_model(model: str, directory: str, download: bool) -> Any:
    from faster_whisper import WhisperModel
    from huggingface_hub.errors import LocalEntryNotFoundError

    options = dict(device="cpu", compute_type="int8", cpu_threads=2, num_workers=1, download_root=directory)
    try:
        return WhisperModel(model, local_files_only=True, **options)
    except (FileNotFoundError, LocalEntryNotFoundError):
        if not download:
            raise
        return WhisperModel(model, local_files_only=False, **options)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["tiny", "base"], required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    model = None
    for line in sys.stdin:
        try:
            if len(line) > 1_000_000:
                raise ValueError("Audio limit")
            request = json.loads(line)
            if model is None:
                model = create_model(args.model, args.directory, args.download)
            segments, _ = model.transcribe(
                io.BytesIO(base64.b64decode(request["wav"], validate=True)),
                language="ru",
                beam_size=1,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            result = {"text": "".join(segment.text for segment in segments).strip()[:8000]}
        except ImportError:
            result = {"error": "Локальное STT не установлено. Запустите install_local_stt.bat"}
        except Exception:
            result = {"error": "Whisper недоступен: проверьте кэш модели, разрешение загрузки и сеть"}
        print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
