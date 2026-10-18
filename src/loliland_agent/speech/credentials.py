"""API secrets in Windows Credential Manager; preferences contain no secrets."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import SecretStr

if TYPE_CHECKING:
    from loliland_agent.config import Settings

SERVICE = "LoliLand AI Copilot"


def vault() -> Any:
    if sys.platform != "win32":
        raise RuntimeError(
            "Сохранение ключей доступно через Windows Credential Manager; в этом сеансе ключ можно использовать без сохранения"
        )
    from keyring.backends.Windows import WinVaultKeyring

    return WinVaultKeyring()


def save_settings(settings: Settings, path: Path = Path("data/settings.json")) -> None:
    if not settings.free_mode:
        backend = vault()
        for name in ("api_key", "speech_api_key"):
            value = getattr(settings, name).get_secret_value()
            if value:
                backend.set_password(SERVICE, name, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            settings.model_dump(mode="json", exclude={"api_key", "speech_api_key", "database_path"}),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_settings(path: Path = Path("data/settings.json")) -> Settings:
    from loliland_agent.config import Settings

    data: dict[str, Any] = {}
    if path.is_file():
        try:
            if path.stat().st_size > 50_000:
                raise ValueError("Settings file too large")
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Settings must be an object")
            data.pop("api_key", None)
            data.pop("speech_api_key", None)
        except (ValueError, OSError, TypeError):
            data = {}
    if sys.platform == "win32" and data.get("free_mode") is False:
        try:
            backend = vault()
            for name in ("api_key", "speech_api_key"):
                if value := backend.get_password(SERVICE, name):
                    data[name] = SecretStr(value)
        except Exception:
            pass
    # Explicit process environment overrides saved GUI preferences; .env supplies other fields.
    for name in Settings.model_fields:
        if name.upper() in os.environ:
            data[name] = os.environ[name.upper()]
    from dotenv import dotenv_values

    file_values = {key.lower(): value for key, value in dotenv_values(".env").items() if value is not None}
    data = {**file_values, **data}
    if str(data.get("free_mode", "true")).lower() not in {"false", "0"}:
        data["free_mode"] = True
        data["api_key"] = SecretStr("")
        data["speech_api_key"] = SecretStr("")
        if data.get("provider") not in {"ollama", "offline"}:
            data["provider"] = "ollama"
        if data.get("model") not in {"qwen3-vl:4b", "qwen3-vl:2b"}:
            data["model"] = "qwen3-vl:4b"
        from urllib.parse import urlsplit

        if urlsplit(str(data.get("api_base_url", ""))).hostname not in {"127.0.0.1", "localhost"}:
            data["api_base_url"] = "http://127.0.0.1:11434/v1"
        if data.get("tts_provider") not in {"piper", "system"}:
            data["tts_provider"] = "piper"
        if data.get("stt_provider") not in {"disabled", "local"}:
            data["stt_provider"] = "local"
    return Settings(**data)
