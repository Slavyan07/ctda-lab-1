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
    if sys.platform == "win32":
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
    return Settings(**data)
