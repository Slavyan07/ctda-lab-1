from __future__ import annotations

import re
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    provider: Literal["offline", "openai", "compatible", "ollama"] = "offline"
    model: str = "gpt-4.1-mini"
    api_base_url: str = "https://api.openai.com/v1"
    api_key: SecretStr = SecretStr("")
    vision: bool = True
    request_timeout: float = Field(default=45, ge=1, le=120)
    interval_seconds: float = Field(default=8, ge=3, le=3600)
    analysis_cache_seconds: float = Field(default=30, ge=0, le=300)
    pack_version: str = Field(default="unspecified", min_length=1, max_length=250)
    speech_enabled: bool = False
    speech_volume: int = Field(default=80, ge=0, le=100)
    database_path: Path = Path("data/copilot.sqlite3")
    window_title_pattern: str = r"(?i)minecraft|loliland|LoliLand Agent Test"
    input_price_per_million: float | None = Field(default=None, ge=0)
    output_price_per_million: float | None = Field(default=None, ge=0)

    @field_validator("input_price_per_million", "output_price_per_million", mode="before")
    @classmethod
    def empty_price(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("window_title_pattern")
    @classmethod
    def valid_pattern(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Window title pattern cannot be empty")
        re.compile(value)
        return value

    @model_validator(mode="after")
    def validate_provider(self) -> Settings:
        url = urlsplit(self.api_base_url)
        if not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("Invalid API URL; credentials belong in API_KEY")
        if url.scheme != "https" and not (
            url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1"}
        ):
            raise ValueError("API requires HTTPS except on loopback")
        if self.provider == "openai" and url.hostname != "api.openai.com":
            raise ValueError("Use compatible for another endpoint")
        if self.provider == "ollama" and url.hostname not in {"localhost", "127.0.0.1"}:
            raise ValueError("Use compatible for remote endpoints")
        if self.provider != "offline" and not self.vision:
            raise ValueError("Copilot requires a multimodal model and VISION=true")
        if self.provider in {"openai", "compatible"} and not self.api_key.get_secret_value():
            raise ValueError("Set API_KEY in local .env")
        return self
