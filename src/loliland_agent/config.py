from __future__ import annotations

import re
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    provider: Literal["offline", "openai", "compatible", "ollama"] = "ollama"
    model: str = "qwen3-vl:4b"
    api_base_url: str = "http://127.0.0.1:11434/v1"
    api_key: SecretStr = SecretStr("")
    vision: bool = True
    request_timeout: float = Field(default=180, ge=1, le=600)
    interval_seconds: float = Field(default=8, ge=3, le=3600)
    analysis_cache_seconds: float = Field(default=30, ge=0, le=300)
    free_mode: bool = True
    ollama_num_ctx: int = Field(default=4096, ge=2048, le=8192)
    ollama_num_predict: int = Field(default=1200, ge=256, le=2400)
    ollama_num_gpu: int = Field(default=0, ge=0, le=12)
    ollama_threads: int = Field(default=4, ge=1, le=8)
    max_model_ram_mb: int = Field(default=5120, ge=1024, le=8192)
    max_model_vram_mb: int = Field(default=1536, ge=0, le=4096)
    min_free_ram_mb: int = Field(default=2048, ge=512, le=8192)
    unload_after_request: bool = True
    local_image_max_edge: int = Field(default=768, ge=384, le=1280)
    max_failed_attempts: int = Field(default=3, ge=2, le=6)
    pack_version: str = Field(default="unspecified", min_length=1, max_length=250)
    speech_enabled: bool = False
    speech_volume: int = Field(default=80, ge=0, le=100)
    speech_rate: float = Field(default=1.0, ge=0.5, le=2)
    tts_provider: Literal["system", "cloud", "piper"] = "piper"
    tts_voice: str = "ru_RU-irina-medium"
    tts_model: str = "gpt-4o-mini-tts"
    speech_api_base_url: str = "https://api.openai.com/v1"
    speech_api_key: SecretStr = SecretStr("")
    stt_provider: Literal["disabled", "cloud", "local"] = "local"
    stt_model: str = "gpt-4o-mini-transcribe"
    whisper_model: Literal["tiny", "base"] = "tiny"
    allow_model_download: bool = False
    voice_mode: Literal["ptt", "hands_free"] = "ptt"
    ptt_hotkey: str = "CTRL+ALT+F6"
    vad_threshold: float = Field(default=0.015, ge=0.001, le=0.3)
    vad_silence: float = Field(default=0.7, ge=0.3, le=2)
    microphone_device: int | None = None
    audio_output_device: str = ""
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
        if self.free_mode:
            if (
                self.provider not in {"ollama", "offline"}
                or self.stt_provider == "cloud"
                or self.tts_provider == "cloud"
            ):
                raise ValueError(
                    "Free mode permits local Ollama/Whisper/Piper only; cloud fallback is disabled"
                )
            if self.provider == "ollama" and self.model not in {"qwen3-vl:4b", "qwen3-vl:2b"}:
                raise ValueError("Choose qwen3-vl:4b or qwen3-vl:2b")
        speech_url = urlsplit(self.speech_api_base_url)
        if (
            not speech_url.hostname
            or speech_url.username
            or speech_url.password
            or speech_url.query
            or speech_url.fragment
            or (
                speech_url.scheme != "https"
                and not (speech_url.scheme == "http" and speech_url.hostname in {"localhost", "127.0.0.1"})
            )
        ):
            raise ValueError("Invalid speech API URL")
        from loliland_agent.speech.hotkeys import parse_hotkey

        parse_hotkey(self.ptt_hotkey)
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
