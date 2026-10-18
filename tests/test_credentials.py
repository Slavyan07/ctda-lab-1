import json

from pydantic import SecretStr

from loliland_agent.config import Settings
from loliland_agent.speech.credentials import load_settings, save_settings


def test_saved_preferences_never_contain_api_keys(tmp_path, monkeypatch):
    stored = {}

    class Vault:
        def set_password(self, service, name, value):
            stored[name] = value

    monkeypatch.setattr("loliland_agent.speech.credentials.vault", lambda: Vault())
    settings = Settings(
        _env_file=None,
        free_mode=False,
        api_key=SecretStr("secret-llm"),
        speech_api_key=SecretStr("secret-speech"),
    )
    path = tmp_path / "settings.json"
    save_settings(settings, path)
    assert stored == {"api_key": "secret-llm", "speech_api_key": "secret-speech"}
    assert "secret-" not in path.read_text() and "api_key" not in json.loads(path.read_text())


def test_corrupt_preferences_fall_back_safely(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("broken")
    assert load_settings(path).provider == "ollama"
