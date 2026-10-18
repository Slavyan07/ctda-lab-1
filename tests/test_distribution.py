import hashlib
import importlib.util
import zipfile
from pathlib import Path


def builder():
    spec = importlib.util.spec_from_file_location("copilot_zip_builder", Path("tools/build_zip.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_zip_is_self_contained_without_git_and_excludes_runtime_secrets(tmp_path, monkeypatch):
    module = builder()
    archive = tmp_path / "LoliLand_AI_Copilot_Windows.zip"
    module.build(archive)
    with zipfile.ZipFile(archive) as data:
        names = data.namelist()
        for required in [
            "install.bat",
            "start.bat",
            "check.bat",
            "README_RU.md",
            "WINDOWS_SETUP.md",
            "requirements-local-lock.txt",
            "src/loliland_agent/speech/local_runner.py",
            "src/loliland_agent/ui/voice_panel.py",
        ]:
            assert "LoliLand_AI_Copilot/" + required in names
        assert not any(
            "/.git/" in n or "/.venv/" in n or "/data/" in n or n.endswith("/.env") or "__pycache__" in n
            for n in names
        )
        for line in data.read("LoliLand_AI_Copilot/MANIFEST.sha256").decode().splitlines():
            digest, name = line.split("  ", 1)
            assert hashlib.sha256(data.read("LoliLand_AI_Copilot/" + name)).hexdigest() == digest


def test_windows_scripts_use_utf8_crlf_existing_entrypoints_and_no_git():
    for name in ["install.bat", "install_local_stt.bat", "start.bat", "check.bat"]:
        data = Path(name).read_bytes()
        text = data.decode("utf-8")
        assert "chcp 65001" in text and b"\r\n" in data and b"\n" not in data.replace(b"\r\n", b"")
        assert "git clone" not in text.lower() and "github" not in text.lower()
    assert Path("src/loliland_agent/__main__.py").is_file()
    assert Path("src/loliland_agent/diagnostics.py").is_file()
