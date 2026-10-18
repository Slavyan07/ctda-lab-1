"""Build a clean Windows source distribution without requiring Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = {
    ".env.example",
    ".gitattributes",
    ".python-version",
    "pyproject.toml",
    "uv.lock",
    "requirements-lock.txt",
    "requirements-local-lock.txt",
    "README.md",
    "README_RU.md",
    "WINDOWS_SETUP.md",
    "install.bat",
    "install_local_stt.bat",
    "start.bat",
    "check.bat",
}
DIRECTORIES = {"src", "tests", "examples", "docs", "scripts", "tools"}
EXCLUDE = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".git",
    ".venv",
    "data",
    "logs",
    "recordings",
    "dist",
    "models",
}


def build(destination: Path) -> None:
    entries = {}
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDE for part in relative.parts) or path.is_symlink() or not path.is_file():
            continue
        if relative.parts[0] not in DIRECTORIES and relative.as_posix() not in FILES:
            continue
        if (
            path.suffix in {".pyc", ".pyo", ".db", ".sqlite3", ".wav", ".bin"}
            or path.name == "PR_DESCRIPTION.md"
        ):
            continue
        entries[relative.as_posix()] = path.read_bytes()
    for required in (
        "install.bat",
        "start.bat",
        "check.bat",
        "README_RU.md",
        "WINDOWS_SETUP.md",
        ".env.example",
        "requirements-lock.txt",
        "requirements-local-lock.txt",
        "src/loliland_agent/__main__.py",
        "src/loliland_agent/assets/demo-initial.png",
        "src/loliland_agent/assets/demo-inventory.png",
        "src/loliland_agent/speech/local_runner.py",
    ):
        if required not in entries:
            raise RuntimeError("Missing distribution file: " + required)
    entries["BUILD_INFO.json"] = json.dumps(
        {
            "product": "LoliLand AI Copilot",
            "version": "0.4.0",
            "python": "3.11 x64",
            "delivery": "Source ZIP + Windows installers; Internet required for dependencies/API; local voice model optional and opt-in",
        },
        ensure_ascii=False,
        indent=2,
    ).encode()
    manifest = "".join(
        hashlib.sha256(data).hexdigest() + "  " + name + "\n" for name, data in sorted(entries.items())
    )
    entries["MANIFEST.sha256"] = manifest.encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(entries.items()):
            archive.writestr("LoliLand_AI_Copilot/" + name, data)
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        for line in manifest.splitlines():
            digest, name = line.split("  ", 1)
            assert hashlib.sha256(archive.read("LoliLand_AI_Copilot/" + name)).hexdigest() == digest
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(".zip.sha256").write_text(digest + "  " + destination.name + "\n")
    print(f"{destination}: {len(entries)} files, {destination.stat().st_size} bytes, SHA256 {digest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    build(args.destination)
