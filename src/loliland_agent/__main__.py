from __future__ import annotations

import argparse
import asyncio
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import httpx
from pydantic import ValidationError

from loliland_agent.config import Settings
from loliland_agent.instructions.engine import Copilot
from loliland_agent.memory import Memory
from loliland_agent.perception.sources import ReplayEnded, ReplaySource
from loliland_agent.planning.providers import make_provider


def configure_logging() -> None:
    Path("logs").mkdir(exist_ok=True)
    for name, filename in [
        ("loliland.copilot", "copilot.jsonl"),
        ("loliland.instructions", "instructions.txt"),
    ]:
        logger = logging.getLogger(name)
        if logger.handlers:
            continue
        handler = RotatingFileHandler(
            Path("logs") / filename, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)


async def headless(settings: Settings, recording: Path, goal: str) -> None:
    memory, source = Memory(settings.database_path), ReplaySource(recording)

    def emit(event: dict[str, object]) -> None:
        if event["kind"] != "snapshot":
            print(json.dumps(event, ensure_ascii=False))

    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            service = Copilot(settings, memory, make_provider(settings, client), source, emit)
            await service.start(goal, 50, 0)
            while True:
                await service.check(0)
    except ReplayEnded:
        pass
    finally:
        source.close()
        memory.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="LoliLand AI Copilot: человек выполняет игровые действия")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--demo", action="store_true", help="Artificial screenshots, deterministic provider, no API"
    )
    inputs.add_argument("--image", type=Path, help="Analyze a saved screenshot")
    inputs.add_argument("--replay", type=Path, help="Read JSONL screenshots as human observations")
    parser.add_argument("--headless", action="store_true", help="Process a replay without GUI")
    parser.add_argument(
        "--goal", default="Изучить видимое состояние сборки", help="Russian language goal for headless replay"
    )
    parser.add_argument("--record", type=Path, help="New directory for local screenshot recordings")
    args = parser.parse_args()
    if args.headless and args.replay is None:
        parser.error("--headless requires --replay")
    if args.record and args.replay:
        parser.error("--record cannot be combined with --replay")
    try:
        settings = Settings()
    except ValidationError:
        print("Ошибка .env: проверьте провайдера, HTTPS URL, API_KEY и VISION. Значения скрыты.")
        return 2
    configure_logging()
    if args.headless:
        try:
            asyncio.run(headless(settings, args.replay, args.goal))
        except KeyboardInterrupt:
            return 130
        except Exception as error:
            print(f"Операция не выполнена ({type(error).__name__}); проверьте запись и настройки API.")
            return 1
        return 0
    from loliland_agent.ui.app import launch

    return launch(settings, replay=args.replay, image=args.image, demo=args.demo, record=args.record)


if __name__ == "__main__":
    raise SystemExit(main())
