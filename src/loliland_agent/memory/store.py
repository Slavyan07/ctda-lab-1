from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class Memory:
    """One transaction per significant event; WAL survives process restarts."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, timeout=10, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript("""
        CREATE TABLE IF NOT EXISTS goals (
          id INTEGER PRIMARY KEY, parent_id INTEGER REFERENCES goals(id),
          description TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', updated REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS events (
          id INTEGER PRIMARY KEY, at REAL NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS knowledge (
          category TEXT NOT NULL, name TEXT NOT NULL, payload TEXT NOT NULL,
          source TEXT NOT NULL, updated REAL NOT NULL, PRIMARY KEY(category,name));
        CREATE TABLE IF NOT EXISTS usage (
          id INTEGER PRIMARY KEY, at REAL NOT NULL, input_tokens INTEGER NOT NULL,
          output_tokens INTEGER NOT NULL, cost_usd REAL);
        PRAGMA user_version=1;
        """)
        # Legacy goal rows remain paused after an interrupted session.
        with self._db:
            self._db.execute("UPDATE goals SET status='paused' WHERE status='active'")

    def goal(self, description: str) -> int:
        description = description.strip()
        if not description or len(description) > 2000:
            raise ValueError("Goal must have 1–2000 characters")
        with self._lock, self._db:
            row = self._db.execute(
                "SELECT id FROM goals WHERE description=? AND status NOT IN ('complete','cancelled') ORDER BY id DESC LIMIT 1",
                (description,),
            ).fetchone()
            if row:
                return int(row["id"])
            cursor = self._db.execute(
                "INSERT INTO goals(description,updated) VALUES (?,?)", (description, time.time())
            )
            assert cursor.lastrowid is not None
            return cursor.lastrowid

    def latest_goal(self) -> str:
        with self._lock:
            row = self._db.execute(
                "SELECT description FROM goals WHERE status NOT IN ('complete','cancelled') ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return (
                str(row[0]) if row else "Осмотреть текущее состояние и определить безопасную следующую цель"
            )

    def set_goal_status(self, goal_id: int, status: str) -> None:
        if status not in {"pending", "active", "paused", "blocked", "complete", "cancelled"}:
            raise ValueError("Unknown goal status")
        with self._lock, self._db:
            self._db.execute(
                "UPDATE goals SET status=?, updated=? WHERE id=?", (status, time.time(), goal_id)
            )

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO events(at,kind,payload) VALUES (?,?,?)",
                (time.time(), kind, json.dumps(payload, ensure_ascii=False)),
            )

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (max(1, min(limit, 100)),)
            ).fetchall()
            return [
                {"id": row["id"], "at": row["at"], "kind": row["kind"], "payload": json.loads(row["payload"])}
                for row in rows
            ]

    def remember(self, category: str, name: str, payload: dict[str, Any], source: str) -> None:
        if not source.strip():
            raise ValueError("Knowledge requires provenance")
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO knowledge VALUES (?,?,?,?,?) ON CONFLICT(category,name) DO UPDATE SET payload=excluded.payload, source=excluded.source, updated=excluded.updated",
                (category, name, json.dumps(payload, ensure_ascii=False), source, time.time()),
            )

    def recall(self, category: str, name: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT payload,source FROM knowledge WHERE category=? AND name=?", (category, name)
            ).fetchone()
            return {"value": json.loads(row[0]), "source": row[1]} if row else None

    def list_knowledge(self, category: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT name,payload,source,updated FROM knowledge WHERE category=? ORDER BY name",
                (category,),
            ).fetchall()
            return [
                {"name": row[0], "value": json.loads(row[1]), "source": row[2], "updated": row[3]}
                for row in rows
            ]

    def record_usage(self, input_tokens: int, output_tokens: int, cost_usd: float | None) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO usage(at,input_tokens,output_tokens,cost_usd) VALUES (?,?,?,?)",
                (time.time(), input_tokens, output_tokens, cost_usd),
            )

    def totals(self) -> dict[str, int | float | None]:
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*),COALESCE(SUM(input_tokens),0),COALESCE(SUM(output_tokens),0),SUM(cost_usd),COUNT(cost_usd) FROM usage"
            ).fetchone()
            return {
                "calls": row[0],
                "input_tokens": row[1],
                "output_tokens": row[2],
                "cost_usd": row[3] if row[0] == row[4] else None,
            }

    def close(self) -> None:
        with self._lock:
            self._db.close()
