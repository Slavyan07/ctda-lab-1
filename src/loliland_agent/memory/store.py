from __future__ import annotations

import json
import re
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
        CREATE TABLE IF NOT EXISTS conversation_messages (
          id INTEGER PRIMARY KEY, at REAL NOT NULL, role TEXT NOT NULL, channel TEXT NOT NULL, text TEXT NOT NULL);
        PRAGMA user_version=2;
        """)
        try:
            self._db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS skill_search USING fts5(category UNINDEXED, name UNINDEXED, text, tokenize='unicode61')"
            )
            self._db.execute("DELETE FROM skill_search")
            self._db.execute(
                "INSERT INTO skill_search(category,name,text) SELECT category,name,payload FROM knowledge WHERE category LIKE 'skills:%'"
            )
            self._fts = True
        except sqlite3.OperationalError:
            self._fts = False
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
            if self._fts and category.startswith("skills:"):
                self._db.execute("DELETE FROM skill_search WHERE category=? AND name=?", (category, name))
                self._db.execute(
                    "INSERT INTO skill_search(category,name,text) VALUES (?,?,?)",
                    (category, name, json.dumps(payload, ensure_ascii=False)),
                )

    def search_knowledge(self, category: str, query: str, limit: int = 40) -> list[dict[str, Any]]:
        terms = re.findall(r"\w+", query.casefold())[:30]
        if not terms or not self._fts:
            return self.list_knowledge(category)
        match = " OR ".join('"' + term + '"' for term in terms)
        with self._lock:
            rows = self._db.execute(
                "SELECT k.name,k.payload,k.source,k.updated FROM skill_search s JOIN knowledge k ON k.category=s.category AND k.name=s.name WHERE skill_search MATCH ? AND s.category=? ORDER BY rank LIMIT ?",
                (match, category, max(1, min(limit, 200))),
            ).fetchall()
            return [
                {"name": row[0], "value": json.loads(row[1]), "source": row[2], "updated": row[3]}
                for row in rows
            ]

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

    def message(self, role: str, channel: str, text: str) -> int:
        if (
            role not in {"user", "assistant"}
            or channel not in {"voice", "text"}
            or not text.strip()
            or len(text) > 8000
        ):
            raise ValueError("Неверное сообщение диалога")
        with self._lock, self._db:
            cursor = self._db.execute(
                "INSERT INTO conversation_messages(at,role,channel,text) VALUES (?,?,?,?)",
                (time.time(), role, channel, text),
            )
            assert cursor.lastrowid is not None
            return cursor.lastrowid

    def messages(self, limit: int = 12, *, after: int = 0, oldest: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM conversation_messages WHERE id>? ORDER BY id {'ASC' if oldest else 'DESC'} LIMIT ?",
                (after, max(1, min(limit, 200))),
            ).fetchall()
            return [dict(row) for row in (rows if oldest else reversed(rows))]

    def close(self) -> None:
        with self._lock:
            self._db.close()
