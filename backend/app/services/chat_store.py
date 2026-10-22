"""SQLite access for chat sessions and messages."""
import json
import uuid
from typing import Optional

from app.database.db import Database
from app.services.document_store import now_iso


class ChatStore:
    def __init__(self, db: Database):
        self.db = db

    def create_session(self, title: Optional[str] = None) -> dict:
        session_id = uuid.uuid4().hex
        now = now_iso()
        title = (title or "New chat").strip() or "New chat"
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO chat_sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title, now, now),
            )
        return {"id": session_id, "title": title, "created_at": now, "updated_at": now}

    def get_session(self, session_id: str) -> Optional[dict]:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None

    def list_sessions(self) -> list[dict]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM chat_sessions ORDER BY updated_at DESC, rowid DESC").fetchall()
        return [dict(r) for r in rows]

    def delete_session(self, session_id: str) -> bool:
        with self.db.connect() as conn:
            cur = conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
            return cur.rowcount > 0

    def rename_session(self, session_id: str, title: str) -> None:
        with self.db.connect() as conn:
            conn.execute("UPDATE chat_sessions SET title=? WHERE id=?", (title, session_id))

    def add_message(self, session_id: str, role: str, content: str, sources: Optional[list[dict]] = None) -> None:
        now = now_iso()
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO chat_messages (session_id, role, content, sources_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, json.dumps(sources) if sources else None, now),
            )
            conn.execute("UPDATE chat_sessions SET updated_at=? WHERE id=?", (now, session_id))

    def get_messages(self, session_id: str) -> list[dict]:
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY id", (session_id,)
            ).fetchall()
        messages = []
        for row in rows:
            item = dict(row)
            item["sources"] = json.loads(item.pop("sources_json")) if item.get("sources_json") else []
            item.pop("session_id", None)
            messages.append(item)
        return messages
