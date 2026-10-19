"""SQLite access for document metadata."""
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from app.database.db import Database


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DocumentStore:
    def __init__(self, db: Database):
        self.db = db

    def create(self, doc_id: str, filename: str, stored_name: str, size_bytes: int) -> bool:
        """Insert a 'processing' record. Returns False if the id already exists."""
        now = now_iso()
        try:
            with self.db.connect() as conn:
                conn.execute(
                    "INSERT INTO documents (id, filename, stored_name, status, size_bytes, created_at, updated_at) "
                    "VALUES (?, ?, ?, 'processing', ?, ?, ?)",
                    (doc_id, filename, stored_name, size_bytes, now, now),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def get(self, doc_id: str) -> Optional[dict]:
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def list_all(self) -> list[dict]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC, rowid DESC").fetchall()
        return [dict(r) for r in rows]

    def list_ready_ids(self) -> list[str]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT id FROM documents WHERE status = 'ready'").fetchall()
        return [r["id"] for r in rows]

    def mark_ready(self, doc_id: str, page_count: int, chunk_count: int) -> None:
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE documents SET status='ready', page_count=?, chunk_count=?, error=NULL, updated_at=? WHERE id=?",
                (page_count, chunk_count, now_iso(), doc_id),
            )

    def mark_failed(self, doc_id: str, error: str) -> None:
        with self.db.connect() as conn:
            conn.execute(
                "UPDATE documents SET status='failed', error=?, updated_at=? WHERE id=?",
                (error, now_iso(), doc_id),
            )

    def fail_interrupted(self) -> int:
        """Called on startup: documents left 'processing' by a crash/restart can never finish."""
        with self.db.connect() as conn:
            cur = conn.execute(
                "UPDATE documents SET status='failed', error=?, updated_at=? WHERE status='processing'",
                ("Processing was interrupted by a restart. Upload the file again.", now_iso()),
            )
            return cur.rowcount

    def delete(self, doc_id: str) -> None:
        with self.db.connect() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
