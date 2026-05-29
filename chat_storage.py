import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Optional


log = logging.getLogger("chat_storage")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class StoredMessage:
    role: str  # 'user' | 'assistant' | 'system'
    content: str
    created_at: str


class ChatStorage:
    """
    Хранилище переписки в SQLite (файлом на диске).

    Почему SQLite:
    - без внешней инфраструктуры (для демо/проверки работоспособности)
    - быстрые чтения последних N сообщений (контекстное окно)
    - WAL + индексы дают нормальную производительность на небольших объемах

    Соединение открывается на каждую операцию (connect-per-op) для безопасной
    работы с параллельными обработчиками TeleBot.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._apply_pragmas_once()
        with self._connection() as conn:
            self._ensure_schema(conn)

    def close(self) -> None:
        """Нет долгоживущего соединения — метод оставлен для совместимости API."""
        return

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys=ON;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _apply_pragmas_once(self) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute("PRAGMA temp_store=MEMORY;")
            cur.execute("PRAGMA foreign_keys=ON;")
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        cur = conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                tg_chat_id INTEGER NOT NULL,
                title TEXT,
                context_window TEXT NOT NULL DEFAULT '{"window_size": 5, "messages": [], "summary": ""}',
                message_count INTEGER NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                CHECK (message_count >= 0)
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_tg_chat_id ON chat_sessions(tg_chat_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_active ON chat_sessions(is_active);
            CREATE INDEX IF NOT EXISTS idx_sessions_updated ON chat_sessions(updated_at);

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                citations TEXT NOT NULL DEFAULT '[]',
                metadata TEXT NOT NULL DEFAULT '{}',
                token_count INTEGER,
                in_response_to TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE,
                CHECK (role IN ('user', 'assistant', 'system'))
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session_created ON messages(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_messages_in_response_to ON messages(in_response_to);
            """
        )
        self._migrate_drop_unique_tg_chat_id(conn)
        self._repair_messages_fk_if_needed(conn)

    def _rebuild_messages_table_with_fk(self, conn: sqlite3.Connection, *, fk_parent: str) -> None:
        """
        Пересоздаёт таблицу messages с FK(session_id) -> fk_parent(id).
        Нужно после RENAME chat_sessions -> chat_sessions_old: SQLite может оставить FK, указывающий на _old.
        """
        log.info("Rebuilding messages table: FK parent=%s", fk_parent)
        cur = conn.cursor()
        cur.executescript(
            f"""
            PRAGMA foreign_keys=OFF;
            BEGIN;
            ALTER TABLE messages RENAME TO messages_old;
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                citations TEXT NOT NULL DEFAULT '[]',
                metadata TEXT NOT NULL DEFAULT '{{}}',
                token_count INTEGER,
                in_response_to TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES {fk_parent}(id) ON DELETE CASCADE,
                CHECK (role IN ('user', 'assistant', 'system'))
            );
            INSERT INTO messages (id, session_id, role, content, citations, metadata, token_count, in_response_to, created_at)
            SELECT id, session_id, role, content, citations, metadata, token_count, in_response_to, created_at
            FROM messages_old;
            DROP TABLE messages_old;
            CREATE INDEX IF NOT EXISTS idx_messages_session_created ON messages(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_messages_in_response_to ON messages(in_response_to);
            COMMIT;
            PRAGMA foreign_keys=ON;
            """
        )

    def _repair_messages_fk_if_needed(self, conn: sqlite3.Connection) -> None:
        cur = conn.cursor()
        try:
            rows = cur.execute("PRAGMA foreign_key_list(messages);").fetchall()
        except Exception:
            rows = []

        refs = []
        for r in rows or []:
            try:
                refs.append(str(r["table"]))
            except Exception:
                continue

        needs = any(t == "chat_sessions_old" for t in refs)

        if needs:
            self._rebuild_messages_table_with_fk(conn, fk_parent="chat_sessions")

        try:
            bad = cur.execute("PRAGMA foreign_key_check;").fetchall()
        except Exception:
            bad = []

        if bad:
            log.warning("foreign_key_check: после repair остались нарушения FK (%s строк)", len(bad))

    def _migrate_drop_unique_tg_chat_id(self, conn: sqlite3.Connection) -> None:
        """
        Исторически tg_chat_id был UNIQUE, что не позволяет иметь несколько сессий на один чат.
        Для кнопки "новая сессия" нужно разрешить несколько записей и выбирать активную (is_active=1).
        """
        cur = conn.cursor()
        try:
            idxs = cur.execute("PRAGMA index_list(chat_sessions);").fetchall()
        except Exception:
            return
        unique_indexes: list[str] = []
        for r in idxs or []:
            try:
                name = str(r["name"])
                is_unique = int(r["unique"])
            except Exception:
                continue
            if is_unique:
                unique_indexes.append(name)

        if not unique_indexes:
            return

        has_unique_tg_chat_id = False
        for idx_name in unique_indexes:
            try:
                cols = cur.execute(f"PRAGMA index_info({json.dumps(idx_name)});").fetchall()
            except Exception:
                continue
            for c in cols or []:
                try:
                    col_name = str(c["name"])
                except Exception:
                    continue
                if col_name == "tg_chat_id":
                    has_unique_tg_chat_id = True
                    break
            if has_unique_tg_chat_id:
                break

        if not has_unique_tg_chat_id:
            return

        log.info("Migrating chat_sessions: drop UNIQUE(tg_chat_id)")

        cur.executescript(
            """
            PRAGMA foreign_keys=OFF;
            BEGIN;
            ALTER TABLE chat_sessions RENAME TO chat_sessions_old;
            CREATE TABLE chat_sessions (
                id TEXT PRIMARY KEY,
                tg_chat_id INTEGER NOT NULL,
                title TEXT,
                context_window TEXT NOT NULL DEFAULT '{"window_size": 5, "messages": [], "summary": ""}',
                message_count INTEGER NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                CHECK (message_count >= 0)
            );
            INSERT INTO chat_sessions (id, tg_chat_id, title, context_window, message_count, is_active, created_at, updated_at)
            SELECT id, tg_chat_id, title, context_window, message_count, is_active, created_at, updated_at
            FROM chat_sessions_old;
            DROP TABLE chat_sessions_old;
            CREATE INDEX IF NOT EXISTS idx_sessions_tg_chat_id ON chat_sessions(tg_chat_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_active ON chat_sessions(is_active);
            CREATE INDEX IF NOT EXISTS idx_sessions_updated ON chat_sessions(updated_at);
            COMMIT;
            PRAGMA foreign_keys=ON;
            """
        )

    def _get_or_create_session(
        self,
        conn: sqlite3.Connection,
        tg_chat_id: int,
        title: Optional[str] = None,
    ) -> str:
        cur = conn.cursor()
        row = cur.execute(
            "SELECT id FROM chat_sessions WHERE tg_chat_id = ? AND is_active = 1",
            (tg_chat_id,),
        ).fetchone()
        if row:
            return str(row["id"])

        session_id = str(uuid.uuid4())
        now = _utc_now_iso()
        context_window = json.dumps({"window_size": 5, "messages": [], "summary": ""}, ensure_ascii=False)
        cur.execute(
            """
            INSERT INTO chat_sessions (id, tg_chat_id, title, context_window, message_count, is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, 0, 1, ?, ?)
            """,
            (session_id, tg_chat_id, title, context_window, now, now),
        )
        return session_id

    def _close_active_session(self, conn: sqlite3.Connection, tg_chat_id: int) -> None:
        now = _utc_now_iso()
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE chat_sessions
            SET is_active = 0, updated_at = ?
            WHERE tg_chat_id = ? AND is_active = 1
            """,
            (now, int(tg_chat_id)),
        )

    def get_or_create_session(self, tg_chat_id: int, title: Optional[str] = None) -> str:
        with self._connection() as conn:
            return self._get_or_create_session(conn, tg_chat_id, title=title)

    def close_active_session(self, tg_chat_id: int) -> None:
        with self._connection() as conn:
            self._close_active_session(conn, tg_chat_id)

    def start_new_session(self, tg_chat_id: int, title: Optional[str] = None) -> str:
        with self._connection() as conn:
            self._close_active_session(conn, int(tg_chat_id))
            return self._get_or_create_session(conn, int(tg_chat_id), title=title)

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        citations: Optional[dict | list] = None,
        metadata: Optional[dict] = None,
        token_count: Optional[int] = None,
        in_response_to: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> str:
        msg_id = str(uuid.uuid4())
        now = created_at or _utc_now_iso()
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO messages (id, session_id, role, content, citations, metadata, token_count, in_response_to, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    msg_id,
                    session_id,
                    role,
                    content,
                    json.dumps(citations or [], ensure_ascii=False),
                    json.dumps(metadata or {}, ensure_ascii=False),
                    token_count,
                    in_response_to,
                    now,
                ),
            )
            cur.execute(
                """
                UPDATE chat_sessions
                SET message_count = message_count + 1, updated_at = ?
                WHERE id = ?
                """,
                (now, session_id),
            )
        return msg_id

    def load_recent_messages(self, session_id: str, limit: int) -> list[StoredMessage]:
        with self._connection() as conn:
            cur = conn.cursor()
            rows = cur.execute(
                """
                SELECT role, content, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, int(limit)),
            ).fetchall()
        out = [StoredMessage(role=str(r["role"]), content=str(r["content"]), created_at=str(r["created_at"])) for r in rows]
        out.reverse()
        return out

    def prune_session_messages(self, session_id: str, keep_last_messages: int) -> None:
        if keep_last_messages <= 0:
            return
        with self._connection() as conn:
            cur = conn.cursor()
            n = cur.execute(
                "SELECT COUNT(*) AS c FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            total = int(n["c"]) if n else 0
            if total <= keep_last_messages:
                return

            rows = cur.execute(
                """
                SELECT id FROM messages
                WHERE session_id = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (session_id, int(keep_last_messages)),
            ).fetchall()
            keep_ids = [str(r["id"]) for r in rows]
            if not keep_ids:
                return

            placeholders = ",".join("?" * len(keep_ids))
            cur.execute(
                f"DELETE FROM messages WHERE session_id = ? AND id NOT IN ({placeholders})",
                [session_id, *keep_ids],
            )
            new_count = cur.execute(
                "SELECT COUNT(*) AS c FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            cnt = int(new_count["c"]) if new_count else 0
            now = _utc_now_iso()
            cur.execute(
                """
                UPDATE chat_sessions
                SET message_count = ?, updated_at = ?
                WHERE id = ?
                """,
                (cnt, now, session_id),
            )

    def clear_session_history(self, session_id: str) -> None:
        now = _utc_now_iso()
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cur.execute(
                """
                UPDATE chat_sessions
                SET message_count = 0, updated_at = ?
                WHERE id = ?
                """,
                (now, session_id),
            )

    def set_session_title(self, session_id: str, title: Optional[str]) -> None:
        now = _utc_now_iso()
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, session_id),
            )

    def healthcheck(self) -> dict:
        t0 = time.perf_counter()
        with self._connection() as conn:
            conn.execute("SELECT 1;").fetchone()
        return {"ok": True, "db_path": self._db_path, "ms": round((time.perf_counter() - t0) * 1000, 2)}
