"""Локальный отладочный журнал retrieval/LLM в SQLite.

Назначение: дать возможность открыть SQLite (DBeaver / sqlite3) и SELECT-ом
разобрать каждый запрос пользователя — что искалось, какие токены ушли в
OR ilike, какие чанки нашлись из обоих путей с их сырыми скорами, как RRF их
перетасовал, в каком порядке топ-чанки ушли в LLM, что ответил LLM, сколько
работало всё/только LLM.

Контракт:
- is_enabled() — считывает INTERACTION_DB_ENABLED (default "0"; включать только для локальной отладки).
- db_path() — считывает INTERACTION_DB_PATH (default data/interactions.sqlite3).
- init_db(path=None) — создаёт каталог и применяет DDL из
  data/interactions_schema.sql (idempotent CREATE TABLE IF NOT EXISTS).
- log(...) — пишет одну строку в interactions и N строк в interaction_chunks.

Любая ошибка записи логируется warning'ом и не роняет бота.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger("interaction_db")

DEFAULT_DB_PATH = os.path.join("data", "interactions.sqlite3")
SCHEMA_FILE = os.path.join("data", "interactions_schema.sql")

_INIT_DONE: Dict[str, bool] = {}


def _env_truthy(name: str, default: str = "1") -> bool:
    v = (os.getenv(name) or default).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def is_enabled() -> bool:
    return _env_truthy("INTERACTION_DB_ENABLED", default="0")


def db_path() -> str:
    return (os.getenv("INTERACTION_DB_PATH") or DEFAULT_DB_PATH).strip() or DEFAULT_DB_PATH


def _read_schema() -> str:
    try:
        with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        log.warning("Schema-файл не найден: %s — interaction_db.init_db пропустит DDL", SCHEMA_FILE)
        return ""


def init_db(path: Optional[str] = None) -> None:
    """Создаёт файл/каталог и применяет DDL из schema-файла.

    Идемпотентно. Запоминает уже инициализированные пути в _INIT_DONE,
    чтобы не делать executescript на каждый вызов log().
    """
    target = (path or db_path()).strip() or DEFAULT_DB_PATH
    if _INIT_DONE.get(target):
        return

    try:
        d = os.path.dirname(target)
        if d:
            os.makedirs(d, exist_ok=True)
    except Exception as e:
        log.warning("interaction_db: не удалось создать каталог для %s: %s", target, e)

    schema_sql = _read_schema()
    try:
        with sqlite3.connect(target) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            if schema_sql:
                conn.executescript(schema_sql)
            conn.commit()
        _INIT_DONE[target] = True
    except Exception as e:
        log.warning("interaction_db: init_db не удался для %s: %s", target, e)


def _coerce_designation(ch: Dict[str, Any]) -> Optional[str]:
    if ch.get("designation"):
        return str(ch.get("designation"))
    dm = ch.get("document_metadata") or {}
    if isinstance(dm, dict) and dm.get("designation"):
        return str(dm.get("designation"))
    return None


def log(
    *,
    chat_id: int,
    user_query: str,
    rag_kind: str,
    lexical_tokens: List[str],
    chunks: List[Dict[str, Any]],
    vector_raw_by_id: Dict[str, float],
    lexical_raw_by_id: Dict[str, float],
    answer: str,
    model: str,
    latency_total_ms: float,
    latency_llm_ms: float,
    path: Optional[str] = None,
) -> None:
    """Записать одну строку в interactions + по строке на чанк.

    Любая ошибка — log.warning, не роняем вызывающий код.
    """
    if not is_enabled():
        return

    target = (path or db_path()).strip() or DEFAULT_DB_PATH
    if not _INIT_DONE.get(target):
        init_db(target)

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    try:
        tokens_json = json.dumps(list(lexical_tokens or []), ensure_ascii=False)
    except Exception:
        tokens_json = "[]"

    try:
        with sqlite3.connect(target) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO interactions (
                    ts_utc, chat_id, user_query, rag_kind, lexical_tokens_json,
                    answer, model, latency_total_ms, latency_llm_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    int(chat_id) if chat_id is not None else None,
                    str(user_query or ""),
                    str(rag_kind or ""),
                    tokens_json,
                    str(answer or ""),
                    str(model or ""),
                    float(latency_total_ms) if latency_total_ms is not None else None,
                    float(latency_llm_ms) if latency_llm_ms is not None else None,
                ),
            )
            interaction_id = cur.lastrowid

            rows = []
            for i, ch in enumerate(chunks or []):
                if not isinstance(ch, dict):
                    continue
                rid = str(ch.get("id") or "")
                if not rid:
                    continue
                br = ch.get("rrf_breakdown") or {}
                rank_vec = br.get("rank_vec")
                rank_lex = br.get("rank_lex")
                source_set = br.get("source_set")
                rrf_score = ch.get("score")
                rows.append(
                    (
                        interaction_id,
                        i + 1,
                        rid,
                        ch.get("clause_display"),
                        _coerce_designation(ch),
                        ch.get("section_path"),
                        source_set,
                        int(rank_vec) if isinstance(rank_vec, (int, float)) else None,
                        int(rank_lex) if isinstance(rank_lex, (int, float)) else None,
                        vector_raw_by_id.get(rid),
                        lexical_raw_by_id.get(rid),
                        float(rrf_score) if rrf_score is not None else None,
                    )
                )
            if rows:
                cur.executemany(
                    """
                    INSERT INTO interaction_chunks (
                        interaction_id, seq, chunk_id, clause_display, designation,
                        section_path, source_set, rank_vec, rank_lex,
                        vector_score_raw, lexical_score_raw, rrf_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            conn.commit()
    except Exception as e:
        log.warning("interaction_db.log пропущен: %s", e)


__all__ = ["is_enabled", "db_path", "init_db", "log"]
