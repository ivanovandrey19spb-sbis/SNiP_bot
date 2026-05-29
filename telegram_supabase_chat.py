"""
Опциональное дублирование переписки из SQLite (ChatStorage) в Supabase chat_sessions / messages.

Режим SUPABASE_CHAT_PRIMARY=1: источник истории для LLM — Supabase; SQLite — узкий резерв (см. bot.py).

В схеме data1/ сообщения привязаны к users.id (UUID). В Telegram у нас только tg user id —
без строки в public.users запись в Supabase невозможна или нарушит FK.

По умолчанию (SUPABASE_USER_RESOLVE_MODE=db) бот пытается найти/создать пользователя в public.users
по полю users.id_telegram (BIGINT) и использовать users.id (UUID) в chat_sessions/messages.

Альтернатива (legacy):
  - SUPABASE_USER_RESOLVE_MODE=json
  - TELEGRAM_USER_MAP_PATH=./data/telegram_user_map.json
  - Формат: {"123456789": "uuid-пользователя-в-supabase"}
  - SUPABASE_CHAT_MIRROR=1 (если не используется SUPABASE_CHAT_PRIMARY)
  - Карта сессий пишется в SUPABASE_SESSION_MAP_PATH (по умолчанию ./data/supabase_session_map.json):
    локальный session_id (SQLite) → uuid chat_sessions в Supabase.

Если резолвинг пользователя не удался — вызов безопасно no-op.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

Resolver = Callable[[int], Optional[str]]


def _load_json_map(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        return {str(k): str(v) for k, v in raw.items()}
    except Exception as e:
        log.warning("json map: не прочитан %s: %s", path, e)
        return {}


def _save_json_map(path: Path, data: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_telegram_user_map(path: str) -> dict[str, str]:
    return _load_json_map(Path(path))


def default_telegram_user_resolver() -> Resolver:
    path = (os.getenv("TELEGRAM_USER_MAP_PATH") or "").strip()
    if not path:
        return lambda _tg_id: None
    mapping = _load_telegram_user_map(path)

    def _resolve(tg_user_id: int) -> Optional[str]:
        return mapping.get(str(tg_user_id))

    return _resolve


def _resolve_or_create_supabase_user_id_by_telegram(supabase_client: Any, tg_user_id: int) -> Optional[str]:
    """
    Возвращает users.id (UUID) для Telegram-пользователя.

    Алгоритм:
    1) SELECT users.id WHERE id_telegram = tg_user_id
    2) Если не найдено — INSERT users(id_telegram=...) и забрать returning id
    3) Если returning пустой или insert упал из-за гонки — повторить SELECT
    """
    try:
        q = (
            supabase_client.table("users")
            .select("id")
            .eq("id_telegram", int(tg_user_id))
            .limit(1)
            .execute()
        )
        rows = getattr(q, "data", None) or []
        if rows and isinstance(rows[0], dict) and rows[0].get("id"):
            return str(rows[0]["id"])
    except Exception as e:
        log.warning("user resolve(select) failed tg_user_id=%s: %s", tg_user_id, e)

    try:
        ins = supabase_client.table("users").insert({"id_telegram": int(tg_user_id)}).execute()
        data = getattr(ins, "data", None) or []
        if data and isinstance(data[0], dict) and data[0].get("id"):
            return str(data[0]["id"])
    except Exception as e:
        # Может быть гонка/unique violation — попробуем дочитать select-ом.
        log.warning("user resolve(insert) failed tg_user_id=%s: %s", tg_user_id, e)

    try:
        q2 = (
            supabase_client.table("users")
            .select("id")
            .eq("id_telegram", int(tg_user_id))
            .limit(1)
            .execute()
        )
        rows2 = getattr(q2, "data", None) or []
        if rows2 and isinstance(rows2[0], dict) and rows2[0].get("id"):
            return str(rows2[0]["id"])
    except Exception as e:
        log.warning("user resolve(select2) failed tg_user_id=%s: %s", tg_user_id, e)

    return None


def _default_user_id_resolver(supabase_client: Any) -> Resolver:
    """
    Возвращает resolver tg_user_id -> users.id (UUID).

    SUPABASE_USER_RESOLVE_MODE:
      - db (default): users.id_telegram -> users.id (find-or-create)
      - json: использовать TELEGRAM_USER_MAP_PATH (legacy)
      - db_then_json: сначала db, затем json
    """
    mode = (os.getenv("SUPABASE_USER_RESOLVE_MODE") or "db").strip().lower()
    json_resolver = default_telegram_user_resolver()

    if mode == "json":
        return json_resolver

    if mode == "db_then_json":
        def _resolve(tg_id: int) -> Optional[str]:
            return _resolve_or_create_supabase_user_id_by_telegram(supabase_client, tg_id) or json_resolver(tg_id)

        return _resolve

    # default: db
    def _resolve(tg_id: int) -> Optional[str]:
        return _resolve_or_create_supabase_user_id_by_telegram(supabase_client, tg_id)

    return _resolve


def get_user_id_resolver(supabase_client: Any) -> Resolver:
    """tg_user_id -> users.id (UUID); для bot.py и скриптов."""
    return _default_user_id_resolver(supabase_client)


def _session_map_path() -> Path:
    p = (os.getenv("SUPABASE_SESSION_MAP_PATH") or "").strip()
    return Path(p) if p else Path(os.getcwd()) / "data" / "supabase_session_map.json"


def get_or_create_remote_session_id(
    supabase_client: Any,
    *,
    supabase_user_id: str,
    local_session_id: str,
    session_title: Optional[str],
    project_id: Optional[str],
) -> Optional[str]:
    """Локальный session_id (SQLite UUID) → uuid chat_sessions в Supabase; при отсутствии — insert и запись в карту."""
    path = _session_map_path()
    m = _load_json_map(path)
    existing = m.get(local_session_id)
    if existing:
        return existing

    try:
        ins = supabase_client.table("chat_sessions").insert(
            {
                "user_id": supabase_user_id,
                "project_id": project_id,
                "title": session_title,
                "is_active": True,
            }
        ).execute()
        data = getattr(ins, "data", None) or []
        sid = None
        if data and isinstance(data[0], dict) and data[0].get("id"):
            sid = str(data[0]["id"])
        if not sid:
            # fallback: попробуем найти только что созданную запись
            q = (
                supabase_client.table("chat_sessions")
                .select("id")
                .eq("user_id", supabase_user_id)
                .eq("title", session_title)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            qrows = getattr(q, "data", None) or []
            if qrows and isinstance(qrows[0], dict) and qrows[0].get("id"):
                sid = str(qrows[0]["id"])
        if not sid:
            log.warning(
                "get_or_create_remote_session_id: chat_sessions insert не вернул id "
                "(returning пустой и fallback select не нашёл строку)"
            )
            return None
        m[local_session_id] = sid
        _save_json_map(path, m)
        return sid
    except Exception as e:
        log.warning("get_or_create_remote_session_id: %s", e)
        return None


def fetch_recent_messages(
    supabase_client: Any,
    remote_session_id: str,
    limit: int,
) -> list[dict[str, str]]:
    """
    Последние `limit` сообщений сессии в хронологическом порядке (старые → новые).
    Ключи: role, content, created_at (ISO или пустая строка).
    """
    if limit <= 0:
        return []
    try:
        q = (
            supabase_client.table("messages")
            .select("role,content,created_at")
            .eq("session_id", remote_session_id)
            .order("created_at", desc=True)
            .limit(int(limit))
            .execute()
        )
        rows = getattr(q, "data", None) or []
    except Exception as e:
        log.warning("fetch_recent_messages: %s", e)
        raise

    out: list[dict[str, str]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        role = r.get("role")
        content = r.get("content")
        if role is None or content is None:
            continue
        ca = r.get("created_at")
        out.append(
            {
                "role": str(role),
                "content": str(content),
                "created_at": str(ca) if ca is not None else "",
            }
        )
    out.reverse()
    return out


def append_chat_message(
    supabase_client: Any,
    *,
    remote_session_id: str,
    role: str,
    content: str,
    metadata: Optional[dict] = None,
    citations: Optional[list] = None,
) -> None:
    """Одна строка в public.messages (контракт как в verify_supabase_message_write)."""
    payload: dict[str, Any] = {
        "session_id": remote_session_id,
        "role": role,
        "content": content or "",
        "metadata": metadata or {},
        "citations": citations if citations is not None else [],
    }
    try:
        supabase_client.table("messages").insert(payload).execute()
    except Exception as e:
        log.warning("append_chat_message: %s", e)
        raise


def mirror_append_last_turn(
    supabase_client: Any,
    *,
    tg_user_id: int,
    local_session_id: str,
    user_text: str,
    assistant_text: str,
    user_metadata: Optional[dict] = None,
    assistant_metadata: Optional[dict] = None,
    resolver: Optional[Resolver] = None,
) -> None:
    """Legacy: дозаписывает пару user/assistant при SUPABASE_CHAT_MIRROR=1 (без SUPABASE_CHAT_PRIMARY)."""
    if (os.getenv("SUPABASE_CHAT_MIRROR") or "").strip().lower() not in ("1", "true", "yes"):
        return
    res = resolver or _default_user_id_resolver(supabase_client)
    uid = res(int(tg_user_id))
    if not uid:
        return

    project_id = (os.getenv("SUPABASE_DEFAULT_PROJECT_ID") or "").strip() or None
    remote_sid = get_or_create_remote_session_id(
        supabase_client,
        supabase_user_id=uid,
        local_session_id=local_session_id,
        session_title=(user_text[:120] if user_text else None),
        project_id=project_id,
    )
    if not remote_sid:
        return

    try:
        append_chat_message(
            supabase_client,
            remote_session_id=remote_sid,
            role="user",
            content=user_text or "",
            metadata=user_metadata or {},
        )
        append_chat_message(
            supabase_client,
            remote_session_id=remote_sid,
            role="assistant",
            content=assistant_text or "",
            metadata=assistant_metadata or {},
        )
    except Exception:
        log.warning("mirror_append_last_turn: вставка сообщений не удалась", exc_info=True)
