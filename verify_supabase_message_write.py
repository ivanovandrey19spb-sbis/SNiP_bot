"""
Smoke-test: запись сообщения пользователя в Supabase (chat_sessions/messages) и проверка чтением.

Этапность:
  - write: вставляет chat_session + message, печатает данные и завершает работу
  - python verify_supabase_message_write.py write
  - check: ждёт 3 секунды, затем читает message обратно и проверяет наличие токена
  - python verify_supabase_message_write.py check --session-id <uuid> --token <NNNNNNAAA>
  - cleanup: удаляет chat_session (messages удалятся каскадно)
  - python verify_supabase_message_write.py cleanup --session-id <uuid>

Требования к env:
  - SUPABASE_URL + (SUPABASE_EMAIL/SUPABASE_PASSWORD + SUPABASE_PUBLIC_KEY_LONG) ИЛИ service role ключи
  - SUPABASE_SMOKE_TEST_USER_ID: UUID строки public.users.id (должна существовать)
  - SUPABASE_SMOKE_CLEANUP: 1/true/yes (опционально, для удаления в check)
  - SUPABASE_SMOKE_TEST_PROJECT_ID или SUPABASE_DEFAULT_PROJECT_ID (опционально; должен существовать в projects)
"""

from __future__ import annotations

import argparse
import os
import random
import string
import sys
import time
from typing import Any, Optional

from dotenv import load_dotenv

from supabase_helper import SupabaseHelper


def _is_truthy(v: Optional[str]) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "y", "on")


def _require_env(name: str) -> str:
    v = (os.getenv(name) or "").strip()
    if not v:
        raise RuntimeError(f"Не задано {name} в переменных окружения.")
    return v


def _gen_token() -> str:
    digits = "".join(random.choice(string.digits) for _ in range(6))
    letters = "".join(random.choice(string.ascii_uppercase) for _ in range(3))
    return digits + letters


def _get_project_id() -> Optional[str]:
    pid = (os.getenv("SUPABASE_SMOKE_TEST_PROJECT_ID") or "").strip()
    if pid:
        return pid
    pid = (os.getenv("SUPABASE_DEFAULT_PROJECT_ID") or "").strip()
    return pid or None


def _create_client() -> Any:
    # .env already loaded by caller
    return SupabaseHelper(load_env=False).get_bot_supabase_client()


def cmd_write(_args: argparse.Namespace) -> int:
    user_id = _require_env("SUPABASE_SMOKE_TEST_USER_ID")
    project_id = _get_project_id()
    token = _gen_token()
    ts = int(time.time())
    content = f"smoke_write token={token} ts={ts}"

    client = _create_client()

    session_payload: dict[str, Any] = {
        "user_id": user_id,
        "title": f"smoke_session_{ts}",
        "is_active": True,
    }
    if project_id:
        session_payload["project_id"] = project_id

    # postgrest-py: после insert() нельзя chain-ить .select(); при default returning=representation
    # тело ответа уже содержит вставленную строку (в т.ч. id).
    ins_sess = client.table("chat_sessions").insert(session_payload).execute()
    sdata = getattr(ins_sess, "data", None) or []
    session_id = str(sdata[0]["id"]) if sdata and sdata[0].get("id") else ""
    if not session_id:
        q = (
            client.table("chat_sessions")
            .select("id")
            .eq("user_id", user_id)
            .eq("title", session_payload["title"])
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        qrows = getattr(q, "data", None) or []
        if qrows and qrows[0].get("id"):
            session_id = str(qrows[0]["id"])
    if not session_id:
        raise RuntimeError("INSERT chat_sessions: не удалось получить id (ответ пустой и fallback select не нашёл строку).")

    msg_payload: dict[str, Any] = {
        "session_id": session_id,
        "role": "user",
        "content": content,
        "citations": [],
        "metadata": {"source": "verify_supabase_message_write", "token": token},
    }
    ins_msg = client.table("messages").insert(msg_payload).execute()
    mdata = getattr(ins_msg, "data", None) or []
    message_id = str(mdata[0]["id"]) if mdata and mdata[0].get("id") else ""
    if not message_id:
        qm = (
            client.table("messages")
            .select("id")
            .eq("session_id", session_id)
            .eq("content", content)
            .limit(1)
            .execute()
        )
        mrows = getattr(qm, "data", None) or []
        if mrows and mrows[0].get("id"):
            message_id = str(mrows[0]["id"])
    if not message_id:
        raise RuntimeError("INSERT messages: не удалось получить id (ответ пустой и fallback select не нашёл строку).")

    print("OK write")
    print("session_id:", session_id)
    print("message_id:", message_id)
    print("token:", token)
    print("content:", content)
    print()
    print("Дальше проверь в Supabase и запусти:")
    print(f"  python verify_supabase_message_write.py check --session-id {session_id} --token {token}")

    return 0


def _select_messages_for_session(client: Any, session_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
    res = (
        client.table("messages")
        .select("id, session_id, role, content, created_at, metadata")
        .eq("session_id", session_id)
        .order("created_at", desc=True)
        .limit(int(limit))
        .execute()
    )
    rows = getattr(res, "data", None) or []
    return [r for r in rows if isinstance(r, dict)]


def _cleanup_session(client: Any, session_id: str) -> None:
    # messages удалятся каскадно (FK ON DELETE CASCADE).
    client.table("chat_sessions").delete().eq("id", session_id).execute()


def cmd_check(args: argparse.Namespace) -> int:
    session_id = (args.session_id or "").strip()
    token = (args.token or "").strip()
    if not session_id:
        raise RuntimeError("Нужен --session-id")
    if not token:
        raise RuntimeError("Нужен --token")

    client = _create_client()

    print("check: session_id=", session_id)
    print("check: token=", token)
    print("sleep 3s...")
    time.sleep(3)

    rows = _select_messages_for_session(client, session_id)
    found = None
    for r in rows:
        c = str(r.get("content") or "")
        if token in c:
            found = r
            break

    if not found:
        print("NOT FOUND: message with token in content")
        print("last_messages_count:", len(rows))
        if rows:
            print("last_message_preview:", str(rows[0].get("content") or "")[:240])
        return 1

    print("OK found")
    print("message_id:", found.get("id"))
    print("role:", found.get("role"))
    print("content:", found.get("content"))
    print("created_at:", found.get("created_at"))

    if _is_truthy(os.getenv("SUPABASE_SMOKE_CLEANUP")):
        try:
            _cleanup_session(client, session_id)
            print("cleanup: deleted chat_sessions id=", session_id)
        except Exception as e:
            print("cleanup: FAILED:", e, file=sys.stderr)

    return 0


def cmd_cleanup(args: argparse.Namespace) -> int:
    session_id = (args.session_id or "").strip()
    if not session_id:
        raise RuntimeError("Нужен --session-id")
    client = _create_client()
    _cleanup_session(client, session_id)
    print("cleanup: deleted chat_sessions id=", session_id)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="verify_supabase_message_write.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("write", help="insert chat_session+message and print data")
    w.set_defaults(func=cmd_write)

    c = sub.add_parser("check", help="sleep 3s and select message back")
    c.add_argument("--session-id", required=True)
    c.add_argument("--token", required=True)
    c.set_defaults(func=cmd_check)

    cl = sub.add_parser("cleanup", help="delete chat_session (messages cascade)")
    cl.add_argument("--session-id", required=True)
    cl.set_defaults(func=cmd_cleanup)

    return p


def main(argv: list[str]) -> int:
    load_dotenv(override=True)
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

