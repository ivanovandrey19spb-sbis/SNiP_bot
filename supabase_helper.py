"""Клиент Supabase: URL + ключ из окружения, healthcheck, create_client."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from dotenv import load_dotenv

log = logging.getLogger(__name__)

try:
    from supabase import Client, create_client
except ImportError:  # pragma: no cover
    Client = Any  # type: ignore[misc, assignment]
    create_client = None  # type: ignore[misc, assignment]


def _resolve_supabase_key() -> str:
    """Серверный бот: предпочитаем service_role; иначе anon / публичный ключ из env.example."""
    return (
        (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
        or (os.getenv("SUPABASE_PUBLIC_KEY_LONG") or "").strip()
    )


def _supabase_auth_email_password_configured() -> bool:
    e = (os.getenv("SUPABASE_EMAIL") or "").strip()
    p = (os.getenv("SUPABASE_PASSWORD") or "").strip()
    return bool(e and p)


class SupabaseHelper:
    """Обёртка над supabase-py: загрузка .env и фабрика клиента."""

    def __init__(
        self,
        *,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        load_env: bool = True,
    ):
        if load_env:
            load_dotenv(override=True)
        self.supabase_url = (supabase_url or os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
        self.supabase_key = (supabase_key or _resolve_supabase_key()).strip()

    def get_supabase_client(self) -> "Client":
        if create_client is None:
            raise RuntimeError("Пакет supabase не установлен. Установите: pip install supabase")
        if not self.supabase_url:
            raise RuntimeError("Не задан SUPABASE_URL")
        if not self.supabase_key:
            raise RuntimeError(
                "Не задан ключ Supabase: SUPABASE_SERVICE_ROLE_KEY или SUPABASE_PUBLIC_KEY_LONG"
            )
        return create_client(self.supabase_url, self.supabase_key)

    def get_bot_supabase_client(self) -> "Client":
        """
        Клиент для bot.py (RAG и зеркало чата).

        Если заданы SUPABASE_EMAIL и SUPABASE_PASSWORD — как в supabase_smoketest: create_client
        только с SUPABASE_PUBLIC_KEY_LONG (anon), затем sign_in_with_password; PostgREST видит JWT
        пользователя (RLS для authenticated).

        Иначе — create_client с SUPABASE_SERVICE_ROLE_KEY или публичным ключом (без Auth).
        """
        if create_client is None:
            raise RuntimeError("Пакет supabase не установлен. Установите: pip install supabase")
        if not self.supabase_url:
            raise RuntimeError("Не задан SUPABASE_URL")

        if _supabase_auth_email_password_configured():
            anon = (os.getenv("SUPABASE_PUBLIC_KEY_LONG") or "").strip()
            if not anon:
                raise RuntimeError(
                    "При использовании SUPABASE_EMAIL/SUPABASE_PASSWORD задайте SUPABASE_PUBLIC_KEY_LONG "
                    "(anon key). Service role в create_client перед sign_in не используйте."
                )
            email = (os.getenv("SUPABASE_EMAIL") or "").strip()
            password = (os.getenv("SUPABASE_PASSWORD") or "").strip()
            client = create_client(self.supabase_url, anon)
            client.auth.sign_in_with_password({"email": email, "password": password})
            log.info("Supabase Auth: сессия пользователя установлена (email=%s)", email)
            return client

        if not self.supabase_key:
            raise RuntimeError(
                "Не задан ключ Supabase: SUPABASE_SERVICE_ROLE_KEY или SUPABASE_PUBLIC_KEY_LONG"
            )
        return create_client(self.supabase_url, self.supabase_key)

    def healthcheck(
        self,
        *,
        client: Optional[Any] = None,
        probe_tables: Optional[tuple[str, ...]] = None,
    ) -> dict[str, Any]:
        """
        Проверка доступа к PostgREST: HEAD-запрос с count по первой доступной таблице.
        Порядок таблиц можно переопределить через SUPABASE_HEALTHCHECK_TABLE (одно имя).
        """
        custom = (os.getenv("SUPABASE_HEALTHCHECK_TABLE") or "").strip()
        tables = (custom,) if custom else (probe_tables or DEFAULT_HEALTHCHECK_TABLES)

        client = client or self.get_supabase_client()
        last_err: Optional[str] = None
        for table in tables:
            try:
                res = client.table(table).select("*", count="exact", head=True).execute()  # type: ignore[union-attr]
                cnt = getattr(res, "count", None)
                return {
                    "ok": True,
                    "table": table,
                    "count": cnt,
                    "error": None,
                }
            except Exception as e:  # pragma: no cover
                last_err = str(e)
                log.debug("healthcheck probe %s failed: %s", table, e)
                continue

        return {
            "ok": False,
            "table": None,
            "count": None,
            "error": last_err or "no table responded",
        }


# Таблицы по умолчанию из схемы data1/ (сначала маленькие/справочники).
DEFAULT_HEALTHCHECK_TABLES = (
    "subscription_plans",
    "query_cache",
    "users",
    "workspaces",
    "documents",
    "chunks",
)
