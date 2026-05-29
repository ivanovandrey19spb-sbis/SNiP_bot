"""Минимальный MVP: один обработчик текста, LLM + опционально RAG (Supabase chunks). Точка входа: python bot.py"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Optional

import telebot
from dotenv import load_dotenv
from llm_compat import AIMessage, ChatOpenAI, HumanMessage, SystemMessage
from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup

from chat_storage import ChatStorage, StoredMessage
from rag import SupabaseChunksRagStore
import interaction_db

load_dotenv(override=True)


class _PassthroughRagStore:
    """Заглушка, если Supabase недоступен или не настроен."""

    rag_kind = "passthrough"

    def enrich_user_prompt(self, user_text: str, k: int = 4) -> str:
        return user_text


# -----------------------------
# Настройки
# -----------------------------
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "10"))
MAX_USER_MESSAGE_CHARS = int(os.getenv("MAX_USER_MESSAGE_CHARS", "4000"))
MAX_LLM_USER_BLOB_CHARS = int(os.getenv("MAX_LLM_USER_BLOB_CHARS", "12000"))

# OpenRouter (OpenAI-compatible)
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "z-ai/glm-4.5-air:free")

# Telegram
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Хранилище переписки (SQLite-файл)
CHAT_DB_PATH = os.getenv("CHAT_DB_PATH", os.path.join(os.getcwd(), "data", "chat.sqlite3"))

# История в Supabase (primary) + локальный резерв N пар Q&A (см. LOCAL_CHAT_BACKUP_QA_PAIRS)
LOCAL_CHAT_BACKUP_QA_PAIRS = int(os.getenv("LOCAL_CHAT_BACKUP_QA_PAIRS", "5"))


def _env_truthy(name: str, default: str = "0") -> bool:
    return (os.getenv(name, default) or "").strip().lower() in ("1", "true", "yes", "y", "on")

# Системный промпт
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Ты дружелюбный и вежливый помощник в Telegram, но не фамильярный. Ниже к сообщению пользователя может быть добавлен релевантный контекст из базы знаний — опирайся на него, если он относится к вопросу.",)

USER_FACING_ERROR_MESSAGE = (
    "Произошла ошибка при обработке вашего запроса. Попробуйте позже."
)
USER_FACING_CALLBACK_ERROR = "❌ Ошибка при обработке запроса"

# -----------------------------
# Логирование (минимально достаточное)
# -----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    encoding="utf-8",
)


class RedactingFormatter(logging.Formatter):
    def __init__(self, secrets: list[str], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._secrets = [s for s in secrets if s]
        self._tg_token_in_url_re = re.compile(r"(api\.telegram\.org/bot)([^/]+)(/)", re.IGNORECASE)

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)

        # Always redact Telegram token in API URLs, even if token isn't in secrets list.
        msg = self._tg_token_in_url_re.sub(r"\1<REDACTED>\3", msg)

        for secret in self._secrets:
            msg = msg.replace(secret, "<REDACTED>")

        return msg


def install_log_redaction() -> None:
    secret_env_keys = (
        "BOT_TOKEN",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_PUBLIC_KEY_LONG",
    )
    secrets = [os.getenv(k) for k in secret_env_keys]

    root = logging.getLogger()
    for handler in root.handlers:
        old = handler.formatter
        fmt = getattr(getattr(old, "_style", None), "_fmt", None) if old else None
        datefmt = getattr(old, "datefmt", None) if old else None
        handler.setFormatter(RedactingFormatter(secrets, fmt=fmt, datefmt=datefmt))


install_log_redaction()
log = logging.getLogger("bot")

# -----------------------------
# RAG / LLM логирование (JSONL)
# -----------------------------

def _interaction_env_truthy(name: str, default: str = "0") -> bool:
    v = (os.getenv(name) or default).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


LOG_FILE = os.getenv("INTERACTION_LOG_PATH", "rag_logs.jsonl")
INTERACTION_JSONL_ENABLED = _interaction_env_truthy("INTERACTION_JSONL_ENABLED", "0")
INTERACTION_JSONL_VERBOSE = _interaction_env_truthy("INTERACTION_JSONL_VERBOSE", "0")


def _build_verbose_interaction_chunks(chunks: list, max_chunk_chars: int) -> list:
    safe_chunks = []
    for ch in chunks:
        if isinstance(ch, dict):
            sc = ch.get("score")
            item = {
                "chunk_id": ch.get("id"),
                "text": (ch.get("text") or "")[:max_chunk_chars],
                "score": sc,
                "retrieval_score": sc,
            }
            for opt_key in (
                "doc_id",
                "designation",
                "official_title",
                "section_id",
                "section_path",
                "clause_display",
            ):
                if ch.get(opt_key) is not None:
                    item[opt_key] = ch.get(opt_key)
            safe_chunks.append(item)
        else:
            safe_chunks.append(
                {
                    "chunk_id": None,
                    "text": str(ch)[:max_chunk_chars],
                    "score": None,
                    "retrieval_score": None,
                }
            )
    return safe_chunks


def log_interaction(
    chat_id: int,
    user_query: str,
    chunks: list,
    answer: str,
    model: str,
    latency_ms: float,
    *,
    rag_kind: str,
):
    if not INTERACTION_JSONL_ENABLED:
        return

    answer_text = answer or ""
    query_text = user_query or ""

    if INTERACTION_JSONL_VERBOSE:
        max_chunk_chars = int(os.getenv("INTERACTION_LOG_CHUNK_MAX_CHARS", "8000"))
        max_answer_chars = int(os.getenv("INTERACTION_LOG_ANSWER_MAX_CHARS", "50000"))
        record = {
            "log_mode": "verbose",
            "timestamp": time.time(),
            "chat_id": chat_id,
            "query": query_text,
            "rag_kind": rag_kind,
            "chunks": _build_verbose_interaction_chunks(chunks, max_chunk_chars),
            "answer": answer_text[:max_answer_chars],
            "model": model,
            "latency_ms": latency_ms,
            "feedback": None,
        }
    else:
        record = {
            "log_mode": "production",
            "timestamp": time.time(),
            "rag_kind": rag_kind,
            "chunk_count": len(chunks or []),
            "query_len": len(query_text),
            "answer_len": len(answer_text),
            "model": model,
            "latency_ms": latency_ms,
            "feedback": None,
        }

    try:
        dirpath = os.path.dirname(LOG_FILE)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    except Exception as e:
        log.warning("Ошибка записи interaction-лога: %s", e)


# -----------------------------
# Инициализация компонентов
# -----------------------------
_rag_debug = os.getenv("RAG_DEBUG", "").lower() in ("1", "true", "yes")

_supabase_client = None
rag_store: SupabaseChunksRagStore | _PassthroughRagStore = _PassthroughRagStore()

if (os.getenv("SUPABASE_URL") or "").strip() and (
    (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    or (os.getenv("SUPABASE_PUBLIC_KEY_LONG") or "").strip()
):
    try:
        from supabase_helper import SupabaseHelper

        _h = SupabaseHelper(load_env=False)
        _supabase_client = _h.get_bot_supabase_client()
        rag_store = SupabaseChunksRagStore(_supabase_client, debug=_rag_debug)
        log.info("Supabase healthcheck: %s", _h.healthcheck(client=_supabase_client))
    except Exception:
        log.exception("Supabase недоступен (проверьте ключи и URL) — RAG отключён")
        _supabase_client = None
        rag_store = _PassthroughRagStore()
else:
    log.warning("Supabase не настроен — RAG отключён (нужны SUPABASE_URL и ключ)")

os.makedirs(os.path.dirname(CHAT_DB_PATH), exist_ok=True)
storage = ChatStorage(CHAT_DB_PATH)

if interaction_db.is_enabled():
    try:
        interaction_db.init_db()
    except Exception as _e:
        log.warning("interaction_db.init_db пропущен: %s", _e)

if not BOT_TOKEN:
    raise RuntimeError("Не задан BOT_TOKEN в переменных окружения.")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Не задан OPENROUTER_API_KEY (или OPENAI_API_KEY) в переменных окружения.")

llm = ChatOpenAI(
    openai_api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    model=MODEL_NAME,
    default_headers={
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", ""),
        "X-Title": os.getenv("OPENROUTER_APP_TITLE", "w_tg_b"),
    },
)

bot = telebot.TeleBot(BOT_TOKEN)

log.info(
    "Старт бота. model=%s base_url=%s db=%s",
    MODEL_NAME,
    OPENROUTER_BASE_URL,
    CHAT_DB_PATH,
)
if isinstance(rag_store, _PassthroughRagStore):
    log.warning("RAG не активен (passthrough); при настройке Supabase включится поиск по chunks.")
else:
    log.info(
        "RAG подключён (rag_kind=%s); см. rag.py и переменные RAG_SUPABASE_*.",
        getattr(rag_store, "rag_kind", os.getenv("RAG_MODE", "unknown")),
    )

try:
    log.info("DB healthcheck: %s", storage.healthcheck())
except Exception:
    log.exception("DB healthcheck не прошёл")


def get_session_keyboard() -> InlineKeyboardMarkup:
    keyboard = InlineKeyboardMarkup(row_width=2)
    keyboard.add(
        InlineKeyboardButton(text="🔄 Начать новую сессию", callback_data="new_session"),
        InlineKeyboardButton(text="🗑️ Очистить историю", callback_data="clear_session"),
    )
    return keyboard


def _stored_to_langchain_messages(stored):
    out = []
    for m in stored:
        if m.role == "user":
            out.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            out.append(AIMessage(content=m.content))
        elif m.role == "system":
            out.append(SystemMessage(content=m.content))
    return out


def _truncate_text(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "[…]"


def _format_response_for_telegram(text: str) -> list[str]:
    max_len = 4096
    if len(text) <= max_len:
        return [text]
    parts: list[str] = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 <= max_len - 10:
            current += line + "\n"
        else:
            if current:
                parts.append(current.rstrip())
            current = line + "\n"
    if current:
        parts.append(current.rstrip())
    return parts


def _prepare_llm_prompt(user_input: str, context_messages: list) -> list:
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        *context_messages,
        HumanMessage(content=user_input),
    ]


@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
    chat_id = call.message.chat.id
    try:
        if call.data == "new_session":
            _ = storage.start_new_session(
                chat_id,
                title=f"Session_{chat_id}_{int(time.time())}",
            )
            bot.answer_callback_query(call.id, "✅ Новая сессия создана!")
            bot.send_message(
                chat_id,
                "🔄 Новая сессия запущена! История очищена, можно начать новый диалог.",
                reply_markup=get_session_keyboard(),
            )
        elif call.data == "clear_session":
            _ = storage.start_new_session(
                chat_id,
                title=f"Session_{chat_id}_{int(time.time())}",
            )
            bot.answer_callback_query(call.id, "🗑️ История очищена!")
            bot.send_message(
                chat_id,
                "🗑️ История текущей сессии очищена. Можно продолжить общение или начать новую сессию.",
                reply_markup=get_session_keyboard(),
            )
        else:
            bot.answer_callback_query(call.id)
    except Exception:
        log.exception("Ошибка в handle_callback chat_id=%s", chat_id)
        try:
            bot.answer_callback_query(call.id, USER_FACING_CALLBACK_ERROR)
        except Exception:
            pass
        bot.send_message(
            chat_id,
            USER_FACING_ERROR_MESSAGE,
            reply_markup=get_session_keyboard(),
        )


@bot.message_handler(func=lambda message: True)
def handle_llm_message(message):
    t0 = time.perf_counter()

    chat_id = int(message.chat.id)
    user_text = (message.text or "").strip()

    if not user_text:
        return

    if len(user_text) > MAX_USER_MESSAGE_CHARS:
        bot.send_message(
            chat_id,
            f"⚠️ Сообщение слишком длинное (более {MAX_USER_MESSAGE_CHARS} символов). Сократите текст и попробуйте снова.",
            reply_markup=get_session_keyboard(),
        )
        return

    log.info("incoming chat_id=%s len=%s", chat_id, len(user_text))

    try:
        if user_text == "/start":
            bot.send_message(
                chat_id,
                "👋 Добро пожаловать! Используйте кнопки ниже для управления сессией:",
                reply_markup=get_session_keyboard(),
            )
            return

        session_id = storage.get_or_create_session(chat_id, title=None)
        tg_user_id = int(message.from_user.id) if message.from_user else chat_id

        use_primary = _env_truthy("SUPABASE_CHAT_PRIMARY", "0") and _supabase_client is not None
        remote_sid: Optional[str] = None
        if use_primary:
            try:
                from telegram_supabase_chat import get_or_create_remote_session_id, get_user_id_resolver

                uid = get_user_id_resolver(_supabase_client)(tg_user_id)
                if uid:
                    project_id = (os.getenv("SUPABASE_DEFAULT_PROJECT_ID") or "").strip() or None
                    remote_sid = get_or_create_remote_session_id(
                        _supabase_client,
                        supabase_user_id=uid,
                        local_session_id=session_id,
                        session_title=(user_text[:120] if user_text else None),
                        project_id=project_id,
                    )
            except Exception:
                log.warning("Supabase chat primary: подготовка remote-сессии не удалась", exc_info=True)
                remote_sid = None

        recent: list[StoredMessage]
        if use_primary and remote_sid:
            try:
                from telegram_supabase_chat import fetch_recent_messages

                raw = fetch_recent_messages(_supabase_client, remote_sid, MAX_HISTORY_MESSAGES)
                recent = [
                    StoredMessage(role=r["role"], content=r["content"], created_at=r.get("created_at", ""))
                    for r in raw
                ]
            except Exception:
                log.warning("Supabase chat primary: чтение истории не удалось, fallback SQLite", exc_info=True)
                recent = storage.load_recent_messages(session_id, limit=MAX_HISTORY_MESSAGES)
        else:
            recent = storage.load_recent_messages(session_id, limit=MAX_HISTORY_MESSAGES)

        if recent and recent[-1].role == "user":
            recent = recent[:-1]

        context_messages = _stored_to_langchain_messages(recent)

        if use_primary and remote_sid:
            try:
                from telegram_supabase_chat import append_chat_message

                append_chat_message(
                    _supabase_client,
                    remote_session_id=remote_sid,
                    role="user",
                    content=user_text,
                    metadata={},
                )
            except Exception:
                log.warning("Supabase chat primary: запись user не удалась", exc_info=True)

        user_msg_id = storage.append_message(session_id, "user", user_text)

        try:
            user_for_llm = rag_store.enrich_user_prompt(user_text, k=RAG_TOP_K)

            if len(user_for_llm) > MAX_LLM_USER_BLOB_CHARS:
                user_for_llm = _truncate_text(user_for_llm, MAX_LLM_USER_BLOB_CHARS)
        except Exception as rag_err:
            log.warning("RAG недоступен, продолжаю без контекста: %s", rag_err)
            user_for_llm = user_text
        chunks = []
        try:
            chunks = getattr(rag_store, "last_chunks", []) or []
        except Exception:
            chunks = []

        messages = _prepare_llm_prompt(user_for_llm, context_messages)

        t_llm = time.perf_counter()
        response = llm.invoke(messages)
        response_text = getattr(response, "content", str(response))
        llm_ms = round((time.perf_counter() - t_llm) * 1000, 2)

        if use_primary and remote_sid:
            try:
                from telegram_supabase_chat import append_chat_message

                append_chat_message(
                    _supabase_client,
                    remote_session_id=remote_sid,
                    role="assistant",
                    content=response_text or "",
                    metadata={"provider": "openrouter", "model": MODEL_NAME, "llm_ms": llm_ms},
                )
            except Exception:
                log.warning("Supabase chat primary: запись assistant не удалась", exc_info=True)

        storage.append_message(
            session_id,
            "assistant",
            response_text,
            in_response_to=user_msg_id,
            metadata={"provider": "openrouter", "model": MODEL_NAME, "llm_ms": llm_ms},
        )

        if LOCAL_CHAT_BACKUP_QA_PAIRS > 0:
            storage.prune_session_messages(session_id, keep_last_messages=LOCAL_CHAT_BACKUP_QA_PAIRS * 2)

        log_interaction(
            chat_id=chat_id,
            user_query=user_text,
            chunks=chunks,
            answer=response_text,
            model=MODEL_NAME,
            latency_ms=llm_ms,
            rag_kind=str(getattr(rag_store, "rag_kind", os.getenv("RAG_MODE", "unknown"))),
        )

        total_ms = round((time.perf_counter() - t0) * 1000, 2)
        if interaction_db.is_enabled():
            try:
                interaction_db.log(
                    chat_id=chat_id,
                    user_query=user_text,
                    rag_kind=str(getattr(rag_store, "rag_kind", "unknown")),
                    lexical_tokens=list(getattr(rag_store, "last_lexical_tokens", []) or []),
                    chunks=chunks,
                    vector_raw_by_id=dict(getattr(rag_store, "last_vector_raw", {}) or {}),
                    lexical_raw_by_id=dict(getattr(rag_store, "last_lexical_raw", {}) or {}),
                    answer=response_text or "",
                    model=MODEL_NAME,
                    latency_total_ms=total_ms,
                    latency_llm_ms=llm_ms,
                )
            except Exception as _e:
                log.warning("interaction_db.log пропущен: %s", _e)

        if _supabase_client is not None and not use_primary:
            try:
                from telegram_supabase_chat import mirror_append_last_turn

                mirror_append_last_turn(
                    _supabase_client,
                    tg_user_id=tg_user_id,
                    local_session_id=session_id,
                    user_text=user_text,
                    assistant_text=response_text or "",
                    user_metadata={},
                    assistant_metadata={"provider": "openrouter", "model": MODEL_NAME, "llm_ms": llm_ms},
                )
            except Exception:
                log.debug("Supabase chat mirror пропущен", exc_info=True)

        response_parts = _format_response_for_telegram(response_text)
        for i, part in enumerate(response_parts):
            bot.send_message(
                chat_id,
                part,
                reply_markup=get_session_keyboard() if i == len(response_parts) - 1 else None,
            )

        total_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.info("done chat_id=%s llm_ms=%s total_ms=%s", chat_id, llm_ms, total_ms)

    except Exception:
        log.exception("Ошибка обработки chat_id=%s", chat_id)
        bot.send_message(
            chat_id,
            USER_FACING_ERROR_MESSAGE,
            reply_markup=get_session_keyboard(),
        )


def _startup_llm_smoke() -> None:
    test_prompt = "Ответь просто 'ok'."
    test_response = llm.invoke([HumanMessage(content=test_prompt)])
    test_answer = getattr(test_response, "content", "").strip().lower()
    if "ok" not in test_answer:
        raise RuntimeError(f"LLM не отвечает корректно: получил '{test_answer}'")


_EMBEDDING_WARMUP_RETRY_SEC = 10


def _should_warmup_embedding() -> bool:
    if isinstance(rag_store, _PassthroughRagStore):
        return False
    return _env_truthy("RAG_USE_QDRANT", "0")


def _try_startup_embedding_warmup() -> bool:
    from query_embedding import warmup

    for attempt in (1, 2):
        try:
            dim = warmup(probe=True)
            log.info("Модель эмбеддинга готова (попытка %s/2), dim=%s", attempt, dim)
            return True
        except Exception:
            log.warning(
                "Warmup эмбеддинга не удался (попытка %s/2)",
                attempt,
                exc_info=True,
            )
            if attempt == 1:
                time.sleep(_EMBEDDING_WARMUP_RETRY_SEC)
    log.warning(
        "Модель эмбеддинга будет загружена при первом запросе с векторным RAG"
    )
    return False


def main():
    health = storage.healthcheck()
    if not health.get("ok"):
        raise RuntimeError(f"Хранилище недоступно: {health}")

    do_smoke = os.getenv("BOT_STARTUP_LLM_CHECK", "1").lower() in ("1", "true", "yes")
    if do_smoke:
        try:
            _startup_llm_smoke()
            log.info("Предварительная проверка LLM пройдена")
        except Exception:
            log.exception("Предварительная проверка LLM не прошла")
            raise
    else:
        log.info("BOT_STARTUP_LLM_CHECK отключён — пропускаем тестовый запрос к LLM")

    if _should_warmup_embedding():
        _try_startup_embedding_warmup()

    log.info("Запуск Telegram-бота (infinity_polling)...")
    try:
        bot.infinity_polling(
            timeout=30,
            long_polling_timeout=20,
            allowed_updates=["message", "callback_query"],
        )
    except KeyboardInterrupt:
        log.info("Завершение работы по KeyboardInterrupt")


if __name__ == "__main__":
    main()
