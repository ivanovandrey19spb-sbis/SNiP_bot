"""Минимальная замена langchain_core.messages + langchain_openai для bot.py (OpenAI SDK)."""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

log = logging.getLogger(__name__)

_TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_NON_RETRYABLE_TYPES = (
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    BadRequestError,
    UnprocessableEntityError,
)


def _float_env(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    return float(raw)


def _int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    return int(raw)


def _is_transient_error(exc: BaseException) -> bool:
    if isinstance(exc, _NON_RETRYABLE_TYPES):
        return False
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)):
        return True
    if isinstance(exc, APIStatusError):
        code = getattr(exc, "status_code", None)
        return code in _TRANSIENT_STATUS_CODES
    return False


def _retry_after_seconds(exc: BaseException) -> Optional[float]:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _compute_delay(exc: BaseException, attempt: int, base_sec: float) -> float:
    backoff = base_sec * (2**attempt) + random.uniform(0, 0.3)
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return max(retry_after, backoff)
    return backoff


@dataclass
class HumanMessage:
    content: str
    role: ClassVar[str] = "user"


@dataclass
class AIMessage:
    content: str
    role: ClassVar[str] = "assistant"


@dataclass
class SystemMessage:
    content: str
    role: ClassVar[str] = "system"


@dataclass
class _Response:
    content: str


class ChatOpenAI:
    """Минимальная замена langchain_openai.ChatOpenAI с методом invoke()."""

    def __init__(
        self,
        *,
        openai_api_key: str,
        base_url: str,
        model: str,
        default_headers: Optional[dict] = None,
        **_: Any,
    ) -> None:
        self._timeout_sec = _float_env("OPENROUTER_TIMEOUT_SEC", 90.0)
        self._max_retries = _int_env("OPENROUTER_MAX_RETRIES", 2)
        self._retry_base_sec = _float_env("OPENROUTER_RETRY_BASE_SEC", 1.0)
        self._client = OpenAI(
            api_key=openai_api_key,
            base_url=base_url,
            default_headers=default_headers or {},
            timeout=self._timeout_sec,
            max_retries=0,
        )
        self._model = model

    def invoke(self, messages: list) -> _Response:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        last_exc: Optional[BaseException] = None
        max_attempts = self._max_retries + 1

        for attempt in range(max_attempts):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=payload,
                )
                return _Response(content=resp.choices[0].message.content or "")
            except Exception as e:
                last_exc = e
                if attempt >= self._max_retries or not _is_transient_error(e):
                    raise
                delay = _compute_delay(e, attempt, self._retry_base_sec)
                log.warning(
                    "LLM retry %s/%s after %.2fs (%s): %s",
                    attempt + 1,
                    self._max_retries,
                    delay,
                    type(e).__name__,
                    e,
                )
                time.sleep(delay)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM invoke failed without exception")
