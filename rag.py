"""
RAG из таблицы chunks (+ documents) в Supabase.

- Lexical: токены запроса → один OR-запрос (`or=(text_content.ilike.<pat1>,...)`) → локальный
  скоринг по числу совпавших токенов.
- Vector (опционально): эмбеддинг запроса → Qdrant → chunk_id из payload → догрузка из chunks;
  слияние с lexical через merge_vector_then_lexical.

Qdrant: только чтение / search (коллекцию создаёт и наполняет DE, приложение не меняет Qdrant).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

from rag_common import context_header_lines

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+", re.UNICODE)
CHUNK_SELECT_FIELDS = "*"
DOCUMENT_SELECT_FIELDS = "*"
SECTION_SELECT_FIELDS = "*"

# Protected-token patterns. Лемматизация не применяется.
# Составные коды извлекаются из ИСХОДНОЙ строки до _TOKEN_RE, иначе точки/дефисы
# режут "5.2.1" → "5","2","1" и protected-правила не сработают.
_RE_COMPOUND_PROTECTED = re.compile(
    r"[0-9A-Za-zА-Яа-яЁё]+(?:[.\-][0-9A-Za-zА-Яа-яЁё]+)+"
)
# Дополнительно — для одиночных токенов (после _TOKEN_RE):
_RE_NUMERIC_DOTTED = re.compile(r"^\d+(?:\.\d+)+$")
_RE_CODE_WITH_SEP = re.compile(r"^[0-9A-Za-zА-Яа-яЁё]+(?:[.\-][0-9A-Za-zА-Яа-яЁё]+){1,}$")
_RE_UPPER_ABBR = re.compile(r"^[A-ZА-ЯЁ]{2,}$")

_RU_STOPWORDS_CACHE: Optional[frozenset[str]] = None


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _ru_stopwords() -> frozenset[str]:
    """Лениво загружает RU stopwords из библиотеки `stop-words`.

    При отсутствии библиотеки возвращает пустой frozenset — фильтр стоп-слов
    отключается, остаются только protected-правила и dedupe.
    """
    global _RU_STOPWORDS_CACHE
    if _RU_STOPWORDS_CACHE is not None:
        return _RU_STOPWORDS_CACHE

    try:
        from stop_words import get_stop_words

        words = {str(w).strip().lower() for w in get_stop_words("russian") if w}
    except Exception as e:
        log.warning("RU stopwords недоступны (stop-words не установлен или ошибка): %s", e)
        words = set()

    _RU_STOPWORDS_CACHE = frozenset(words)
    return _RU_STOPWORDS_CACHE


def _is_protected_token(t_raw: str) -> bool:
    """True для токенов, которые нельзя выкидывать stop-words фильтром."""
    if not t_raw:
        return False
    if _RE_NUMERIC_DOTTED.match(t_raw):
        return True
    if _RE_UPPER_ABBR.match(t_raw):
        return True
    if _RE_CODE_WITH_SEP.match(t_raw):
        return True
    return False


def _select_lexical_tokens(query: str, *, max_tokens: int) -> List[str]:
    """Собрать токены для Supabase OR ilike.

    Шаги:
    1) Из исходной строки вырезать составные protected (`4.9.7`, `113.13330.2023`,
       `12.3.002-75`, `EN-1991-1-1`).
    2) Остаток токенизировать `_TOKEN_RE`. Внутри: одиночные uppercase-аббревиатуры
       (`СП`, `ГОСТ`, `ISO`) — protected; остальное — фильтр по длине и stopwords.
    3) Объединить protected → normal с сохранением порядка и dedupe по lower-case.
    4) Fallback: если список пуст, не делать пустой OR — вернуть `_tokenize`+drop_short.
    5) Обрезать по `max_tokens`.
    """
    text = query or ""
    if not text:
        return []

    stopwords = _ru_stopwords()
    seen: set[str] = set()
    kept_protected: List[str] = []
    kept_normal: List[str] = []

    text_for_normal_parts: List[str] = []
    last_end = 0
    for m in _RE_COMPOUND_PROTECTED.finditer(text):
        text_for_normal_parts.append(text[last_end : m.start()])
        token = m.group(0)
        t_low = token.lower()
        if t_low and t_low not in seen:
            seen.add(t_low)
            kept_protected.append(t_low)
        last_end = m.end()
    text_for_normal_parts.append(text[last_end:])
    text_for_normal = " ".join(text_for_normal_parts)

    for t in _TOKEN_RE.findall(text_for_normal):
        if _is_protected_token(t):
            t_low = t.lower()
            if t_low and t_low not in seen:
                seen.add(t_low)
                kept_protected.append(t_low)
            continue

        t_low = t.lower()
        if len(t_low) <= 1:
            continue
        if t_low in stopwords:
            continue
        if t_low in seen:
            continue
        seen.add(t_low)
        kept_normal.append(t_low)

    tokens = kept_protected + kept_normal
    if not tokens:
        # Fallback: ничего не осталось — не делаем пустой OR.
        tokens = list(dict.fromkeys(t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1))

    if max_tokens > 0:
        tokens = tokens[:max_tokens]
    return tokens


def _ilike_pattern(token: str) -> str:
    """Экранирование % и _ для PostgREST ilike."""
    t = token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{t}%"


def _postgrest_or_escape(s: str) -> str:
    """Экранирование значений для inline-DSL `or=(...)` PostgREST.

    В `or_` значения разделяются запятыми, а круглые скобки и пробелы могут
    ломать разбор условий. Экранируем их обратной косой чертой.
    Не трогаем `%` и `_` — это часть semantics ilike (см. `_ilike_pattern`).
    """
    return (
        s.replace("\\", "\\\\")
         .replace(",", "\\,")
         .replace("(", "\\(")
         .replace(")", "\\)")
         .replace(" ", "\\ ")
    )


def _env_truthy(name: str, *, default: str = "0") -> bool:
    v = (os.getenv(name) or default).strip().lower()
    return v in ("1", "true", "yes", "on")


def _rag_use_qdrant() -> bool:
    return _env_truthy("RAG_USE_QDRANT", default="0")


def _stopwords_enabled() -> bool:
    return _env_truthy("RAG_STOPWORDS_ENABLED", default="1")


def _rrf_enabled() -> bool:
    return _env_truthy("RAG_RRF_ENABLED", default="1")


def merge_vector_then_lexical(
    vector_chunks: List[dict],
    lexical_chunks: List[dict],
    *,
    max_total: int,
) -> List[dict]:
    """Сначала векторные чанки (порядок релевантности Qdrant), затем lexical без дубликатов по id."""
    seen: set[str] = set()
    out: List[dict] = []
    for ch in vector_chunks:
        rid = str(ch.get("id", "") or "")
        if not rid or rid in seen:
            continue
        seen.add(rid)
        out.append(ch)
        if len(out) >= max_total:
            return out
    for ch in lexical_chunks:
        rid = str(ch.get("id", "") or "")
        if not rid or rid in seen:
            continue
        seen.add(rid)
        out.append(ch)
        if len(out) >= max_total:
            break
    return out


def rrf_fuse(
    vector_chunks: List[dict],
    lexical_chunks: List[dict],
    *,
    k_const: int = 60,
    top_k: int,
) -> List[dict]:
    """Reciprocal Rank Fusion двух ранжированных списков по полю 'id'.

    Формула: RRF(id) = 1/(C+rank_vec) + 1/(C+rank_lex);
    отсутствие в списке → соответствующее слагаемое = 0.

    Возвращает топ-k чанков с переопределёнными:
      - score = rrf_score (float)
      - source = "rrf"
      - rrf_breakdown = {rank_vec, rank_lex, source_set}

    Метаданные берём из первого встреченного чанка по id (приоритет vector_chunks),
    т.к. _format_chunk_row одинаково форматирует строки в обеих ветках.
    """
    rank_vec: Dict[str, int] = {}
    rank_lex: Dict[str, int] = {}
    chunk_by_id: Dict[str, dict] = {}

    for i, ch in enumerate(vector_chunks):
        rid = str(ch.get("id", "") or "")
        if not rid or rid in rank_vec:
            continue
        rank_vec[rid] = i + 1
        chunk_by_id.setdefault(rid, ch)

    for i, ch in enumerate(lexical_chunks):
        rid = str(ch.get("id", "") or "")
        if not rid or rid in rank_lex:
            continue
        rank_lex[rid] = i + 1
        chunk_by_id.setdefault(rid, ch)

    c = max(1, int(k_const))
    all_ids = set(rank_vec) | set(rank_lex)

    scored: List[tuple[float, int, int, str]] = []
    for rid in all_ids:
        rv = rank_vec.get(rid)
        rl = rank_lex.get(rid)
        score = (1.0 / (c + rv) if rv else 0.0) + (1.0 / (c + rl) if rl else 0.0)
        # для устойчивого тай-брейка используем «бесконечный» ранг там, где нет вхождения
        rv_tb = rv if rv is not None else 10**9
        rl_tb = rl if rl is not None else 10**9
        scored.append((score, rv_tb, rl_tb, rid))

    # сортировка: score desc, затем rank_vec asc, затем rank_lex asc
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))

    out: List[dict] = []
    for score, _rv_tb, _rl_tb, rid in scored[: max(1, int(top_k))]:
        base = chunk_by_id.get(rid)
        if not base:
            continue
        rv = rank_vec.get(rid)
        rl = rank_lex.get(rid)
        if rv and rl:
            source_set = "vector+lexical"
        elif rv:
            source_set = "vector_only"
        else:
            source_set = "lexical_only"
        ch = dict(base)
        ch["score"] = float(score)
        ch["source"] = "rrf"
        ch["rrf_breakdown"] = {
            "rank_vec": int(rv) if rv else 0,
            "rank_lex": int(rl) if rl else 0,
            "source_set": source_set,
        }
        out.append(ch)
    return out


class SupabaseChunksRagStore:
    """
    Хранилище RAG поверх Supabase (+ опционально Qdrant).

    ENV (опционально):
    - RAG_SUPABASE_LIMIT_PER_TOKEN, RAG_SUPABASE_MAX_TOKENS, RAG_SUPABASE_WORKSPACE_ID
    - RAG_USE_QDRANT=1 — включить гибрид (эмбеддинг + Qdrant + merge)
    - RAG_VECTOR_TOP_K (по умолчанию 20), RAG_MERGE_MAX_TOTAL (по умолчанию RAG_VECTOR_TOP_K + k retrieve)
    - QDRANT_*, EMBEDDING_MODEL / HF_EMBED_MODEL, HF_DEVICE, HF_MAX_LENGTH
    """

    def __init__(
        self,
        supabase_client: Any,
        *,
        debug: bool = False,
        max_chunk_chars: int = 2500,
        limit_per_token: Optional[int] = None,
        max_query_tokens: Optional[int] = None,
        workspace_id: Optional[str] = None,
    ):
        self._client = supabase_client
        self._debug = bool(debug)
        self._max_chunk_chars = int(max_chunk_chars)
        self._limit_per_token = int(
            limit_per_token if limit_per_token is not None else os.getenv("RAG_SUPABASE_LIMIT_PER_TOKEN", "20")
        )
        self._max_query_tokens = int(
            max_query_tokens
            if max_query_tokens is not None
            else os.getenv("RAG_SUPABASE_MAX_TOKENS", "6")
        )
        ws = workspace_id if workspace_id is not None else (os.getenv("RAG_SUPABASE_WORKSPACE_ID") or "").strip()
        self._workspace_id = ws or None
        self.last_chunks: List[dict] = []
        # Поля для отладочного журнала interaction_db (см. interaction_db.py).
        # Заполняются на каждом retrieve(); читаются bot.py после enrich_user_prompt.
        self.last_lexical_tokens: List[str] = []
        self.last_vector_raw: Dict[str, float] = {}
        self.last_lexical_raw: Dict[str, float] = {}
        if _rag_use_qdrant():
            self.rag_kind = "qdrant+rrf" if _rrf_enabled() else "supabase+qdrant"
        else:
            self.rag_kind = "supabase"

    def _base_chunk_query(self):
        q = self._client.table("chunks").select(CHUNK_SELECT_FIELDS)
        if self._workspace_id:
            q = q.eq("workspace_id", self._workspace_id)
        return q

    def _fetch_documents(self, doc_ids: List[str]) -> Dict[str, dict]:
        if not doc_ids:
            return {}
        try:
            res = self._client.table("documents").select(DOCUMENT_SELECT_FIELDS).in_(
                "id", doc_ids
            ).execute()
            data = getattr(res, "data", None) or []
            return {str(d["id"]): d for d in data}
        except Exception as e:
            log.warning("Supabase RAG: не удалось загрузить documents: %s", e)
            return {}

    def _fetch_sections_for_docs(self, doc_ids: List[str]) -> tuple[Dict[str, dict], Dict[tuple[str, str], dict]]:
        if not doc_ids:
            return {}, {}
        try:
            res = self._client.table("document_sections").select(SECTION_SELECT_FIELDS).in_(
                "doc_id", doc_ids
            ).execute()
            data = getattr(res, "data", None) or []
        except Exception as e:
            log.warning("Supabase RAG: не удалось загрузить document_sections: %s", e)
            return {}, {}

        by_id: Dict[str, dict] = {}
        by_doc_code: Dict[tuple[str, str], dict] = {}
        for section in data:
            sid = str(section.get("id") or "")
            doc_id = str(section.get("doc_id") or "")
            code = str(section.get("section_code") or "").strip()
            if sid:
                by_id[sid] = section
            if doc_id and code:
                by_doc_code[(doc_id, code)] = section
        return by_id, by_doc_code

    def _fetch_chunk_rows_by_ids(self, ids: List[str]) -> Dict[str, dict]:
        if not ids:
            return {}
        q = self._client.table("chunks").select(CHUNK_SELECT_FIELDS).in_("id", ids)
        if self._workspace_id:
            q = q.eq("workspace_id", self._workspace_id)
        try:
            res = q.execute()
            data = getattr(res, "data", None) or []
            return {str(r["id"]): r for r in data if r.get("id")}
        except Exception as e:
            log.warning("Supabase RAG: не удалось загрузить chunks по id: %s", e)
            return {}

    @staticmethod
    def _section_summary(section: dict) -> dict:
        return {
            "id": str(section.get("id")) if section.get("id") else None,
            "section_code": section.get("section_code"),
            "section_title": section.get("section_title"),
            "hierarchy_path": section.get("hierarchy_path"),
            "level": section.get("level"),
        }

    @staticmethod
    def _section_code_from_path_part(part: str) -> str:
        part = (part or "").strip()
        if not part:
            return ""
        # section_path иногда содержит только код, а иногда "4.9 Название".
        m = re.match(r"^([0-9]+(?:\.[0-9]+)*|[А-Яа-яA-Za-z]+\s*[0-9A-Za-zА-Яа-я.-]*)", part)
        return (m.group(1).strip() if m else part)

    def _build_section_hierarchy(
        self,
        row: dict,
        sections_by_id: Dict[str, dict],
        sections_by_doc_code: Dict[tuple[str, str], dict],
    ) -> list[dict]:
        doc_id = str(row.get("doc_id") or "")
        section_id = str(row.get("section_id") or "")
        current = sections_by_id.get(section_id) if section_id else None

        chain: list[dict] = []
        if current:
            seen: set[str] = set()
            node: Optional[dict] = current
            while node:
                sid = str(node.get("id") or "")
                if not sid or sid in seen:
                    break
                seen.add(sid)
                chain.append(self._section_summary(node))
                parent_id = str(node.get("parent_section_id") or "")
                node = sections_by_id.get(parent_id) if parent_id else None
            chain.reverse()

        section_path = str(row.get("section_path") or "")
        path_codes = [
            self._section_code_from_path_part(part)
            for part in section_path.split(">")
            if self._section_code_from_path_part(part)
        ]
        if path_codes and len(chain) < len(path_codes):
            fallback_chain: list[dict] = []
            for code in path_codes:
                section = sections_by_doc_code.get((doc_id, code))
                if section:
                    fallback_chain.append(self._section_summary(section))
                else:
                    fallback_chain.append(
                        {
                            "id": None,
                            "section_code": code,
                            "section_title": None,
                            "hierarchy_path": section_path or None,
                            "level": None,
                        }
                    )
            chain = fallback_chain

        return chain

    def _format_chunk_row(
        self,
        row: dict,
        score: float,
        source: str,
        docs: Dict[str, dict],
        sections_by_id: Dict[str, dict],
        sections_by_doc_code: Dict[tuple[str, str], dict],
    ) -> dict:
        rid = str(row.get("id"))
        text = (row.get("text_content") or "").strip()
        text = text[: self._max_chunk_chars]
        doc_id = row.get("doc_id")
        doc = docs.get(str(doc_id)) if doc_id else None
        designation = (doc or {}).get("designation") or ""
        clause = row.get("clause_display") or ""
        section_id = row.get("section_id")
        section_hierarchy = self._build_section_hierarchy(row, sections_by_id, sections_by_doc_code)
        section_title = None
        if section_hierarchy:
            section_title = section_hierarchy[-1].get("section_title")

        return {
            "id": rid,
            "score": float(round(score, 6)),
            "text": text,
            "text_content": text,
            "chunk_metadata": dict(row),
            "doc_id": str(doc_id) if doc_id else None,
            "section_id": str(section_id) if section_id else None,
            "designation": designation or None,
            "official_title": (doc or {}).get("official_title"),
            "valid_from": (doc or {}).get("valid_from"),
            "valid_to": (doc or {}).get("valid_to"),
            "document_metadata": doc or None,
            "clause_display": clause or None,
            "section_path": row.get("section_path"),
            "section_title": section_title,
            "section_hierarchy": section_hierarchy,
            "source": source,
        }

    @staticmethod
    def _format_chunk_for_prompt(ch: dict) -> str:
        lines: list[str] = []
        designation = ch.get("designation")
        official_title = ch.get("official_title")
        if designation or official_title:
            doc_label = " — ".join(str(p) for p in (designation, official_title) if p)
            lines.append(f"Документ: {doc_label}")

        if ch.get("valid_from") or ch.get("valid_to") is not None:
            valid_to = ch.get("valid_to") or "действующий"
            lines.append(f"Действует с: {ch.get('valid_from') or 'не указано'}; действует до: {valid_to}")

        if ch.get("clause_display"):
            lines.append(f"Пункт: {ch.get('clause_display')}")
        if ch.get("section_path"):
            lines.append(f"Путь раздела: {ch.get('section_path')}")
        if ch.get("section_title"):
            lines.append(f"Раздел: {ch.get('section_title')}")

        hierarchy = ch.get("section_hierarchy") or []
        if hierarchy:
            lines.append("Иерархия разделов:")
            for section in hierarchy:
                code = section.get("section_code")
                title = section.get("section_title")
                label = " ".join(str(p) for p in (code, title) if p)
                lines.append(f"- {label or 'без названия'}")

        lines.append("Текст чанка:")
        lines.append(ch.get("text_content") or ch.get("text") or "")
        return "\n".join(lines)

    def _retrieve_lexical(self, query: str, *, k: int) -> List[dict]:
        if _stopwords_enabled():
            tokens = _select_lexical_tokens(query, max_tokens=self._max_query_tokens)
        else:
            tokens = list(dict.fromkeys(t for t in _tokenize(query) if len(t) > 1))[: self._max_query_tokens]

        # Сохраняем выбранные токены для отладочного журнала (interaction_db).
        self.last_lexical_tokens = list(tokens)

        if self._debug:
            raw = _TOKEN_RE.findall(query or "")
            compound = [m.lower() for m in _RE_COMPOUND_PROTECTED.findall(query or "")]
            log.debug(
                "lexical tokens: raw=%s compound_protected=%s kept=%s",
                raw,
                compound,
                tokens,
            )

        if not tokens:
            return []

        or_expr = ",".join(
            f"text_content.ilike.{_postgrest_or_escape(_ilike_pattern(tok))}" for tok in tokens
        )
        total_limit = self._limit_per_token * max(1, len(tokens))

        try:
            res = (
                self._base_chunk_query()
                .or_(or_expr)
                .limit(total_limit)
                .execute()
            )
            rows = getattr(res, "data", None) or []
        except Exception as e:
            if self._debug:
                log.exception("Supabase RAG retrieve (or_): %s", e)
            else:
                log.warning("Supabase RAG retrieve (or_) пропущен: %s", e)
            return []

        scores: dict[str, float] = {}
        by_id: dict[str, dict] = {}
        for row in rows:
            rid = str(row.get("id") or "")
            if not rid:
                continue
            text_lower = (row.get("text_content") or "").lower()
            hit = sum(1 for tok in tokens if tok in text_lower)
            if hit <= 0:
                continue
            scores[rid] = float(hit)
            by_id[rid] = row

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: max(1, int(k))]
        doc_ids = list({str(by_id[i]["doc_id"]) for i, _ in ranked if i in by_id and by_id[i].get("doc_id")})
        docs = self._fetch_documents(doc_ids)
        sections_by_id, sections_by_doc_code = self._fetch_sections_for_docs(doc_ids)

        out: List[dict] = []
        for rid, score in ranked:
            row = by_id.get(rid)
            if not row:
                continue
            out.append(self._format_chunk_row(row, float(score), "lexical", docs, sections_by_id, sections_by_doc_code))
        return out

    def _retrieve_vector_chunks(self, query: str, *, limit: int) -> List[dict]:
        import qdrant_rag as qr
        from query_embedding import embed_query

        vec = embed_query(query)
        cfg = qr.qdrant_config_from_env()
        col = (cfg.get("collection") or "").strip()
        if not col:
            raise RuntimeError("Не задано QDRANT_COLLECTION")

        hits = qr.search_chunk_hits(
            vec,
            limit=limit,
            workspace_id=self._workspace_id,
            collection_name=col,
        )
        ids = [h["chunk_id"] for h in hits if h.get("chunk_id")]
        score_by_id = {str(h["chunk_id"]): float(h.get("score") or 0.0) for h in hits if h.get("chunk_id")}
        rows = self._fetch_chunk_rows_by_ids(ids)
        doc_ids = list({str(r["doc_id"]) for r in rows.values() if r.get("doc_id")})
        docs = self._fetch_documents(doc_ids)
        sections_by_id, sections_by_doc_code = self._fetch_sections_for_docs(doc_ids)

        out: List[dict] = []
        for cid in ids:
            row = rows.get(str(cid))
            if not row:
                continue
            out.append(
                self._format_chunk_row(
                    row,
                    score_by_id.get(str(cid), 0.0),
                    "vector",
                    docs,
                    sections_by_id,
                    sections_by_doc_code,
                )
            )
        return out

    @staticmethod
    def _raw_score_map(chunks: List[dict]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for c in chunks or []:
            rid = str(c.get("id") or "")
            if not rid:
                continue
            try:
                out[rid] = float(c.get("score") or 0.0)
            except (TypeError, ValueError):
                out[rid] = 0.0
        return out

    def retrieve(self, query: str, *, k: int = 4) -> List[dict]:
        # Сбрасываем «прошлые» данные для отладочного журнала. last_lexical_tokens
        # перезапишется в _retrieve_lexical(); raw-скоры — здесь по факту fetch'а.
        self.last_vector_raw = {}
        self.last_lexical_raw = {}

        if _rag_use_qdrant() and _rrf_enabled():
            try:
                n = int(os.getenv("RAG_RRF_VECTOR_TOP_K") or "80")
            except ValueError:
                n = 80
            try:
                m = int(os.getenv("RAG_RRF_LEXICAL_TOP_K") or "40")
            except ValueError:
                m = 40
            try:
                c = int(os.getenv("RAG_RRF_K_CONSTANT") or "60")
            except ValueError:
                c = 60
            try:
                vector = self._retrieve_vector_chunks(query, limit=max(1, n))
                lexical = self._retrieve_lexical(query, k=max(1, m))
                # Фиксируем сырые скоры ДО RRF (rrf_fuse перезапишет chunk["score"]).
                self.last_vector_raw = self._raw_score_map(vector)
                self.last_lexical_raw = self._raw_score_map(lexical)
                out = rrf_fuse(vector, lexical, k_const=max(1, c), top_k=max(1, k))
                if self._debug and out:
                    for ch in out[:3]:
                        br = ch.get("rrf_breakdown") or {}
                        log.debug(
                            "rrf top: id=%s score=%.4f rank_vec=%s rank_lex=%s set=%s",
                            ch.get("id"),
                            float(ch.get("score") or 0.0),
                            br.get("rank_vec"),
                            br.get("rank_lex"),
                            br.get("source_set"),
                        )
                return out
            except Exception as e:
                if self._debug:
                    log.exception("RRF fusion пропущен: %s", e)
                else:
                    log.warning("RRF fusion пропущен, fallback к legacy: %s", e)

        lexical = self._retrieve_lexical(query, k=k)
        # legacy-ветка: lexical уже отфильтрован до k, но сырые hit-count'ы там есть.
        self.last_lexical_raw = self._raw_score_map(lexical)
        if not _rag_use_qdrant():
            return lexical

        vk = int(os.getenv("RAG_VECTOR_TOP_K") or "20")
        default_max = vk + max(1, int(k))
        max_total = int(os.getenv("RAG_MERGE_MAX_TOTAL") or str(default_max))
        max_total = max(1, max_total)

        try:
            vector_chunks = self._retrieve_vector_chunks(query, limit=vk)
            self.last_vector_raw = self._raw_score_map(vector_chunks)
            return merge_vector_then_lexical(vector_chunks, lexical, max_total=max_total)
        except Exception as e:
            if self._debug:
                log.exception("Vector RAG: %s", e)
            else:
                log.warning("Vector RAG пропущен: %s", e)
            return lexical

    def enrich_user_prompt(self, user_text: str, k: int = 4) -> str:
        chunks = self.retrieve(user_text, k=k)
        self.last_chunks = chunks

        if not chunks:
            return user_text

        ctx_lines: List[str] = []
        ctx_lines.extend(
            context_header_lines(rag_kind=self.rag_kind, block_title="Relevant context (Supabase chunks)")
        )
        for ch in chunks:
            src = ch.get("source", "lexical")
            ctx_lines.append(f"[chunk id={ch['id']} | source={src} | retrieval_score={ch.get('score')}]")
            ctx_lines.append(self._format_chunk_for_prompt(ch))
            ctx_lines.append("")
        ctx = "\n".join(ctx_lines).strip()

        return (
            f"{user_text}\n\n"
            f"---\n"
            f"{ctx}\n"
            f"---\n"
            f"Инструкция: используй контекст выше только если он действительно относится к вопросу; "
            f"если контекст не помогает — отвечай без него."
        )


__all__ = ["SupabaseChunksRagStore", "merge_vector_then_lexical", "rrf_fuse"]
