"""
Смоук-проверка: Supabase (chunks) + Qdrant (поиск) + подгрузка текстов по id из Qdrant payload.

Запуск: python verify_rag_connections.py

Эмбеддинг запроса пока не считается — вектор случайный, цель: убедиться, что подключения
живы и цепочка возвращает хоть какой-то ответ.

Модель эмбеддинга для продакшена (от DE, для следующих шагов):
  EMBEDDING_MODEL = "ai-forever/ru-en-RoSBERTa"
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from supabase_helper import SupabaseHelper

# Согласовано с DE; в этом скрипте не используется (зарезервировано для следующего шага).
EMBEDDING_MODEL = "ai-forever/ru-en-RoSBERTa"


def _print_step(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def check_supabase_list_chunks(client: Any) -> list[dict[str, Any]]:
    res = client.table("chunks").select("id, text_content").limit(3).execute()
    rows = getattr(res, "data", None) or []
    print(f"SUPABASE_URL: {(os.getenv('SUPABASE_URL') or '').strip()}")
    print(f"Строк из chunks (limit=3): {len(rows)}")
    for i, row in enumerate(rows, 1):
        tid = row.get("id")
        text = (row.get("text_content") or "")[:200]
        print(f"  [{i}] id={tid!s} text_preview={text!r}...")
    return rows


def check_qdrant_random_search() -> list[Any]:
    host = (os.getenv("QDRANT_HOST") or "localhost").strip()
    port = int(os.getenv("QDRANT_PORT") or "6333")
    collection = (os.getenv("QDRANT_COLLECTION") or "test_chunks").strip()
    vector_size = int(os.getenv("QDRANT_VECTOR_SIZE") or "1024")

    print(f"Qdrant: {host}:{port}, collection={collection}, vector_size={vector_size} (random)")
    client = QdrantClient(host=host, port=port)
    vector = np.random.default_rng().random(vector_size, dtype=np.float64).tolist()
    hits = client.search(collection_name=collection, query_vector=vector, limit=3)
    print(f"Хитов: {len(hits)}")
    for hit in hits:
        pl = getattr(hit, "payload", None) or {}
        print(f"  id={hit.id} score={hit.score} payload={pl}")
    return hits


def fetch_chunk_texts_by_ids(client: Any, ids: list[str]) -> list[str]:
    if not ids:
        return []
    res = client.table("chunks").select("id, text_content").in_("id", ids).execute()
    rows = getattr(res, "data", None) or []
    return [(row.get("text_content") or "") for row in rows]


def main() -> int:
    load_dotenv(override=True)

    print("(Справка) Модель эмбеддинга от DE (пока не используется):", EMBEDDING_MODEL)

    ok = True
    supabase = SupabaseHelper(load_env=False).get_bot_supabase_client()

    _print_step("1) Supabase: выборка из chunks")
    try:
        check_supabase_list_chunks(supabase)
    except Exception as e:
        ok = False
        print("ОШИБКА:", e, file=sys.stderr)

    _print_step("2) Qdrant: поиск со случайным вектором")
    hits: list[Any] = []
    try:
        hits = check_qdrant_random_search()
    except Exception as e:
        ok = False
        print("ОШИБКА:", e, file=sys.stderr)

    _print_step("3) Supabase: тексты по chunk_id из payload Qdrant")
    try:
        ids: list[str] = []
        for h in hits:
            pl = getattr(h, "payload", None) or {}
            if not isinstance(pl, dict):
                continue
            cid = pl.get("chunk_id")
            if cid is not None:
                ids.append(str(cid))
        ids = list(dict.fromkeys(ids))
        print("chunk_id из payload:", ids)
        if not ids:
            print("Нет chunk_id в payload — шаг пропущен.")
        else:
            texts = fetch_chunk_texts_by_ids(supabase, ids)
            print(f"Получено текстов: {len(texts)}")
            for i, t in enumerate(texts, 1):
                preview = (t or "")[:300]
                print(f"  [{i}] {preview}")
                print("  " + "-" * 50)
    except Exception as e:
        ok = False
        print("ОШИБКА:", e, file=sys.stderr)

    _print_step("Итог")
    if ok:
        print("Смоук завершён без исключений.")
        return 0
    print("Смоук завершён с ошибками (см. выше).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
