"""
Qdrant: только подключение и поиск (read-only). Коллекции создаёт и наполняет DE.

Подключение: QDRANT_URL (+ QDRANT_API_KEY) или QDRANT_HOST + QDRANT_PORT (как в verify_rag_connections).
Payload: chunk_id (UUID чанка в Supabase); опционально workspace_id для фильтра.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

log = logging.getLogger(__name__)


def qdrant_config_from_env() -> dict[str, Any]:
    host = (os.getenv("QDRANT_HOST") or "").strip()
    port = int(os.getenv("QDRANT_PORT") or "6333")
    return {
        "url": (os.getenv("QDRANT_URL") or "").strip(),
        "api_key": (os.getenv("QDRANT_API_KEY") or "").strip() or None,
        "host": host or None,
        "port": port,
        "collection": (os.getenv("QDRANT_COLLECTION") or "test_chunks").strip(),
        "vector_size": int(os.getenv("QDRANT_VECTOR_SIZE") or "1024"),
    }


def get_qdrant_client() -> QdrantClient:
    cfg = qdrant_config_from_env()
    if cfg["url"]:
        return QdrantClient(url=cfg["url"], api_key=cfg["api_key"])
    host = cfg["host"] or "localhost"
    return QdrantClient(host=host, port=int(cfg["port"]))


def search_chunk_hits(
    query_vector: Sequence[float],
    *,
    limit: int = 8,
    workspace_id: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> List[dict]:
    """
    Поиск в Qdrant. Возвращает список dict: chunk_id (str из payload), score (float), point_id.
    chunk_id берётся из payload[\"chunk_id\"] (как в verify_rag_connections).
    """
    cfg = qdrant_config_from_env()
    name = (collection_name or cfg["collection"] or "").strip()
    if not name:
        return []

    client = get_qdrant_client()
    q_filter: Filter | None = None
    ws = (workspace_id or "").strip()
    if ws:
        q_filter = Filter(
            must=[
                FieldCondition(
                    key="workspace_id",
                    match=MatchValue(value=ws),
                )
            ]
        )

    hits = client.search(
        collection_name=name,
        query_vector=list(query_vector),
        limit=max(1, int(limit)),
        query_filter=q_filter,
        with_payload=True,
    )

    out: List[dict] = []
    for h in hits:
        pl = getattr(h, "payload", None) or {}
        if not isinstance(pl, dict):
            continue
        cid = pl.get("chunk_id")
        if cid is None:
            continue
        out.append(
            {
                "chunk_id": str(cid),
                "score": float(getattr(h, "score", 0.0) or 0.0),
                "point_id": getattr(h, "id", None),
            }
        )
    return out


def search_points_stub(
    _query_vector: Sequence[float],
    *,
    limit: int = 8,
    workspace_id: str | None = None,
) -> List[dict]:
    """Обратная совместимость: делегирует search_chunk_hits."""
    return search_chunk_hits(_query_vector, limit=limit, workspace_id=workspace_id)
