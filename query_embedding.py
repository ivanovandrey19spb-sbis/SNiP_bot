"""
Эмбеддинг запроса пользователя (HF transformers), модель по умолчанию — ai-forever/ru-en-RoSBERTa.

Mean pooling по last_hidden_state + L2-нормализация (распространённый паттерн для семантического поиска).
Размерность берётся из config модели; должна совпадать с QDRANT_VECTOR_SIZE / коллекцией.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional

log = logging.getLogger(__name__)

_model = None
_tokenizer = None
_embedding_dim: Optional[int] = None
_load_lock = threading.Lock()


def _model_name() -> str:
    return (os.getenv("EMBEDDING_MODEL") or os.getenv("HF_EMBED_MODEL") or "ai-forever/ru-en-RoSBERTa").strip()


def _device() -> str:
    d = (os.getenv("HF_DEVICE") or "cpu").strip().lower()
    if d in ("auto", ""):
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return d


def _mean_pooling(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def is_loaded() -> bool:
    return _model is not None


def _load():
    global _model, _tokenizer, _embedding_dim
    if _model is not None:
        return
    with _load_lock:
        if _model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        name = _model_name()
        log.info("Загрузка модели эмбеддинга: %s", name)
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
        model.eval()
        model.to(_device())
        dim = int(getattr(model.config, "hidden_size", 0) or 0)
        if dim <= 0:
            raise RuntimeError("Не удалось определить hidden_size модели эмбеддинга")
        _tokenizer = tokenizer
        _model = model
        _embedding_dim = dim


def warmup(*, probe: bool = True) -> int:
    """Идемпотентно загружает модель; при probe — один короткий forward."""
    _load()
    dim = embedding_dimension()
    if probe:
        embed_query("warmup", max_length=32)
    return dim


def embedding_dimension() -> int:
    _load()
    assert _embedding_dim is not None
    return _embedding_dim


def embed_query(text: str, *, max_length: int | None = None) -> List[float]:
    """
    Один текст → один вектор (float list).
    """
    import torch

    _load()
    assert _model is not None and _tokenizer is not None

    ml = max_length if max_length is not None else int(os.getenv("HF_MAX_LENGTH", "512"))
    t = (text or "").strip()
    if not t:
        raise ValueError("Пустой текст для эмбеддинга")

    enc = _tokenizer(
        [t],
        padding=True,
        truncation=True,
        max_length=ml,
        return_tensors="pt",
    )
    enc = {k: v.to(_device()) for k, v in enc.items()}

    with torch.no_grad():
        out = _model(**enc)
    emb = _mean_pooling(out.last_hidden_state, enc["attention_mask"])
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    vec = emb[0].detach().float().cpu().tolist()

    expected = int(os.getenv("QDRANT_VECTOR_SIZE") or "0")
    if expected and len(vec) != expected:
        raise RuntimeError(
            f"Размерность эмбеддинга {len(vec)} не совпадает с QDRANT_VECTOR_SIZE={expected}. "
            "Согласуйте коллекцию Qdrant и модель с DE."
        )
    return vec
