"""Утилиты для генерации ID ссылок."""

import hashlib
from typing import Any

from loguru import logger
from slugify import slugify


def generate_ref_id(ref: dict[str, Any], trace_id: str | None = None) -> str:
    """Генерация уникального ID для ссылки.

    Приоритет: arxiv_id → slugify(doi) → slugify(title) + hash

    Args:
        ref: Словарь с полями ссылки
        trace_id: Идентификатор трейса для логирования

    Returns:
        Строковый ID
    """
    if ref.get("arxiv_id"):
        return ref["arxiv_id"]

    if ref.get("doi"):
        return slugify(ref["doi"], lowercase=False)

    # Генерация из title + hash
    title = ref.get("title", "")
    if title:
        title_slug = slugify(title, max_length=50)
        title_hash = hashlib.sha256(title.encode("utf-8")).hexdigest()[:8]
        return f"{title_slug}-{title_hash}"

    # Fallback: hash всего словаря
    ref_str = str(sorted(ref.items()))
    ref_hash = hashlib.sha256(ref_str.encode("utf-8")).hexdigest()[:12]
    return f"ref-{ref_hash}"

