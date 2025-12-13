"""Объединение метаданных из разных источников."""

from typing import Any

from loguru import logger

from oasis.sources.base import Metadata


def merge_metadata(base: dict[str, Any], metadata: Metadata | None) -> dict[str, Any]:
    """Объединяет метаданные из источника с базовыми данными ссылки.

    Args:
        base: Базовый словарь ссылки
        metadata: Metadata объект из источника

    Returns:
        Обновлённый словарь ссылки
    """
    if not metadata:
        return base

    merged = base.copy()

    # Обновляем только пустые поля
    if not merged.get("title") and metadata.title:
        merged["title"] = metadata.title

    if not merged.get("authors") and metadata.authors:
        merged["authors"] = metadata.authors

    if not merged.get("year") and metadata.year:
        merged["year"] = metadata.year

    if not merged.get("venue") and metadata.venue:
        merged["venue"] = metadata.venue

    if not merged.get("doi") and metadata.doi:
        merged["doi"] = metadata.doi

    if not merged.get("arxiv_id") and metadata.arxiv_id:
        merged["arxiv_id"] = metadata.arxiv_id

    # URLs: объединяем
    if metadata.article_url:
        if not merged.get("article_url"):
            merged["article_url"] = metadata.article_url
            logger.debug(
                f"merge_metadata: article_url применен: {metadata.article_url[:60]}..."
            )
        elif metadata.article_url not in (merged.get("additional_urls") or []):
            if not merged.get("additional_urls"):
                merged["additional_urls"] = []
            merged["additional_urls"].append(metadata.article_url)
            logger.debug(
                f"merge_metadata: article_url добавлен в additional_urls (article_url уже был): {metadata.article_url[:60]}..."
            )

    if metadata.pdf_url:
        # PDF URL обрабатывается отдельно с валидацией
        if not merged.get("pdf_url"):
            merged["pdf_url"] = metadata.pdf_url
            logger.debug(
                f"merge_metadata: pdf_url применен: {metadata.pdf_url[:60]}..."
            )

    if metadata.additional_urls:
        if not merged.get("additional_urls"):
            merged["additional_urls"] = []
        for url in metadata.additional_urls:
            if url and url not in merged["additional_urls"]:
                merged["additional_urls"].append(url)

    return merged

