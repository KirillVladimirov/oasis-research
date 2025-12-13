"""Стратегия поиска PDF через Unpaywall."""

from typing import Any

from loguru import logger

from oasis.enrichment.sources.aggregator import SourceAggregator
from oasis.enrichment.sources.merger import merge_metadata


def search_unpaywall_pdf(
    enriched: dict[str, Any],
    doi: str,
    aggregator: SourceAggregator,
    enrichment_log: dict[str, Any],
    trace_prefix: str = "",
) -> dict[str, Any]:
    """Выполняет поиск PDF через Unpaywall.
    
    Args:
        enriched: Обогащаемая ссылка
        doi: DOI для поиска
        aggregator: Агрегатор источников
        enrichment_log: Лог обогащения
        trace_prefix: Префикс для логов
        
    Returns:
        Обновленная ссылка
    """
    if not enriched.get("pdf_url"):
        enrichment_log["steps"].append({"step": "unpaywall_pdf", "doi": doi})
        unpaywall_meta = aggregator.get_by_doi(doi, sources=["unpaywall"])
        
        if unpaywall_meta and unpaywall_meta.pdf_url:
            enriched = merge_metadata(enriched, unpaywall_meta)
            enrichment_log["steps"][-1]["result"] = "success"
            enrichment_log["steps"][-1]["pdf_url"] = unpaywall_meta.pdf_url
            logger.debug(f"{trace_prefix}Найден PDF через Unpaywall: {doi}")
        else:
            enrichment_log["steps"][-1]["result"] = "not_found"
            logger.debug(f"{trace_prefix}PDF не найден через Unpaywall для DOI: {doi}")
    
    return enriched

