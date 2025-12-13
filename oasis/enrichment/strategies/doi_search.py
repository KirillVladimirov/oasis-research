"""Стратегия поиска по DOI."""

from typing import Any

from loguru import logger

from oasis.enrichment.sources.aggregator import SourceAggregator
from oasis.enrichment.sources.merger import merge_metadata


def search_by_doi(
    enriched: dict[str, Any],
    doi: str,
    aggregator: SourceAggregator,
    enrichment_log: dict[str, Any],
    trace_prefix: str = "",
) -> dict[str, Any]:
    """Выполняет поиск метаданных по DOI.
    
    Args:
        enriched: Обогащаемая ссылка
        doi: DOI для поиска
        aggregator: Агрегатор источников
        enrichment_log: Лог обогащения
        trace_prefix: Префикс для логов
        
    Returns:
        Обновленная ссылка
    """
    enrichment_log["steps"].append({
        "step": "doi_search",
        "doi": doi,
        "sources": ["openalex", "crossref", "semanticscholar"]
    })
    
    metadata = aggregator.get_by_doi(doi, sources=["openalex", "crossref", "semanticscholar"])
    
    if metadata:
        enriched = merge_metadata(enriched, metadata)
        enrichment_log["steps"][-1]["result"] = "success"
        enrichment_log["steps"][-1]["found"] = {
            "title": metadata.title,
            "year": metadata.year,
            "venue": metadata.venue,
            "doi": metadata.doi,
        }
        logger.debug(f"{trace_prefix}Обогащено по DOI: {doi}")
    else:
        enrichment_log["steps"][-1]["result"] = "not_found"
        enrichment_log["warnings"].append(f"DOI {doi} не найден ни в одном источнике")
        logger.debug(f"{trace_prefix}DOI {doi} не найден в источниках")
    
    return enriched

