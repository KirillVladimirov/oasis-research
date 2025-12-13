"""Pipeline для обогащения ссылок."""

import os
from typing import Any, Callable

from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests

from oasis.enrichment.sources.aggregator import SourceAggregator
from oasis.enrichment.validate import validate_pdf_matches_reference
from oasis.enrichment.workflow import EnrichmentWorkflow
from oasis.io import get_cache, get_registry
from oasis.sources import (
    ArxivSource,
    CrossrefSource,
    OpenAlexSource,
    SemanticScholarSource,
    UnpaywallSource,
)


def setup_enrichment_sources(cache=None) -> SourceAggregator:
    """Настраивает и регистрирует источники данных для обогащения.

    Args:
        cache: Экземпляр кэша (по умолчанию глобальный)

    Returns:
        SourceAggregator с настроенными источниками
    """
    if cache is None:
        cache = get_cache()

    registry = get_registry()

    # Регистрируем все источники
    openalex_mailto = os.getenv("OPENALEX_MAILTO", "").strip()
    registry.register("openalex", OpenAlexSource(cache=cache), enabled=True)
    registry.register("crossref", CrossrefSource(cache=cache), enabled=True)
    registry.register("semanticscholar", SemanticScholarSource(cache=cache), enabled=True)
    registry.register("arxiv", ArxivSource(cache=cache), enabled=True)

    unpaywall_mailto = os.getenv("UNPAYWALL_MAILTO", openalex_mailto)
    if unpaywall_mailto:
        registry.register("unpaywall", UnpaywallSource(cache=cache), enabled=True)
    else:
        registry.register("unpaywall", UnpaywallSource(cache=cache), enabled=False)

    return SourceAggregator(registry=registry, cache=cache)


def enrich_single_reference(
    reference: dict[str, Any],
    aggregator: SourceAggregator,
    *,
    validate_pdf_fn: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    trace_id: str | None = None,
    enable_selenium: bool = False,
    selenium_delay: float = 8.0,
) -> dict[str, Any]:
    """Совместимый обёртка над EnrichmentWorkflow для обогащения одной ссылки.

    Args:
        reference: Словарь с полями ссылки (doi/title/...).
        aggregator: Настроенный SourceAggregator.
        validate_pdf_fn: Кастомная функция валидации PDF (опционально).
        trace_id: Идентификатор для логов.
        enable_selenium: Включить ли Selenium-поиск (False по умолчанию).
        selenium_delay: Задержка между Selenium-запросами (секунды).

    Returns:
        Обогащённая ссылка (dict).
    """

    logger.warning(
        "enrich_single_reference() устарела; используйте EnrichmentWorkflow напрямую. "
        "Функция сохранена для обратной совместимости."
    )
    workflow = EnrichmentWorkflow(aggregator=aggregator, validate_pdf_fn=validate_pdf_fn)
    if enable_selenium:
        workflow.enable_selenium_search(delay_seconds=selenium_delay)
    return workflow.execute(reference, trace_id=trace_id)

