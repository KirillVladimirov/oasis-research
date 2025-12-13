"""Стадия 2: Обогащение ссылок внешними источниками."""

import json
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from oasis.enrichment.dedupe import dedupe_references
from oasis.enrichment.metrics import calculate_reference_metrics
from oasis.enrichment.pipeline import setup_enrichment_sources
from oasis.enrichment.validate import validate_pdf_matches_reference
from oasis.enrichment.workflow import EnrichmentWorkflow
from oasis.io.writers import save_references
from oasis.models.reference import Reference
from oasis.parsing.normalization import normalize_reference
from oasis.utils.ids import generate_ref_id


def enrich_references(
    raw_refs: list[dict[str, Any]],
    topic: str,
    trace_id: str | None = None,
    progress_callback: Callable[[str, float, str], None] | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Обогащение ссылок с использованием SourceAggregator.

    Args:
        raw_refs: Список сырых ссылок из стадии 1
        topic: Название темы исследования
        trace_id: Идентификатор трейса для логирования
        progress_callback: Функция callback для обновления прогресса

    Returns:
        Кортеж: (путь к CSV, список обогащённых ссылок, словарь метрик)
    """
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    logger.info(f"{trace_prefix}Стадия 2: Обогащение {len(raw_refs)} ссылок")
    t_total = time.perf_counter()
    timings_enrich: dict[str, float] = {}

    def _progress(stage: str, progress: float, message: str = ""):
        if progress_callback:
            progress_callback(stage, progress, message)

    # Настройка источников
    _progress("setup", 0.0, "Настройка источников данных...")
    aggregator = setup_enrichment_sources()

    # Создание сессии для валидации
    def make_session(user_agent: str) -> requests.Session:
        s = requests.Session()
        retries = Retry(
            total=4,
            connect=3,
            read=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
        return s

    sess = make_session("OASIS-Enrich/0.1")

    # Обёртка для валидации PDF
    def validate_pdf_wrapper(pdf_url: str, expected: dict[str, Any]) -> dict[str, Any]:
        return validate_pdf_matches_reference(
            session=sess,
            pdf_url=pdf_url,
            expected=expected,
            source_hint=None,
            get_crossref_fn=None,
            arxiv_api_url="https://export.arxiv.org/api/query",
        )

    # Создаем EnrichmentWorkflow с Selenium
    workflow = EnrichmentWorkflow(
        aggregator=aggregator,
        validate_pdf_fn=validate_pdf_wrapper,
    )
    # Включаем Selenium поиск
    workflow.enable_selenium_search(delay_seconds=8.0)
    
    # Обогащение ссылок
    t_loop = time.perf_counter()
    enriched_refs = []
    for idx, ref in enumerate(raw_refs):
        if progress_callback:
            progress = 0.1 + (idx / len(raw_refs)) * 0.7
            _progress("enrich", progress, f"Обогащение ссылки {idx + 1}/{len(raw_refs)}...")

        # Используем новый workflow с Selenium
        enriched = workflow.execute(
            ref,
            trace_id=trace_id,
        )

        # Генерация ref_id и нормализация
        enriched["ref_id"] = generate_ref_id(enriched, trace_id=trace_id)
        enriched = normalize_reference(enriched, trace_id=trace_id)

        enriched_refs.append(enriched)

    timings_enrich["enrich_loop_sec"] = time.perf_counter() - t_loop

    # Сохранение кэша
    cache = aggregator.cache
    if cache:
        cache.save()

    # Дедупликация
    t_dedupe = time.perf_counter()
    _progress("dedupe", 0.9, "Дедупликация обогащённых ссылок...")
    deduped_refs = dedupe_references(enriched_refs, trace_id=trace_id)
    timings_enrich["dedupe_sec"] = time.perf_counter() - t_dedupe

    # Сохранение в CSV
    t_save = time.perf_counter()
    _progress("save", 0.95, "Сохранение в CSV...")
    output_dir = Path(f"data/{topic}")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "references.csv"

    # Преобразуем в Reference объекты для сохранения
    refs_objects = [Reference.from_dict(ref) for ref in deduped_refs]
    save_references(refs_objects, csv_path)
    
    # Сохранение детальных логов обогащения в JSON
    logs_path = output_dir / "enrichment_logs.json"
    logs_data = []
    for ref in deduped_refs:
        title_value = ref.get("title", "") or ""
        if not isinstance(title_value, str):
            title_value = str(title_value)
        log_entry = {
            "ref_number": ref.get("ref_number"),
            "title": title_value[:100],  # Первые 100 символов
            "pdf_found": bool(ref.get("pdf_url")),
            "pdf_url": ref.get("pdf_url", ""),
            "doi": ref.get("doi", ""),
            "arxiv_id": ref.get("arxiv_id", ""),
            "year": ref.get("year"),
            "enrichment_log": ref.get("_enrichment_log", {}),
        }
        logs_data.append(log_entry)
    
    try:
        with open(logs_path, "w", encoding="utf-8") as f:
            json.dump(logs_data, f, ensure_ascii=False, indent=2)
        logger.info(f"{trace_prefix}Детальные логи сохранены в: {logs_path}")
    except Exception as e:
        logger.warning(f"{trace_prefix}Не удалось сохранить детальные логи: {e}")

    timings_enrich["save_csv_sec"] = time.perf_counter() - t_save
    timings_enrich["total_sec"] = time.perf_counter() - t_total

    # Расчёт метрик
    df_deduped = pd.DataFrame(deduped_refs)
    metrics = calculate_reference_metrics(df_deduped)
    metrics["timings"] = timings_enrich
    total_count = metrics["total"]

    _progress("complete", 1.0, f"Завершено: {total_count} обогащённых ссылок")
    logger.info(
        f"{trace_prefix}Стадия 2 завершена: {total_count} ссылок, "
        f"с DOI/arXiv: {metrics.get('with_doi_or_arxiv', 0)} ({metrics.get('pct_with_id', 0):.1f}%)"
    )
    logger.info(f"{trace_prefix}Детальные логи JSON: {logs_path}")

    return str(csv_path), deduped_refs, metrics

