"""Извлечение библиографических ссылок из PDF обзоров через GROBID и fallback-парсер."""

import csv
import hashlib
import re
from pathlib import Path
import time
from typing import Any, Callable
import os
import json

import fitz  # PyMuPDF
import pandas as pd
import requests
from lxml import etree
from loguru import logger
from rapidfuzz import fuzz
from slugify import slugify
from tenacity import retry, stop_after_attempt, wait_exponential
from unidecode import unidecode

from oasis.enrichment.metrics import calculate_reference_metrics
from oasis.enrichment.validate import (
    validate_url as validate_url_impl,
    validate_pdf_url as validate_pdf_url_impl,
    validate_pdf_matches_reference as validate_pdf_matches_reference_impl,
)
from oasis.parsing.guardrails import filter_phantom_references, validate_ref_number
from oasis.parsing.normalization import normalize_authors, normalize_reference
from oasis.parsing.pdf_refs import (
    GrobidClient,
    extract_references_from_pdf_text,
    parse_bibl_struct,
    parse_grobid_fulltext,
    parse_grobid_references,
)
from oasis.parsing.regexes import find_arxiv_id, find_doi, find_urls, parse_references_regex
from oasis.utils.text import clean_text_value, normalize_text, validate_doi_format

# Реэкспорт для обратной совместимости
__all__ = [
    "extract_reference_titles",
    "enrich_references",
    "dedupe_references",
    "parse_grobid_references",
    "parse_grobid_fulltext",
    "parse_references_regex",
    "extract_references_from_pdf_text",
    "parse_bibl_struct",
    "_parse_bibl_struct",  # Для обратной совместимости
    "normalize_reference",
    "normalize_authors",
    "_normalize_authors",  # Для обратной совместимости
    "generate_ref_id",
    "merge_info",
    "clean_text_value",
    "normalize_text",
    "validate_doi_format",
    "find_doi",
    "find_arxiv_id",
    "find_urls",
    "GrobidClient",
    "filter_phantom_references",
    "validate_ref_number",
]

GROBID_NAMESPACE = "http://www.tei-c.org/ns/1.0"
GROBID_NSMAP = {"tei": GROBID_NAMESPACE}


# Реэкспорт для обратной совместимости (функции импортированы из oasis.utils.text)

# Старая функция extract_reference_titles удалена - используйте oasis.pipelines.stage1_extract.extract_reference_titles
# Реэкспорт для обратной совместимости:
from oasis.pipelines.stage1_extract import extract_reference_titles

# Удалено тело старой функции extract_reference_titles (было ~400 строк)
# Оставлен только реэкспорт выше

# Старая функция merge_info удалена - используйте oasis.enrichment.merge.merge_info
# Реэкспорт для обратной совместимости:
from oasis.enrichment.merge import merge_info

# Старая функция generate_ref_id удалена - используйте oasis.utils.ids.generate_ref_id
# Реэкспорт для обратной совместимости:
from oasis.utils.ids import generate_ref_id

# Старая функция enrich_references удалена - используйте oasis.pipelines.stage2_enrich.enrich_references
# Для обратной совместимости оставляем реэкспорт:
from oasis.pipelines.stage2_enrich import enrich_references

# Старая функция dedupe_references удалена - используйте oasis.enrichment.dedupe.dedupe_references
# Реэкспорт для обратной совместимости:
from oasis.enrichment.dedupe import dedupe_references

# Реэкспорт для обратной совместимости
from oasis.pipelines.stage1_extract import extract_reference_titles

# parse_grobid_fulltext импортирована из oasis.parsing.pdf_refs


# extract_references_from_pdf_text импортирована из oasis.parsing.pdf_refs


# parse_grobid_references импортирована из oasis.parsing.pdf_refs


# _parse_bibl_struct - псевдоним для обратной совместимости
_parse_bibl_struct = parse_bibl_struct


# parse_references_regex импортирована из oasis.parsing.regexes


# clean_text_value импортирована из oasis.utils.text


# _normalize_authors - псевдоним для обратной совместимости
_normalize_authors = normalize_authors

# Реэкспорт для обратной совместимости
from oasis.enrichment.merge import merge_info


# validate_doi_format импортирована из oasis.utils.text


# normalize_reference импортирована из oasis.parsing.normalization


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


# Старая функция enrich_references удалена - используйте oasis.pipelines.stage2_enrich.enrich_references
# Реэкспорт уже сделан выше (строка 90)

# Старая функция dedupe_references удалена - используйте oasis.enrichment.dedupe.dedupe_references
# Реэкспорт уже сделан выше (строка 94)


def extract_references_from_pdf(
    pdf_path: str,
    topic: str,
    trace_id: str | None = None,
    progress_callback: Callable[[str, float, str], None] | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Главная функция: извлечение, нормализация, дедупликация и сохранение ссылок.

    Args:
        pdf_path: Путь к PDF файлу обзора
        topic: Название темы исследования
        trace_id: Идентификатор трейса для логирования
        progress_callback: Функция callback для обновления прогресса

    Returns:
        Кортеж: (путь к CSV, список обогащённых ссылок, словарь метрик)
    """
    # Используем новую функцию из pipelines
    csv_path, raw_refs = extract_reference_titles(pdf_path, topic, trace_id, progress_callback)
    enriched_csv_path, enriched_refs, metrics = enrich_references(raw_refs, topic, trace_id, progress_callback)
    return enriched_csv_path, enriched_refs, metrics


def extract_references_from_pdf(
    pdf_path: str,
    topic: str,
    trace_id: str | None = None,
    progress_callback: Callable[[str, float, str], None] | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Главная функция: извлечение, нормализация, дедупликация и сохранение ссылок.

    Args:
        pdf_path: Путь к PDF файлу обзора
        topic: Название темы исследования
        trace_id: Идентификатор трейса для логирования
        progress_callback: Функция callback для обновления прогресса (stage, progress, message)

    Returns:
        Кортеж: (путь к CSV, список ссылок, словарь метрик)
    """
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    logger.info(
        f"{trace_prefix}Начало извлечения ссылок из {pdf_path} для темы '{topic}'"
    )

    def _progress(stage: str, progress: float, message: str = ""):
        if progress_callback:
            progress_callback(stage, progress, message)

    # Попытка через GROBID
    _progress("grobid", 0.1, "Отправка PDF в GROBID...")
    refs = parse_grobid_references(pdf_path, trace_id=trace_id)

    # Fallback на regex при отсутствии результатов
    if not refs:
        logger.warning(f"{trace_prefix}GROBID не вернул результатов, используем regex fallback")
        _progress("regex", 0.1, "GROBID недоступен, использование regex fallback...")
        refs = parse_references_regex(pdf_path, trace_id=trace_id)

    if not refs:
        logger.warning(f"{trace_prefix}Не удалось извлечь ссылки из PDF")
        refs = []

    _progress("normalize", 0.5, f"Нормализация {len(refs)} ссылок...")
    # Нормализация
    logger.info(f"{trace_prefix}Нормализация {len(refs)} ссылок")
    normalized_refs = [normalize_reference(ref, trace_id=trace_id) for ref in refs]

    _progress("dedupe", 0.7, "Дедупликация ссылок...")
    # Дедупликация
    deduped_refs = dedupe_references(normalized_refs, trace_id=trace_id)

    _progress("generate_ids", 0.8, "Генерация идентификаторов...")
    # Генерация ref_id
    for ref in deduped_refs:
        ref["ref_id"] = generate_ref_id(ref, trace_id=trace_id)

    # Подготовка данных для CSV
    csv_data = []
    for ref in deduped_refs:
        csv_data.append(
            {
                "ref_id": ref.get("ref_id", ""),
                "title": ref.get("title", ""),
                "year": ref.get("year") if ref.get("year") else "",
                "doi": ref.get("doi", ""),
                "arxiv_id": ref.get("arxiv_id", ""),
                "venue": ref.get("venue", ""),
                "first_author": ref.get("first_author", ""),
            }
        )

    # Сохранение в CSV
    output_dir = Path(f"data/{topic}")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "references.csv"

    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # Статистика
    total = len(deduped_refs)
    with_doi_or_arxiv = sum(
        1 for r in deduped_refs if r.get("doi") or r.get("arxiv_id")
    )
    pct_with_id = (with_doi_or_arxiv / total * 100) if total > 0 else 0

    logger.info(
        f"{trace_prefix}Сохранено {total} ссылок в {csv_path}\n"
        f"  - С DOI или arXiv ID: {with_doi_or_arxiv} ({pct_with_id:.1f}%)"
    )

    if total > 0 and pct_with_id < 70:
        logger.warning(
            f"{trace_prefix}Только {pct_with_id:.1f}% ссылок имеют DOI/arXiv ID "
            f"(требуется ≥70%)"
        )

    # Формирование метрик
    metrics = {
        "total": total,
        "with_doi": sum(1 for r in deduped_refs if r.get("doi")),
        "with_arxiv_id": sum(1 for r in deduped_refs if r.get("arxiv_id")),
        "with_doi_or_arxiv": with_doi_or_arxiv,
        "pct_with_id": pct_with_id,
        "with_title": sum(1 for r in deduped_refs if r.get("title")),
        "with_year": sum(1 for r in deduped_refs if r.get("year")),
        "with_venue": sum(1 for r in deduped_refs if r.get("venue")),
        "with_author": sum(1 for r in deduped_refs if r.get("first_author")),
    }

    _progress("complete", 1.0, f"Завершено: {total} ссылок")
    return str(csv_path), deduped_refs, metrics

