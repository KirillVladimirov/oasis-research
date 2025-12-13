"""Извлечение библиографических ссылок из PDF (API обертка)."""

from typing import Any

import requests
from loguru import logger

from oasis.parsing.grobid import GrobidClient, parse_bibl_struct
from oasis.parsing.grobid.xml_parser import GROBID_NAMESPACE, GROBID_NSMAP
from oasis.parsing.extractors import extract_references_from_pdf_text as direct_extract

# Реэкспорт для обратной совместимости
__all__ = [
    "GrobidClient",
    "parse_bibl_struct",
    "parse_grobid_references",
    "parse_grobid_fulltext",
    "extract_references_from_pdf_text",
    "GROBID_NAMESPACE",
    "GROBID_NSMAP",
]


def parse_grobid_references(
    pdf_path: str, grobid_url: str = "http://localhost:8070", trace_id: str | None = None
) -> list[dict[str, Any]]:
    """Извлечь ссылки из PDF через GROBID API.

    Args:
        pdf_path: Путь к PDF файлу
        grobid_url: URL GROBID сервиса
        trace_id: Идентификатор трейса для логирования

    Returns:
        Список словарей с полями ссылок
    """
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    logger.info(f"{trace_prefix}Отправка PDF в GROBID: {pdf_path}")

    client = GrobidClient(grobid_url)
    if not client.is_alive():
        logger.warning(f"{trace_prefix}GROBID недоступен, пропускаем")
        return []

    try:
        return client.process_references(pdf_path)
    except (requests.exceptions.RequestException, Exception) as e:
        logger.warning(f"{trace_prefix}Ошибка запроса к GROBID: {e}")
        return []


def parse_grobid_fulltext(
    pdf_path: str, grobid_url: str = "http://localhost:8070", trace_id: str | None = None
) -> list[dict[str, Any]]:
    """Извлечь все ссылки из PDF через GROBID processFulltextDocument.

    Args:
        pdf_path: Путь к PDF файлу
        grobid_url: URL GROBID сервиса
        trace_id: Идентификатор трейса для логирования

    Returns:
        Список словарей с полями ссылок
    """
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    logger.info(f"{trace_prefix}Извлечение через processFulltextDocument: {pdf_path}")

    client = GrobidClient(grobid_url)
    if not client.is_alive():
        logger.warning(f"{trace_prefix}GROBID недоступен")
        return []

    try:
        return client.process_fulltext(pdf_path)
    except (requests.exceptions.RequestException, Exception) as e:
        logger.warning(f"{trace_prefix}Ошибка запроса к GROBID: {e}")
        return []


def extract_references_from_pdf_text(
    pdf_path: str, trace_id: str | None = None
) -> list[dict[str, Any]]:
    """Прямое извлечение раздела References из PDF текста.

    Тонкая обертка над extractors.extract_references_from_pdf_text.

    Args:
        pdf_path: Путь к PDF файлу
        trace_id: Идентификатор трейса для логирования

    Returns:
        Список словарей с сырыми данными ссылок
    """
    return direct_extract(pdf_path, trace_id)
