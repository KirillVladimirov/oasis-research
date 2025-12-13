"""Нормализация полей библиографических ссылок."""

import re
from typing import Any

from unidecode import unidecode

from oasis.utils.text import clean_text_value, normalize_text, validate_doi_format


def normalize_authors(authors: Any) -> str:
    """Нормализация авторов из различных форматов в строку.

    Args:
        authors: Может быть списком, строкой или None

    Returns:
        Нормализованная строка с авторами через "; "
    """
    if isinstance(authors, list):
        return "; ".join([clean_text_value(a) for a in authors if a])
    return clean_text_value(authors)


def normalize_reference(ref: dict[str, Any], trace_id: str | None = None) -> dict[str, Any]:
    """Нормализация полей ссылки.

    Args:
        ref: Словарь с полями ссылки
        trace_id: Идентификатор трейса для логирования

    Returns:
        Нормализованный словарь
    """
    normalized = ref.copy()

    # Нормализация title и raw_text
    if normalized.get("title"):
        title_raw = normalized["title"]
        if isinstance(title_raw, str):
            normalized["title"] = normalize_text(unidecode(title_raw))
        else:
            normalized["title"] = ""

    if normalized.get("raw_text"):
        raw_text_raw = normalized.get("raw_text", "")
        if isinstance(raw_text_raw, str):
            normalized["raw_text"] = normalize_text(unidecode(raw_text_raw))
        else:
            normalized["raw_text"] = ""

    # Нормализация authors
    if isinstance(normalized.get("authors"), list):
        normalized["authors"] = [
            normalize_text(unidecode(str(a))) for a in normalized["authors"] if a
        ]
        if normalized.get("first_author"):
            normalized["first_author"] = normalize_text(
                unidecode(str(normalized["first_author"]))
            )
        else:
            normalized["first_author"] = ""

    # Нормализация DOI
    if normalized.get("doi") is not None:
        doi_raw = normalized.get("doi")
        try:
            import pandas as _pd

            is_nan = _pd.isna(doi_raw)
        except Exception:
            is_nan = False

        doi_str = "" if doi_raw is None or is_nan else str(doi_raw)
        if doi_str:
            # Удаляем префикс https://doi.org/
            doi_str = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi_str, flags=re.IGNORECASE)
            doi_clean = doi_str.lower().strip()
            # Проверяем валидность формата DOI
            if validate_doi_format(doi_clean):
                normalized["doi"] = doi_clean
            else:
                normalized["doi"] = ""
        else:
            normalized["doi"] = ""

    # Нормализация arXiv ID (формат: YYYY.NNNNN)
    if normalized.get("arxiv_id") is not None:
        arxiv_raw = normalized.get("arxiv_id")
        try:
            import pandas as _pd

            is_nan = _pd.isna(arxiv_raw)
        except Exception:
            is_nan = False

        arxiv_str = "" if arxiv_raw is None or is_nan else str(arxiv_raw)
        if arxiv_str:
            # Убираем префиксы и оставляем только формат YYYY.NNNNN
            arxiv_match = re.search(r"(\d{4}\.\d{4,})", arxiv_str)
            if arxiv_match:
                normalized["arxiv_id"] = arxiv_match.group(1)
            else:
                normalized["arxiv_id"] = arxiv_str
        else:
            normalized["arxiv_id"] = ""

    # Нормализация venue
    if normalized.get("venue") is not None:
        venue_raw = normalized.get("venue")
        if isinstance(venue_raw, str):
            normalized["venue"] = normalize_text(unidecode(venue_raw))
        else:
            normalized["venue"] = ""

    return normalized

