"""Извлечение идентификаторов и заголовков из ссылок."""

import math
import re
from typing import Any

from oasis.parsing.regexes import find_arxiv_id, find_doi

try:  # noqa: SIM105 - допускаем любые исключения при импорте pandas
    import pandas as _pd  # type: ignore
except Exception:  # pragma: no cover - pandas может быть недоступен при тестах
    _pd = None  # type: ignore


def safe_str(value: Any) -> str:
    """Безопасно преобразует значение в строку, возвращая пустую строку для NaN/None."""

    if value is None:
        return ""

    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.lower() in {"nan", "none", "null", ""}:
            return ""
        return cleaned

    if isinstance(value, (int, bool)):
        return str(value)

    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return str(value).strip()

    if isinstance(value, (list, tuple, set)):
        parts = [safe_str(v) for v in value]
        joined = "; ".join([p for p in parts if p])
        return joined

    if _pd is not None:
        try:
            if _pd.isna(value):
                return ""
        except Exception:
            pass

    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _validate_arxiv_id(arxiv_id: str) -> str | None:
    """Валидирует и нормализует arXiv ID."""
    if not arxiv_id:
        return None

    # Очищаем от версии
    arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id

    # Проверяем формат: YYYY.NNNNN или YYYY.NNNNNN
    if re.match(r"^\d{4}\.\d{4,6}$", arxiv_id_clean):
        return arxiv_id_clean

    return None


def extract_arxiv_id_from_ref(ref: dict[str, Any]) -> str | None:
    """Извлекает arXiv ID из ссылки.

    Проверяет все возможные поля в правильном порядке приоритета.

    Args:
        ref: Словарь с данными ссылки

    Returns:
        arXiv ID (формат YYYY.NNNNN) или None
    """
    # Приоритет 1: DOI (самый надежный источник arXiv ID)
    doi = safe_str(ref.get("doi"))
    if doi and ("arxiv" in doi.lower() or "10.48550/arxiv" in doi.lower()):
        arxiv_id_from_doi = find_arxiv_id(doi)
        if arxiv_id_from_doi:
            validated = _validate_arxiv_id(arxiv_id_from_doi)
            if validated:
                return validated

    # Приоритет 2: article_url
    article_url = safe_str(ref.get("article_url"))
    if article_url:
        arxiv_id_from_url = find_arxiv_id(article_url)
        if arxiv_id_from_url:
            validated = _validate_arxiv_id(arxiv_id_from_url)
            if validated:
                return validated

    # Приоритет 3: pdf_url
    pdf_url = safe_str(ref.get("pdf_url"))
    if pdf_url:
        arxiv_id_from_url = find_arxiv_id(pdf_url)
        if arxiv_id_from_url:
            validated = _validate_arxiv_id(arxiv_id_from_url)
            if validated:
                return validated

    # Приоритет 4: raw_text
    raw_text = safe_str(ref.get("raw_text"))
    if raw_text:
        arxiv_id_from_text = find_arxiv_id(raw_text)
        if arxiv_id_from_text:
            validated = _validate_arxiv_id(arxiv_id_from_text)
            if validated:
                return validated

    # Приоритет 5: прямое поле arxiv_id (только если прошло валидацию)
    arxiv_id = safe_str(ref.get("arxiv_id"))
    if arxiv_id:
        validated = _validate_arxiv_id(arxiv_id)
        if validated:
            return validated

    return None


def extract_doi_from_ref(ref: dict[str, Any]) -> str | None:
    """Извлекает DOI из ссылки.

    Args:
        ref: Словарь с данными ссылки

    Returns:
        DOI или None
    """
    # Проверяем прямое поле doi
    doi = safe_str(ref.get("doi"))
    if doi:
        # Очищаем от префикса https://doi.org/
        doi_clean = (
            doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
        )
        if doi_clean and doi_clean.startswith("10."):
            return doi_clean

    # Проверяем article_url
    article_url = safe_str(ref.get("article_url"))
    if article_url:
        doi_from_url = find_doi(article_url)
        if doi_from_url:
            return doi_from_url

    # Проверяем raw_text
    raw_text = safe_str(ref.get("raw_text"))
    if raw_text:
        doi_from_text = find_doi(raw_text)
        if doi_from_text:
            return doi_from_text

    return None


def extract_title_from_ref(ref: dict[str, Any]) -> str | None:
    """Извлекает title из ссылки.

    Args:
        ref: Словарь с данными ссылки

    Returns:
        Title или None
    """
    title = safe_str(ref.get("title"))
    if title and len(title) >= 10:  # Минимальная длина title
        return title
    return None
