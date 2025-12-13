"""Расчёт метрик по обогащённым ссылкам."""

from typing import Any, Callable

import pandas as pd


def calculate_reference_metrics(
    df: pd.DataFrame,
    extract_pdf_fn: Callable[[str], str] | None = None,
    is_valid_pdf_fn: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Рассчитывает метрики по обогащённым ссылкам из DataFrame.

    Args:
        df: DataFrame с колонками: doi, arxiv_id, title, year, venue, authors, pdf_url, additional_urls
        extract_pdf_fn: Опциональная функция для извлечения PDF из additional_urls (если pdf_url пуст)
        is_valid_pdf_fn: Опциональная функция для валидации PDF URL

    Returns:
        Словарь с метриками:
        {
            "total": int,
            "with_doi": int,
            "with_arxiv_id": int,
            "with_title": int,
            "with_year": int,
            "with_venue": int,
            "with_author": int,
            "with_pdf": int,
            "pct_doi": float,
            "pct_arxiv": float,
            "with_doi_or_arxiv": int,
            "pct_with_id": float,
        }
    """
    total = len(df)

    # Подсчёт полей (используем комбинацию .notna() и проверки на пустые строки)
    def _count_non_empty(series: pd.Series) -> int:
        """Подсчитывает непустые значения (не NaN и не пустая строка)."""
        count = 0
        for val in series:
            # Проверяем на None
            if val is None:
                continue
            # Проверяем на pandas/float NaN
            try:
                if pd.isna(val):
                    continue
            except (TypeError, ValueError):
                pass
            # Проверяем на float NaN (math.isnan)
            if isinstance(val, float):
                try:
                    import math
                    if math.isnan(val):
                        continue
                except (TypeError, ValueError, ImportError):
                    pass
            # Преобразуем в строку и проверяем на пустоту
            val_str = str(val).strip()
            if val_str and val_str.lower() not in {"nan", "none", "null", ""}:
                count += 1
        return count
    
    with_doi = _count_non_empty(df["doi"]) if "doi" in df.columns else 0
    with_arxiv_id = _count_non_empty(df["arxiv_id"]) if "arxiv_id" in df.columns else 0
    with_title = _count_non_empty(df["title"]) if "title" in df.columns else 0
    # Для year используем .notna(), так как это числовое поле
    with_year = df["year"].notna().sum() if "year" in df.columns else 0
    with_venue = _count_non_empty(df["venue"]) if "venue" in df.columns else 0
    with_author = _count_non_empty(df["authors"]) if "authors" in df.columns else 0

    # Подсчёт PDF: считаем непустые pdf_url (без валидации - все уже проверено на этапе 2)
    # ВАЖНО: Не валидируем PDF URL для метрик - просто считаем непустые значения
    pdf_count = 0
    if "pdf_url" in df.columns:
        for _, row in df.iterrows():
            pdf_url = str(row.get("pdf_url", "")).strip()
            # Простая проверка: непустой URL с правильным форматом (без HTTP-запросов)
            if pdf_url and pdf_url.lower() not in {"nan", "none", "null", ""}:
                if pdf_url.startswith(("http://", "https://")):
                    pdf_count += 1

    # Проценты
    pct_doi = (with_doi / total * 100) if total > 0 else 0.0
    pct_arxiv = (with_arxiv_id / total * 100) if total > 0 else 0.0

    # Дополнительные метрики: количество ссылок с DOI ИЛИ arXiv ID
    if "doi" in df.columns and "arxiv_id" in df.columns:
        has_doi = df["doi"].astype(str).str.strip() != ""
        has_arxiv = df["arxiv_id"].astype(str).str.strip() != ""
        with_doi_or_arxiv = (has_doi | has_arxiv).sum()
    else:
        with_doi_or_arxiv = with_doi + with_arxiv_id
    pct_with_id = (with_doi_or_arxiv / total * 100) if total > 0 else 0.0

    return {
        "total": total,
        "with_doi": with_doi,
        "with_arxiv_id": with_arxiv_id,
        "with_title": with_title,
        "with_year": with_year,
        "with_venue": with_venue,
        "with_author": with_author,
        "with_pdf": pdf_count,
        "pct_doi": pct_doi,
        "pct_arxiv": pct_arxiv,
        "with_doi_or_arxiv": with_doi_or_arxiv,
        "pct_with_id": pct_with_id,
    }

