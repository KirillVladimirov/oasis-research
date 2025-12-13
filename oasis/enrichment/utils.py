"""Утилиты для обогащения ссылок."""

import json
from typing import Any

import pandas as pd


def safe_strip(val: Any) -> str:
    """Безопасно извлекает и очищает строковое значение.
    
    Обрабатывает None, NaN (float и pandas), и строковые представления пустых значений.
    
    Args:
        val: Значение для обработки
        
    Returns:
        Очищенная строка или пустая строка
    """
    if val is None:
        return ""
    # Проверяем на pandas NA (нужно проверять до всех других операций)
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        # Если pd.isna не может обработать тип (например, строка), продолжаем
        pass
    # Проверяем на pandas nan или float nan
    if isinstance(val, float) and pd.isna(val):
        return ""
    # Проверяем на обычный float nan (на случай если pandas не распознал)
    if isinstance(val, float) and val != val:  # nan != nan всегда True
        return ""
    # Проверяем на пустые строки после преобразования
    # Используем прямое преобразование для чисел (включая 0)
    if isinstance(val, (int, float)):
        result = str(val).strip()
    else:
        result = str(val or "").strip()
    # Проверяем что результат не стал "nan" строкой
    if result.lower() == "nan":
        return ""
    return result


def extract_pdf_from_additional(additional: Any) -> str:
    """Возвращает первую PDF ссылку из additional_urls.
    
    Args:
        additional: Список URL или JSON строка со списком URL
        
    Returns:
        Первая найденная PDF ссылка или пустая строка
    """
    urls: list[str] = []

    if not additional:
        return ""

    if isinstance(additional, list):
        urls = [str(u).strip() for u in additional if isinstance(u, (str, int, float))]
    elif isinstance(additional, str):
        try:
            loaded = json.loads(additional)
            if isinstance(loaded, list):
                urls = [str(u).strip() for u in loaded if isinstance(u, (str, int, float))]
        except json.JSONDecodeError:
            urls = []

    for url in urls:
        if not url:
            continue
        lower = url.lower()
        if lower in {"nan", "none", "null"}:
            continue
        if not url.startswith(("http://", "https://")):
            continue
        if lower.endswith(".pdf") or "/pdf" in lower.split("?")[0]:
            return url
    return ""

