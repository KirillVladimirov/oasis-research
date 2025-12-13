"""Утилиты для работы с текстом: нормализация, очистка, валидация форматов."""

import re
from typing import Any


def clean_text_value(val: Any) -> str:
    """Очистка текстового значения от лишних пробелов.
    
    Args:
        val: Любое значение, которое нужно преобразовать в строку
        
    Returns:
        Очищенная строка
    """
    return re.sub(r"\s+", " ", str(val or "").strip())


def normalize_text(text: str) -> str:
    """Нормализация текста: удаление лишних переносов, пробелов и символов.

    Агрессивная очистка всех видов пробелов, переносов и управляющих символов.

    Args:
        text: Исходный текст

    Returns:
        Нормализованный текст без переносов и лишних пробелов
    """
    if not text:
        return ""
    
    # Преобразуем в строку на случай если передан не str
    if not isinstance(text, str):
        text = str(text)
    
    # Удаляем все виды табуляций и заменяем на пробел
    text = text.replace("\t", " ")
    # Удаляем все виды переносов строк: \r, \n, \r\n
    text = re.sub(r"[\r\n]+", " ", text)
    # Удаляем неразрывные пробелы и другие Unicode пробелы
    text = re.sub(r"[\u00A0\u2000-\u200B\u202F\u205F\u3000]", " ", text)
    # Заменяем все множественные пробелы (любого вида) на один обычный пробел
    text = re.sub(r"\s+", " ", text)
    # Убираем пробелы в начале и конце
    text = text.strip()
    # Удаляем управляющие символы
    # Удаляем: \x00-\x08, \x0B-\x0C, \x0E-\x1F, \x7F (DEL)
    text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)
    # Финальная очистка пробелов
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def validate_doi_format(doi: str) -> bool:
    """Проверяет валидность DOI по формату (без проверки доступности).

    Args:
        doi: DOI строка для проверки

    Returns:
        True если DOI соответствует формату (10.xxxx/xxxxx)
    """
    if not doi or not isinstance(doi, str):
        return False
    doi_clean = doi.strip().lower()
    if not doi_clean:
        return False
    # Проверка формата DOI: должен начинаться с 10., затем минимум 4 цифры, затем /, затем непустой suffix
    doi_pattern = re.compile(r"^10\.\d{4,}/.+")
    return bool(doi_pattern.match(doi_clean))

