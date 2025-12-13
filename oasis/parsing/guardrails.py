"""Guardrails для фильтрации "фантомных" ссылок и валидации."""

import re
from typing import Any


def validate_ref_number(ref_number: str | int | None, max_ref_number: int = 500) -> bool:
    """Валидация номера ссылки на аномалии.

    Args:
        ref_number: Номер ссылки (может быть строкой или числом)
        max_ref_number: Максимальный допустимый номер ссылки (по умолчанию 500 для обзоров)

    Returns:
        True если номер валидный, False если аномальный
    """
    if ref_number is None:
        return False

    # Преобразуем в строку для проверки
    ref_num_str = str(ref_number).strip()

    # Извлекаем число из строки (может быть в формате "[1]", "1", "(1)" и т.д.)
    # Ищем первое число в строке
    num_match = re.search(r'\d+', ref_num_str)
    if not num_match:
        return False
    
    try:
        ref_num_int = int(num_match.group(0))
        if ref_num_int > max_ref_number:  # Слишком большой номер для ссылки
            return False
        if ref_num_int < 1:  # Номер должен быть положительным
            return False
    except (ValueError, TypeError):
        return False

    # Проверяем что это не биография или другой раздел
    ref_num_lower = ref_num_str.lower()
    if any(word in ref_num_lower for word in ["biography", "acknowledgment", "appendix"]):
        return False

    return True


def filter_phantom_references(
    refs: list[dict[str, Any]], max_ref_number: int = 500
) -> list[dict[str, Any]]:
    """Фильтрует "фантомные" ссылки (биографии, номера страниц, аномальные ref_number).

    Args:
        refs: Список словарей ссылок
        max_ref_number: Максимальный допустимый номер ссылки (по умолчанию 500 для обзоров)

    Returns:
        Отфильтрованный список ссылок
    """
    filtered = []

    for ref in refs:
        # Проверка ref_number
        ref_number = ref.get("ref_number")
        if not validate_ref_number(ref_number, max_ref_number=max_ref_number):
            continue

        # Проверка минимальной длины raw_text
        raw_text = ref.get("raw_text", "")
        if not raw_text or len(raw_text.strip()) < 15:  # Увеличена минимальная длина
            continue

        # Проверка на слишком короткие title (одна буква, инициал)
        title = ref.get("title", "").strip()
        title_lower = title.lower()
        raw_text_lower = raw_text.lower()

        # Фильтруем ссылки с title длиной менее 10 символов или состоящие только из одной буквы
        if len(title) < 10 or re.match(r"^[A-Z]\s*$", title):
            # Если title слишком короткий, проверяем raw_text - возможно, title не был правильно извлечён
            if len(raw_text.strip()) < 20:
                continue

        # Проверка на биографии (ключевые слова в title или raw_text)
        biography_keywords = [
            "biography",
            "biographical",
            "about the author",
            "about the authors",
            "author biography",
            "he was also",
            "visiting associate professor",
            "in 2016",
            "ieee transactions",
        ]
        if any(keyword in title_lower or keyword in raw_text_lower for keyword in biography_keywords):
            continue

        # Проверка на странные паттерны (например, только одна буква в начале)
        # Если raw_text начинается с одной буквы и точки, это может быть ошибка парсинга
        if re.match(r"^[A-Z]\s*\.?\s*$", raw_text.strip()):
            continue

        filtered.append(ref)

    return filtered

