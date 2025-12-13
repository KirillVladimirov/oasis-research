"""Нормализация и сравнение заголовков статей."""

import re

from unidecode import unidecode

# Список стоп-слов для нормализации заголовков
_TITLE_STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "is",
    "was",
    "are",
    "were",
    "been",
    "be",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "should",
    "could",
    "may",
    "might",
    "must",
    "can",
    "this",
    "that",
    "these",
    "those",
}


def normalize_title_for_match(title: str) -> str:
    """Нормализует заголовок для сравнения: убирает пунктуацию, стоп-слова, приводит к lowercase.

    Args:
        title: Исходный заголовок

    Returns:
        Нормализованный заголовок
    """
    if not title:
        return ""

    # Приводим к lowercase и убираем unicode проблемы
    normalized = unidecode(title.lower().strip())

    # Убираем пунктуацию, оставляем только буквы, цифры и пробелы
    normalized = re.sub(r"[^\w\s]", " ", normalized)

    # Убираем множественные пробелы
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # Убираем стоп-слова (опционально, можно закомментировать если нужно)
    # words = normalized.split()
    # words = [w for w in words if w not in _TITLE_STOP_WORDS]
    # normalized = " ".join(words)

    return normalized
