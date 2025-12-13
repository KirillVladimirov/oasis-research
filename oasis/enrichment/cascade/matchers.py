"""Валидация совпадений кандидатов с оригинальными ссылками."""

import re
from typing import Any

from loguru import logger
from rapidfuzz import fuzz
from unidecode import unidecode

from oasis.enrichment.cascade.extractors import safe_str


def _normalize_title(title: str) -> str:
    """Нормализует название для сравнения.
    
    Преобразования:
    1. Разбивает CamelCase на слова (CurriculumLearning → Curriculum Learning)
    2. Заменяет дефисы и подчеркивания на пробелы (active-learning → active learning)
    3. Приводит к нижнему регистру
    4. Убирает акценты (unidecode)
    5. Удаляет пунктуацию
    6. Нормализует пробелы
    """
    # 1. Разбиваем CamelCase на слова (CurriculumLearning → Curriculum Learning)
    title_spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', title)
    
    # 2. Заменяем дефисы и подчеркивания на пробелы
    title_spaced = re.sub(r'[-_]', ' ', title_spaced)
    
    # 3. Приводим к нижнему регистру и убираем акценты
    title_lower = unidecode(title_spaced.lower())
    
    # 4. Удаляем пунктуацию и лишние пробелы
    title_clean = re.sub(r"[^\w\s]", "", title_lower)
    title_clean = re.sub(r"\s+", " ", title_clean).strip()
    
    return title_clean


def normalize_title_for_match(title: str) -> str:
    """Нормализует название для строгого сравнения (алиас для _normalize_title)."""
    return _normalize_title(title)


def strict_title_match(expected: str, candidate: str, threshold: float = 0.90) -> bool:
    """Строгая проверка совпадения названий.
    
    Требования:
    1. fuzz.ratio >= threshold (не partial_ratio!)
    2. Длина названий отличается < 20%
    3. Все ключевые слова (>4 букв) из expected есть в candidate
    
    Args:
        expected: Ожидаемое название статьи
        candidate: Название кандидата
        threshold: Минимальный порог similarity (по умолчанию 0.90)
        
    Returns:
        True если совпадение строгое, False иначе
    """
    if not expected or not candidate:
        return False
    
    # Нормализация
    exp_norm = normalize_title_for_match(expected)
    cand_norm = normalize_title_for_match(candidate)
    
    if not exp_norm or not cand_norm:
        return False
    
    # 1. Полное совпадение (не partial!)
    ratio = fuzz.ratio(exp_norm, cand_norm) / 100.0
    if ratio < threshold:
        logger.debug(
            f"Строгая валидация: ratio {ratio:.2f} < {threshold:.2f} для '{expected[:50]}...' vs '{candidate[:50]}...'"
        )
        return False
    
    # 2. Длина
    len_diff = abs(len(exp_norm) - len(cand_norm)) / max(len(exp_norm), len(cand_norm))
    if len_diff > 0.20:
        logger.debug(
            f"Строгая валидация: разница длины {len_diff:.2%} > 20% для '{expected[:50]}...' vs '{candidate[:50]}...'"
        )
        return False
    
    # 3. Ключевые слова
    exp_words = {w for w in exp_norm.split() if len(w) > 4}
    if not exp_words:  # Если нет ключевых слов, используем только ratio
        return True
    
    cand_words = set(cand_norm.split())
    missing_words = exp_words - cand_words
    missing_ratio = len(missing_words) / len(exp_words)
    
    if missing_ratio > 0.10:  # Не более 10% ключевых слов отсутствует
        logger.debug(
            f"Строгая валидация: {missing_ratio:.1%} ключевых слов отсутствует для '{expected[:50]}...' "
            f"(отсутствуют: {list(missing_words)[:3]}...)"
        )
        return False
    
    return True


def validate_match(
    original_title: str, candidate: dict[str, Any], ref: dict[str, Any], use_strict: bool = True
) -> bool:
    """Валидирует совпадение кандидата с оригинальной ссылкой.

    Проверяет:
    - Similarity title (>= 0.90 в строгом режиме, >= 0.85 в мягком)
    - Разница в длине (< 20% в строгом режиме, < 30% в мягком)
    - Ключевые слова (в строгом режиме)
    - Год (разница <= 2 года)
    - Авторы (если есть)

    Args:
        original_title: Оригинальное название статьи
        candidate: Словарь с метаданными кандидата (Metadata.to_dict() или dict)
        ref: Словарь с данными оригинальной ссылки
        use_strict: Использовать строгую валидацию (по умолчанию True)

    Returns:
        True если совпадение валидно, False иначе
    """
    original_title = safe_str(original_title)
    if not original_title:
        return False

    candidate_title = safe_str(candidate.get("title"))
    if not candidate_title:
        return False

    # Используем строгую валидацию по умолчанию
    if use_strict:
        if not strict_title_match(original_title, candidate_title, threshold=0.90):
            return False
    else:
        # Старая логика для обратной совместимости
        original_normalized = _normalize_title(original_title)
        candidate_normalized = _normalize_title(candidate_title)

        # Проверка similarity
        similarity = fuzz.ratio(original_normalized, candidate_normalized) / 100.0
        if similarity < 0.85:
            logger.debug(
                f"Низкая similarity: {similarity:.2f} для '{original_title}' vs '{candidate_title}'"
            )
            return False

        # Проверка разницы в длине
        length_diff = abs(len(original_normalized) - len(candidate_normalized)) / max(
            len(original_normalized), 1
        )
        if length_diff > 0.3:
            logger.debug(
                f"Большая разница в длине: {length_diff:.2%} для '{original_title}' vs '{candidate_title}'"
            )
            return False

    # Проверка года (если есть) - СТРОГАЯ проверка для избежания путаницы
    ref_year = ref.get("year")
    candidate_year = candidate.get("year")
    if ref_year and candidate_year:
        year_diff = abs(ref_year - candidate_year)
        if year_diff > 2:  # Допускаем разницу до 2 лет (опечатки, препринты)
            logger.warning(
                f" Большая разница в годе: {year_diff} лет для '{original_title[:50]}...' "
                f"(ref: {ref_year}, candidate: {candidate_year}). ОТКЛОНЯЕМ для избежания путаницы."
            )
            return False  # БЛОКИРУЕМ - это разные статьи!

    # Проверка авторов (если есть)
    ref_authors = safe_str(ref.get("authors"))
    candidate_authors = safe_str(candidate.get("authors"))
    if ref_authors and candidate_authors:
        # Простая проверка: есть ли хотя бы один общий автор
        ref_author_list = [
            a.strip().lower() for a in ref_authors.split(";") if a.strip()
        ]
        candidate_author_list = [
            a.strip().lower() for a in candidate_authors.split(";") if a.strip()
        ]
        if ref_author_list and candidate_author_list:
            # Проверяем совпадение по фамилиям (последнее слово)
            ref_surnames = {a.split()[-1] if " " in a else a for a in ref_author_list}
            candidate_surnames = {
                a.split()[-1] if " " in a else a for a in candidate_author_list
            }
            if not ref_surnames.intersection(candidate_surnames):
                logger.warning(
                    f"️ Нет общих авторов для '{original_title[:50]}...' vs '{candidate_title[:50]}...'. "
                    f"Возможно неполные данные об авторах, НЕ блокируем."
                )
                # Не отбрасываем - авторы могут быть неполными или с ошибками

    return True
