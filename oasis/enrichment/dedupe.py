"""Дедупликация ссылок."""

from typing import Any

from loguru import logger
from rapidfuzz import fuzz
from unidecode import unidecode


def dedupe_references(
    refs: list[dict[str, Any]],
    threshold: float = 90.0,
    max_length_diff: float = 0.3,
    require_year_match: bool = True,
    require_author_match: bool = True,
    trace_id: str | None = None,
) -> list[dict[str, Any]]:
    """Дедупликация ссылок по DOI/arXiv ID и fuzzy-сравнению title с улучшенными проверками.

    Args:
        refs: Список словарей ссылок
        threshold: Порог для fuzzy matching (0-100), по умолчанию 90.0
        max_length_diff: Максимальная разница в длине названий (0.0-1.0), по умолчанию 0.3 (30%)
        require_year_match: Требовать совпадение года для fuzzy matching
        require_author_match: Требовать совпадение хотя бы одного автора для fuzzy matching
        trace_id: Идентификатор трейса для логирования

    Returns:
        Список уникальных ссылок
    """
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    logger.info(f"{trace_prefix}Дедупликация {len(refs)} ссылок (порог={threshold}%, max_length_diff={max_length_diff})")

    seen_by_id: set[str] = set()
    seen_refs: list[dict[str, Any]] = []  # Храним полные ссылки для проверки года и авторов
    deduped = []

    # Опасные фразы, требующие особой обработки
    dangerous_phrases = ["survey", "active learning", "curriculum learning", "variational", "adversarial"]

    def extract_author_surnames(authors: Any) -> set[str]:
        """Извлекает фамилии авторов из строки или списка."""
        surnames = set()
        if not authors:
            return surnames
        
        authors_str = ""
        if isinstance(authors, str):
            authors_str = authors
        elif isinstance(authors, list):
            authors_str = ", ".join(str(a) for a in authors)
        else:
            authors_str = str(authors)
        
        # Парсим фамилии (обычно первое слово или часть до запятой)
        for author in authors_str.split(","):
            author = author.strip()
            if author:
                # Берем первую часть (фамилия обычно первая)
                parts = author.split()
                if parts:
                    surname = parts[0].strip().lower()
                    if surname:
                        surnames.add(surname)
        return surnames

    for ref in refs:
        # Точное совпадение по DOI или arXiv ID
        deduped_by_id = False
        if ref.get("doi"):
            doi_key = ref["doi"].lower().strip()
            if doi_key and doi_key in seen_by_id:
                logger.debug(f"{trace_prefix}Дубликат по DOI: {doi_key}")
                continue
            if doi_key:
                seen_by_id.add(doi_key)
                deduped_by_id = True

        if ref.get("arxiv_id"):
            arxiv_key = str(ref["arxiv_id"]).strip()
            if arxiv_key and arxiv_key in seen_by_id:
                logger.debug(f"{trace_prefix}Дубликат по arXiv ID: {arxiv_key}")
                continue
            if arxiv_key:
                seen_by_id.add(arxiv_key)
                deduped_by_id = True

        # Fuzzy matching по title с улучшенными проверками
        if ref.get("title"):
            title = unidecode(ref["title"].lower().strip())
            title_len = len(title)
            
            # Исключаем очень короткие заголовки из нечеткой дедупликации
            if title_len < 20:
                logger.debug(f"{trace_prefix}Пропуск короткого заголовка: {title[:50]}...")
                deduped.append(ref)
                seen_refs.append(ref)
                continue

            is_duplicate = False
            ref_year = ref.get("year")
            ref_authors = extract_author_surnames(ref.get("authors"))

            for seen_ref in seen_refs:
                seen_title = unidecode(seen_ref.get("title", "").lower().strip())
                if not seen_title:
                    continue

                # Проверка длины названий
                seen_len = len(seen_title)
                if seen_len > 0:
                    length_diff = abs(title_len - seen_len) / max(title_len, seen_len)
                    if length_diff > max_length_diff:
                        # Разница в длине слишком большая - не дубликат
                        continue

                # Проверка similarity (используем и partial_ratio и ratio)
                partial_sim = fuzz.partial_ratio(title, seen_title)
                full_sim = fuzz.ratio(title, seen_title)
                
                # Для длинных заголовков используем полное совпадение, для коротких - partial
                similarity = max(partial_sim, full_sim) if title_len > 50 else partial_sim

                if similarity >= threshold:
                    # Дополнительные проверки перед признанием дубликатом
                    checks_passed = True
                    check_details = []

                    # Проверка года
                    if require_year_match and ref_year and seen_ref.get("year"):
                        if ref_year != seen_ref.get("year"):
                            checks_passed = False
                            check_details.append(f"год не совпадает: {ref_year} vs {seen_ref.get('year')}")

                    # Проверка авторов
                    if require_author_match and ref_authors:
                        seen_authors = extract_author_surnames(seen_ref.get("authors"))
                        if seen_authors and not ref_authors.intersection(seen_authors):
                            checks_passed = False
                            check_details.append("авторы не совпадают")

                    # Проверка на опасные фразы
                    has_dangerous_phrase = any(phrase in title.lower() for phrase in dangerous_phrases)
                    if has_dangerous_phrase:
                        # Для опасных фраз требуем более высокое совпадение
                        if similarity < 95.0:
                            checks_passed = False
                            check_details.append(f"опасная фраза, требуется более высокое совпадение (было {similarity:.1f}%)")

                    if checks_passed:
                        logger.debug(
                            f"{trace_prefix}Найден дубликат (similarity={similarity:.1f}%, length_diff={length_diff:.1%}): "
                            f'"{title[:50]}..." ~ "{seen_title[:50]}..."'
                        )
                        is_duplicate = True
                        break
                    else:
                        logger.debug(
                            f"{trace_prefix}Высокое совпадение ({similarity:.1f}%), но не дубликат: "
                            f'"{title[:50]}..." ~ "{seen_title[:50]}..." '
                            f"(причины: {', '.join(check_details)})"
                        )

            if is_duplicate and not deduped_by_id:
                continue

            seen_refs.append(ref)

        deduped.append(ref)

    logger.info(f"{trace_prefix}После дедупликации: {len(deduped)} уникальных ссылок (удалено {len(refs) - len(deduped)})")
    return deduped

