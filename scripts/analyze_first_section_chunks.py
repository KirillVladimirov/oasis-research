#!/usr/bin/env python3
"""Анализ чанков первой секции первой статьи из Weaviate."""

import sys
from pathlib import Path

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent))

import weaviate
from loguru import logger
from collections import Counter

WEAVIATE_URL = "http://localhost:8081"


def analyze_first_section_chunks():
    """Анализирует чанки первой секции первой статьи."""
    logger.info("=" * 80)
    logger.info("АНАЛИЗ ЧАНКОВ ПЕРВОЙ СЕКЦИИ ПЕРВОЙ СТАТЬИ")
    logger.info("=" * 80)
    
    # Подключение
    client = weaviate.Client(WEAVIATE_URL, startup_period=10)
    logger.info(f"Подключено к Weaviate: {WEAVIATE_URL}")
    
    # Находим первую статью
    logger.info("\nПоиск первой статьи...")
    try:
        result = client.query.get("Paper", ["paper_id", "title"]).with_limit(1).do()
        papers = result.get("data", {}).get("Get", {}).get("Paper", [])
        
        if not papers:
            logger.error("Статьи не найдены")
            return 1
        
        first_paper = papers[0]
        first_paper_id = first_paper.get("paper_id")
        first_paper_title = first_paper.get("title", "N/A")
        
        logger.info(f"Первая статья: {first_paper_id}")
        logger.info(f"Название: {first_paper_title}")
    except Exception as e:
        logger.exception(f"Ошибка поиска статьи: {e}")
        return 1
    
    # Находим первую секцию этой статьи
    logger.info(f"\nПоиск секций статьи {first_paper_id}...")
    try:
        result = client.query.get("Section", ["section_id", "section_heading", "section_number", "full_text"]).with_where({
            "path": ["paper_id"],
            "operator": "Equal",
            "valueString": first_paper_id
        }).with_limit(10).do()
        
        sections = result.get("data", {}).get("Get", {}).get("Section", [])
        
        if not sections:
            logger.error("Секции не найдены")
            return 1
        
        # Сортируем по section_number
        sections.sort(key=lambda s: s.get("section_number", 0))
        first_section = sections[0]
        first_section_id = first_section.get("section_id")
        first_section_heading = first_section.get("section_heading", "N/A")
        first_section_text = first_section.get("full_text", "")
        
        logger.info(f"Первая секция: {first_section_id}")
        logger.info(f"Заголовок: {first_section_heading}")
        logger.info(f"Длина текста секции: {len(first_section_text)} символов")
        logger.info(f"Количество секций в статье: {len(sections)}")
    except Exception as e:
        logger.exception(f"Ошибка поиска секций: {e}")
        return 1
    
    # Находим все чанки этой секции
    logger.info(f"\nПоиск чанков секции {first_section_id}...")
    try:
        result = client.query.get(
            "Chunk",
            ["chunk_id", "text", "start_char", "end_char", "section_heading"]
        ).with_where({
            "path": ["section_id"],
            "operator": "Equal",
            "valueString": first_section_id
        }).with_limit(1000).do()
        
        chunks = result.get("data", {}).get("Get", {}).get("Chunk", [])
        
        logger.info(f"Найдено чанков: {len(chunks)}")
        
        if not chunks:
            logger.warning("Чанки не найдены")
            return 0
        
        # Сортируем по start_char
        chunks.sort(key=lambda c: c.get("start_char", 0))
        
    except Exception as e:
        logger.exception(f"Ошибка поиска чанков: {e}")
        return 1
    
    # Анализ чанков
    logger.info("\n" + "=" * 80)
    logger.info("АНАЛИЗ ЧАНКОВ")
    logger.info("=" * 80)
    
    logger.info(f"\nОбщая статистика:")
    logger.info(f"  - Всего чанков: {len(chunks)}")
    logger.info(f"  - Длина текста секции: {len(first_section_text)} символов")
    logger.info(f"  - Средний размер чанка: {sum(len(c.get('text', '')) for c in chunks) / len(chunks) if chunks else 0:.0f} символов")
    
    # Анализ размеров чанков
    chunk_sizes = [len(c.get("text", "")) for c in chunks]
    logger.info(f"\nРазмеры чанков:")
    logger.info(f"  - Минимальный: {min(chunk_sizes)} символов")
    logger.info(f"  - Максимальный: {max(chunk_sizes)} символов")
    logger.info(f"  - Медиана: {sorted(chunk_sizes)[len(chunk_sizes)//2]} символов")
    
    # Проверка перекрытий
    logger.info(f"\nАнализ перекрытий:")
    overlaps = []
    for i in range(len(chunks) - 1):
        curr_end = chunks[i].get("end_char", 0)
        next_start = chunks[i + 1].get("start_char", 0)
        if next_start < curr_end:
            overlap = curr_end - next_start
            overlaps.append(overlap)
    
    if overlaps:
        logger.info(f"  - Найдено перекрытий: {len(overlaps)}")
        logger.info(f"  - Среднее перекрытие: {sum(overlaps) / len(overlaps):.0f} символов")
        logger.info(f"  - Минимальное перекрытие: {min(overlaps)} символов")
        logger.info(f"  - Максимальное перекрытие: {max(overlaps)} символов")
    else:
        logger.info(f"  - Перекрытий не найдено (возможно, чанки не последовательные)")
    
    # Показываем первые 10 чанков
    logger.info(f"\nПервые 10 чанков:")
    logger.info("-" * 80)
    for i, chunk in enumerate(chunks[:10], 1):
        chunk_id = chunk.get("chunk_id", "N/A")
        chunk_text = chunk.get("text", "")
        start_char = chunk.get("start_char", 0)
        end_char = chunk.get("end_char", 0)
        
        logger.info(f"\nЧанк {i}: {chunk_id}")
        logger.info(f"  Позиция: {start_char}-{end_char} ({end_char - start_char} символов)")
        logger.info(f"  Длина текста: {len(chunk_text)} символов")
        logger.info(f"  Первые 150 символов:")
        logger.info(f"    {chunk_text[:150]}...")
        logger.info(f"  Последние 50 символов:")
        logger.info(f"    ...{chunk_text[-50:]}")
    
    # Проверка на дубликаты
    logger.info(f"\nПроверка на дубликаты:")
    chunk_texts = [c.get("text", "") for c in chunks]
    text_counter = Counter(chunk_texts)
    duplicates = {text: count for text, count in text_counter.items() if count > 1}
    
    if duplicates:
        logger.warning(f"  Найдено {len(duplicates)} дублирующихся текстов!")
        for text, count in list(duplicates.items())[:5]:
            logger.warning(f"    Дубликат (встречается {count} раз): {text[:100]}...")
    else:
        logger.info(f"  Дубликатов не найдено")
    
    # Проверка последовательности
    logger.info(f"\nПроверка последовательности чанков:")
    gaps = []
    for i in range(len(chunks) - 1):
        curr_end = chunks[i].get("end_char", 0)
        next_start = chunks[i + 1].get("start_char", 0)
        if next_start > curr_end:
            gap = next_start - curr_end
            gaps.append(gap)
    
    if gaps:
        logger.info(f"  Найдено пропусков: {len(gaps)}")
        logger.info(f"  Средний пропуск: {sum(gaps) / len(gaps):.0f} символов")
        logger.info(f"  Максимальный пропуск: {max(gaps)} символов")
    
    # Проверка покрытия текста секции
    logger.info(f"\nПроверка покрытия текста секции:")
    covered_chars = set()
    for chunk in chunks:
        start = chunk.get("start_char", 0)
        end = chunk.get("end_char", 0)
        covered_chars.update(range(start, end))
    
    section_length = len(first_section_text)
    coverage = len(covered_chars) / section_length * 100 if section_length > 0 else 0
    logger.info(f"  Длина секции: {section_length} символов")
    logger.info(f"  Покрыто чанками: {len(covered_chars)} символов ({coverage:.1f}%)")
    
    # Показываем последние 5 чанков
    logger.info(f"\nПоследние 5 чанков:")
    logger.info("-" * 80)
    for i, chunk in enumerate(chunks[-5:], len(chunks) - 4):
        chunk_id = chunk.get("chunk_id", "N/A")
        chunk_text = chunk.get("text", "")
        start_char = chunk.get("start_char", 0)
        end_char = chunk.get("end_char", 0)
        
        logger.info(f"\nЧанк {i}: {chunk_id}")
        logger.info(f"  Позиция: {start_char}-{end_char} ({end_char - start_char} символов)")
        logger.info(f"  Длина текста: {len(chunk_text)} символов")
        logger.info(f"  Первые 150 символов:")
        logger.info(f"    {chunk_text[:150]}...")
    
    logger.success("\n" + "=" * 80)
    logger.success("АНАЛИЗ ЗАВЕРШЕН")
    logger.success("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(analyze_first_section_chunks())

