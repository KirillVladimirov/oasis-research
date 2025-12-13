"""Стадия 1: Извлечение ссылок из PDF обзора."""

import csv
import re
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from loguru import logger
from oasis.parsing.guardrails import filter_phantom_references
from oasis.parsing.normalization import normalize_reference
from oasis.parsing.pdf_refs import (
    extract_references_from_pdf_text,
    parse_grobid_fulltext,
)
from oasis.utils.text import normalize_text


def extract_reference_titles(
    pdf_path: str,
    topic: str,
    trace_id: str | None = None,
    progress_callback: Callable[[str, float, str], None] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Стадия 1: Извлечение всех названий статей из раздела References.

    Использует GROBID processFulltextDocument для получения всех ссылок,
    затем извлекает раздел References из PDF напрямую для максимального покрытия.

    Args:
        pdf_path: Путь к PDF файлу обзора
        topic: Название темы исследования
        trace_id: Идентификатор трейса для логирования
        progress_callback: Функция callback для обновления прогресса

    Returns:
        Кортеж: (путь к CSV, список сырых ссылок с минимальными данными)
    """
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    logger.info(f"{trace_prefix}Стадия 1: Извлечение названий из {pdf_path}")

    def _progress(stage: str, progress: float, message: str = ""):
        if progress_callback:
            progress_callback(stage, progress, message)

    timings: dict[str, float] = {}

    t_step = time.perf_counter()
    _progress("grobid_fulltext", 0.1, "Извлечение через GROBID processFulltextDocument...")
    # Метод 1: Попытка через GROBID processFulltextDocument
    refs_grobid = parse_grobid_fulltext(pdf_path, trace_id=trace_id)
    timings["grobid_fulltext_sec"] = time.perf_counter() - t_step

    t_step = time.perf_counter()
    _progress("direct_extract", 0.4, "Прямое извлечение из PDF раздела References...")
    # Метод 2: Прямое извлечение из PDF раздела References
    refs_direct = extract_references_from_pdf_text(pdf_path, trace_id=trace_id)
    timings["direct_extract_sec"] = time.perf_counter() - t_step

    t_step = time.perf_counter()
    _progress("merge", 0.6, "Объединение результатов...")

    # Объединение результатов (приоритет GROBID, дополняем прямым парсингом)
    all_refs_dict = {}
    ref_counter = 1

    def _extract_ref_number(ref: dict[str, Any], default_counter: int) -> int:
        """Извлекает числовой номер ссылки из ref_number."""
        ref_num = ref.get("ref_number")
        if ref_num is None:
            return default_counter
        # Пытаемся извлечь число из строки типа "[1]", "1", "[ 1 ]"
        if isinstance(ref_num, str):
            match = re.search(r"\d+", ref_num)
            if match:
                return int(match.group())
        elif isinstance(ref_num, int):
            return ref_num
        return default_counter

    # Сначала добавляем из GROBID (уже нормализованные в parse_bibl_struct)
    for ref in refs_grobid:
        ref_num = _extract_ref_number(ref, ref_counter)
        # Дополнительная нормализация на всякий случай
        if ref.get("title"):
            ref["title"] = normalize_text(ref["title"])
        if ref.get("raw_text"):
            ref["raw_text"] = normalize_text(ref["raw_text"])
        if ref.get("venue"):
            ref["venue"] = normalize_text(ref["venue"])

        if ref_num not in all_refs_dict:
            all_refs_dict[ref_num] = ref
            ref_counter = max(ref_counter, ref_num + 1)

    # Затем дополняем из прямого парсинга
    for ref in refs_direct:
        ref_num = _extract_ref_number(ref, ref_counter)
        # Нормализуем данные перед добавлением
        if ref.get("title"):
            ref["title"] = normalize_text(ref["title"])
        if ref.get("raw_text"):
            ref["raw_text"] = normalize_text(ref["raw_text"])
        if ref.get("venue"):
            ref["venue"] = normalize_text(ref["venue"])

        if ref_num not in all_refs_dict:
            all_refs_dict[ref_num] = ref
            ref_counter = max(ref_counter, ref_num + 1)
        else:
            # Если есть дубликат, объединяем данные
            existing = all_refs_dict[ref_num]
            if not existing.get("title") and ref.get("title"):
                existing["title"] = normalize_text(ref["title"])
            if not existing.get("raw_text") and ref.get("raw_text"):
                # Гарантируем нормализацию при объединении
                existing["raw_text"] = normalize_text(ref["raw_text"])
            # Сохраняем article_url из прямого парсинга, если его нет в GROBID ссылке
            # GROBID обычно не извлекает URL, поэтому приоритет у прямого парсинга
            if not existing.get("article_url") and ref.get("article_url"):
                existing["article_url"] = ref.get("article_url")
            # Сохраняем arxiv_id из прямого парсинга, если его нет в GROBID ссылке
            # Прямой парсинг лучше извлекает arXiv ID из URL
            if not existing.get("arxiv_id") and ref.get("arxiv_id"):
                existing["arxiv_id"] = ref.get("arxiv_id")

    all_refs = sorted(all_refs_dict.values(), key=lambda x: x.get("ref_number", 9999))

    timings["merge_sec"] = time.perf_counter() - t_step

    t_step = time.perf_counter()
    _progress("filter_phantoms", 0.65, f"Фильтрация фантомных ссылок из {len(all_refs)} ссылок...")
    # Применяем guardrails для фильтрации фантомных ссылок
    all_refs = filter_phantom_references(all_refs, max_ref_number=500)
    logger.info(f"{trace_prefix}После фильтрации фантомных ссылок осталось {len(all_refs)} ссылок")
    timings["filter_phantoms_sec"] = time.perf_counter() - t_step

    t_step = time.perf_counter()
    _progress("normalize", 0.7, f"Нормализация текстов {len(all_refs)} ссылок...")
    # КРИТИЧНО: Нормализуем данные В СЛОВАРЕ перед подготовкой CSV
    # Это гарантирует что данные в возвращаемом списке refs тоже нормализованы
    for ref in all_refs:
        if ref.get("title"):
            ref["title"] = normalize_text(str(ref["title"]))
            # Дополнительная очистка
            ref["title"] = ref["title"].replace("\n", " ").replace("\r", " ").replace("\t", " ")
            ref["title"] = re.sub(r"\s+", " ", ref["title"]).strip()
        if ref.get("raw_text"):
            ref["raw_text"] = normalize_text(str(ref["raw_text"]))
            # Дополнительная очистка
            ref["raw_text"] = ref["raw_text"].replace("\n", " ").replace("\r", " ").replace("\t", " ")
            ref["raw_text"] = re.sub(r"\s+", " ", ref["raw_text"]).strip()
        if ref.get("venue"):
            ref["venue"] = normalize_text(str(ref.get("venue", "")))
            ref["venue"] = ref["venue"].replace("\n", " ").replace("\r", " ").replace("\t", " ")
            ref["venue"] = re.sub(r"\s+", " ", ref["venue"]).strip()
        if ref.get("authors"):
            if isinstance(ref["authors"], list):
                ref["authors"] = [
                    normalize_text(str(a)).replace("\n", " ").replace("\r", " ").replace("\t", " ")
                    for a in ref["authors"]
                ]
            else:
                ref["authors"] = normalize_text(str(ref["authors"])).replace("\n", " ").replace("\r", " ").replace("\t", " ")
            ref["authors"] = re.sub(r"\s+", " ", str(ref["authors"])).strip()

    # Подготовка данных для CSV с нормализацией текстов
    csv_data = []
    for ref in all_refs:
        # АГРЕССИВНАЯ нормализация title и raw_text перед сохранением
        # Убеждаемся что данные полностью очищены
        title_raw = ref.get("title", "") or ""
        raw_text_raw = ref.get("raw_text", "") or ""

        # Множественная нормализация для гарантии очистки
        title = normalize_text(str(title_raw))
        raw_text = normalize_text(str(raw_text_raw))

        # Ещё раз нормализуем на случай если были скрытые символы
        title = normalize_text(title)
        raw_text = normalize_text(raw_text)

        # Если title пустой, пытаемся извлечь из raw_text (первые слова)
        if not title and raw_text:
            # Берём первые слова до точки, запятой или 100 символов
            title_match = re.match(r"^([^\.]{1,100}?)(?:\.|,|$)", raw_text)
            if title_match:
                title = normalize_text(title_match.group(1))

        # ФИНАЛЬНАЯ проверка raw_text перед добавлением в CSV
        # Убеждаемся что нет переносов строк - если есть, заменяем СРАЗУ
        raw_text_final = raw_text[:800] if len(raw_text) > 800 else raw_text
        # КРИТИЧНО: Удаляем ВСЕ переносы и табуляции, даже если они единственные символы
        raw_text_final = raw_text_final.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        raw_text_final = re.sub(r"\s+", " ", raw_text_final).strip()
        # Если после очистки осталась только пустая строка или пробелы - делаем пустым
        if not raw_text_final or raw_text_final.isspace():
            raw_text_final = ""

        # Дополнительная проверка title
        title = title.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        title = re.sub(r"\s+", " ", title).strip()
        if not title or title.isspace():
            title = ""

        # Нормализуем authors и venue тоже
        authors_final = normalize_text(
            ", ".join(ref["authors"]) if isinstance(ref.get("authors"), list) else str(ref.get("authors", ""))
        )
        authors_final = authors_final.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        authors_final = re.sub(r"\s+", " ", authors_final).strip()

        venue_final = normalize_text(ref.get("venue", ""))
        venue_final = venue_final.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        venue_final = re.sub(r"\s+", " ", venue_final).strip()

        # КРИТИЧНО: Убеждаемся что все пустые поля - это действительно пустые строки, а не None или другие значения
        doi_val = str(ref.get("doi", "")).strip() if ref.get("doi") else ""
        arxiv_val = str(ref.get("arxiv_id", "")).strip() if ref.get("arxiv_id") else ""
        year_val = str(ref.get("year", "")) if ref.get("year") else ""
        article_url_val = str(ref.get("article_url", "")).strip() if ref.get("article_url") else ""

        # Удаляем переносы из всех полей, включая doi, arxiv_id и article_url
        doi_val = doi_val.replace("\n", "").replace("\r", "").replace("\t", "").strip()
        arxiv_val = arxiv_val.replace("\n", "").replace("\r", "").replace("\t", "").strip()
        article_url_val = article_url_val.replace("\n", "").replace("\r", "").replace("\t", "").strip()

        # Галочка наличия article_url
        has_article_url = "" if article_url_val else ""

        csv_data.append(
            {
                "ref_number": str(ref.get("ref_number", "")),
                "title": title,
                "raw_text": raw_text_final,
                "doi": doi_val,
                "arxiv_id": arxiv_val,
                "year": year_val,
                "authors": authors_final,
                "venue": venue_final,
                "article_url": article_url_val,
                "has_article_url": has_article_url,
            }
        )

    # Сохранение в CSV
    output_dir = Path(f"data/{topic}")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "references_raw.csv"

    timings["normalize_sec"] = time.perf_counter() - t_step

    t_step = time.perf_counter()
    _progress("save", 0.9, "Сохранение в CSV...")
    df = pd.DataFrame(csv_data)

    # КРИТИЧНО: Проверяем что в csv_data НЕТ переносов перед созданием DataFrame
    # Это гарантирует что DataFrame не будет содержать переносов
    for idx, row_data in enumerate(csv_data):
        for key, value in row_data.items():
            if isinstance(value, str) and ("\n" in value or "\r" in value or "\t" in value):
                # Если найден перенос - заменяем принудительно в исходных данных
                # Многократная очистка до полного отсутствия переносов
                cleaned = str(value)
                iterations = 0
                while ("\n" in cleaned or "\r" in cleaned or "\t" in cleaned) and iterations < 10:
                    cleaned = cleaned.replace("\n", " ").replace("\r", " ").replace("\t", " ")
                    cleaned = re.sub(r"\s+", " ", cleaned).strip()
                    iterations += 1
                # Финальная очистка через split/join
                if "\n" in cleaned or "\r" in cleaned or "\t" in cleaned:
                    cleaned = " ".join(cleaned.split())
                csv_data[idx][key] = cleaned

    # Пересоздаём DataFrame с очищенными данными
    df = pd.DataFrame(csv_data)

    # ПРИНУДИТЕЛЬНАЯ агрессивная нормализация всех текстовых колонок перед сохранением
    # Включаем ВСЕ текстовые колонки, включая числовые которые могут быть строками
    text_columns = ["title", "raw_text", "venue", "authors", "doi", "arxiv_id", "year", "ref_number"]

    def aggressive_normalize(x):
        """Максимально агрессивная нормализация с проверкой результата."""
        if pd.isna(x) or x == "" or str(x).lower() in ["nan", "none"]:
            return ""
        x_str = str(x)
        # СРАЗУ удаляем все переносы и табуляции перед нормализацией
        x_str = x_str.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        x_str = re.sub(r"\s+", " ", x_str).strip()
        # Затем нормализуем
        normalized = normalize_text(x_str)
        prev = ""
        # Нормализуем пока результат не перестанет меняться
        while normalized != prev:
            prev = normalized
            normalized = normalize_text(normalized)
        # Финальная проверка: не должно быть переносов строк и табуляций
        if "\n" in normalized or "\r" in normalized or "\t" in normalized:
            # Агрессивная замена всех видов переносов и табуляций
            normalized = normalized.replace("\n", " ").replace("\r", " ").replace("\t", " ")
            normalized = re.sub(r"\s+", " ", normalized).strip()
            normalized = normalize_text(normalized)
        # Убеждаемся что нет никаких управляющих символов
        normalized = re.sub(r"[\x00-\x1F\x7F]", "", normalized)
        # Финальная проверка - если всё ещё есть переносы, заменяем вручную
        if "\n" in normalized or "\r" in normalized or "\t" in normalized:
            normalized = " ".join(normalized.split())  # split() убирает все пробельные символы
        return normalized

    for col in text_columns:
        if col in df.columns:
            # Финальная нормализация через apply для гарантии
            def final_clean(x):
                if pd.isna(x):
                    return ""
                x_str = str(x)
                # Удаляем все переносы и табуляции
                x_str = re.sub(r"[\r\n\t]+", " ", x_str)
                # Нормализуем пробелы
                x_str = re.sub(r"\s+", " ", x_str)
                # Удаляем управляющие символы
                x_str = re.sub(r"[\x00-\x1F\x7F]", "", x_str)
                return x_str.strip()

            # Сначала агрессивная нормализация
            df[col] = df[col].apply(aggressive_normalize)
            # Затем финальная очистка
            df[col] = df[col].apply(final_clean)

    # ФИНАЛЬНАЯ проверка перед сохранением - убеждаемся что нет переносов
    for col in text_columns:
        if col in df.columns:
            # Проверяем каждое значение и заменяем переносы если они всё ещё есть
            def force_remove_newlines(val):
                if pd.isna(val):
                    return ""
                val_str = str(val) if val is not None else ""
                # КРИТИЧНО: Если строка состоит только из пробельных символов или переносов - делаем пустой
                if not val_str or val_str.strip() in ["", "\n", "\r", "\t", "\n\r", "\r\n"]:
                    return ""
                # Агрессивная замена всех видов переносов
                # Для пустых полей (doi, arxiv_id) просто удаляем переносы, не заменяем на пробелы
                if col in ["doi", "arxiv_id"] and len(val_str.strip()) == 0:
                    return ""
                val_str = val_str.replace("\n", " ").replace("\r", " ").replace("\t", " ")
                val_str = re.sub(r"\s+", " ", val_str).strip()
                # Если после очистки ничего не осталось - возвращаем пустую строку
                if not val_str:
                    return ""
                return val_str

            df[col] = df[col].apply(force_remove_newlines)

            # Дополнительная проверка: убеждаемся что в DataFrame действительно нет переносов
            # Проверяем все значения и заменяем если найден перенос
            for idx in df.index:
                val = df.at[idx, col]
                if isinstance(val, str) and ("\n" in val or "\r" in val or "\t" in val):
                    # Принудительная замена
                    cleaned = val.replace("\n", " ").replace("\r", " ").replace("\t", " ")
                    cleaned = re.sub(r"\s+", " ", cleaned).strip()
                    df.at[idx, col] = cleaned if cleaned else ""

    # ФИНАЛЬНАЯ проверка DataFrame: убеждаемся что ВСЕ значения не содержат переносов
    # Проверяем каждый элемент DataFrame напрямую
    for col in df.columns:
        for idx in df.index:
            val = df.at[idx, col]
            if isinstance(val, str) and ("\n" in val or "\r" in val or "\t" in val):
                # КРИТИЧНО: Если найден перенос - заменяем принудительно
                cleaned = val.replace("\n", " ").replace("\r", " ").replace("\t", " ")
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                # Для пустых полей после очистки - делаем полностью пустым
                if not cleaned or cleaned in ["\n", "\r", "\t"]:
                    cleaned = ""
                df.at[idx, col] = cleaned

    # ДОПОЛНИТЕЛЬНАЯ проверка: убеждаемся что DataFrame не содержит переносов
    # Проверяем через сериализацию в строку перед сохранением
    csv_string = df.to_csv(index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    if isinstance(csv_string, str) and "\n" in csv_string and csv_string.count('"') > 0:
        # Проверяем есть ли переносы внутри кавычек (многострочные поля)
        import io

        reader = csv.reader(io.StringIO(csv_string))
        rows = list(reader)
        # Если количество строк не совпадает с количеством записей + заголовок - есть проблема
        if len(rows) != len(df) + 1:
            logger.warning(f"{trace_prefix}Обнаружены многострочные поля, применяем дополнительную нормализацию")
            # Перенормализуем все текстовые колонки ещё раз
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace("\n", " ").str.replace("\r", " ").str.replace("\t", " ")
                    df[col] = df[col].str.replace(r"\s+", " ", regex=True).str.strip()

    # КРИТИЧНО: Последняя проверка - убеждаемся что DataFrame НЕ содержит переносов
    # Проверяем каждый элемент ещё раз перед сохранением
    for col in df.columns:
        for idx in df.index:
            val = df.at[idx, col]
            if isinstance(val, str) and ("\n" in val or "\r" in val or "\t" in val):
                # Принудительная очистка
                cleaned = val.replace("\n", " ").replace("\r", " ").replace("\t", " ")
                cleaned = re.sub(r"\s+", " ", cleaned).strip()
                if "\n" in cleaned or "\r" in cleaned or "\t" in cleaned:
                    cleaned = " ".join(cleaned.split())
                df.at[idx, col] = cleaned

    # Сохраняем CSV с настройками для избежания многострочных полей
    # Используем quoting=csv.QUOTE_ALL + escapechar='\\' чтобы pandas экранировал переносы
    # Это решает проблему отображения в редакторах
    # line_terminator='\n' явно задаёт окончание строк
    # КРИТИЧНО: Используем na_rep='' чтобы None/NaN сохранялись как пустые строки, не как строки "nan"
    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,  # Все поля в кавычках
        lineterminator="\n",
        escapechar="\\",  # КРИТИЧНО: Экранирование переносов вместо создания многострочных полей
        doublequote=True,
        na_rep="",  # КРИТИЧНО: None/NaN должны сохраняться как пустые строки, не как строки "nan"
    )
    timings["save_csv_sec"] = time.perf_counter() - t_step

    # Подсчёт метрик
    total_refs = len(all_refs)
    with_article_url = sum(1 for ref in all_refs if ref.get("article_url"))
    pct_with_url = (with_article_url / total_refs * 100) if total_refs > 0 else 0

    _progress("complete", 1.0, f"Завершено: {total_refs} ссылок")
    logger.info(
        f"{trace_prefix}Стадия 1 завершена: найдено {total_refs} ссылок, сохранено в {csv_path}"
    )
    logger.info(
        f"{trace_prefix}Метрики ссылок: с article_url - {with_article_url} ({pct_with_url:.1f}%)"
    )
    # Лог таймингов по шагам (только лог, без изменения интерфейса)
    try:
        logger.info(
            f"{trace_prefix}Тайминги Стадия 1: "
            f"grobid_fulltext={timings.get('grobid_fulltext_sec', 0):.2f}s, "
            f"direct_extract={timings.get('direct_extract_sec', 0):.2f}s, "
            f"merge={timings.get('merge_sec', 0):.2f}s, "
            f"normalize={timings.get('normalize_sec', 0):.2f}s, "
            f"save_csv={timings.get('save_csv_sec', 0):.2f}s"
        )
    except Exception:
        pass

    return str(csv_path), all_refs

