"""Извлечение тем из обзорных статей с помощью LLM (Шаг 1 конвейера)."""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from loguru import logger
from openai import OpenAI

from oasis.io.loaders import load_survey_papers


def _create_openai_client(timeout: float | None = None) -> OpenAI:
    """Создаёт клиент OpenAI для работы с LLM.

    Returns:
        OpenAI клиент

    Raises:
        RuntimeError: Если не установлены переменные окружения
    """
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE", "https://api.aitunnel.ru/v1/")

    if not api_key:
        raise RuntimeError("Не установлена переменная окружения OPENAI_API_KEY")

    if timeout is None:
        timeout_env = os.getenv("OPENAI_TIMEOUT")
        if timeout_env:
            try:
                timeout = float(timeout_env)
            except ValueError:
                logger.warning("Неверное значение OPENAI_TIMEOUT: %s", timeout_env)
                timeout = None

    client_kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url}
    if timeout is not None:
        client_kwargs["timeout"] = timeout
        logger.info(f"Используется таймаут OpenAI {timeout} секунд")

    return OpenAI(**client_kwargs)


def extract_sections_from_pdf(
    pdf_path: str | Path, paper_id: str, trace_id: str | None = None
) -> list[dict[str, Any]]:
    """Извлекает текст из PDF и разбивает на разделы по заголовкам.

    Args:
        pdf_path: Путь к PDF файлу
        paper_id: Идентификатор статьи
        trace_id: Идентификатор трейса для логирования

    Returns:
        Список словарей с разделами: {section_id, section_heading, section_text, page_start, page_end}

    Raises:
        FileNotFoundError: Если PDF файл не найден
        IOError: Если не удалось прочитать PDF
    """
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF файл не найден: {pdf_path}")

    logger.info(f"{trace_prefix}Извлечение разделов из PDF: {path.name}")

    try:
        doc = fitz.open(str(path))
        full_text_pages = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            full_text_pages.append((page_num + 1, text))  # Нумерация с 1 для пользователя
        doc.close()
    except Exception as e:
        logger.error(f"{trace_prefix}Ошибка чтения PDF {path.name}: {e}")
        raise IOError(f"Не удалось прочитать PDF: {e}") from e

    # Объединяем весь текст
    full_text = "\n".join(text for _, text in full_text_pages)

    # Ищем начало раздела References/Bibliography в тексте (более точно)
    references_pattern = re.compile(
        r"(?:^|\n)\s*(?:references|bibliography)\s*(?:\n|$)", re.IGNORECASE | re.MULTILINE
    )
    references_match = references_pattern.search(full_text)
    if references_match:
        references_start_pos = references_match.start()
        full_text = full_text[:references_start_pos]
        logger.debug(f"{trace_prefix}Найден раздел References на позиции {references_start_pos}, обрезаем текст")

    # Паттерны для поиска заголовков разделов
    # 1. Нумерованные разделы: "1. Introduction", "2.1 Background", "3.2.1 Methods"
    # Исключаем паттерны, которые выглядят как ссылки (например, "2021 Conference...")
    numbered_pattern = re.compile(
        r"^(?:\d+\.)+\s+([A-Z][a-z][^\n]{5,100})$", re.MULTILINE
    )
    # 2. Заголовки уровня 1: "INTRODUCTION", "METHODS", "RESULTS" (все заглавные, но не слишком длинные)
    uppercase_pattern = re.compile(
        r"^([A-Z][A-Z\s]{3,50})$", re.MULTILINE
    )
    # 3. Заголовки с форматированием: "1 Introduction", "2. Background"
    # Исключаем паттерны, начинающиеся с года (например, "2021 Conference...")
    simple_numbered_pattern = re.compile(
        r"^(\d+\.?\s+[A-Z][a-z][^\n]{5,100})$", re.MULTILINE
    )

    sections = []
    section_matches = []

    # Собираем все возможные заголовки
    for match in numbered_pattern.finditer(full_text):
        section_matches.append((match.start(), match.group(0).strip(), match.group(1).strip()))

    for match in uppercase_pattern.finditer(full_text):
        # Пропускаем слишком короткие или слишком длинные (вероятно, не заголовки)
        heading = match.group(1).strip()
        if 5 <= len(heading) <= 50 and heading.isupper():
            section_matches.append((match.start(), heading, heading))

    for match in simple_numbered_pattern.finditer(full_text):
        section_matches.append((match.start(), match.group(1).strip(), match.group(1).strip()))

    # Сортируем по позиции в тексте
    section_matches.sort(key=lambda x: x[0])

    # Фильтруем дубликаты и слишком близкие заголовки (вероятно, один и тот же)
    filtered_matches = []
    prev_pos = -1000
    references_start_pos = None
    
    for pos, full_heading, clean_heading in section_matches:
        # Пропускаем заголовки, которые слишком близко к предыдущему (менее 100 символов)
        if pos - prev_pos < 100:
            continue
        
        # Фильтруем заголовки, которые выглядят как ссылки из References
        # (начинаются с года, содержат "Conference", "Proceedings", имена авторов и т.д.)
        if re.search(r"^(19|20)\d{2}\s", clean_heading):  # Начинается с года
            continue
        if re.search(r"(Conference|Proceedings|Workshop|Journal|IEEE|ACM|Springer|Elsevier)", clean_heading, re.IGNORECASE):
            continue
        if re.search(r"^[A-Z]\.\s+[A-Z][a-z]+\s+[A-Z]", clean_heading):  # Паттерн имени автора
            continue
        
        # Ищем начало раздела References/Bibliography
        if re.search(r"^(references|bibliography)", clean_heading, re.IGNORECASE):
            references_start_pos = pos
            logger.debug(f"{trace_prefix}Найден раздел References на позиции {pos}, прекращаем обработку")
            break  # Прекращаем обработку после References
        # Пропускаем другие служебные разделы
        if re.search(r"^(acknowledgments?|acknowledgements?|appendix|author\s+biography)", clean_heading, re.IGNORECASE):
            logger.debug(f"{trace_prefix}Пропущен служебный раздел: {clean_heading}")
            continue
        filtered_matches.append((pos, full_heading, clean_heading))
        prev_pos = pos
    
    # Обрезаем текст до начала References, если он найден
    if references_start_pos is not None:
        full_text = full_text[:references_start_pos]

    # Извлекаем текст для каждого раздела
    for idx, (start_pos, full_heading, clean_heading) in enumerate(filtered_matches):
        # Определяем конец раздела (начало следующего или конец документа)
        if idx + 1 < len(filtered_matches):
            end_pos = filtered_matches[idx + 1][0]
        else:
            end_pos = len(full_text)

        section_text = full_text[start_pos:end_pos].strip()

        # Пропускаем слишком короткие разделы (менее 200 символов)
        if len(section_text) < 200:
            logger.debug(f"{trace_prefix}Пропущен короткий раздел: {clean_heading} ({len(section_text)} символов)")
            continue

        # Определяем страницы раздела
        page_start = 1
        page_end = len(full_text_pages)
        for page_num, page_text in full_text_pages:
            if start_pos <= full_text.find(page_text) + len(page_text):
                page_start = page_num
                break
        for page_num, page_text in full_text_pages:
            if end_pos <= full_text.find(page_text) + len(page_text):
                page_end = page_num
                break

        # Создаём section_id
        section_num = idx + 1
        section_id = f"{paper_id}::sec_{section_num}"

        sections.append({
            "section_id": section_id,
            "section_heading": clean_heading,
            "section_text": section_text,
            "page_start": page_start,
            "page_end": page_end,
        })

        logger.debug(f"{trace_prefix}Извлечён раздел: {clean_heading} (страницы {page_start}-{page_end}, {len(section_text)} символов)")

    logger.info(f"{trace_prefix}Извлечено {len(sections)} разделов из PDF {path.name}")
    return sections


def create_topic_extraction_prompt(
    article_id: str,
    article_title: str,
    section_id: str,
    section_heading: str,
    section_text: str,
) -> tuple[str, str]:
    """Создаёт промпт для извлечения тем из раздела обзора.

    Args:
        article_id: Идентификатор статьи
        article_title: Название статьи
        section_id: Идентификатор раздела
        section_heading: Заголовок раздела
        section_text: Текст раздела

    Returns:
        Кортеж (system_prompt, user_prompt)
    """
    system_prompt = """You are an expert in Deep Active Learning and ontology engineering.
Analyze the given section of a survey paper and identify key domain topics discussed.
For each topic, provide:
- a concise title (3–10 words),
- a brief description (2–4 sentences) explaining the topic (entities/processes/metrics involved),
- an indication of which part of the text the topic is based on.
Output JSON only.

Rules:
- Ignore general sections like "Introduction", "Conclusion", "Abstract" - focus on substantive topics
- Concentrate on content-rich topics that describe domain concepts, methods, or relationships
- Each topic should be specific and meaningful, not generic
- Output format: JSON object with "topics" array, each topic having "local_topic_id", "title", "description", "text_span"
"""

    user_prompt = f"""article_id: {article_id}
article_title: "{article_title}"
section_id: {section_id}
section_heading: "{section_heading}"
section_text: {section_text}

Extract key topics from this section. Return JSON in the following format:
{{
  "topics": [
    {{
      "local_topic_id": "T001",
      "title": "Topic title (3-10 words)",
      "description": "Detailed description (2-4 sentences)",
      "text_span": "Brief indication of where in the text this topic is based (e.g., 'lines 45-67 of section' or 'first paragraph')"
    }}
  ]
}}
"""

    return system_prompt, user_prompt


def call_llm_for_topic_extraction(
    system_prompt: str,
    user_prompt: str,
    trace_id: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> dict[str, Any]:
    """Вызывает LLM для извлечения тем из раздела обзора.

    Args:
        system_prompt: System промпт для LLM
        user_prompt: User промпт для LLM
        trace_id: Идентификатор трейса для логирования
        max_retries: Максимальное количество попыток при ошибке
        retry_delay: Начальная задержка между попытками (секунды)

    Returns:
        Словарь с ответом LLM: {"content": str, "finish_reason": str}

    Raises:
        RuntimeError: Если не удалось получить ответ после всех попыток
        ValueError: Если ответ не является валидным JSON
    """
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    client = _create_openai_client()

    for attempt in range(max_retries):
        try:
            logger.debug(f"{trace_prefix}Вызов LLM (попытка {attempt + 1}/{max_retries})")

            response = client.chat.completions.create(
                model="gpt-5.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=50000,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            if not content:
                raise ValueError("Пустой ответ от LLM")

            # Парсим JSON для валидации
            try:
                parsed = json.loads(content)
                if "topics" not in parsed:
                    raise ValueError("Ответ LLM не содержит поле 'topics'")
            except json.JSONDecodeError as e:
                raise ValueError(f"Ответ LLM не является валидным JSON: {e}") from e

            logger.debug(f"{trace_prefix}Успешный ответ от LLM (finish_reason: {finish_reason})")
            return {"content": content, "finish_reason": finish_reason}

        except Exception as e:
            if attempt < max_retries - 1:
                delay = retry_delay * (2 ** attempt)  # Экспоненциальная задержка
                logger.warning(f"{trace_prefix}Ошибка при вызове LLM (попытка {attempt + 1}): {e}. Повтор через {delay:.1f}с")
                time.sleep(delay)
            else:
                logger.error(f"{trace_prefix}Не удалось получить ответ от LLM после {max_retries} попыток: {e}")
                raise RuntimeError(f"Ошибка вызова LLM: {e}") from e

    raise RuntimeError("Неожиданное завершение цикла retry")


def extract_topics_from_survey_papers(
    survey_papers_json: str | Path | None = None,
    output_jsonl: str | Path | None = None,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Извлекает темы из всех обзорных статей с помощью LLM.

    Args:
        survey_papers_json: Путь к JSON файлу со списком обзорных статей.
            Если None, используется outputs/deep_active_learning/survey_papers.json
        output_jsonl: Путь к выходному JSONL файлу для сохранения результатов.
            Если None, используется outputs/deep_active_learning/topics_raw.jsonl
        trace_id: Идентификатор трейса для логирования

    Returns:
        Словарь со статистикой обработки

    Raises:
        FileNotFoundError: Если файл со списком статей не найден
    """
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    logger.info(f"{trace_prefix}Начало извлечения тем из обзорных статей")

    # Определяем пути по умолчанию
    if survey_papers_json is None:
        project_root = Path(__file__).parent.parent.parent
        survey_papers_json = project_root / "outputs" / "deep_active_learning" / "survey_papers.json"

    if output_jsonl is None:
        project_root = Path(__file__).parent.parent.parent
        output_jsonl = project_root / "outputs" / "deep_active_learning" / "topics_raw.jsonl"

    # Загружаем список обзорных статей
    try:
        papers = load_survey_papers(survey_papers_json)
        logger.info(f"{trace_prefix}Загружено {len(papers)} обзорных статей")
    except Exception as e:
        logger.error(f"{trace_prefix}Ошибка загрузки списка статей: {e}")
        raise

    # Создаём выходную директорию если нужно
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Статистика
    stats = {
        "total_papers": len(papers),
        "processed_papers": 0,
        "total_sections": 0,
        "processed_sections": 0,
        "successful_extractions": 0,
        "failed_extractions": 0,
        "started_at": datetime.now().isoformat(),
    }

    # Обрабатываем каждую статью
    with output_path.open("w", encoding="utf-8") as f:
        for paper_idx, paper in enumerate(papers, 1):
            paper_id = paper["paper_id"]
            paper_title = paper["title"]
            pdf_path = paper["pdf_path"]

            logger.info(f"{trace_prefix}Обработка статьи {paper_idx}/{len(papers)}: {paper_id} - {paper_title}")

            # Проверяем существование PDF файла
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                # Пробуем найти файл относительно корня проекта
                project_root = Path(__file__).parent.parent.parent
                pdf_file = project_root / pdf_path
                if not pdf_file.exists():
                    logger.error(f"{trace_prefix}PDF файл не найден: {pdf_path}")
                    stats["failed_extractions"] += 1
                    continue

            # Извлекаем разделы из PDF
            try:
                sections = extract_sections_from_pdf(pdf_file, paper_id, trace_id)
                stats["total_sections"] += len(sections)
                logger.info(f"{trace_prefix}Извлечено {len(sections)} разделов из статьи {paper_id}")
            except Exception as e:
                logger.error(f"{trace_prefix}Ошибка извлечения разделов из {paper_id}: {e}")
                stats["failed_extractions"] += 1
                continue

            # Обрабатываем каждый раздел
            for section in sections:
                section_id = section["section_id"]
                section_heading = section["section_heading"]
                section_text = section["section_text"]

                logger.debug(f"{trace_prefix}Обработка раздела: {section_heading}")

                # Создаём промпт
                system_prompt, user_prompt = create_topic_extraction_prompt(
                    article_id=paper_id,
                    article_title=paper_title,
                    section_id=section_id,
                    section_heading=section_heading,
                    section_text=section_text,
                )

                # Вызываем LLM
                try:
                    llm_response = call_llm_for_topic_extraction(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        trace_id=trace_id,
                    )

                    # Парсим ответ
                    topics_data = json.loads(llm_response["content"])

                    # Формируем запись для JSONL
                    record = {
                        "article_id": paper_id,
                        "article_title": paper_title,
                        "section_id": section_id,
                        "section_heading": section_heading,
                        "topics": topics_data.get("topics", []),
                        "extracted_at": datetime.now().isoformat(),
                        "trace_id": trace_id,
                    }

                    # Сохраняем в JSONL
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()

                    stats["processed_sections"] += 1
                    stats["successful_extractions"] += 1

                    logger.info(
                        f"{trace_prefix}Извлечено {len(topics_data.get('topics', []))} тем из раздела {section_heading}"
                    )

                except Exception as e:
                    logger.error(f"{trace_prefix}Ошибка извлечения тем из раздела {section_id}: {e}")
                    stats["failed_extractions"] += 1
                    continue

            stats["processed_papers"] += 1

    stats["completed_at"] = datetime.now().isoformat()
    stats["duration_seconds"] = (
        datetime.fromisoformat(stats["completed_at"]) - datetime.fromisoformat(stats["started_at"])
    ).total_seconds()

    logger.info(f"{trace_prefix}Завершено извлечение тем. Статистика: {stats}")

    return stats

