"""Прямое извлечение ссылок из PDF текста (без GROBID)."""

from typing import Any

import fitz  # PyMuPDF
from loguru import logger

from oasis.parsing.regexes import find_urls, find_arxiv_id
from oasis.utils.text import normalize_text

def extract_references_from_pdf_text(
    pdf_path: str, trace_id: str | None = None
) -> list[dict[str, Any]]:
    """Прямое извлечение раздела References из PDF текста.

    Ищет раздел References/Bibliography и извлекает все строки как потенциальные ссылки.

    Args:
        pdf_path: Путь к PDF файлу
        trace_id: Идентификатор трейса для логирования

    Returns:
        Список словарей с сырыми данными ссылок
    """
    import re

    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    logger.info(f"{trace_prefix}Прямое извлечение раздела References из {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
        full_text_pages = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            full_text_pages.append((page_num, text))

        doc.close()
    except Exception as e:
        logger.error(f"{trace_prefix}Ошибка чтения PDF: {e}")
        return []

    # Поиск начала раздела References
    references_start = None
    references_pattern = re.compile(
        r"(?:^|\n)\s*(?:references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE
    )

    for page_num, text in full_text_pages:
        match = references_pattern.search(text)
        if match:
            references_start = (page_num, match.start())
            logger.info(f"{trace_prefix}Найден раздел References на странице {page_num + 1}")
            break

    if references_start is None:
        logger.warning(f"{trace_prefix}Раздел References не найден, ищем ссылки во всём документе")
        start_page = int(len(full_text_pages) * 0.8)
        full_text = "\n".join(text for _, text in full_text_pages[start_page:])
    else:
        start_page, start_pos = references_start
        full_text = "\n".join(
            text[start_pos:] if page_num == start_page else text
            for page_num, text in full_text_pages[start_page:]
        )

    # Обнаружение конца раздела References
    # Останавливаем парсинг при обнаружении разделов: Biographies, Acknowledgments, Appendix, etc.
    end_section_pattern = re.compile(
        r"^(?:biographies?|acknowledgments?|acknowledgements?|appendix|appendices|author\s+biography|about\s+the\s+author)",
        re.IGNORECASE | re.MULTILINE,
    )
    end_match = end_section_pattern.search(full_text)
    if end_match:
        end_pos = end_match.start()
        logger.info(f"{trace_prefix}Обнаружен конец раздела References на позиции {end_pos}, обрезаем текст")
        full_text = full_text[:end_pos]

    # Поиск нумерованных ссылок
    # Улучшенные паттерны: более точный захват полного текста ссылки
    # Используем более жадный захват, который захватывает текст до следующей ссылки
    # Включая возможные URL на следующих строках после "Available:"
    numbered_pattern1 = re.compile(
        r"^\[\s*(\d+)\s*\][\s\.]*(.+?)(?=^\[\s*\d+\s*\]|^\s*(?:biographies?|acknowledgments?|appendix)|$)",
        re.MULTILINE | re.DOTALL,
    )
    numbered_pattern2 = re.compile(
        r"^(\d+)[\s\.]+(.+?)(?=^\d+[\s\.]+|^\s*(?:biographies?|acknowledgments?|appendix)|$)",
        re.MULTILINE | re.DOTALL,
    )
    numbered_pattern3 = re.compile(
        r"^\(\s*(\d+)\s*\)[\s\.]*(.+?)(?=^\(\s*\d+\s*\)|^\s*(?:biographies?|acknowledgments?|appendix)|$)",
        re.MULTILINE | re.DOTALL,
    )

    references = []
    seen_numbers = set()

    def extract_from_pattern(pattern, ref_list):
        for match in pattern.finditer(full_text):
            ref_num = int(match.group(1))
            raw_text_extracted = match.group(2)
            match_end = match.end()

            # Улучшенный захват: проверяем, есть ли уже URL в захваченном тексте
            # Нормализуем текст для проверки (убираем переносы строк)
            raw_text_for_check = raw_text_extracted.replace("\n", " ").replace("\r", " ").replace("\t", " ")
            raw_text_for_check = re.sub(r"\s+", " ", raw_text_for_check)
            has_url_in_text = bool(re.search(r"https?://", raw_text_for_check, re.IGNORECASE))
            
            if not has_url_in_text:
                # Ищем следующую ссылку того же типа
                next_match = pattern.search(full_text, match_end)
                if next_match:
                    # Захватываем текст между текущей и следующей ссылкой
                    gap_text = full_text[match_end:next_match.start()]
                    # Ищем URL в промежутке (обычно после "Available:" или "[Online]")
                    # Используем улучшенный поиск с нормализацией для разорванных URL
                    gap_normalized = gap_text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
                    gap_normalized = re.sub(r"\s+", " ", gap_normalized)
                    url_match = re.search(r"https?://[^\s\)\]]+", gap_normalized, re.IGNORECASE)
                    if url_match:
                        # Добавляем URL к тексту ссылки
                        url_found = url_match.group(0).rstrip(".,;)")
                        raw_text_extracted += " " + url_found
                else:
                    # Если следующей ссылки нет, ищем URL в оставшемся тексте
                    remaining_text = full_text[match_end:match_end + 500]  # Ограничиваем поиск
                    # Нормализуем для поиска разорванных URL
                    remaining_normalized = remaining_text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
                    remaining_normalized = re.sub(r"\s+", " ", remaining_normalized)
                    url_match = re.search(r"https?://[^\s\)\]]+", remaining_normalized, re.IGNORECASE)
                    if url_match:
                        url_found = url_match.group(0).rstrip(".,;)")
                        raw_text_extracted += " " + url_found

            # Нормализация сразу после извлечения
            raw_text_cleaned = str(raw_text_extracted).replace("\n", " ").replace("\r", " ").replace("\t", " ")
            raw_text_cleaned = re.sub(r"\s+", " ", raw_text_cleaned)

            raw_text_normalized = normalize_text(raw_text_cleaned)

            if "\n" in raw_text_normalized or "\r" in raw_text_normalized or "\t" in raw_text_normalized:
                raw_text_normalized = raw_text_normalized.replace("\n", " ").replace("\r", " ").replace("\t", " ")
                raw_text_normalized = re.sub(r"\s+", " ", raw_text_normalized).strip()
                raw_text_normalized = normalize_text(raw_text_normalized)

            if ref_num not in seen_numbers and len(raw_text_normalized) > 10:
                seen_numbers.add(ref_num)

                # Извлекаем базовую информацию
                doi_match = re.search(
                    r"(?:doi\.org/|doi[:\s]+)(10\.\d{4,}/[^\s\)]+)", raw_text_normalized, re.IGNORECASE
                )
                arxiv_match = re.search(
                    r"(?:arxiv[:\s]+|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,})",
                    raw_text_normalized,
                    re.IGNORECASE,
                )

                # Улучшенное извлечение title:
                # 1. Ищем все заголовки в кавычках и берём самый длинный (наиболее надёжный)
                # 2. Если кавычек нет, ищем паттерн после авторов
                # 3. Fallback: берём первые значимые слова (но не менее 10 символов)
                title_normalized = ""
                
                # Паттерн 1: ищем все тексты в кавычках и берём самый длинный (обычно это заголовок)
                quoted_matches = list(re.finditer(r'"([^"]{10,200})"', raw_text_normalized))
                if quoted_matches:
                    # Берём самый длинный текст в кавычках
                    longest_quoted = max(quoted_matches, key=lambda m: len(m.group(1)))
                    title_normalized = normalize_text(longest_quoted.group(1))
                
                # Паттерн 2: заголовок без кавычек после авторов
                if not title_normalized or len(title_normalized) < 10:
                    # Ищем последовательность: авторы, запятая, пробел, затем заголовок
                    # Заголовок обычно длиннее 15 символов и заканчивается на запятую или точку
                    author_pattern = r"(?:[A-Z][.\s]*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s*,\s*[A-Z][.\s]*[A-Z][a-z]+)*(?:\s+and\s+[A-Z][.\s]*[A-Z][a-z]+)*)"
                    title_after_authors_no_quotes = re.compile(
                        author_pattern + r"[,\s]+" + r"([^,]{15,200})(?:,|\.|$)",  # заголовок без кавычек
                        re.IGNORECASE,
                    )
                    match = title_after_authors_no_quotes.search(raw_text_normalized)
                    if match:
                        candidate = normalize_text(match.group(1).strip())
                        # Убираем лишние пробелы и проверяем длину
                        candidate = re.sub(r"\s+", " ", candidate).strip()
                        # Проверяем, что это не часть авторов (например, "and M. Potthast")
                        if len(candidate) >= 15 and not candidate.lower().startswith(("and ", "et al")):
                            title_normalized = candidate
                
                # Паттерн 4: fallback - берём первые значимые слова (но не менее 15 символов)
                # Пропускаем короткие инициалы и ищем более длинный текст
                if not title_normalized or len(title_normalized) < 15:
                    # Ищем последовательность слов, начинающуюся с заглавной буквы
                    # Минимум 15 символов, до точки, запятой или конца строки
                    # Но пропускаем паттерны авторов (одна буква + точка)
                    fallback_match = re.search(
                        r"(?:^|,\s+)([A-Z][A-Za-z\s]{14,150}?)(?:\.|,|$)", raw_text_normalized
                    )
                    if fallback_match:
                        potential_title = fallback_match.group(1).strip()
                        # Убираем инициалы авторов в начале (паттерн: одна буква + точка + пробел)
                        potential_title = re.sub(r"^[A-Z]\.\s+", "", potential_title)
                        potential_title = re.sub(r"\s+", " ", potential_title).strip()
                        # Проверяем, что это не просто инициал, не часть авторов ("and X", "et al")
                        if (
                            len(potential_title) >= 15
                            and not re.match(r"^[A-Z]\s*$", potential_title)
                            and not potential_title.lower().startswith(("and ", "et al"))
                        ):
                            title_normalized = normalize_text(potential_title)

                # Извлечение URL из raw_text
                # Ищем все URL и выбираем первый валидный (ссылка на статью, не PDF)
                article_url = ""
                all_urls = find_urls(raw_text_normalized)
                for url in all_urls:
                    url_lower = url.lower()
                    # Проверяем, что это не PDF-ссылка
                    # Исключаем ссылки, которые явно указывают на PDF
                    is_pdf = (
                        url_lower.endswith(".pdf")
                        or "/pdf/" in url_lower
                        or url_lower.endswith("/pdf")
                        or "filetype=pdf" in url_lower
                    )
                    if not is_pdf:
                        article_url = url
                        break

                # Улучшенное извлечение arXiv ID (приоритет: URL в raw_text > article_url > паттерн в тексте > DOI):
                # 1. Сначала проверяем все URL в raw_text (наиболее надежный способ для разорванных URL)
                # 2. Затем проверяем article_url
                # 3. Затем проверяем паттерн в raw_text_normalized (fallback для текстовых упоминаний)
                # 4. В конце проверяем DOI (10.48550/arXiv.XXXX.XXXXX)
                arxiv_id_final = ""
                
                # Шаг 1: Проверяем все URL в raw_text (включая разорванные)
                for url in all_urls:
                    arxiv_from_url = find_arxiv_id(url)
                    if arxiv_from_url:
                        arxiv_id_final = arxiv_from_url
                        break
                
                # Шаг 2: Проверяем article_url
                if not arxiv_id_final and article_url:
                    arxiv_from_url = find_arxiv_id(article_url)
                    if arxiv_from_url:
                        arxiv_id_final = arxiv_from_url
                
                # Шаг 3: Проверяем паттерн в raw_text_normalized (fallback для текстовых упоминаний типа "arXiv:1506.02158")
                if not arxiv_id_final and arxiv_match:
                    arxiv_id_final = arxiv_match.group(1)
                
                # Шаг 4: Проверяем DOI (10.48550/arXiv.XXXX.XXXXX)
                if not arxiv_id_final and doi_match:
                    doi_value = doi_match.group(1).lower()
                    if "arxiv" in doi_value:
                        # Ищем паттерн arXiv.XXXX.XXXXX в DOI
                        arxiv_in_doi = re.search(r"arxiv\.(\d{4}\.\d{4,})", doi_value, re.IGNORECASE)
                        if arxiv_in_doi:
                            arxiv_id_final = arxiv_in_doi.group(1)

                ref_list.append(
                    {
                        "ref_number": ref_num,
                        "raw_text": raw_text_normalized,
                        "title": title_normalized,
                        "doi": doi_match.group(1).lower() if doi_match else "",
                        "arxiv_id": arxiv_id_final,
                        "authors": "",
                        "year": None,
                        "venue": "",
                        "article_url": article_url,
                    }
                )

    extract_from_pattern(numbered_pattern1, references)
    extract_from_pattern(numbered_pattern2, references)
    extract_from_pattern(numbered_pattern3, references)

    logger.info(f"{trace_prefix}Прямое извлечение нашло {len(references)} ссылок")
    return references

