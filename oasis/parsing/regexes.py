"""Regex парсеры для извлечения DOI, arXiv ID, URL и авторов из текста."""

import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from loguru import logger


def find_doi(text: str) -> str | None:
    """Находит DOI в тексте.

    Args:
        text: Текст для поиска

    Returns:
        DOI строка или None если не найден
    """
    if not text or not isinstance(text, str):
        return None

    doi_pattern = re.compile(
        r"(?:doi\.org/|doi[:\s]+)(10\.\d{4,}/[^\s\)]+)", re.IGNORECASE
    )
    match = doi_pattern.search(text)
    if match:
        doi = match.group(1).rstrip(".,;)").lower()
        return doi
    return None


def find_arxiv_id(text: str) -> str | None:
    """Находит arXiv ID в тексте.

    Args:
        text: Текст для поиска

    Returns:
        arXiv ID строка (формат YYYY.NNNNN) или None если не найден
    """
    if not text or not isinstance(text, str):
        return None

    arxiv_pattern = re.compile(
        r"(?:arxiv[:\s]+|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,})", re.IGNORECASE
    )
    match = arxiv_pattern.search(text)
    if match:
        return match.group(1)
    return None


def find_urls(text: str) -> list[str]:
    """Находит все URL в тексте, включая разорванные на строки.

    Args:
        text: Текст для поиска

    Returns:
        Список найденных URL
    """
    if not text or not isinstance(text, str):
        return []

    # Сначала нормализуем текст: заменяем переносы строк и табуляции на пробелы
    # Это нужно для обработки URL, разорванных на несколько строк в PDF
    normalized_text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # Убираем множественные пробелы, но сохраняем один пробел
    normalized_text = re.sub(r"\s+", " ", normalized_text)

    # Паттерн для поиска URL: https:// или http://, затем всё до пробела, скобки, запятой или точки с запятой
    # URL может содержать почти любые символы, но не пробелы, закрывающие скобки/квадратные скобки
    url_pattern = re.compile(
        r"https?://[^\s\)\]]+", re.IGNORECASE
    )
    
    matches = url_pattern.findall(normalized_text)
    
    # Очищаем URL от завершающих символов
    cleaned = []
    for url in matches:
        # Убираем завершающие точки, запятые, скобки и точки с запятой
        url_clean = url.rstrip(".,;)")
        # Убираем завершающие пробелы
        url_clean = url_clean.strip()
        if url_clean and url_clean.startswith(("http://", "https://")):
            cleaned.append(url_clean)
    
    return cleaned


def parse_references_regex(pdf_path: str, trace_id: str | None = None) -> list[dict[str, Any]]:
    """Fallback-парсер ссылок через regex при недоступности GROBID.

    Args:
        pdf_path: Путь к PDF файлу
        trace_id: Идентификатор трейса для логирования

    Returns:
        Список словарей с минимальными полями (doi, arxiv_id)
    """
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    logger.info(f"{trace_prefix}Fallback: парсинг ссылок через regex из {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
        full_text = "\n".join(page.get_text() for page in doc)
        doc.close()
    except Exception as e:
        logger.error(f"{trace_prefix}Ошибка чтения PDF: {e}")
        return []

    references = []
    seen_dois = set()
    seen_arxiv = set()

    # Поиск DOI
    for doi in find_urls(full_text):
        if "doi.org" in doi.lower():
            doi_match = re.search(r"10\.\d{4,}/[^\s\)]+", doi, re.IGNORECASE)
            if doi_match:
                doi_val = doi_match.group(0).lower()
                if doi_val not in seen_dois:
                    references.append({"doi": doi_val, "arxiv_id": "", "title": ""})
                    seen_dois.add(doi_val)

    # Поиск arXiv ID
    arxiv_id = find_arxiv_id(full_text)
    if arxiv_id and arxiv_id not in seen_arxiv:
        # Проверяем, не добавили ли уже с DOI
        found = False
        for ref in references:
            if ref["arxiv_id"] == arxiv_id:
                found = True
                break
        if not found:
            references.append({"doi": "", "arxiv_id": arxiv_id, "title": ""})
            seen_arxiv.add(arxiv_id)

    logger.info(f"{trace_prefix}Regex нашел {len(references)} уникальных идентификаторов")
    return references

