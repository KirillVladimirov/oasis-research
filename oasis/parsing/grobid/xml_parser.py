"""Парсер отдельных biblStruct элементов из GROBID TEI XML."""

import re
from typing import Any

from lxml import etree

from oasis.parsing.regexes import find_urls, find_arxiv_id
from oasis.utils.text import normalize_text

GROBID_NAMESPACE = "http://www.tei-c.org/ns/1.0"
GROBID_NSMAP = {"tei": GROBID_NAMESPACE}


def clean_raw_text(raw_text: str) -> str:
    """Очищает сырой текст от управляющих символов и лишних пробелов.
    
    Args:
        raw_text: Исходный текст
        
    Returns:
        Очищенный текст
    """
    # Агрессивная нормализация
    text = raw_text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = normalize_text(text)
    
    # Многократная проверка и очистка
    iterations = 0
    max_iterations = 10
    while ("\n" in text or "\r" in text or "\t" in text) and iterations < max_iterations:
        text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        text = re.sub(r"\s+", " ", text).strip()
        text = normalize_text(text)
        iterations += 1
    
    if "\n" in text or "\r" in text or "\t" in text:
        text = " ".join(text.split())
        text = normalize_text(text)
    
    return text


def extract_raw_text(bibl_struct: etree._Element) -> str:
    """Извлекает сырой текст из biblStruct элемента.
    
    Args:
        bibl_struct: XML элемент biblStruct
        
    Returns:
        Сырой текст ссылки
    """
    raw_text_full = ""
    for text_part in bibl_struct.itertext():
        if text_part:
            cleaned_part = str(text_part)
            cleaned_part = cleaned_part.replace("\n", " ").replace("\r", " ").replace("\t", " ")
            cleaned_part = re.sub(r"[\u00A0\u2000-\u200B\u202F\u205F\u3000]", " ", cleaned_part)
            cleaned_part = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", cleaned_part)
            raw_text_full += cleaned_part + " "
    
    return clean_raw_text(raw_text_full)


def extract_arxiv_id(ref: dict[str, Any]) -> str:
    """Извлекает arXiv ID из различных полей ссылки.
    
    Args:
        ref: Словарь с данными ссылки
        
    Returns:
        arXiv ID или пустая строка
    """
    arxiv_id_final = ""
    
    # 1. Из raw_text
    if ref.get("raw_text"):
        arxiv_match = re.search(
            r"(?:arxiv[:\s]+|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,})", 
            ref["raw_text"], 
            re.IGNORECASE
        )
        if arxiv_match:
            arxiv_id_final = arxiv_match.group(1)
    
    # 2. Из title
    if not arxiv_id_final:
        arxiv_match = re.search(
            r"arxiv[:\s]+(\d{4}\.\d{4,})", 
            ref.get("title", ""), 
            re.IGNORECASE
        )
        if arxiv_match:
            arxiv_id_final = arxiv_match.group(1)
    
    # 3. Из article_url
    if not arxiv_id_final and ref.get("article_url"):
        arxiv_from_url = find_arxiv_id(ref["article_url"])
        if arxiv_from_url:
            arxiv_id_final = arxiv_from_url
    
    # 4. Из DOI
    if not arxiv_id_final and ref.get("doi"):
        doi_value = ref["doi"].lower()
        if "arxiv" in doi_value:
            arxiv_in_doi = re.search(r"arxiv\.(\d{4}\.\d{4,})", doi_value, re.IGNORECASE)
            if arxiv_in_doi:
                arxiv_id_final = arxiv_in_doi.group(1)
    
    # 5. Из всех URL в raw_text
    if not arxiv_id_final and ref.get("raw_text"):
        all_urls_check = find_urls(ref["raw_text"])
        for url in all_urls_check:
            arxiv_from_url = find_arxiv_id(url)
            if arxiv_from_url:
                arxiv_id_final = arxiv_from_url
                break
    
    return arxiv_id_final


def extract_article_url(raw_text: str) -> str:
    """Извлекает URL статьи (не PDF) из текста.
    
    Args:
        raw_text: Сырой текст
        
    Returns:
        URL статьи или пустая строка
    """
    if not raw_text:
        return ""
    
    all_urls = find_urls(raw_text)
    for url in all_urls:
        url_lower = url.lower()
        # Проверяем, что это не PDF-ссылка
        is_pdf = (
            url_lower.endswith(".pdf")
            or "/pdf/" in url_lower
            or url_lower.endswith("/pdf")
            or "filetype=pdf" in url_lower
        )
        if not is_pdf:
            return url
    
    return ""


def parse_bibl_struct(bibl_struct: etree._Element) -> dict[str, Any] | None:
    """Парсинг одного biblStruct элемента TEI XML.

    Args:
        bibl_struct: XML элемент biblStruct

    Returns:
        Словарь с полями ссылки или None при ошибке
    """
    ref: dict[str, Any] = {
        "title": "",
        "raw_text": "",
        "authors": [],
        "year": None,
        "doi": "",
        "arxiv_id": "",
        "venue": "",
        "first_author": "",
    }

    # Сохраняем сырой текст ссылки
    ref["raw_text"] = extract_raw_text(bibl_struct)

    # Title из analytic/title или monogr/title
    title_nodes = bibl_struct.xpath(
        ".//tei:analytic/tei:title | .//tei:monogr/tei:title",
        namespaces=GROBID_NSMAP,
    )
    if title_nodes:
        ref["title"] = normalize_text(" ".join(title_nodes[0].itertext()).strip())

    # Authors из analytic/author или monogr/author
    author_nodes = bibl_struct.xpath(
        ".//tei:author//tei:persName", namespaces=GROBID_NSMAP
    )
    authors = []
    for author in author_nodes:
        forenames = author.xpath(".//tei:forename", namespaces=GROBID_NSMAP)
        surnames = author.xpath(".//tei:surname", namespaces=GROBID_NSMAP)
        forename_str = " ".join(fn.text or "" for fn in forenames).strip()
        surname_str = " ".join(sn.text or "" for sn in surnames).strip()
        if forename_str or surname_str:
            authors.append(f"{forename_str} {surname_str}".strip())
    ref["authors"] = authors
    ref["first_author"] = authors[0] if authors else ""

    # Year из date@when или date text
    date_nodes = bibl_struct.xpath(
        ".//tei:date[@when] | .//tei:date", namespaces=GROBID_NSMAP
    )
    for date_node in date_nodes:
        when_attr = date_node.get("when", "")
        if when_attr:
            year_match = re.search(r"(\d{4})", when_attr)
            if year_match:
                year = int(year_match.group(1))
                if 1800 <= year <= 2100:
                    ref["year"] = year
                    break
        date_text = " ".join(date_node.itertext())
        year_match = re.search(r"\b(19|20)\d{2}\b", date_text)
        if year_match:
            year = int(year_match.group(0))
            if 1800 <= year <= 2100:
                ref["year"] = year
                break

    # DOI из idno type="DOI"
    doi_nodes = bibl_struct.xpath(
        './/tei:idno[@type="DOI"]', namespaces=GROBID_NSMAP
    )
    if doi_nodes:
        ref["doi"] = " ".join(doi_nodes[0].itertext()).strip()

    # Venue (journal/conference) из monogr/title level="j"
    venue_nodes = bibl_struct.xpath(
        './/tei:monogr/tei:title[@level="j"]', namespaces=GROBID_NSMAP
    )
    if not venue_nodes:
        venue_nodes = bibl_struct.xpath(
            ".//tei:monogr/tei:title[not(ancestor::tei:analytic)]",
            namespaces=GROBID_NSMAP,
        )
    if venue_nodes:
        ref["venue"] = normalize_text(" ".join(venue_nodes[0].itertext()).strip())

    # Извлечение URL статьи
    ref["article_url"] = extract_article_url(ref["raw_text"])
    
    # Извлечение arXiv ID
    ref["arxiv_id"] = extract_arxiv_id(ref)

    return ref

