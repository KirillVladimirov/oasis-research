"""Утилиты для работы с URL."""

import json
import re
from typing import Any

import pandas as pd


def extract_pdf_from_additional_urls(add_urls_str: str) -> str:
    """Извлекает первую валидную PDF ссылку из additional_urls JSON.

    Args:
        add_urls_str: JSON строка со списком URLs

    Returns:
        Первая найденная PDF ссылка или пустая строка
    """
    if not add_urls_str or pd.isna(add_urls_str):
        return ""
    try:
        urls = json.loads(add_urls_str) if isinstance(add_urls_str, str) else []
        if not isinstance(urls, list):
            return ""
        for url in urls:
            url_str = str(url).strip()
            if not url_str:
                continue
            # Проверяем что это PDF ссылка
            if (
                "arxiv.org/pdf" in url_str.lower()
                or "/pdf" in url_str.lower()
                or url_str.lower().endswith(".pdf")
                or ".pdf" in url_str.lower().split("?")[0]
            ):
                # Исключаем localhost и /nan
                if not ("localhost" in url_str.lower() or "/nan" in url_str.lower()):
                    return url_str
    except Exception:
        pass
    return ""


def pdf_url_to_article_url(pdf_url: str) -> str:
    """Преобразует PDF URL в article URL (для arXiv, IEEE, ACM и др.).

    Args:
        pdf_url: URL на PDF файл

    Returns:
        URL на страницу публикации или пустая строка
    """
    if not pdf_url or not isinstance(pdf_url, str):
        return ""

    pdf_url = pdf_url.strip()
    if not pdf_url:
        return ""

    # arXiv: https://arxiv.org/pdf/2111.02275v2 → https://arxiv.org/abs/2111.02275
    # или: http://arxiv.org/pdf/2111.02275v2.pdf → https://arxiv.org/abs/2111.02275
    if "arxiv.org/pdf" in pdf_url.lower():
        # Извлекаем arXiv ID
        match = re.search(r"arxiv\.org/pdf/(\d{4}\.\d{4,})(v\d+)?(?:\.pdf)?", pdf_url, re.IGNORECASE)
        if match:
            arxiv_id = match.group(1)  # Без версии
            return f"https://arxiv.org/abs/{arxiv_id}"

    # IEEE: может быть /stamp/stamp.jsp?tp=&arnumber=X или /document/X
    # Преобразуем в /document/X
    if "ieee" in pdf_url.lower() and "arnumber=" in pdf_url:
        match = re.search(r"arnumber=(\d+)", pdf_url)
        if match:
            doc_id = match.group(1)
            return f"https://ieeexplore.ieee.org/document/{doc_id}"

    # ACM: https://dl.acm.org/doi/pdf/10.1145/X → https://dl.acm.org/doi/10.1145/X
    if "acm.org" in pdf_url.lower() and "/pdf/" in pdf_url:
        return pdf_url.replace("/pdf/", "/")

    # Для остальных случаев - убираем .pdf в конце если есть
    if pdf_url.endswith(".pdf"):
        return pdf_url[:-4]

    return ""
