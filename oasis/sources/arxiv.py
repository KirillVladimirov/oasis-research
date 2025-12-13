"""arXiv источник данных."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import feedparser
import requests
from loguru import logger

from oasis.io.cache import ApiCache
from oasis.sources.base import Metadata, PdfCandidate, SourcePort
from oasis.utils.text import clean_text_value

if TYPE_CHECKING:
    from oasis.models.reference import Reference


class ArxivSource(SourcePort):
    """Источник данных arXiv."""

    def __init__(self, api_url: str = "https://export.arxiv.org/api/query", cache: ApiCache | None = None):
        """Инициализация источника.

        Args:
            api_url: URL arXiv API
            cache: Экземпляр кэша для API ответов
        """
        self.api_url = api_url
        self.cache = cache

    @property
    def name(self) -> str:
        """Имя источника."""
        return "arxiv"

    def get_by_doi(self, doi: str) -> Metadata | None:
        """Получает метаданные по DOI (arXiv не поддерживает поиск по DOI напрямую).

        Args:
            doi: DOI публикации

        Returns:
            None (arXiv не поддерживает поиск по DOI)
        """
        # arXiv не предоставляет поиск по DOI
        return None

    def search_by_title(self, title: str) -> list[Metadata]:
        """Ищет публикации по названию через arXiv API.

        Args:
            title: Название публикации

        Returns:
            Список Metadata объектов
        """
        if not title:
            return []

        # Извлекаем ключевые токены из названия
        tokens = [t for t in re.findall(r"[A-Za-z0-9\-]{3,}", title)][:6]
        if not tokens:
            return []

        # Формируем запрос
        query = " AND ".join([f'all:"{t}"' for t in tokens])

        try:
            response = requests.get(
                self.api_url,
                params={
                    "search_query": query,
                    "start": 0,
                    "max_results": 1,
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                },
                timeout=60,
            )
            if response.status_code != 200:
                return []

            feed = feedparser.parse(response.text)
            if not feed.entries:
                return []

            entry = feed.entries[0]

            # Извлекаем arXiv ID
            arxiv_id = ""
            for link in entry.get("links", []):
                if link.get("rel") == "alternate" and "arxiv.org/abs/" in link.get("href", ""):
                    match = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", link["href"])
                    if match:
                        arxiv_id = match.group(1)
                        break

            # Извлекаем PDF
            pdf_url = ""
            for link in entry.get("links", []):
                if link.get("type") == "application/pdf":
                    pdf_url = link.get("href", "")
                    break

            # Авторы
            authors = "; ".join(
                [clean_text_value(a.get("name", "")) for a in (entry.get("authors") or [])]
            ) if entry.get("authors") else ""

            # Год
            updated = (entry.get("updated") or entry.get("published") or "")[:4]
            year = int(updated) if updated.isdigit() else None

            # Article URL
            article_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""

            return [
                Metadata(
                    title=clean_text_value(entry.get("title", "")),
                    authors=authors,
                    year=year,
                    venue="arXiv",
                    arxiv_id=arxiv_id,
                    article_url=article_url,
                    pdf_url=clean_text_value(pdf_url),
                    additional_urls=[article_url, pdf_url] if article_url or pdf_url else None,
                    source_name=self.name,
                    raw_data=entry,
                )
            ]
        except Exception as e:
            logger.debug(f"Ошибка при запросе arXiv по title: {e}")
            return []

    def get_pdf_candidates(self, ref: dict[str, Any] | Reference) -> list[PdfCandidate]:
        """Получает кандидатов на PDF файл для ссылки.

        Для arXiv: если есть arxiv_id, формируем PDF URL напрямую без запроса к API.

        Args:
            ref: Словарь или Reference объект с данными ссылки

        Returns:
            Список PdfCandidate объектов
        """
        from oasis.models.reference import Reference

        candidates = []

        # Поддерживаем как dict, так и Reference
        if isinstance(ref, dict):
            arxiv_id = ref.get("arxiv_id", "")
        else:
            arxiv_id = ref.arxiv_id

        # Если есть arxiv_id, формируем PDF URL напрямую
        if arxiv_id:
            arxiv_id_clean = str(arxiv_id).strip()
            # Проверяем формат arXiv ID (YYYY.NNNNN или YYYY.NNNNNN)
            if re.match(r"\d{4}\.\d{4,}", arxiv_id_clean):
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"
                candidates.append(
                    PdfCandidate(
                        url=pdf_url,
                        source=self.name,
                        confidence=1.0,  # Высокая уверенность для arXiv
                        validated=False,
                    )
                )
                logger.debug(f"Сформирован PDF URL для arXiv ID {arxiv_id_clean}: {pdf_url}")

        # Если не нашли по arxiv_id, пробуем базовый метод (по DOI или title)
        if not candidates:
            candidates = super().get_pdf_candidates(ref)

        return candidates

