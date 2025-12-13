"""Semantic Scholar источник данных."""

from typing import Any

import requests
from loguru import logger

from oasis.io.cache import ApiCache
from oasis.sources.base import Metadata, PdfCandidate, SourcePort
from oasis.utils.text import clean_text_value


class SemanticScholarSource(SourcePort):
    """Источник данных Semantic Scholar."""

    def __init__(
        self,
        search_url: str = "https://api.semanticscholar.org/graph/v1/paper/search",
        paper_url: str = "https://api.semanticscholar.org/graph/v1/paper/",
        cache: ApiCache | None = None,
    ):
        """Инициализация источника.

        Args:
            search_url: URL для поиска
            paper_url: URL для получения по DOI/ID
            cache: Экземпляр кэша для API ответов
        """
        self.search_url = search_url
        self.paper_url = paper_url
        self.cache = cache

    @property
    def name(self) -> str:
        """Имя источника."""
        return "semanticscholar"

    def get_by_doi(self, doi: str) -> Metadata | None:
        """Получает метаданные по DOI.

        Args:
            doi: DOI публикации

        Returns:
            Metadata объект или None если не найдено
        """
        if not doi:
            return None

        # Проверяем кэш
        cache_key = f"s2:doi:{doi}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return self._parse_response(cached)

        url = f"{self.paper_url}DOI:{requests.utils.quote(doi)}"
        try:
            response = requests.get(
                url, params={"fields": "title,authors,venue,year,url,openAccessPdf"}, timeout=60
            )
            if response.status_code == 429:
                # Rate limit - не ошибка сети
                return None
            if response.status_code != 200:
                return None

            js = response.json()
            if self.cache:
                self.cache.set(cache_key, js)

            return self._parse_response(js)
        except Exception as e:
            logger.debug(f"Ошибка при запросе Semantic Scholar по DOI: {e}")
            return None

    def search_by_title(self, title: str) -> list[Metadata]:
        """Ищет публикации по названию.

        Args:
            title: Название публикации

        Returns:
            Список Metadata объектов
        """
        if not title:
            return []

        # Проверяем кэш
        import hashlib
        cache_key = f"s2:title:{hashlib.sha256(title.encode('utf-8')).hexdigest()[:16]}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return [self._parse_response(cached)] if cached else []

        try:
            response = requests.get(
                self.search_url,
                params={
                    "query": title,
                    "limit": 1,
                    "fields": "title,authors,venue,year,url,openAccessPdf,externalIds",
                },
                timeout=60,
            )
            if response.status_code == 429:
                # Rate limit
                return []
            if response.status_code != 200:
                return []

            arr = response.json().get("data", [])
            if not arr:
                return []

            js = arr[0]
            if self.cache:
                self.cache.set(cache_key, js)

            metadata = self._parse_response(js)
            return [metadata] if metadata else []
        except Exception as e:
            logger.debug(f"Ошибка при запросе Semantic Scholar по title: {e}")
            return []

    def _parse_response(self, js: dict[str, Any]) -> Metadata | None:
        """Парсит ответ от Semantic Scholar API в Metadata.

        Args:
            js: JSON ответ от API

        Returns:
            Metadata объект или None
        """
        if not js:
            return None

        # Извлекаем авторов
        authors = js.get("authors") or []
        authors_str = "; ".join([clean_text_value(a.get("name", "")) for a in authors])

        # Извлекаем externalIds для DOI и arXiv
        external_ids = js.get("externalIds") or {}
        doi = external_ids.get("DOI", "")
        arxiv_id = external_ids.get("ArXiv", "")

        # PDF URL
        open_access_pdf = js.get("openAccessPdf") or {}
        pdf_url = clean_text_value(open_access_pdf.get("url", ""))

        return Metadata(
            title=clean_text_value(js.get("title", "")),
            authors=authors_str,
            year=js.get("year"),
            venue=clean_text_value(js.get("venue", "")),
            doi=clean_text_value(doi),
            arxiv_id=clean_text_value(arxiv_id),
            article_url=clean_text_value(js.get("url", "")),
            pdf_url=pdf_url,
            additional_urls=[js.get("url", ""), pdf_url] if js.get("url") or pdf_url else None,
            source_name=self.name,
            raw_data=js,
        )

    def get_pdf_candidates(self, ref: dict[str, Any]) -> list[PdfCandidate]:
        """Получает кандидатов на PDF файл для ссылки через Semantic Scholar.

        Использует DOI или title для поиска Open Access PDF.

        Args:
            ref: Словарь с данными ссылки

        Returns:
            Список PdfCandidate объектов
        """
        candidates = []

        # Приоритет 1: Поиск по DOI
        doi = ref.get("doi", "")
        if doi and isinstance(doi, str):
            doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
            try:
                metadata = self.get_by_doi(doi_clean)
                if metadata and metadata.pdf_url:
                    # Проверяем что это действительно PDF URL, а не DOI
                    if not metadata.pdf_url.startswith("https://doi.org/"):
                        candidates.append(
                            PdfCandidate(
                                url=metadata.pdf_url,
                                source=self.name,
                                confidence=0.85,  # Средняя уверенность для SS PDF
                            )
                        )
                        logger.debug(f"Найден PDF через Semantic Scholar по DOI {doi_clean}: {metadata.pdf_url}")
                        return candidates  # Возвращаем сразу если нашли по DOI
            except Exception as e:
                logger.debug(f"Ошибка при получении PDF по DOI {doi_clean} через Semantic Scholar: {e}")

        # Приоритет 2: Поиск по title (если нет DOI или не нашли по DOI)
        title = ref.get("title", "")
        if title:
            try:
                results = self.search_by_title(title)
                # Проверяем первые несколько результатов
                for result in results[:3]:
                    if result.pdf_url and not result.pdf_url.startswith("https://doi.org/"):
                        candidates.append(
                            PdfCandidate(
                                url=result.pdf_url,
                                source=self.name,
                                confidence=0.75,  # Ниже уверенность для поиска по title
                            )
                        )
                        logger.debug(f"Найден PDF через Semantic Scholar по title '{title[:50]}...': {result.pdf_url}")
            except Exception as e:
                logger.debug(f"Ошибка при получении PDF по title через Semantic Scholar: {e}")

        return candidates

