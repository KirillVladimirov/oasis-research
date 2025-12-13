"""Unpaywall источник данных."""

import os
from typing import Any

import requests
from loguru import logger

from oasis.io.cache import ApiCache
from oasis.sources.base import Metadata, PdfCandidate, SourcePort
from oasis.utils.text import clean_text_value


class UnpaywallSource(SourcePort):
    """Источник данных Unpaywall (для получения PDF по DOI)."""

    def __init__(
        self,
        api_url: str = "https://api.unpaywall.org/v2/",
        cache: ApiCache | None = None,
    ):
        """Инициализация источника.

        Args:
            api_url: Базовый URL API Unpaywall
            cache: Экземпляр кэша для API ответов
        """
        self.api_url = api_url
        self.cache = cache
        self.mailto = os.getenv("UNPAYWALL_MAILTO", os.getenv("OPENALEX_MAILTO", "").strip())

    @property
    def name(self) -> str:
        """Имя источника."""
        return "unpaywall"

    def get_by_doi(self, doi: str) -> Metadata | None:
        """Получает метаданные по DOI (главным образом для получения PDF).

        Args:
            doi: DOI публикации

        Returns:
            Metadata объект или None если не найдено
        """
        if not doi or not self.mailto:
            return None

        # Проверяем кэш
        cache_key = f"unpaywall:{doi}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return self._parse_response(cached, doi)

        try:
            response = requests.get(
                f"{self.api_url}{requests.utils.quote(doi)}",
                params={"email": self.mailto},
                timeout=60,
            )
            if response.status_code != 200:
                return None

            js = response.json()
            if self.cache:
                self.cache.set(cache_key, js)

            return self._parse_response(js, doi)
        except Exception as e:
            logger.debug(f"Ошибка при запросе Unpaywall по DOI: {e}")
            return None

    def search_by_title(self, title: str) -> list[Metadata]:
        """Ищет публикации по названию (Unpaywall не поддерживает поиск по title).

        Args:
            title: Название публикации

        Returns:
            Пустой список
        """
        return []

    def get_pdf_candidates(self, ref: dict[str, Any]) -> list[PdfCandidate]:
        """Получает кандидатов на PDF файл для ссылки через Unpaywall.

        Использует DOI для поиска Open Access PDF.

        Args:
            ref: Словарь с данными ссылки

        Returns:
            Список PdfCandidate объектов
        """
        candidates = []

        # Извлекаем DOI
        doi = ref.get("doi", "")
        if not doi or not isinstance(doi, str):
            return candidates

        # Очищаем DOI от префикса
        doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()

        if not doi_clean or not self.mailto:
            return candidates

        try:
            # Получаем метаданные по DOI
            metadata = self.get_by_doi(doi_clean)
            if metadata and metadata.pdf_url:
                # Проверяем что это действительно PDF URL, а не DOI
                if not metadata.pdf_url.startswith("https://doi.org/"):
                    candidates.append(
                        PdfCandidate(
                            url=metadata.pdf_url,
                            source=self.name,
                            confidence=0.95,  # Высокая уверенность для Unpaywall OA PDF
                        )
                    )
                    logger.debug(f"Найден OA PDF через Unpaywall для DOI {doi_clean}: {metadata.pdf_url}")
        except Exception as e:
            logger.debug(f"Ошибка при получении PDF для DOI {doi_clean} через Unpaywall: {e}")

        return candidates

    def _parse_response(self, js: dict[str, Any], doi: str) -> Metadata | None:
        """Парсит ответ от Unpaywall API в Metadata.

        Args:
            js: JSON ответ от API
            doi: DOI для которого был запрос

        Returns:
            Metadata объект или None
        """
        if not js:
            return None

        best_oa_location = js.get("best_oa_location") or {}
        article_url = best_oa_location.get("url", "")
        pdf_url = best_oa_location.get("url_for_pdf", "")

        return Metadata(
            title="",  # Unpaywall не предоставляет title
            authors="",  # Unpaywall не предоставляет authors
            year=None,  # Unpaywall не предоставляет year
            venue="",
            doi=doi,
            article_url=clean_text_value(article_url),
            pdf_url=clean_text_value(pdf_url),
            additional_urls=[article_url, pdf_url] if article_url or pdf_url else None,
            source_name=self.name,
            raw_data=js,
        )

