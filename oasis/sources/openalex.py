"""OpenAlex источник данных."""

import os
from typing import Any

import requests
from loguru import logger

from oasis.io.cache import ApiCache
from oasis.sources.base import Metadata, SourcePort
from oasis.utils.text import clean_text_value


class OpenAlexSource(SourcePort):
    """Источник данных OpenAlex."""

    def __init__(self, base_url: str = "https://api.openalex.org/works", cache: ApiCache | None = None):
        """Инициализация источника.

        Args:
            base_url: Базовый URL API OpenAlex
            cache: Экземпляр кэша для API ответов
        """
        self.base_url = base_url
        self.cache = cache
        self.mailto = os.getenv("OPENALEX_MAILTO", "").strip()

    @property
    def name(self) -> str:
        """Имя источника."""
        return "openalex"

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
        cache_key = f"oa:doi:{doi}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return self._parse_response(cached)

        params = {"fields": "title,doi,authorships,publication_year,host_venue,primary_location"}
        if self.mailto:
            params["mailto"] = self.mailto

        try:
            # Пробуем оба варианта URL
            for path in (f"{self.base_url}/doi:{doi}", f"{self.base_url}/https://doi.org/{doi}"):
                try:
                    response = requests.get(path, params=params, timeout=60)
                    if response.status_code == 200:
                        js = response.json()
                        if self.cache:
                            self.cache.set(cache_key, js)
                        return self._parse_response(js)
                    if response.status_code in (400, 404):
                        break
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"Ошибка при запросе OpenAlex по DOI: {e}")

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
        cache_key = f"oa:title:{hashlib.sha256(title.encode('utf-8')).hexdigest()[:16]}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                result = self._parse_response(cached)
                return [result] if result else []

        params = {
            "search": f'title.search:"{title}"',
            "per_page": 1,
            "sort": "relevance_score:desc",
            "fields": "title,doi,authorships,publication_year,host_venue,primary_location",
        }
        if self.mailto:
            params["mailto"] = self.mailto

        try:
            response = requests.get(self.base_url, params=params, timeout=60)
            if response.status_code != 200:
                return []

            js = response.json().get("results") or []
            if not js:
                return []

            result = js[0]
            if self.cache:
                self.cache.set(cache_key, result)

            metadata = self._parse_response(result)
            return [metadata] if metadata else []
        except Exception as e:
            logger.debug(f"Ошибка при запросе OpenAlex по title: {e}")
            return []

    def _parse_response(self, js: dict[str, Any]) -> Metadata | None:
        """Парсит ответ от OpenAlex API в Metadata.

        Args:
            js: JSON ответ от API

        Returns:
            Metadata объект или None
        """
        if not js:
            return None

        # Извлекаем авторов
        authorships = js.get("authorships") or []
        authors_str = "; ".join(
            [
                clean_text_value((a or {}).get("author", {}).get("display_name"))
                for a in authorships
            ]
        )

        # Извлекаем venue
        host_venue = js.get("host_venue") or {}
        venue = clean_text_value(host_venue.get("display_name", ""))

        # Извлекаем URLs
        primary_location = js.get("primary_location") or {}
        article_url = clean_text_value(primary_location.get("landing_page_url", ""))
        pdf_url = clean_text_value(primary_location.get("pdf_url", ""))

        # DOI
        doi = clean_text_value(js.get("doi", ""))
        if doi and doi.startswith("https://doi.org/"):
            doi = doi.replace("https://doi.org/", "")

        return Metadata(
            title=clean_text_value(js.get("title") or js.get("display_name", "")),
            authors=authors_str,
            year=js.get("publication_year"),
            venue=venue,
            doi=doi,
            article_url=article_url,
            pdf_url=pdf_url,
            additional_urls=[article_url, pdf_url] if article_url or pdf_url else None,
            source_name=self.name,
            raw_data=js,
        )

