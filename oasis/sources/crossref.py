"""Crossref источник данных."""

from typing import Any

import requests
from loguru import logger

from oasis.io.cache import ApiCache
from oasis.sources.base import Metadata, SourcePort
from oasis.utils.text import clean_text_value


class CrossrefSource(SourcePort):
    """Источник данных Crossref."""

    def __init__(self, base_url: str = "https://api.crossref.org/works/", cache: ApiCache | None = None):
        """Инициализация источника.

        Args:
            base_url: Базовый URL API Crossref
            cache: Экземпляр кэша для API ответов
        """
        self.base_url = base_url
        self.cache = cache

    @property
    def name(self) -> str:
        """Имя источника."""
        return "crossref"

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
        cache_key = f"cr:{doi}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return self._parse_response(cached)

        try:
            response = requests.get(
                self.base_url + requests.utils.quote(doi),
                headers={"Accept": "application/json"},
                timeout=60,
            )
            if response.status_code != 200:
                return None

            msg = response.json().get("message", {})
            if self.cache:
                self.cache.set(cache_key, msg)

            return self._parse_response(msg)
        except Exception as e:
            logger.debug(f"Ошибка при запросе Crossref по DOI: {e}")
            return None

    def search_by_title(self, title: str) -> list[Metadata]:
        """Ищет публикации по названию (Crossref не поддерживает поиск по title напрямую).

        Args:
            title: Название публикации

        Returns:
            Пустой список (Crossref не поддерживает поиск по title)
        """
        # Crossref не предоставляет поиск по title через API
        return []

    def _parse_response(self, msg: dict[str, Any]) -> Metadata | None:
        """Парсит ответ от Crossref API в Metadata.

        Args:
            msg: JSON ответ от API (поле "message")

        Returns:
            Metadata объект или None
        """
        if not msg:
            return None

        # Извлекаем авторов
        authors = msg.get("author") or []
        authors_str = "; ".join(
            [
                clean_text_value(f"{a.get('given', '')} {a.get('family', '')}")
                for a in authors
            ]
        )

        # Извлекаем venue
        container_title = msg.get("container-title") or []
        venue = clean_text_value(container_title[0]) if container_title else ""

        # Извлекаем год
        year = None
        for field in ("published-print", "published-online", "issued"):
            comp = (msg.get(field, {}) or {}).get("date-parts", [])
            if comp and comp[0]:
                try:
                    year = int(comp[0][0])
                    break
                except Exception:
                    pass

        # URL
        article_url = msg.get("URL", "")

        return Metadata(
            title=clean_text_value((msg.get("title") or [""])[0]),
            authors=authors_str,
            year=year,
            venue=venue,
            doi=clean_text_value(msg.get("DOI", "")),
            article_url=clean_text_value(article_url),
            additional_urls=[article_url] if article_url else None,
            source_name=self.name,
            raw_data=msg,
        )

