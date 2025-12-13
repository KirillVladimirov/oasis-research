"""DuckDuckGo поиск для нахождения PDF и метаданных статей."""

import re
import time
from typing import Any

from duckduckgo_search import DDGS
from loguru import logger

from oasis.enrichment.cascade.matchers import strict_title_match
from oasis.sources.base import Metadata, SourcePort


class DuckDuckGoWrapper(SourcePort):
    """Источник данных через DuckDuckGo поиск."""

    def __init__(self, delay_seconds: float = 1.0, max_retries: int = 3):
        """Инициализация DuckDuckGo wrapper.

        Args:
            delay_seconds: Задержка между запросами (по умолчанию 1 сек)
            max_retries: Максимальное количество повторных попыток
        """
        self.delay_seconds = delay_seconds
        self.max_retries = max_retries
        self._last_request_time = 0.0

    @property
    def name(self) -> str:
        """Имя источника."""
        return "duckduckgo"

    def _rate_limit(self):
        """Применяет rate limiting между запросами."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)
        self._last_request_time = time.time()

    def search_by_title(
        self, title: str, max_results: int = 5, require_pdf: bool = False
    ) -> list[Metadata]:
        """Ищет статьи по названию через DuckDuckGo.

        Args:
            title: Название статьи
            max_results: Максимальное количество результатов
            require_pdf: Искать только PDF файлы

        Returns:
            Список объектов Metadata
        """
        if not title or len(title) < 10:
            return []

        # Формируем запрос
        if require_pdf:
            query = f'"{title}" filetype:pdf'
        else:
            query = f'"{title}" (pdf OR doi OR arxiv)'

        logger.debug(f"DuckDuckGo поиск: {query[:80]}...")

        results = []
        try:
            self._rate_limit()

            # Используем DuckDuckGo text search
            with DDGS() as ddgs:
                search_results = list(
                    ddgs.text(query, max_results=max_results * 2)  # Берем больше для фильтрации
                )

            logger.debug(f"DuckDuckGo вернул {len(search_results)} результатов")

            for item in search_results[:max_results]:
                url = item.get("href", "")
                snippet_title = item.get("title", "")
                body = item.get("body", "")

                if not url:
                    continue

                # Пропускаем неакадемические домены
                if any(
                    domain in url.lower()
                    for domain in [
                        "youtube.com",
                        "facebook.com",
                        "twitter.com",
                        "linkedin.com",
                        "pinterest.com",
                        "amazon.com",
                        "ebay.com",
                        "reddit.com",
                        "wikipedia.org",
                    ]
                ):
                    continue

                # Извлекаем метаданные
                doi = self._extract_doi_from_text(url + " " + body)
                arxiv_id = self._extract_arxiv_id_from_text(url + " " + body)
                pdf_url = url if url.lower().endswith(".pdf") else None

                # Используем title из сниппета, если он похож на исходный
                result_title = title  # По умолчанию используем исходный
                if snippet_title and strict_title_match(
                    title, snippet_title, threshold=0.70
                ):
                    result_title = snippet_title

                results.append(
                    Metadata(
                        title=result_title,
                        article_url=url,
                        pdf_url=pdf_url,
                        doi=doi,
                        arxiv_id=arxiv_id,
                        source_name=self.name,
                    )
                )

                if len(results) >= max_results:
                    break

        except Exception as e:
            logger.warning(f"Ошибка при поиске в DuckDuckGo: {e}")

        logger.info(
            f"DuckDuckGo поиск для '{title[:60]}...' вернул {len(results)} релевантных результатов"
        )
        return results

    def _extract_doi_from_text(self, text: str) -> str | None:
        """Извлекает DOI из текста."""
        if not text:
            return None

        # Паттерн для DOI
        match = re.search(r"\b(10\.\d{4,}/[^\s]+)", text, re.IGNORECASE)
        if match:
            doi = match.group(1).rstrip(".,;)")
            # Базовая валидация
            if len(doi) > 7 and "/" in doi:
                return doi
        return None

    def _extract_arxiv_id_from_text(self, text: str) -> str | None:
        """Извлекает arXiv ID из текста."""
        if not text:
            return None

        # Паттерн для arXiv ID
        match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,})", text, re.IGNORECASE)
        if match:
            return match.group(1)

        # Альтернативный паттерн (без URL)
        match = re.search(r"\b(\d{4}\.\d{4,}(?:v\d+)?)\b", text)
        if match:
            arxiv_id = match.group(1)
            # Проверяем, что это похоже на arXiv ID (год >= 1991)
            year = int(arxiv_id[:4])
            if 1991 <= year <= 2030:
                return arxiv_id.split("v")[0]  # Убираем версию

        return None

    # Методы для совместимости с SourcePort (не используются)
    def get_by_doi(self, doi: str) -> Metadata | None:
        """Не поддерживается для DuckDuckGo."""
        return None

    def get_by_arxiv_id(self, arxiv_id: str) -> Metadata | None:
        """Не поддерживается для DuckDuckGo."""
        return None

    def get_pdf_candidates(
        self, doi: str | None = None, title: str | None = None
    ) -> list[dict[str, Any]]:
        """Не поддерживается для DuckDuckGo."""
        return []

