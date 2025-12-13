"""Обертка вокруг библиотеки arxiv для поиска статей."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import arxiv
from loguru import logger

from oasis.io.cache import ApiCache
from oasis.sources.base import Metadata, PdfCandidate, SourcePort

if TYPE_CHECKING:
    from oasis.models.reference import Reference


class ArxivWrapper(SourcePort):
    """Обертка вокруг библиотеки arxiv для поиска статей и PDF."""

    def __init__(self, cache: ApiCache | None = None):
        """Инициализация обертки.

        Args:
            cache: Экземпляр кэша для API ответов (пока не используется, но сохранен для совместимости)
        """
        self.cache = cache
        self.client = arxiv.Client()

    @property
    def name(self) -> str:
        """Имя источника."""
        return "arxiv_wrapper"

    def get_by_arxiv_id(self, arxiv_id: str) -> Metadata | None:
        """Получает метаданные по arXiv ID.

        Args:
            arxiv_id: arXiv ID публикации (например, "1901.00596" или "1901.00596v4")

        Returns:
            Metadata объект или None если не найдено
        """
        if not arxiv_id:
            return None

        try:
            # Очищаем arXiv ID от версии, если есть
            arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id

            search = arxiv.Search(id_list=[arxiv_id_clean], max_results=1)
            results = list(self.client.results(search))

            if not results:
                return None

            paper = results[0]

            # Извлекаем arXiv ID из entry_id
            arxiv_id_from_entry = paper.entry_id.split("/")[-1] if "/" in paper.entry_id else arxiv_id_clean

            # Формируем список авторов
            authors = "; ".join([a.name for a in paper.authors])

            # Извлекаем год
            year = paper.published.year if paper.published else None

            return Metadata(
                title=paper.title,
                authors=authors,
                year=year,
                arxiv_id=arxiv_id_from_entry,
                article_url=paper.entry_id,
                pdf_url=paper.pdf_url,
                source_name=self.name,
                raw_data={"title": paper.title, "authors": [a.name for a in paper.authors], "published": str(paper.published) if paper.published else None},
            )
        except Exception as e:
            logger.debug(f"Ошибка при поиске по arXiv ID {arxiv_id}: {e}")
            return None

    def get_by_doi(self, doi: str) -> Metadata | None:
        """Получает метаданные по DOI (arXiv не поддерживает поиск по DOI напрямую).

        Args:
            doi: DOI публикации

        Returns:
            None (arXiv не поддерживает поиск по DOI)
        """
        # arXiv не предоставляет поиск по DOI
        return None

    def search_by_title(self, title: str, max_results: int = 5) -> list[Metadata]:
        """Ищет публикации по названию через библиотеку arxiv.

        Использует два варианта запроса:
        1. Точный поиск с ti:"title" (дает лучшие результаты)
        2. Простой поиск по title (fallback)

        Args:
            title: Название публикации
            max_results: Максимальное количество результатов

        Returns:
            Список Metadata объектов, отсортированных по релевантности
        """
        if not title:
            return []

        results = []

        # Вариант 1: Точный поиск с ti:"title" (дает лучшие результаты)
        try:
            query_exact = f'ti:"{title}"'
            search = arxiv.Search(query=query_exact, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
            exact_results = list(self.client.results(search))
            results.extend(exact_results)
        except Exception as e:
            logger.debug(f"Ошибка при точном поиске по title '{title}': {e}")

        # Вариант 2: Простой поиск (fallback, если точный не дал результатов)
        if not results:
            try:
                search = arxiv.Search(query=title, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
                simple_results = list(self.client.results(search))
                results.extend(simple_results)
            except Exception as e:
                logger.debug(f"Ошибка при простом поиске по title '{title}': {e}")

        # Преобразуем результаты в Metadata
        metadata_list = []
        seen_ids = set()

        for paper in results:
            # Извлекаем arXiv ID из entry_id
            arxiv_id = paper.entry_id.split("/")[-1] if "/" in paper.entry_id else ""

            # Пропускаем дубликаты
            if arxiv_id and arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)

            # Формируем список авторов
            authors = "; ".join([a.name for a in paper.authors])

            # Извлекаем год
            year = paper.published.year if paper.published else None

            metadata_list.append(
                Metadata(
                    title=paper.title,
                    authors=authors,
                    year=year,
                    arxiv_id=arxiv_id,
                    article_url=paper.entry_id,
                    pdf_url=paper.pdf_url,
                    source_name=self.name,
                    raw_data={"title": paper.title, "authors": [a.name for a in paper.authors], "published": str(paper.published) if paper.published else None},
                )
            )

        return metadata_list

    def get_pdf_candidates(self, ref: dict[str, Any] | Reference) -> list[PdfCandidate]:
        """Получает кандидатов на PDF файл для ссылки.

        Если есть arXiv ID, напрямую возвращает PDF URL.

        Args:
            ref: Словарь или Reference объект с данными ссылки

        Returns:
            Список PdfCandidate объектов
        """
        candidates = []

        # Извлекаем arXiv ID
        arxiv_id = None
        if isinstance(ref, dict):
            arxiv_id = ref.get("arxiv_id", "")
        else:
            arxiv_id = getattr(ref, "arxiv_id", "") or ""

        if not arxiv_id:
            return candidates

        # Очищаем arXiv ID от версии
        arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id

        # Проверяем формат arXiv ID
        if not re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", arxiv_id_clean):
            logger.debug(f"Неверный формат arXiv ID: {arxiv_id}")
            return candidates

        try:
            # Получаем метаданные по arXiv ID
            metadata = self.get_by_arxiv_id(arxiv_id_clean)
            if metadata and metadata.pdf_url:
                candidates.append(
                    PdfCandidate(
                        url=metadata.pdf_url,
                        source=self.name,
                        confidence=1.0,  # Высокая уверенность для прямого arXiv PDF
                    )
                )
        except Exception as e:
            logger.debug(f"Ошибка при получении PDF для arXiv ID {arxiv_id_clean}: {e}")

        return candidates

