"""Обертка вокруг библиотеки pyalex для поиска статей."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pyalex
from loguru import logger

from oasis.io.cache import ApiCache
from oasis.sources.base import Metadata, PdfCandidate, SourcePort

if TYPE_CHECKING:
    from oasis.models.reference import Reference

# Настройка email для OpenAlex polite pool
_openalex_email = os.getenv("OPENALEX_MAILTO", "").strip()
if _openalex_email:
    pyalex.config.email = _openalex_email
    logger.debug(f"Настроен email для OpenAlex: {_openalex_email}")


class OpenAlexWrapper(SourcePort):
    """Обертка вокруг библиотеки pyalex для поиска статей и PDF."""

    def __init__(self, cache: ApiCache | None = None):
        """Инициализация обертки.

        Args:
            cache: Экземпляр кэша для API ответов (пока не используется, но сохранен для совместимости)
        """
        self.cache = cache

    @property
    def name(self) -> str:
        """Имя источника."""
        return "openalex_wrapper"

    def get_by_doi(self, doi: str) -> Metadata | None:
        """Получает метаданные по DOI.

        Args:
            doi: DOI публикации (например, "10.1109/TNNLS.2020.2978386")

        Returns:
            Metadata объект или None если не найдено
        """
        if not doi:
            return None

        try:
            # Очищаем DOI от префикса https://doi.org/ если есть
            doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()

            # Получаем работу по DOI
            work = pyalex.Works()["doi:" + doi_clean]

            if not work:
                return None

            # Извлекаем авторов
            authors_list = work.get("authorships", [])
            authors = "; ".join([a.get("author", {}).get("display_name", "") for a in authors_list[:10]])

            # Извлекаем год
            pub_date = work.get("publication_date", "")
            year = int(pub_date[:4]) if pub_date and len(pub_date) >= 4 else None

            # Извлекаем PDF URL
            pdf_url = work.get("primary_location", {}).get("pdf_url", "") or ""
            # Если pdf_url пустой, проверяем open_access
            if not pdf_url:
                open_access = work.get("open_access", {})
                if open_access.get("is_oa", False):
                    pdf_url = open_access.get("oa_url", "") or ""

            # Извлекаем article URL
            article_url = work.get("primary_location", {}).get("landing_page_url", "") or ""
            if not article_url and doi_clean:
                article_url = f"https://doi.org/{doi_clean}"

            # Извлекаем DOI
            doi_from_work = work.get("doi", "") or ""
            if not doi_from_work and doi_clean:
                doi_from_work = f"https://doi.org/{doi_clean}"

            # Извлекаем arXiv ID (если есть)
            arxiv_id = ""
            for ext_id in work.get("ids", {}):
                if ext_id.startswith("arxiv:"):
                    arxiv_id = ext_id.replace("arxiv:", "")
                    break

            return Metadata(
                title=work.get("title", ""),
                authors=authors,
                year=year,
                doi=doi_from_work,
                arxiv_id=arxiv_id,
                article_url=article_url,
                pdf_url=pdf_url,
                source_name=self.name,
                raw_data={"id": work.get("id"), "doi": doi_from_work, "publication_date": pub_date},
            )
        except Exception as e:
            logger.debug(f"Ошибка при поиске по DOI {doi}: {e}")
            return None

    def search_by_title(self, title: str, max_results: int = 5) -> list[Metadata]:
        """Ищет публикации по названию через библиотеку pyalex.

        Args:
            title: Название публикации
            max_results: Максимальное количество результатов

        Returns:
            Список Metadata объектов, отсортированных по релевантности
        """
        if not title:
            return []

        results = []

        try:
            # Используем search_filter для поиска по полю title
            works = pyalex.Works().search_filter(title=title).get(per_page=max_results)

            for work in works:
                # Извлекаем авторов
                authors_list = work.get("authorships", [])
                authors = "; ".join([a.get("author", {}).get("display_name", "") for a in authors_list[:10]])

                # Извлекаем год
                pub_date = work.get("publication_date", "")
                year = int(pub_date[:4]) if pub_date and len(pub_date) >= 4 else None

                # Извлекаем PDF URL
                pdf_url = work.get("primary_location", {}).get("pdf_url", "") or ""
                # Если pdf_url пустой, проверяем open_access
                if not pdf_url:
                    open_access = work.get("open_access", {})
                    if open_access.get("is_oa", False):
                        pdf_url = open_access.get("oa_url", "") or ""

                # Извлекаем article URL
                article_url = work.get("primary_location", {}).get("landing_page_url", "") or ""

                # Извлекаем DOI
                doi = work.get("doi", "") or ""

                # Извлекаем arXiv ID (если есть)
                arxiv_id = ""
                for ext_id in work.get("ids", {}):
                    if ext_id.startswith("arxiv:"):
                        arxiv_id = ext_id.replace("arxiv:", "")
                        break

                results.append(
                    Metadata(
                        title=work.get("title", ""),
                        authors=authors,
                        year=year,
                        doi=doi,
                        arxiv_id=arxiv_id,
                        article_url=article_url,
                        pdf_url=pdf_url,
                        source_name=self.name,
                        raw_data={"id": work.get("id"), "doi": doi, "publication_date": pub_date},
                    )
                )
        except Exception as e:
            logger.debug(f"Ошибка при поиске по title '{title}': {e}")

        return results

    def get_pdf_candidates(self, ref: dict[str, Any] | Reference) -> list[PdfCandidate]:
        """Получает кандидатов на PDF файл для ссылки.

        Если есть DOI, пытается получить PDF через OpenAlex.

        Args:
            ref: Словарь или Reference объект с данными ссылки

        Returns:
            Список PdfCandidate объектов
        """
        candidates = []

        # Извлекаем DOI
        doi = None
        if isinstance(ref, dict):
            doi = ref.get("doi", "")
        else:
            doi = getattr(ref, "doi", "") or ""

        if not doi:
            return candidates

        # Очищаем DOI от префикса
        doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()

        try:
            # Получаем метаданные по DOI
            metadata = self.get_by_doi(doi_clean)
            if metadata and metadata.pdf_url:
                candidates.append(
                    PdfCandidate(
                        url=metadata.pdf_url,
                        source=self.name,
                        confidence=0.9,  # Высокая уверенность для OpenAlex PDF
                    )
                )
        except Exception as e:
            logger.debug(f"Ошибка при получении PDF для DOI {doi_clean}: {e}")

        return candidates

