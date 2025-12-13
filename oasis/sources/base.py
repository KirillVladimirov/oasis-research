"""Базовый интерфейс для источников данных (порт/адаптер)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from oasis.models.reference import Reference


@dataclass
class Metadata:
    """Метаданные публикации из внешнего источника."""

    title: str = ""
    authors: str = ""  # Строка с авторами через "; "
    year: int | None = None
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    article_url: str = ""
    pdf_url: str = ""
    additional_urls: list[str] | None = None
    source_name: str = ""  # Имя источника: "openalex", "crossref", etc.
    raw_data: dict[str, Any] | None = None  # Сырые данные из API

    def to_dict(self) -> dict[str, Any]:
        """Преобразует Metadata в словарь."""
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "article_url": self.article_url,
            "pdf_url": self.pdf_url,
            "additional_urls": self.additional_urls or [],
            "source_name": self.source_name,
        }


@dataclass
class PdfCandidate:
    """Кандидат на PDF файл."""

    url: str
    source: str  # Источник: "openalex", "crossref", etc.
    confidence: float = 1.0  # Уверенность в том, что это правильный PDF
    validated: bool = False  # Прошёл ли валидацию


class SourcePort(ABC):
    """Абстрактный интерфейс для источников данных.

    Все источники должны реализовывать этот интерфейс для единообразного API.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Имя источника (например, "openalex", "crossref")."""
        pass

    @abstractmethod
    def get_by_doi(self, doi: str) -> Metadata | None:
        """Получает метаданные по DOI.

        Args:
            doi: DOI публикации

        Returns:
            Metadata объект или None если не найдено
        """
        pass

    @abstractmethod
    def search_by_title(self, title: str) -> list[Metadata]:
        """Ищет публикации по названию.

        Args:
            title: Название публикации

        Returns:
            Список Metadata объектов (может быть пустым)
        """
        pass

    def get_pdf_candidates(self, ref: dict[str, Any] | Reference) -> list[PdfCandidate]:
        """Получает кандидатов на PDF файл для ссылки.

        Args:
            ref: Словарь или Reference объект с данными ссылки (title, doi, arxiv_id, etc.)

        Returns:
            Список PdfCandidate объектов
        """
        candidates = []
        # Поддерживаем как dict, так и Reference
        if isinstance(ref, dict):
            doi = ref.get("doi", "")
            title = ref.get("title", "")
            arxiv_id = ref.get("arxiv_id", "")
        else:
            doi = ref.doi
            title = ref.title
            arxiv_id = ref.arxiv_id

        # Пробуем по DOI
        if doi:
            metadata = self.get_by_doi(doi)
            if metadata and metadata.pdf_url:
                candidates.append(
                    PdfCandidate(
                        url=metadata.pdf_url,
                        source=self.name,
                        confidence=0.9,
                    )
                )

        # Пробуем по title
        if title and not candidates:
            results = self.search_by_title(title)
            if results:
                for meta in results[:1]:  # Берем первый результат
                    if meta.pdf_url:
                        candidates.append(
                            PdfCandidate(
                                url=meta.pdf_url,
                                source=self.name,
                                confidence=0.7,
                            )
                        )

        return candidates

