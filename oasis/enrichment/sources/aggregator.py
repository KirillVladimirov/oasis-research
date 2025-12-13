"""Агрегатор источников данных для обогащения ссылок."""

from typing import Any

from loguru import logger

from oasis.io.cache import ApiCache
from oasis.io.registry import SourceRegistry, get_registry
from oasis.sources.base import Metadata, SourcePort


class SourceAggregator:
    """Агрегатор для получения данных из нескольких источников."""

    def __init__(self, registry: SourceRegistry | None = None, cache: ApiCache | None = None):
        """Инициализация агрегатора.

        Args:
            registry: Реестр источников (по умолчанию глобальный)
            cache: Кэш для API ответов
        """
        self.registry = registry or get_registry()
        self.cache = cache

    def get_by_doi(self, doi: str, sources: list[str] | None = None) -> Metadata | None:
        """Получает метаданные по DOI из доступных источников.

        Args:
            doi: DOI публикации
            sources: Список имён источников для использования (None = все включенные)

        Returns:
            Metadata объект или None если не найдено
        """
        if not doi:
            return None

        source_names = sources or [name for name in self.registry.list_all() if self.registry.is_enabled(name)]
        
        for source_name in source_names:
            source = self.registry.get(source_name)
            if not source:
                continue

            try:
                metadata = source.get_by_doi(doi)
                if metadata:
                    logger.debug(f"Найдено по DOI в {source_name}: {doi}")
                    return metadata
            except Exception as e:
                logger.debug(f"Ошибка при запросе {source_name} по DOI {doi}: {e}")
                continue

        return None

    def search_by_title(
        self,
        title: str,
        sources: list[str] | None = None,
        max_results_per_source: int = 3,
        try_all_sources: bool = True,
    ) -> list[Metadata]:
        """Ищет публикации по названию в доступных источниках с улучшенной нормализацией.

        Args:
            title: Название публикации
            sources: Список имён источников для использования (None = все включенные)
            max_results_per_source: Максимальное количество результатов из каждого источника
            try_all_sources: Если True, проверяет все источники, иначе останавливается после первого успешного

        Returns:
            Список Metadata объектов, отсортированных по релевантности
        """
        if not title:
            return []

        from oasis.utils.text import normalize_text
        from unidecode import unidecode
        import re

        # Нормализация заголовка для поиска
        def normalize_title_for_search(t: str) -> str:
            """Нормализует заголовок для поиска: убирает пунктуацию, приводит к lowercase."""
            # Убираем специальные символы и лишние пробелы
            t = unidecode(t.lower().strip())
            # Убираем пунктуацию, оставляем только буквы, цифры и пробелы
            t = re.sub(r"[^\w\s]", " ", t)
            # Убираем множественные пробелы
            t = re.sub(r"\s+", " ", t).strip()
            return t

        normalized_title = normalize_title_for_search(title)
        source_names = sources or [name for name in self.registry.list_all() if self.registry.is_enabled(name)]
        all_results = []

        # Пробуем все источники для максимального покрытия
        for source_name in source_names:
            source = self.registry.get(source_name)
            if not source:
                continue

            try:
                # Пробуем поиск с оригинальным заголовком
                results = source.search_by_title(title)
                
                # Если не нашли, пробуем с нормализованным
                if not results:
                    results = source.search_by_title(normalized_title)
                
                # Ограничиваем количество результатов из каждого источника
                if results:
                    results = results[:max_results_per_source]
                    logger.debug(f"Найдено по title в {source_name}: {len(results)} результатов")
                    all_results.extend(results)
                    
                    # Если включен режим "остановиться после первого успешного"
                    if not try_all_sources and results:
                        break
            except Exception as e:
                logger.debug(f"Ошибка при запросе {source_name} по title: {e}")
                continue

        # Удаляем дубликаты по DOI/arXiv ID, сохраняя первые вхождения
        seen_ids = set()
        unique_results = []
        for result in all_results:
            result_id = None
            if result.doi:
                doi_str = str(result.doi).strip().lower() if result.doi else ""
                if doi_str:
                    result_id = f"doi:{doi_str}"
            elif result.arxiv_id:
                result_id = f"arxiv:{result.arxiv_id}"
            
            if result_id and result_id in seen_ids:
                continue
            if result_id:
                seen_ids.add(result_id)
            
            unique_results.append(result)

        logger.debug(f"Всего уникальных результатов по title: {len(unique_results)}")
        return unique_results

    def get_pdf_candidates(self, ref: dict[str, Any], sources: list[str] | None = None) -> list[dict[str, Any]]:
        """Получает кандидатов на PDF файл из всех источников.

        Args:
            ref: Словарь с данными ссылки
            sources: Список имён источников для использования (None = все включенные)

        Returns:
            Список словарей с полями: url, source, confidence
        """
        source_names = sources or [name for name in self.registry.list_all() if self.registry.is_enabled(name)]
        candidates = []

        for source_name in source_names:
            source = self.registry.get(source_name)
            if not source:
                continue

            try:
                pdf_candidates = source.get_pdf_candidates(ref)
                for candidate in pdf_candidates:
                    candidates.append(
                        {
                            "url": candidate.url,
                            "source": candidate.source,
                            "confidence": candidate.confidence,
                            "validated": candidate.validated,
                        }
                    )
            except Exception as e:
                logger.debug(f"Ошибка при получении PDF кандидатов из {source_name}: {e}")
                continue

        # ВАЖНО: Если есть article_url но нет pdf_url, пытаемся извлечь PDF из article_url
        # для известных доменов (OpenReview, IEEE, ACM и т.д.)
        if not ref.get("pdf_url") and ref.get("article_url"):
            article_url = ref.get("article_url", "")
            # Проверяем что article_url - строка (не NaN из pandas)
            if not isinstance(article_url, str) or not article_url.strip():
                article_url = None
            if article_url:
                try:
                    from oasis.sources.academic_parsers import (
                        OpenReviewParser,
                        IEEEParser,
                        ACMParser,
                        SpringerParser,
                        ECVAParser,
                    )
                    import requests
                    from requests.adapters import HTTPAdapter
                    from urllib3.util.retry import Retry

                    # Создаем сессию с retry
                    session = requests.Session()
                    retry_strategy = Retry(
                        total=2,
                        backoff_factor=0.3,
                        status_forcelist=[429, 500, 502, 503, 504],
                    )
                    adapter = HTTPAdapter(max_retries=retry_strategy)
                    session.mount("http://", adapter)
                    session.mount("https://", adapter)

                    # Определяем парсер по домену
                    parser = None
                    if "openreview.net" in article_url.lower():
                        parser = OpenReviewParser()
                    elif "ieeexplore.ieee.org" in article_url.lower():
                        parser = IEEEParser()
                    elif "dl.acm.org" in article_url.lower():
                        parser = ACMParser()
                    elif "link.springer.com" in article_url.lower() or "springer.com" in article_url.lower():
                        parser = SpringerParser()
                    elif "ecva.net" in article_url.lower() or "thecvf.com" in article_url.lower():
                        parser = ECVAParser()

                    if parser:
                        logger.debug(f"Пытаемся извлечь PDF из article_url: {article_url[:60]}...")
                        response = session.get(article_url, timeout=10, allow_redirects=True)
                        if response.status_code == 200:
                            metadata = parser.extract(article_url, response.text)
                            if metadata.get("pdf_url"):
                                pdf_url = metadata["pdf_url"]
                                candidates.append(
                                    {
                                        "url": pdf_url,
                                        "source": "academic_parser",
                                        "confidence": 0.85,  # Высокая уверенность для парсинга страницы
                                        "validated": False,
                                    }
                                )
                                logger.info(
                                    f" Извлечен PDF из article_url: {pdf_url[:60]}... "
                                    f"(source=academic_parser, domain={parser.__class__.__name__})"
                                )
                except Exception as e:
                    article_url_str = str(article_url)[:60] if article_url else "N/A"
                    logger.debug(f"Ошибка при извлечении PDF из article_url {article_url_str}...: {e}")

        # Сортируем по confidence
        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        return candidates

