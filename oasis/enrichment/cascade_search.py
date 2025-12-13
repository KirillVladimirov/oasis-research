"""Каскадный поиск статей с несколькими fallback-вариантами."""

from typing import Any

from loguru import logger

from oasis.enrichment.cascade import (
    extract_arxiv_id_from_ref,
    extract_doi_from_ref,
    extract_title_from_ref,
    safe_str,
    validate_match,
)
from oasis.enrichment.sources.aggregator import SourceAggregator
from oasis.enrichment.sources.arxiv_wrapper import ArxivWrapper
from oasis.enrichment.sources.openalex_wrapper import OpenAlexWrapper
from oasis.enrichment.sources.researchgate_wrapper import IEEEWrapper, ResearchGateWrapper
from oasis.sources.base import Metadata

# Реэкспорт для обратной совместимости
__all__ = [
    "safe_str",
    "extract_arxiv_id_from_ref",
    "extract_doi_from_ref",
    "extract_title_from_ref",
    "validate_match",
    "cascade_search_reference",
]


def cascade_search_reference(
    ref: dict[str, Any],
    arxiv_wrapper: ArxivWrapper | None = None,
    openalex_wrapper: OpenAlexWrapper | None = None,
    aggregator: SourceAggregator | None = None,
    researchgate_wrapper: ResearchGateWrapper | None = None,
    ieee_wrapper: IEEEWrapper | None = None,
    google_wrapper=None,  # type: ignore
    use_google: bool = False,
    selenium_wrapper=None,  # type: ignore
    use_selenium: bool = False,
) -> Metadata | None:
    """Каскадный поиск статьи с несколькими fallback-вариантами.

    Последовательность:
    1. Прямой запрос по arXiv ID (если есть в исходных данных)
    2. Поиск по DOI через OpenAlex → Semantic Scholar → Crossref → Unpaywall
    3. Selenium поиск (Google Scholar + Google) если use_selenium=True
    4. Поиск по title через Semantic Scholar → OpenAlex → arXiv
    5. Если в результатах найден arXiv ID - используем его напрямую

    Args:
        ref: Словарь с данными ссылки
        arxiv_wrapper: Обертка вокруг библиотеки arxiv (опционально)
        openalex_wrapper: Обертка вокруг библиотеки pyalex (опционально)
        aggregator: SourceAggregator для доступа к другим источникам (опционально)
        researchgate_wrapper: Обертка для ResearchGate (опционально)
        ieee_wrapper: Обертка для IEEE Xplore (опционально)
        google_wrapper: GoogleSearchWrapper для Google поиска (опционально, deprecated)
        use_google: Использовать ли Google поиск (по умолчанию False, deprecated)
        selenium_wrapper: SeleniumSearchWrapper для Selenium поиска (опционально)
        use_selenium: Использовать ли Selenium поиск (по умолчанию False)

    Returns:
        Metadata объект с метаданными найденной статьи или None
    """
    if not arxiv_wrapper:
        arxiv_wrapper = ArxivWrapper()
    if not openalex_wrapper:
        openalex_wrapper = OpenAlexWrapper()
    if not researchgate_wrapper:
        researchgate_wrapper = ResearchGateWrapper()
    if not ieee_wrapper:
        ieee_wrapper = IEEEWrapper()
    if not aggregator:
        # Создаем минимальный aggregator для fallback
        from oasis.enrichment.pipeline import setup_enrichment_sources

        aggregator = setup_enrichment_sources()
    if use_google and not google_wrapper:
        # Создаем Google wrapper только если нужен
        try:
            from oasis.sources.google_search import GoogleSearchWrapper
            google_wrapper = GoogleSearchWrapper(delay_seconds=3.0)
        except ImportError:
            logger.warning("googlesearch-python не установлен, Google поиск отключен")
            use_google = False
    
    if use_selenium and not selenium_wrapper:
        # Создаем Selenium wrapper только если нужен
        try:
            from oasis.sources.selenium_search import SeleniumSearchWrapper
            selenium_wrapper = SeleniumSearchWrapper(delay_seconds=3.0)
        except ImportError:
            logger.warning("selenium не установлен, Selenium поиск отключен")
            use_selenium = False

    enrichment_log: dict[str, Any] = {
        "steps": [],
        "errors": [],
        "warnings": [],
    }
    
    # Извлекаем title в начале для использования в Selenium
    title = extract_title_from_ref(ref)

    # Шаг 1: Проверка arXiv ID в исходных данных
    arxiv_id = extract_arxiv_id_from_ref(ref)
    logger.debug(f"Извлеченный arXiv ID: '{arxiv_id}' из ref: {ref.get('ref_number', '?')}")
    if arxiv_id:
        enrichment_log["steps"].append({"step": "1_arxiv_id", "value": arxiv_id})
        logger.debug(f"Шаг 1: Поиск по arXiv ID: {arxiv_id}")

        try:
            result = arxiv_wrapper.get_by_arxiv_id(arxiv_id)
            if result and result.pdf_url:
                enrichment_log["steps"].append(
                    {"step": "1_arxiv_id", "status": "success", "found": True}
                )
                logger.info(f" Найдено по arXiv ID: {arxiv_id}")
                return result
        except Exception as e:
            enrichment_log["errors"].append({"step": "1_arxiv_id", "error": str(e)})
            logger.debug(f"Ошибка при поиске по arXiv ID {arxiv_id}: {e}")

    # Шаг 2: Поиск по DOI (если есть)
    doi = extract_doi_from_ref(ref)
    if doi:
        enrichment_log["steps"].append({"step": "2_doi", "value": doi})
        logger.debug(f"Шаг 2: Поиск по DOI: {doi}")

        # Вариант 2.1: OpenAlex по DOI
        try:
            result = openalex_wrapper.get_by_doi(doi)
            if result and result.pdf_url and not safe_str(result.pdf_url).startswith(
                "https://doi.org/"
            ):
                enrichment_log["steps"].append(
                    {"step": "2_doi_openalex", "status": "success", "found": True}
                )
                logger.info(f" Найдено по DOI через OpenAlex: {doi}")
                return result
        except Exception as e:
            enrichment_log["errors"].append({"step": "2_doi_openalex", "error": str(e)})
            logger.debug(f"Ошибка при поиске по DOI через OpenAlex {doi}: {e}")

        # Вариант 2.2: Semantic Scholar по DOI
        if aggregator:
            try:
                result = aggregator.get_by_doi(doi, sources=["semanticscholar"])
                if result and result.pdf_url and not safe_str(result.pdf_url).startswith(
                    "https://doi.org/"
                ):
                    enrichment_log["steps"].append(
                        {"step": "2_doi_semanticscholar", "status": "success", "found": True}
                    )
                    logger.info(f" Найдено по DOI через Semantic Scholar: {doi}")
                    return result
            except Exception as e:
                enrichment_log["errors"].append({"step": "2_doi_semanticscholar", "error": str(e)})
                logger.debug(f"Ошибка при поиске по DOI через Semantic Scholar {doi}: {e}")

        # Шаг 2.5: Selenium поиск ОТКЛЮЧЕН (перенесен в конец pipeline - Шаг 5.5)
        # Selenium теперь запускается в workflow.py после всех попыток API
        # Это позволяет сначала найти метаданные (DOI, title, year), а потом искать PDF

    # Шаг 3: Поиск по title через API (проверяем все результаты)
    if title:
        enrichment_log["steps"].append({"step": "3_title", "value": title})
        logger.debug(f"Шаг 3: Поиск по title через API: {title}")

        # Вариант 3.1: Semantic Scholar (с расширенным поиском)
        if aggregator:
            try:
                # Пробуем несколько вариантов поиска для Semantic Scholar
                all_ss_results = []

                # Основной поиск - увеличим до 8 результатов
                results1 = aggregator.search_by_title(
                    title, sources=["semanticscholar"], max_results_per_source=8, try_all_sources=False
                )
                all_ss_results.extend(results1)

                # Поиск с сокращенным title (первые 10 слов) для очень длинных названий
                if len(title.split()) > 10:
                    short_title = " ".join(title.split()[:10])
                    results2 = aggregator.search_by_title(
                        short_title,
                        sources=["semanticscholar"],
                        max_results_per_source=3,
                        try_all_sources=False,
                    )
                    all_ss_results.extend(results2)

                # Убираем дубликаты по DOI
                seen_dois = set()
                unique_results = []
                for result in all_ss_results:
                    doi_val = result.doi if hasattr(result, "doi") else getattr(result, "doi", "")
                    doi_str = safe_str(doi_val)
                    if doi_str and doi_str in seen_dois:
                        continue
                    if doi_str:
                        seen_dois.add(doi_str)
                    unique_results.append(result)

                for candidate in unique_results[:10]:  # Увеличим до 10 для лучшего покрытия
                    # Сначала валидируем совпадение title/authors/year
                    if not validate_match(title, candidate.to_dict(), ref):
                        continue
                    
                    # Если есть arXiv ID - используем его напрямую (максимальный приоритет)
                    if candidate.arxiv_id:
                        enrichment_log["steps"].append(
                            {
                                "step": "3_title_semanticscholar_arxiv_id",
                                "value": candidate.arxiv_id,
                            }
                        )
                        try:
                            arxiv_result = arxiv_wrapper.get_by_arxiv_id(candidate.arxiv_id)
                            if arxiv_result:
                                enrichment_log["steps"].append(
                                    {
                                        "step": "3_title_semanticscholar_arxiv_id",
                                        "status": "success",
                                        "found": True,
                                    }
                                )
                                logger.info(
                                    f" Найдено по title через Semantic Scholar → arXiv ID: {candidate.arxiv_id}"
                                )
                                return arxiv_result
                        except Exception as e:
                            enrichment_log["errors"].append(
                                {"step": "3_title_semanticscholar_arxiv_id", "error": str(e)}
                            )
                            logger.debug(
                                f"Ошибка при поиске по arXiv ID из Semantic Scholar {candidate.arxiv_id}: {e}"
                            )

                    # Проверяем PDF (не DOI)
                    has_pdf = candidate.pdf_url and not safe_str(candidate.pdf_url).startswith("https://doi.org/")
                    
                    if has_pdf:
                        enrichment_log["steps"].append(
                            {"step": "3_title_semanticscholar", "status": "success", "found": True}
                        )
                        logger.info(f" Найдено по title через Semantic Scholar: {candidate.title}")
                        return candidate
                    else:
                        # Результат найден, но нет pdf_url - продолжаем поиск в других источниках
                        logger.debug(
                            f"Semantic Scholar нашел '{candidate.title}', но нет pdf_url, продолжаем поиск"
                        )
            except Exception as e:
                enrichment_log["errors"].append({"step": "3_title_semanticscholar", "error": str(e)})
                logger.debug(f"Ошибка при поиске по title через Semantic Scholar '{title}': {e}")

        # Вариант 3.2: OpenAlex по title
        try:
            results = openalex_wrapper.search_by_title(title, max_results=8)
            for candidate in results:
                # Сначала валидируем совпадение
                if not validate_match(title, candidate.to_dict(), ref):
                    continue
                
                # Проверяем PDF (не DOI)
                has_pdf = candidate.pdf_url and not safe_str(candidate.pdf_url).startswith("https://doi.org/")
                
                if has_pdf:
                    enrichment_log["steps"].append(
                        {"step": "3_title_openalex", "status": "success", "found": True}
                    )
                    logger.info(f" Найдено по title через OpenAlex: {candidate.title}")
                    return candidate
                else:
                    # Результат найден, но нет pdf_url - продолжаем поиск в других источниках
                    logger.debug(
                        f"OpenAlex нашел '{candidate.title}', но нет pdf_url, продолжаем поиск"
                    )
        except Exception as e:
            enrichment_log["errors"].append({"step": "3_title_openalex", "error": str(e)})
            logger.debug(f"Ошибка при поиске по title через OpenAlex '{title}': {e}")

        # Вариант 3.3: arXiv по title
        try:
            results = arxiv_wrapper.search_by_title(title, max_results=8)
            for candidate in results:
                # Сначала валидируем совпадение
                if not validate_match(title, candidate.to_dict(), ref):
                    continue
                
                # arXiv всегда имеет PDF
                if candidate.pdf_url:
                    enrichment_log["steps"].append(
                        {"step": "3_title_arxiv", "status": "success", "found": True}
                    )
                    logger.info(f" Найдено по title через arXiv: {candidate.title}")
                    return candidate
        except Exception as e:
            enrichment_log["errors"].append({"step": "3_title_arxiv", "error": str(e)})
            logger.debug(f"Ошибка при поиске по title через arXiv '{title}': {e}")

    # Шаг 4: Поиск на ResearchGate (веб-скрапинг)
    if title:
        enrichment_log["steps"].append({"step": "4_researchgate", "value": title})
        logger.debug(f"Шаг 4: Поиск на ResearchGate: {title}")

        try:
            results = researchgate_wrapper.search_by_title(title, max_results=3)
            for candidate in results:
                if candidate.pdf_url:
                    if validate_match(title, candidate.to_dict(), ref):
                        enrichment_log["steps"].append(
                            {"step": "4_researchgate", "status": "success", "found": True}
                        )
                        logger.info(f" Найдено по title через ResearchGate: {candidate.title}")
                        return candidate
        except Exception as e:
            enrichment_log["errors"].append({"step": "4_researchgate", "error": str(e)})
            logger.debug(f"Ошибка при поиске на ResearchGate '{title}': {e}")

    # Шаг 5: Поиск на IEEE Xplore (веб-скрапинг)
    if title:
        enrichment_log["steps"].append({"step": "5_ieee", "value": title})
        logger.debug(f"Шаг 5: Поиск на IEEE Xplore: {title}")

        try:
            results = ieee_wrapper.search_by_title(title, max_results=3)
            for candidate in results:
                # ВАЖНО: Проверяем article_url даже если нет pdf_url
                # IEEE Xplore часто имеет article_url (ссылку на публикацию), но не всегда pdf_url
                if candidate.article_url or candidate.pdf_url:
                    if validate_match(title, candidate.to_dict(), ref):
                        enrichment_log["steps"].append(
                            {"step": "5_ieee", "status": "success", "found": True}
                        )
                        logger.info(
                            f" Найдено по title через IEEE Xplore: {candidate.title} "
                            f"(article_url={bool(candidate.article_url)}, pdf_url={bool(candidate.pdf_url)})"
                        )
                        return candidate
        except Exception as e:
            enrichment_log["errors"].append({"step": "5_ieee", "error": str(e)})
            logger.debug(f"Ошибка при поиске на IEEE Xplore '{title}': {e}")

    # Не нашли
    logger.warning(f" Не удалось найти статью: {title or ref.get('ref_number', '?')}")
    return None
