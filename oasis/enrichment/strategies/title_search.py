"""Стратегия fallback поиска по заголовку."""

from typing import Any

from loguru import logger
from rapidfuzz import fuzz
from unidecode import unidecode

from oasis.enrichment.cascade.matchers import strict_title_match
from oasis.enrichment.sources.aggregator import SourceAggregator
from oasis.enrichment.sources.merger import merge_metadata
from oasis.enrichment.utils import safe_strip


def search_by_title_fallback(
    enriched: dict[str, Any],
    title: str,
    doi: str,
    arxiv_id: str,
    aggregator: SourceAggregator,
    enrichment_log: dict[str, Any],
    trace_prefix: str = "",
) -> dict[str, Any]:
    """Выполняет fallback поиск по заголовку с умным выбором лучшего результата.
    
    Args:
        enriched: Обогащаемая ссылка
        title: Заголовок для поиска
        doi: DOI исходной ссылки
        arxiv_id: arXiv ID исходной ссылки
        aggregator: Агрегатор источников
        enrichment_log: Лог обогащения
        trace_prefix: Префикс для логов
        
    Returns:
        Обновленная ссылка
    """
    enrichment_log["steps"].append({
        "step": "title_search",
        "title": title,
        "sources": ["semanticscholar", "openalex", "arxiv"],
    })
    
    # Пробуем источники в порядке приоритета (проверяем все источники)
    title_results = aggregator.search_by_title(
        title, sources=["semanticscholar", "openalex", "arxiv"], max_results_per_source=3, try_all_sources=True
    )
    
    if title_results:
        # Проверяем топ-5 результатов для лучшего совпадения
        best_metadata = None
        best_match_score = 0.0
        
        normalized_title = unidecode(title.lower().strip())
        
        for candidate in title_results[:5]:  # Проверяем топ-5 для лучшего выбора
            candidate_title = candidate.title if candidate.title else ""
            if not candidate_title:
                continue
            
            # СТРОГАЯ ВАЛИДАЦИЯ: используем strict_title_match с порогом 0.90
            if not strict_title_match(title, candidate_title, threshold=0.90):
                logger.debug(
                    f"{trace_prefix}Кандидат отклонён strict_title_match: "
                    f"'{candidate_title[:50]}...'"
                )
                continue
            
            # Если прошёл strict_title_match, вычисляем базовый score
            normalized_title = unidecode(title.lower().strip())
            candidate_title_norm = unidecode(candidate_title.lower().strip())
            similarity = fuzz.ratio(normalized_title, candidate_title_norm) / 100.0
            
            # Бонус за совпадение года
            if enriched.get("year") and candidate.year:
                year_diff = abs(enriched.get("year") - candidate.year)
                if year_diff == 0:
                    similarity += 0.1
                elif year_diff <= 2:
                    similarity += 0.05
                elif year_diff > 5:
                    similarity -= 0.1
            
            # Бонус за совпадение DOI/arXiv
            if doi and candidate.doi:
                if doi.lower() == candidate.doi.lower():
                    similarity += 0.2
            if arxiv_id and candidate.arxiv_id:
                if str(arxiv_id) == str(candidate.arxiv_id):
                    similarity += 0.2
            
            # Проверка авторов
            if enriched.get("authors") and candidate.authors:
                enriched_authors = [a.strip().lower() for a in str(enriched.get("authors")).split(";")]
                candidate_authors = [a.strip().lower() for a in str(candidate.authors).split(";")]
                common_authors = set(enriched_authors) & set(candidate_authors)
                if common_authors:
                    similarity += 0.1
            
            if similarity > best_match_score:
                best_match_score = similarity
                best_metadata = candidate
        
        # Используем лучший результат, если совпадение достаточно хорошее (>=0.90)
        # Уже прошло strict_title_match, но дополнительно проверяем итоговый score
        if best_metadata and best_match_score >= 0.90:
            enriched = merge_metadata(enriched, best_metadata)
            
            # Обработка PDF URL из метаданных
            if best_metadata.pdf_url:
                pdf_url = best_metadata.pdf_url
                if getattr(best_metadata, "source_name", None):
                    enriched["_pdf_source"] = best_metadata.source_name
                
                # Проверяем, является ли это DOI
                if pdf_url.startswith("https://doi.org/") or pdf_url.startswith("http://doi.org/"):
                    logger.debug(f"{trace_prefix}pdf_url из метаданных является DOI, пропускаем: {pdf_url[:50]}...")
                    
                    # ВАЖНО: НЕ назначаем arXiv PDF напрямую без валидации!
                    # Даже если есть arXiv ID, PDF должен пройти валидацию через pdf_search.py
                    # Это предотвращает принятие неправильных статей с неправильным arXiv ID
                    enriched["pdf_url"] = ""
                    enriched.pop("_pdf_source", None)
                    logger.debug(
                        f"{trace_prefix}DOI найден, но arXiv PDF не назначаем без валидации "
                        f"(будет проверен в pdf_search.py)"
                    )
                else:
                    # ВАЖНО: Для arXiv PDF тоже не назначаем напрямую - нужна валидация
                    if "arxiv.org" in pdf_url.lower():
                        logger.debug(
                            f"{trace_prefix}arXiv PDF найден, но не назначаем без валидации "
                            f"(будет проверен в pdf_search.py): {pdf_url[:50]}..."
                        )
                        enriched["pdf_url"] = ""
                        enriched.pop("_pdf_source", None)
                    else:
                        enriched["pdf_url"] = pdf_url
                        logger.debug(f"{trace_prefix}PDF найден из метаданных найденной статьи: {pdf_url[:50]}...")
            
            enrichment_log["steps"][-1]["result"] = "success"
            enrichment_log["steps"][-1]["found_count"] = len(title_results)
            enrichment_log["steps"][-1]["best_match_score"] = best_match_score
            enrichment_log["steps"][-1]["top_result"] = {
                "title": best_metadata.title,
                "year": best_metadata.year,
                "venue": best_metadata.venue,
                "doi": best_metadata.doi,
                "arxiv_id": best_metadata.arxiv_id,
                "pdf_url": best_metadata.pdf_url,
            }
            logger.debug(
                f"{trace_prefix}Обогащено по title: {title[:50]}... "
                f"(найдено {len(title_results)} результатов, выбран с score={best_match_score:.2f})"
            )
        else:
            # Если лучшее совпадение недостаточно хорошее, НЕ используем результаты
            enrichment_log["steps"][-1]["result"] = "low_confidence"
            enrichment_log["steps"][-1]["found_count"] = len(title_results)
            enrichment_log["steps"][-1]["best_match_score"] = best_match_score if best_metadata else 0.0
            if best_metadata:
                enrichment_log["steps"][-1]["top_result"] = {
                    "title": best_metadata.title,
                    "year": best_metadata.year,
                    "venue": best_metadata.venue,
                    "doi": best_metadata.doi,
                    "arxiv_id": best_metadata.arxiv_id,
                }
            enrichment_log["warnings"].append(
                f"Лучшее совпадение по title имеет низкий score ({best_match_score:.2f}), результат не использован"
            )
            logger.warning(
                f"{trace_prefix}Низкое совпадение по title ({best_match_score:.2f}), результат не использован"
            )
    else:
        enrichment_log["steps"][-1]["result"] = "not_found"
        logger.debug(f"{trace_prefix}Ничего не найдено по title: {title[:50]}...")
    
    return enriched

