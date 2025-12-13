"""Стратегия поиска PDF кандидатов."""

from typing import Any, Callable

from loguru import logger
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from oasis.enrichment.sources.aggregator import SourceAggregator
from oasis.parsing.regexes import find_arxiv_id, find_urls


def extract_arxiv_id_from_enriched(
    enriched: dict[str, Any],
    trace_prefix: str = ""
) -> dict[str, Any]:
    """Извлекает arXiv ID из URL если его еще нет.
    
    Args:
        enriched: Обогащаемая ссылка
        trace_prefix: Префикс для логов
        
    Returns:
        Обновленная ссылка
    """
    if not enriched.get("arxiv_id"):
        # Проверяем article_url
        article_url = enriched.get("article_url", "")
        if article_url:
            arxiv_from_url = find_arxiv_id(article_url)
            if arxiv_from_url:
                enriched["arxiv_id"] = arxiv_from_url
                logger.debug(f"{trace_prefix}Извлечен arXiv ID из article_url: {arxiv_from_url}")

        # Если не нашли, проверяем все URL в raw_text
        if not enriched.get("arxiv_id") and enriched.get("raw_text"):
            all_urls = find_urls(enriched["raw_text"])
            for url in all_urls:
                arxiv_from_url = find_arxiv_id(url)
                if arxiv_from_url:
                    enriched["arxiv_id"] = arxiv_from_url
                    logger.debug(f"{trace_prefix}Извлечен arXiv ID из raw_text URL: {arxiv_from_url}")
                    break
    
    return enriched


def search_pdf_candidates(
    enriched: dict[str, Any],
    aggregator: SourceAggregator,
    validate_pdf_fn: Callable[[str, dict[str, Any]], dict[str, Any]] | None,
    enrichment_log: dict[str, Any],
    trace_prefix: str = "",
) -> dict[str, Any]:
    """Ищет и валидирует PDF кандидатов.
    
    Args:
        enriched: Обогащаемая ссылка
        aggregator: Агрегатор источников
        validate_pdf_fn: Функция валидации PDF
        enrichment_log: Лог обогащения
        trace_prefix: Префикс для логов
        
    Returns:
        Обновленная ссылка
    """
    # Извлекаем arXiv ID если еще не найден
    enriched = extract_arxiv_id_from_enriched(enriched, trace_prefix)
    
    # Поиск PDF кандидатов только если PDF нет в метаданных
    if not enriched.get("pdf_url"):
        enrichment_log["steps"].append({"step": "pdf_candidates_search"})
        pdf_candidates = aggregator.get_pdf_candidates(enriched)
        
        if pdf_candidates:
            enrichment_log["steps"][-1]["candidates_count"] = len(pdf_candidates)
            logger.debug(
                f"{trace_prefix}Найдено {len(pdf_candidates)} PDF кандидатов для валидации"
            )
            
            # Сохраняем метаданные для валидации
            expected_for_validation = {
                "title": enriched.get("title", ""),
                "authors": enriched.get("authors", ""),
                "year": enriched.get("year"),
                "doi": enriched.get("doi", ""),
                "arxiv_id": enriched.get("arxiv_id", ""),
            }
            
            # Валидируем каждый кандидат и сохраняем ВСЕ результаты
            validated_candidates = []
            all_candidates_for_additional_urls = []
            
            for candidate in pdf_candidates:
                logger.debug(
                    f"{trace_prefix}PDF кандидат (до валидации): {candidate['url'][:60]}... "
                    f"(source={candidate.get('source', 'N/A')}, confidence={candidate.get('confidence', 0):.2f})"
                )
                if validate_pdf_fn:
                    validation_result = validate_pdf_fn(
                        candidate["url"],
                        expected_for_validation
                    )
                    candidate_with_validation = {
                        **candidate,
                        "validation_confidence": validation_result.get("confidence", 0.0),
                        "validated": validation_result.get("matched", False),
                    }
                    
                    if validation_result.get("matched", False):
                        validated_candidates.append(candidate_with_validation)
                        logger.debug(
                            f"{trace_prefix}PDF кандидат прошел валидацию: {candidate['url'][:50]}... "
                            f"(confidence={validation_result.get('confidence', 0):.2f})"
                        )
                    else:
                        logger.debug(
                            f"{trace_prefix}PDF кандидат не прошел валидацию: {candidate['url'][:50]}... "
                            f"(confidence={validation_result.get('confidence', 0):.2f})"
                        )
                    
                    # Сохраняем все кандидаты для additional_urls
                    all_candidates_for_additional_urls.append(candidate_with_validation)
                else:
                    # Без валидации используем кандидата как есть
                    candidate_with_validation = {**candidate, "validated": False}
                    validated_candidates.append(candidate_with_validation)
                    all_candidates_for_additional_urls.append(candidate_with_validation)
            
            # Выбираем лучший ДЕЙСТВИТЕЛЬНО валидированный кандидат
            # Фильтруем только те, что прошли валидацию (validated=True)
            truly_validated = [c for c in validated_candidates if c.get("validated", False)]
            
            if truly_validated:
                truly_validated.sort(
                    key=lambda x: (x.get("validation_confidence", 0.0), x.get("confidence", 0.0)),
                    reverse=True
                )
                best_candidate = truly_validated[0]
                enriched["pdf_url"] = best_candidate["url"]
                enriched["_pdf_validated"] = True
                enriched["_pdf_source"] = best_candidate.get("source", "")
                logger.info(
                    f"{trace_prefix} PDF назначен из кандидата: {best_candidate['url'][:60]}... "
                    f"(source={best_candidate.get('source', 'N/A')}, "
                    f"validation_confidence={best_candidate.get('validation_confidence', 0):.3f})"
                )
                
                # Сохраняем остальные валидированные кандидаты в additional_urls
                # Дедуплицируем по нормализованному URL (без query params)
                # ВАЖНО: НЕ сохраняем arXiv PDF, которые не прошли валидацию
                seen_urls = {best_candidate["url"].split("?")[0].lower()}
                additional_pdfs = []
                
                for candidate in all_candidates_for_additional_urls:
                    url = candidate["url"]
                    # Фильтруем arXiv PDF, которые не прошли валидацию
                    if "arxiv.org" in url.lower() and not candidate.get("validated", False):
                        logger.debug(
                            f"{trace_prefix}Пропускаем arXiv PDF в additional_urls (не прошел валидацию): {url[:50]}..."
                        )
                        continue
                    
                    normalized_url = url.split("?")[0].lower()
                    if normalized_url not in seen_urls:
                        seen_urls.add(normalized_url)
                        additional_pdfs.append(url)
                
                # Обновляем additional_urls с новыми PDF
                current_additional = enriched.get("additional_urls", [])
                if isinstance(current_additional, str):
                    current_additional = [current_additional] if current_additional else []
                enriched["additional_urls"] = current_additional + additional_pdfs
                
                enrichment_log["steps"][-1]["result"] = "success"
                enrichment_log["steps"][-1]["validated_count"] = len(truly_validated)
                enrichment_log["steps"][-1]["total_candidates_saved"] = len(all_candidates_for_additional_urls)
                enrichment_log["steps"][-1]["best_candidate"] = {
                    "source": best_candidate["source"],
                    "url": best_candidate["url"],
                    "confidence": best_candidate.get("confidence", 0),
                    "validation_confidence": best_candidate.get("validation_confidence", 0.0),
                }
                logger.debug(
                    f"{trace_prefix}Найден валидированный PDF из {best_candidate['source']}: "
                    f"{best_candidate['url'][:50]}... "
                    f"(всего {len(pdf_candidates)} кандидатов, {len(truly_validated)} прошли валидацию, "
                    f"{len(additional_pdfs)} сохранено в additional_urls)"
                )
            else:
                # Даже если не прошли валидацию, сохраним их в additional_urls для анализа
                # ВАЖНО: НЕ сохраняем arXiv PDF, которые не прошли валидацию - они могут быть неправильными
                seen_urls = set()
                additional_pdfs = []
                for candidate in all_candidates_for_additional_urls:
                    url = candidate["url"]
                    # Фильтруем arXiv PDF, которые не прошли валидацию
                    if "arxiv.org" in url.lower() and not candidate.get("validated", False):
                        logger.debug(
                            f"{trace_prefix}Пропускаем arXiv PDF в additional_urls (не прошел валидацию): {url[:50]}..."
                        )
                        continue
                    
                    normalized_url = url.split("?")[0].lower()
                    if normalized_url not in seen_urls:
                        seen_urls.add(normalized_url)
                        additional_pdfs.append(url)
                
                current_additional = enriched.get("additional_urls", [])
                if isinstance(current_additional, str):
                    current_additional = [current_additional] if current_additional else []
                enriched["additional_urls"] = current_additional + additional_pdfs
                
                enrichment_log["steps"][-1]["result"] = "validation_failed"
                enrichment_log["steps"][-1]["saved_to_additional_urls"] = len(additional_pdfs)
                enrichment_log["warnings"].append(
                    f"Найдено {len(pdf_candidates)} PDF кандидатов, но ни один не прошел валидацию (сохранено в additional_urls)"
                )
                logger.warning(
                    f"{trace_prefix}Найдено {len(pdf_candidates)} PDF кандидатов, но ни один не прошел валидацию "
                    f"(сохранено {len(additional_pdfs)} в additional_urls)"
                )
        else:
            enrichment_log["steps"][-1]["result"] = "not_found"
            enrichment_log["errors"].append("PDF кандидаты не найдены ни в одном источнике")
            logger.debug(f"{trace_prefix}PDF кандидаты не найдены")
    else:
        logger.debug(
            f"{trace_prefix}pdf_url уже есть (не ищем кандидатов): {enriched.get('pdf_url')[:60] if enriched.get('pdf_url') else 'N/A'}..."
        )
    
    return enriched

