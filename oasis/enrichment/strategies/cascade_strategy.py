"""Стратегия каскадного поиска."""

from typing import Any

from loguru import logger

from oasis.enrichment.cascade_search import cascade_search_reference
from oasis.enrichment.sources.aggregator import SourceAggregator
from oasis.enrichment.sources.arxiv_wrapper import ArxivWrapper
from oasis.enrichment.sources.openalex_wrapper import OpenAlexWrapper
from oasis.enrichment.sources.researchgate_wrapper import ResearchGateWrapper, IEEEWrapper
from oasis.enrichment.sources.merger import merge_metadata


def execute_cascade_search(
    enriched: dict[str, Any],
    title: str,
    aggregator: SourceAggregator,
    enrichment_log: dict[str, Any],
    trace_prefix: str = "",
    selenium_wrapper=None,
    use_selenium: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Выполняет каскадный поиск по различным источникам.
    
    Args:
        enriched: Обогащаемая ссылка
        title: Заголовок для поиска
        aggregator: Агрегатор источников
        enrichment_log: Лог обогащения
        trace_prefix: Префикс для логов
        selenium_wrapper: SeleniumSearchWrapper для Selenium поиска (опционально)
        use_selenium: Использовать ли Selenium поиск (по умолчанию False)
        
    Returns:
        Кортеж (обновленная ссылка, нужен ли fallback)
    """
    need_title_enrich = title and (
        not enriched.get("venue")
        or not enriched.get("year")
        or not enriched.get("doi")
        or not enriched.get("arxiv_id")
        or not enriched.get("article_url")
        or not enriched.get("pdf_url")
    )
    
    need_fallback = False
    
    if need_title_enrich:
        try:
            arxiv_wrapper = ArxivWrapper()
            openalex_wrapper = OpenAlexWrapper()
            researchgate_wrapper = ResearchGateWrapper()
            ieee_wrapper = IEEEWrapper()

            cascade_result = cascade_search_reference(
                enriched,
                arxiv_wrapper=arxiv_wrapper,
                openalex_wrapper=openalex_wrapper,
                aggregator=aggregator,
                researchgate_wrapper=researchgate_wrapper,
                ieee_wrapper=ieee_wrapper,
                selenium_wrapper=selenium_wrapper,
                use_selenium=use_selenium,
            )
            
            if cascade_result:
                # Логируем что было найдено ДО merge_metadata
                logger.debug(
                    f"{trace_prefix}Каскадный поиск нашел (ДО merge_metadata): "
                    f"article_url={cascade_result.article_url or 'N/A'}, "
                    f"pdf_url={cascade_result.pdf_url or 'N/A'}, "
                    f"doi={cascade_result.doi or 'N/A'}, "
                    f"enriched.article_url={enriched.get('article_url') or 'N/A'}, "
                    f"enriched.pdf_url={enriched.get('pdf_url') or 'N/A'}"
                )
                
                # Объединяем результаты каскадного поиска
                enriched = merge_metadata(enriched, cascade_result)
                
                # Логируем что было найдено ПОСЛЕ merge_metadata
                logger.debug(
                    f"{trace_prefix}Каскадный поиск (ПОСЛЕ merge_metadata): "
                    f"enriched.article_url={enriched.get('article_url') or 'N/A'}, "
                    f"enriched.pdf_url={enriched.get('pdf_url') or 'N/A'}, "
                    f"enriched.doi={enriched.get('doi') or 'N/A'}"
                )
                
                # Если есть pdf_url из каскадного поиска, используем его
                if cascade_result.pdf_url:
                    pdf_url = cascade_result.pdf_url
                    if cascade_result.source_name:
                        enriched["_pdf_source"] = cascade_result.source_name
                    # Проверяем, является ли это DOI
                    if pdf_url.startswith("https://doi.org/") or pdf_url.startswith("http://doi.org/"):
                        # Это DOI, а не PDF - если есть arXiv ID, используем arXiv PDF
                        if cascade_result.arxiv_id:
                            pdf_url = f"https://arxiv.org/pdf/{cascade_result.arxiv_id}.pdf"
                            enriched["pdf_url"] = pdf_url
                            enriched["_pdf_source"] = cascade_result.source_name or "arxiv"
                        else:
                            enriched["pdf_url"] = ""
                            enriched.pop("_pdf_source", None)
                    else:
                        enriched["pdf_url"] = pdf_url
                
                # Логируем финальное состояние после присвоения pdf_url
                logger.debug(
                    f"{trace_prefix}Каскадный поиск (ФИНАЛЬНОЕ состояние): "
                    f"enriched.article_url={enriched.get('article_url') or 'N/A'}, "
                    f"enriched.pdf_url={enriched.get('pdf_url') or 'N/A'}, "
                    f"enriched.doi={enriched.get('doi') or 'N/A'}"
                )
                
                enrichment_log["steps"].append({
                    "step": "cascade_search",
                    "result": "success",
                    "found": True,
                })
                logger.debug(f"{trace_prefix}Каскадный поиск успешен: {cascade_result.title[:50]}...")
            else:
                enrichment_log["steps"].append({
                    "step": "cascade_search",
                    "result": "not_found",
                })
                logger.debug(f"{trace_prefix}Каскадный поиск не дал результатов, переходим к обычному поиску")
                need_fallback = True
                
        except Exception as e:
            enrichment_log["errors"].append({"step": "cascade_search", "error": str(e)})
            logger.warning(f"{trace_prefix}Ошибка при каскадном поиске: {e}, переходим к обычному поиску")
            need_fallback = True
    
    return enriched, need_fallback

