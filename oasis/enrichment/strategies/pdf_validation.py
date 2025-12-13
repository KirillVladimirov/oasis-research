"""Стратегия валидации PDF."""

import json
from typing import Any, Callable

from loguru import logger
from rapidfuzz import fuzz

from oasis.enrichment.utils import safe_strip, extract_pdf_from_additional
from oasis.enrichment.validate import is_trusted_source


def extract_article_url_from_pdf(enriched: dict[str, Any]) -> dict[str, Any]:
    """Извлекает article_url из pdf_url если его еще нет.
    
    Args:
        enriched: Обогащаемая ссылка
        
    Returns:
        Обновленная ссылка
    """
    if enriched.get("pdf_url") and not safe_strip(enriched.get("article_url", "")):
        from oasis.utils.url import pdf_url_to_article_url
        article_url = pdf_url_to_article_url(enriched["pdf_url"])
        if article_url:
            enriched["article_url"] = article_url
    
    return enriched


def validate_pdf_candidate(
    enriched: dict[str, Any],
    validate_pdf_fn: Callable[[str, dict[str, Any]], dict[str, Any]] | None,
    enrichment_log: dict[str, Any],
    trace_prefix: str = "",
) -> dict[str, Any]:
    """Валидирует найденный PDF.
    
    Args:
        enriched: Обогащаемая ссылка
        validate_pdf_fn: Функция валидации PDF
        enrichment_log: Лог обогащения
        trace_prefix: Префикс для логов
        
    Returns:
        Обновленная ссылка
    """
    # Шаг 1: Обновление first_author из authors
    if not enriched.get("first_author"):
        authors_val = enriched.get("authors")
        authors_str = ""
        if isinstance(authors_val, str):
            authors_str = authors_val.strip()
        elif authors_val is not None:
            try:
                import pandas as _pd
                if not _pd.isna(authors_val):
                    authors_str = str(authors_val).strip()
            except Exception:
                authors_str = str(authors_val).strip()
        if authors_str:
            enriched["first_author"] = authors_str.split(";")[0].strip()

    # Шаг 2: Валидация PDF
    pdf_url = safe_strip(enriched.get("pdf_url", ""))
    if pdf_url and validate_pdf_fn:
        expected = {
            "title": enriched.get("title", ""),
            "authors": enriched.get("authors", ""),
            "year": enriched.get("year"),
            "doi": enriched.get("doi", ""),
            "arxiv_id": enriched.get("arxiv_id", ""),
        }
        enrichment_log["steps"].append({"step": "pdf_validation", "pdf_url": pdf_url})
        validation_result = validate_pdf_fn(pdf_url, expected)
        
        # Детальное логирование результата валидации
        extracted_title = validation_result.get("extracted", {}).get("title", "")
        reason = validation_result.get("reason", "")
        matched = validation_result.get("matched", False)
        confidence = validation_result.get("confidence", 0.0)
        
        logger.debug(
            f"{trace_prefix}PDF валидация: url={pdf_url[:60]}..., "
            f"matched={matched}, confidence={confidence:.3f}, "
            f"извлеченный_title='{extracted_title[:60] if extracted_title else 'N/A'}...', "
            f"reason={reason[:100] if reason else 'N/A'}"
        )
        
        enrichment_log["steps"][-1]["validation_result"] = {
            "matched": matched,
            "confidence": confidence,
            "extracted_title": extracted_title[:100] if extracted_title else None,
            "reason": reason[:200] if reason else None,
        }
        
        if not matched:
            pdf_source_hint = enriched.get("_pdf_source", "")
            
            # Проверяем доверенные источники и домены
            # ВАЖНО: arXiv НЕ доверенный для автоматического принятия (может быть неправильный ID)
            trusted_pdf = is_trusted_source(pdf_url) or pdf_source_hint in {
                "openalex",
                "semanticscholar",
                "semantic_scholar",
                "crossref",
                "unpaywall",
            }
            
            # Также проверяем известные академические домены в URL
            # ВАЖНО: arxiv.org ИСКЛЮЧЁН - может быть неправильный ID
            academic_domains = [
                "aaai.org", "ojs.aaai.org",  # AAAI
                "openreview.net",  # OpenReview
                "proceedings.mlr.press",  # PMLR/JMLR
                "dl.acm.org",  # ACM Digital Library
                "hal.science",  # HAL (HAL Archives Ouvertes)
            ]
            if any(domain in pdf_url.lower() for domain in academic_domains):
                trusted_pdf = True
            
            # Принимаем PDF если: 1) доверенный источник, 2) confidence >= 0.6, 3) title совпадает
            title_ratio = 0.0
            if extracted_title:
                title_ratio = fuzz.ratio(
                    enriched.get("title", "").lower(), 
                    extracted_title.lower()
                )
            
            accept_pdf = (
                trusted_pdf or 
                confidence >= 0.60 or 
                (confidence >= 0.50 and title_ratio > 85)
            )
            
            logger.debug(
                f"{trace_prefix}PDF валидация: trusted_pdf={trusted_pdf}, "
                f"confidence={confidence:.3f}, title_ratio={title_ratio:.1f}, "
                f"accept_pdf={accept_pdf}"
            )

            if accept_pdf:
                enrichment_log["steps"][-1]["result"] = "accepted_with_reduced_validation"
                enrichment_log["warnings"].append(
                    f"PDF {pdf_url[:50]}... принят с пониженными требованиями "
                    f"(confidence={confidence:.2f}, источник: {pdf_source_hint or 'trusted domain'})"
                )
                logger.info(
                    f"{trace_prefix}PDF принят с пониженными требованиями: {pdf_url[:60]}... "
                    f"(trusted={trusted_pdf}, confidence={confidence:.3f})"
                )
                additional_list = enriched.get("additional_urls") or []
                if not isinstance(additional_list, list):
                    additional_list = []
                if pdf_url not in additional_list:
                    additional_list.append(pdf_url)
                enriched["additional_urls"] = additional_list
                enriched["_pdf_validated"] = False
                # НЕ удаляем pdf_url!
            else:
                enrichment_log["steps"][-1]["result"] = "failed"
                enrichment_log["errors"].append(
                    f"PDF валидация провалилась (confidence={confidence:.2f}, reason={reason[:100] if reason else 'N/A'})"
                )
                logger.warning(
                    f"{trace_prefix}PDF не соответствует статье (confidence={confidence:.3f}): {pdf_url[:80]}...\n"
                    f"   Ожидаемый title: '{expected.get('title', 'N/A')[:60]}...'\n"
                    f"   Извлеченный title: '{extracted_title[:60] if extracted_title else 'N/A'}...'\n"
                    f"   Причина: {reason[:100] if reason else 'N/A'}"
                )
                existing_add = enriched.get("additional_urls") or []
                if not isinstance(existing_add, list):
                    existing_add = []
                if pdf_url not in existing_add:
                    existing_add.append(pdf_url)
                enriched["additional_urls"] = existing_add
                
                # Логируем ДО очистки pdf_url с деталями почему он очищается
                logger.warning(
                    f"{trace_prefix}️ Очищаем pdf_url (не прошел валидацию): {pdf_url[:80]}...\n"
                    f"   Причина: confidence={confidence:.3f} < threshold, reason={reason[:100] if reason else 'N/A'}\n"
                    f"   Ожидаемый title: '{expected.get('title', 'N/A')[:60]}...'\n"
                    f"   Извлеченный title: '{extracted_title[:60] if extracted_title else 'N/A'}...'\n"
                    f"   PDF сохранен в additional_urls для анализа"
                )
                
                enriched["pdf_url"] = ""
                enriched.pop("_pdf_source", None)
        else:
            enrichment_log["steps"][-1]["result"] = "success"
            enriched["_pdf_validated"] = True
            # ВАЖНО: НЕ перезаписываем title из PDF без строгой проверки!
            # Это может привести к замене правильного заголовка неправильным
            # extracted_title = validation_result.get("extracted", {}).get("title")
            # if extracted_title and len(extracted_title) > len(enriched.get("title", "")):
            #     enriched["title"] = extracted_title

    # Шаг 3: Fallback на additional_urls ОТКЛЮЧЁН
    # ВАЖНО: Больше НЕ используем additional_urls как fallback без валидации
    # Если PDF не прошёл валидацию, лучше оставить пустым чем взять неправильный
    if not safe_strip(enriched.get("pdf_url", "")):
        # Логируем что PDF не найден
        if enriched.get("additional_urls"):
            enrichment_log["warnings"].append(
                "PDF не найден: кандидаты есть в additional_urls, но ни один не прошёл валидацию"
            )
            enriched["_pdf_validated"] = False

    # Шаг 4: Извлекаем article_url из pdf_url
    enriched = extract_article_url_from_pdf(enriched)

    return enriched

