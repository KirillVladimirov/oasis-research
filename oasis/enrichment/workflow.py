"""Оркестратор процесса обогащения ссылок."""

from typing import Any, Callable

from loguru import logger

from oasis.enrichment.sources.aggregator import SourceAggregator
from oasis.enrichment.utils import safe_strip
from oasis.enrichment.strategies import (
    search_by_doi,
    search_unpaywall_pdf,
    execute_cascade_search,
    search_by_title_fallback,
    search_pdf_candidates,
    validate_pdf_candidate,
)


class EnrichmentWorkflow:
    """Оркестратор процесса обогащения библиографических ссылок.
    
    Управляет последовательным выполнением стратегий обогащения:
    1. Поиск по DOI
    2. Поиск PDF через Unpaywall
    3. Каскадный поиск (если нужно)
    4. Fallback поиск по заголовку
    5. Поиск PDF кандидатов
    6. Валидация PDF
    """
    
    def __init__(
        self,
        aggregator: SourceAggregator,
        validate_pdf_fn: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ):
        """Инициализация workflow.
        
        Args:
            aggregator: Агрегатор источников данных
            validate_pdf_fn: Функция валидации PDF (опционально)
        """
        self.aggregator = aggregator
        self.validate_pdf_fn = validate_pdf_fn
        self.selenium_wrapper = None
        self.use_selenium = False  # По умолчанию выключен
    
    def enable_selenium_search(self, delay_seconds: float = 8.0):
        """Включает Selenium поиск для статей без PDF.
        
        Args:
            delay_seconds: Задержка между запросами (по умолчанию 8 сек для обхода CAPTCHA)
        """
        try:
            from oasis.sources.selenium_search import SeleniumSearchWrapper
            self.selenium_wrapper = SeleniumSearchWrapper(delay_seconds=delay_seconds)
            self.use_selenium = True
            logger.info(f"Selenium поиск включен (задержка: {delay_seconds} сек)")
        except ImportError:
            logger.warning("selenium не установлен, Selenium поиск недоступен")
        except Exception as e:
            logger.warning(f"Ошибка при инициализации Selenium: {e}")
    
    def execute(
        self,
        ref: dict[str, Any],
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        """Выполняет полный цикл обогащения ссылки.
        
        Args:
            ref: Исходная ссылка для обогащения
            trace_id: Идентификатор трейса для логирования
            
        Returns:
            Обогащённая ссылка
        """
        trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
        enriched = ref.copy()
        
        # Инициализация лога обогащения
        enrichment_log = {
            "ref_number": ref.get("ref_number", ""),
            "original_title": ref.get("title", ""),
            "steps": [],
            "errors": [],
            "warnings": [],
        }
        
        # Извлекаем основные поля
        doi = safe_strip(enriched.get("doi"))
        title = safe_strip(enriched.get("title"))
        arxiv_id = safe_strip(enriched.get("arxiv_id"))
        
        # Шаг 1: Поиск по DOI
        if doi:
            enriched = search_by_doi(
                enriched, doi, self.aggregator, enrichment_log, trace_prefix
            )
        
        # Шаг 2: Поиск PDF через Unpaywall
        if doi:
            enriched = search_unpaywall_pdf(
                enriched, doi, self.aggregator, enrichment_log, trace_prefix
            )
        
        # Шаг 3: Каскадный поиск (если не хватает данных)
        enriched, need_fallback = execute_cascade_search(
            enriched, title, self.aggregator, enrichment_log, trace_prefix,
            selenium_wrapper=self.selenium_wrapper,
            use_selenium=self.use_selenium,
        )
        
        # Логируем состояние после cascade_search
        logger.debug(
            f"{trace_prefix}После cascade_search: "
            f"article_url={enriched.get('article_url') or 'N/A'}, "
            f"pdf_url={enriched.get('pdf_url') or 'N/A'}, "
            f"doi={enriched.get('doi') or 'N/A'}"
        )
        
        # Шаг 4: Fallback поиск по заголовку
        if need_fallback:
            enriched = search_by_title_fallback(
                enriched, title, doi, arxiv_id, self.aggregator, enrichment_log, trace_prefix
            )
        
        # Шаг 5: Поиск PDF кандидатов
        enriched = search_pdf_candidates(
            enriched, self.aggregator, self.validate_pdf_fn, enrichment_log, trace_prefix
        )
        
        # Логируем состояние после pdf_search
        logger.debug(
            f"{trace_prefix}После pdf_search: "
            f"article_url={enriched.get('article_url') or 'N/A'}, "
            f"pdf_url={enriched.get('pdf_url') or 'N/A'}, "
            f"doi={enriched.get('doi') or 'N/A'}"
        )
        
        # Шаг 5.5: Selenium поиск (Google Scholar + Yandex) если PDF не найден
        if not enriched.get("pdf_url") and title:
            logger.debug(f"{trace_prefix}PDF не найден после всех попыток, запускаем Selenium поиск")
            
            # Включаем Selenium если еще не включен
            if not self.use_selenium:
                self.enable_selenium_search()
            
            # Запускаем Selenium поиск напрямую
            if self.use_selenium and self.selenium_wrapper:
                enrichment_log["steps"].append({"step": "5.5_selenium", "value": title})
                
                # Извлекаем year и authors
                year = enriched.get("year")
                authors = enriched.get("authors", [])
                if isinstance(authors, str):
                    authors = [authors]
                
                try:
                    # Приоритет: Google Scholar
                    logger.debug(f"{trace_prefix}Selenium: пробуем Google Scholar")
                    selenium_results = self.selenium_wrapper.search_google_scholar(
                        title, max_results=3, year=year, authors=authors
                    )
                    
                    # Fallback: Yandex (если Google Scholar не нашел PDF)
                    has_pdf_in_results = any(r.pdf_url for r in selenium_results) if selenium_results else False
                    if not selenium_results or not has_pdf_in_results:
                        logger.debug(f"{trace_prefix}Selenium: Google Scholar не нашел PDF, пробуем Yandex")
                        try:
                            from oasis.sources.yandex_search import YandexSearchWrapper
                            yandex_wrapper = YandexSearchWrapper(delay_seconds=5.0, headless=self.selenium_wrapper.headless)
                            yandex_results = yandex_wrapper.search_by_title(
                                title, year=year, authors=authors
                            )
                            if yandex_results:
                                if has_pdf_in_results:
                                    selenium_results.extend(yandex_results)
                                else:
                                    selenium_results = yandex_results
                                logger.debug(f"{trace_prefix}Yandex вернул {len(yandex_results)} результатов")
                            yandex_wrapper.close()
                        except Exception as e:
                            logger.debug(f"{trace_prefix}Ошибка при Yandex поиске: {e}")
                    
                    # Обрабатываем результаты с СТРОГОЙ валидацией заголовка
                    if selenium_results:
                        from oasis.enrichment.cascade.matchers import strict_title_match
                        
                        for result in selenium_results:
                            # Если есть PDF - проверяем заголовок
                            if result.pdf_url and not result.pdf_url.startswith("https://doi.org/"):
                                # КРИТИЧНО: Проверяем, что заголовок совпадает с исходным запросом
                                original_title = title  # Исходный запрос
                                found_title = result.title  # Найденный заголовок
                                
                                if not strict_title_match(original_title, found_title, threshold=0.90):
                                    logger.warning(
                                        f"{trace_prefix} Selenium нашел PDF, но заголовок НЕ совпадает:\n"
                                        f"   Искали: '{original_title[:60]}...'\n"
                                        f"   Нашли:  '{found_title[:60]}...'\n"
                                        f"   Пропускаем этот результат."
                                    )
                                    continue  # Пропускаем этот результат
                                
                                # Заголовок совпадает - принимаем результат
                                enriched["pdf_url"] = result.pdf_url
                                enriched["article_url"] = enriched.get("article_url") or result.article_url
                                # ВАЖНО: НЕ перезаписываем title! Оставляем оригинальный
                                # enriched["title"] уже содержит правильный title
                                enriched["doi"] = enriched.get("doi") or result.doi
                                enriched["arxiv_id"] = enriched.get("arxiv_id") or result.arxiv_id
                                enrichment_log["steps"].append(
                                    {"step": "5.5_selenium", "status": "success", "found": True}
                                )
                                logger.info(
                                    f"{trace_prefix} PDF найден через Selenium (title совпадает): {result.pdf_url[:60]}..."
                                )
                                break
                    else:
                        logger.debug(f"{trace_prefix}Selenium не нашел результатов")
                        
                except Exception as e:
                    enrichment_log["errors"].append({"step": "5.5_selenium", "error": str(e)})
                    logger.debug(f"{trace_prefix}Ошибка при Selenium поиске: {e}")
        
        # Шаг 6: Валидация PDF и finalization
        enriched = validate_pdf_candidate(
            enriched, self.validate_pdf_fn, enrichment_log, trace_prefix
        )
        
        # Логируем состояние после pdf_validation
        logger.debug(
            f"{trace_prefix}После pdf_validation: "
            f"article_url={enriched.get('article_url') or 'N/A'}, "
            f"pdf_url={enriched.get('pdf_url') or 'N/A'}, "
            f"doi={enriched.get('doi') or 'N/A'}, "
            f"additional_urls_count={len(enriched.get('additional_urls', [])) if isinstance(enriched.get('additional_urls'), list) else 'N/A'}"
        )
        
        # Сохраняем лог обогащения
        enriched["_enrichment_log"] = enrichment_log
        
        # Логируем итоговый результат
        if enrichment_log["errors"]:
            logger.warning(
                f"{trace_prefix}Обогащение [{enriched.get('ref_number', '?')}] завершено с ошибками: "
                f"{len(enrichment_log['errors'])} ошибок, {len(enrichment_log['warnings'])} предупреждений"
            )
        elif enrichment_log["warnings"]:
            logger.info(
                f"{trace_prefix}Обогащение [{enriched.get('ref_number', '?')}] завершено с предупреждениями: "
                f"{len(enrichment_log['warnings'])} предупреждений"
            )
        
        # Вспомогательные служебные поля не сохраняем в итоговый CSV
        enriched.pop("_pdf_source", None)
        
        return enriched

