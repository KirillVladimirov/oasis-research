"""Selenium-based поиск для Google Scholar и Google."""

import random
import re
import time
from typing import Any

from bs4 import BeautifulSoup
from loguru import logger
from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from oasis.enrichment.cascade.matchers import strict_title_match
from oasis.sources.base import Metadata, SourcePort


class SeleniumSearchWrapper(SourcePort):
    """Источник данных через Selenium (Google Scholar + Google)."""

    def __init__(
        self,
        delay_seconds: float = 8.0,
        headless: bool = True,
        timeout: int = 10,
    ):
        """Инициализация Selenium wrapper.

        Args:
            delay_seconds: Задержка между запросами (по умолчанию 8 сек для избежания CAPTCHA)
            headless: Запускать браузер в headless режиме
            timeout: Timeout на загрузку страниц (секунды)
        """
        self.delay_seconds = delay_seconds
        self.headless = headless
        self.timeout = timeout
        self._last_request_time = 0.0
        self._driver = None
        self._user_agents = [
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ]
        self._current_ua_index = 0

    @property
    def name(self) -> str:
        """Имя источника."""
        return "selenium"

    def search_by_title(
        self,
        title: str,
        year: int | None = None,
        authors: list[str] | None = None,
    ) -> list[Metadata]:
        """Ищет публикации по названию (реализация для SourcePort).
        
        Использует приоритет: Google Scholar → обычный Google.
        Пробует разные поисковые запросы для увеличения шансов найти PDF.
        
        Args:
            title: Название публикации
            year: Год публикации (опционально)
            authors: Список авторов (опционально)
            
        Returns:
            Список объектов Metadata
        """
        # Приоритет: Google Scholar с разными запросами
        results = self.search_google_scholar(
            title, max_results=3, year=year, authors=authors
        )
        
        # Fallback: обычный Google с разными запросами
        if not results:
            results = self.search_google(
                title, max_results=3, year=year, authors=authors
            )
        
        return results

    def _get_driver(self) -> webdriver.Chrome:
        """Получает или создает WebDriver."""
        if self._driver is None:
            try:
                chrome_options = Options()
                if self.headless:
                    chrome_options.add_argument("--headless=new")  # Новый headless режим
                
                # Базовые опции
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument("--disable-gpu")
                
                # Обход детектирования автоматизации
                chrome_options.add_argument("--disable-blink-features=AutomationControlled")
                chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
                chrome_options.add_experimental_option("useAutomationExtension", False)
                
                # Дополнительные опции для обхода CAPTCHA
                chrome_options.add_argument("--disable-web-security")
                chrome_options.add_argument("--disable-features=IsolateOrigins,site-per-process")
                chrome_options.add_argument("--disable-site-isolation-trials")
                chrome_options.add_argument("--disable-infobars")
                chrome_options.add_argument("--disable-notifications")
                chrome_options.add_argument("--disable-popup-blocking")
                chrome_options.add_argument("--start-maximized")
                chrome_options.add_argument("--window-size=1920,1080")
                
                # Ротация User-Agent
                user_agent = self._user_agents[self._current_ua_index]
                chrome_options.add_argument(f"user-agent={user_agent}")
                self._current_ua_index = (self._current_ua_index + 1) % len(self._user_agents)
                
                # Дополнительные preferences
                prefs = {
                    "profile.default_content_setting_values.notifications": 2,
                    "profile.managed_default_content_settings.images": 2,  # Отключаем загрузку изображений для скорости
                }
                chrome_options.add_experimental_option("prefs", prefs)

                service = Service(ChromeDriverManager().install())
                self._driver = webdriver.Chrome(service=service, options=chrome_options)
                self._driver.set_page_load_timeout(self.timeout)
                
                # Маскируем WebDriver через JavaScript
                self._driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                    "source": """
                        Object.defineProperty(navigator, 'webdriver', {
                            get: () => undefined
                        });
                        Object.defineProperty(navigator, 'plugins', {
                            get: () => [1, 2, 3, 4, 5]
                        });
                        Object.defineProperty(navigator, 'languages', {
                            get: () => ['en-US', 'en']
                        });
                        window.chrome = {
                            runtime: {}
                        };
                    """
                })
                
                logger.debug("WebDriver создан успешно")
            except Exception as e:
                logger.error(f"Ошибка при создании WebDriver: {e}")
                raise

        return self._driver

    def _rate_limit(self):
        """Применяет rate limiting между запросами с случайной задержкой."""
        elapsed = time.time() - self._last_request_time
        # Добавляем случайную задержку ±2 секунды для имитации человека
        random_delay = self.delay_seconds + random.uniform(-2.0, 2.0)
        random_delay = max(3.0, random_delay)  # Минимум 3 секунды
        
        if elapsed < random_delay:
            time.sleep(random_delay - elapsed)
        self._last_request_time = time.time()
    
    def _human_like_delay(self, min_seconds: float = 1.0, max_seconds: float = 3.0):
        """Случайная задержка для имитации человеческого поведения."""
        time.sleep(random.uniform(min_seconds, max_seconds))
    
    def _simulate_human_behavior(self, driver: webdriver.Chrome):
        """Имитирует человеческое поведение: движения мыши, скроллинг."""
        try:
            # Случайное движение мыши
            action = ActionChains(driver)
            for _ in range(random.randint(1, 3)):
                x_offset = random.randint(-100, 100)
                y_offset = random.randint(-100, 100)
                action.move_by_offset(x_offset, y_offset)
            action.perform()
            
            # Случайный скроллинг
            scroll_amount = random.randint(100, 500)
            driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
            self._human_like_delay(0.5, 1.5)
            
            # Скроллинг обратно
            driver.execute_script(f"window.scrollBy(0, -{scroll_amount});")
            self._human_like_delay(0.5, 1.0)
        except Exception as e:
            logger.debug(f"Ошибка при имитации человеческого поведения: {e}")

    def _check_for_captcha(self, page_source: str) -> bool:
        """Проверяет наличие CAPTCHA на странице."""
        captcha_indicators = [
            "captcha",
            "unusual traffic",
            "automated requests",
            "verify you're not a robot",
        ]
        page_lower = page_source.lower()
        return any(indicator in page_lower for indicator in captcha_indicators)

    def search_google_scholar(
        self,
        title: str,
        max_results: int = 3,
        year: int | None = None,
        authors: list[str] | None = None,
    ) -> list[Metadata]:
        """Ищет статьи по названию через Google Scholar.
        
        Пробует несколько вариантов запроса для увеличения шансов найти PDF:
        1. Базовый запрос (title)
        2. title + year
        3. title + year + author(s)
        4. "paper" + title + "pdf"

        Args:
            title: Название статьи
            max_results: Максимальное количество результатов на запрос
            year: Год публикации (опционально)
            authors: Список авторов (опционально)

        Returns:
            Список объектов Metadata
        """
        if not title or len(title) < 10:
            return []

        # Формируем разные варианты запроса
        queries = [title]  # Базовый запрос
        
        if year:
            queries.append(f"{title} {year}")
        
        if year and authors:
            # Берем 1-2 авторов
            author_names = authors[:2] if len(authors) > 2 else authors
            author_str = " ".join(author_names)
            queries.append(f"{title} {year} {author_str}")
        
        # Запрос с "paper" и "pdf"
        queries.append(f'paper "{title}" pdf')
        
        logger.debug(f"Google Scholar поиск с {len(queries)} вариантами: {title[:60]}...")

        results = []
        
        # Пробуем все варианты запроса
        for query_idx, query in enumerate(queries, 1):
            # Если уже нашли результат с PDF, не продолжаем
            if results and any(r.pdf_url for r in results):
                logger.debug(f"  PDF найден, пропускаем оставшиеся запросы")
                break
                
            try:
                self._rate_limit()
                driver = self._get_driver()

                # Формируем URL для Google Scholar
                encoded_query = query.replace(" ", "+")
                url = f"https://scholar.google.com/scholar?q={encoded_query}"

                logger.debug(f"  Попытка {query_idx}/{len(queries)}: {query[:70]}...")
                driver.get(url)
                time.sleep(2)  # Даем странице загрузиться

                page_source = driver.page_source

                # Проверка CAPTCHA
                if self._check_for_captcha(page_source):
                    logger.warning("Google Scholar показал CAPTCHA, пропускаем")
                    return []

                soup = BeautifulSoup(page_source, "html.parser")

                # Парсим результаты
                result_divs = soup.find_all("div", class_="gs_r gs_or gs_scl")
                if not result_divs:
                    # Альтернативный селектор
                    result_divs = soup.find_all("div", {"data-rp": True})

                logger.debug(f"  Google Scholar нашел {len(result_divs)} результатов для запроса {query_idx}")

                for div in result_divs[:max_results]:
                    try:
                        # Извлекаем заголовок
                        title_elem = div.find("h3", class_="gs_rt")
                        if not title_elem:
                            continue

                        result_title = title_elem.get_text(strip=True)
                        # Убираем префиксы типа [PDF], [HTML]
                        result_title = re.sub(r"^\[(?:PDF|HTML|BOOK)\]\s*", "", result_title)

                        # Извлекаем основную ссылку
                        link_elem = title_elem.find("a")
                        article_url = link_elem["href"] if link_elem and "href" in link_elem.attrs else ""

                        # Извлекаем PDF ссылку
                        pdf_url = None
                        pdf_link = div.find("div", class_="gs_or_ggsm")
                        if pdf_link:
                            pdf_a = pdf_link.find("a")
                            if pdf_a and "href" in pdf_a.attrs:
                                pdf_url = pdf_a["href"]
                                if not pdf_url.lower().endswith(".pdf"):
                                    pdf_url = None

                        # Извлекаем DOI и arXiv ID из текста
                        snippet_text = div.get_text()
                        doi = self._extract_doi_from_text(snippet_text + " " + article_url)
                        arxiv_id = self._extract_arxiv_id_from_text(snippet_text + " " + article_url)

                        # Проверяем совпадение заголовка
                        if not strict_title_match(title, result_title, threshold=0.70):
                            logger.debug(f"  Заголовок не совпадает: '{result_title[:60]}...'")
                            continue

                        results.append(
                            Metadata(
                                title=result_title,
                                article_url=article_url,
                                pdf_url=pdf_url,
                                doi=doi,
                                arxiv_id=arxiv_id,
                                source_name=self.name,
                            )
                        )

                        if len(results) >= max_results:
                            break

                    except Exception as e:
                        logger.debug(f"Ошибка при парсинге результата Google Scholar: {e}")
                        continue

            except TimeoutException:
                logger.warning(f"Timeout при загрузке Google Scholar для запроса {query_idx}")
                continue
            except WebDriverException as e:
                logger.warning(f"WebDriver ошибка для запроса {query_idx}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Ошибка при поиске в Google Scholar для запроса {query_idx}: {e}")
                continue

        logger.info(
            f"Google Scholar поиск для '{title[:60]}...' вернул {len(results)} релевантных результатов"
        )
        return results

    def search_google(
        self,
        title: str,
        max_results: int = 3,
        year: int | None = None,
        authors: list[str] | None = None,
    ) -> list[Metadata]:
        """Ищет статьи по названию через обычный Google.
        
        Пробует несколько вариантов запроса для увеличения шансов найти PDF:
        1. title + filetype:pdf
        2. title + year + filetype:pdf
        3. title + year + author(s) + pdf
        4. "paper" + title + "pdf"

        Args:
            title: Название статьи
            max_results: Максимальное количество результатов на запрос
            year: Год публикации (опционально)
            authors: Список авторов (опционально)

        Returns:
            Список объектов Metadata
        """
        if not title or len(title) < 10:
            return []

        # Формируем разные варианты запроса
        queries = [f'"{title}" filetype:pdf']  # Базовый запрос
        
        if year:
            queries.append(f'"{title}" {year} filetype:pdf')
        
        if year and authors:
            # Берем 1-2 авторов
            author_names = authors[:2] if len(authors) > 2 else authors
            author_str = " ".join(author_names)
            queries.append(f'"{title}" {year} {author_str} pdf')
        
        # Запрос с "paper" и "pdf"
        queries.append(f'paper "{title}" pdf')
        
        logger.debug(f"Google поиск с {len(queries)} вариантами: {title[:60]}...")

        results = []
        
        # Пробуем все варианты запроса
        for query_idx, query in enumerate(queries, 1):
            # Если уже нашли результат с PDF, не продолжаем
            if results and any(r.pdf_url for r in results):
                logger.debug(f"  PDF найден, пропускаем оставшиеся запросы")
                break
                
            try:
                self._rate_limit()
                driver = self._get_driver()

                # Формируем URL для Google
                query_encoded = query.replace(" ", "+").replace('"', "%22")
                url = f"https://www.google.com/search?q={query_encoded}"

                logger.debug(f"  Попытка {query_idx}/{len(queries)}: {query[:70]}...")
                driver.get(url)
                time.sleep(2)  # Даем странице загрузиться

                page_source = driver.page_source

                # Проверка CAPTCHA
                if self._check_for_captcha(page_source):
                    logger.warning(f"Google показал CAPTCHA для запроса {query_idx}, пропускаем")
                    continue

                soup = BeautifulSoup(page_source, "html.parser")

                # Парсим результаты (основной контейнер результатов)
                result_divs = soup.find_all("div", class_="g")
                if not result_divs:
                    # Альтернативный селектор
                    result_divs = soup.find_all("div", {"data-sokoban-container": True})

                logger.debug(f"  Google нашел {len(result_divs)} результатов для запроса {query_idx}")

                for div in result_divs[:max_results * 2]:  # Берем больше для фильтрации
                    try:
                        # Извлекаем ссылку
                        link_elem = div.find("a")
                        if not link_elem or "href" not in link_elem.attrs:
                            continue

                        article_url = link_elem["href"]

                        # Пропускаем неакадемические домены
                        if any(
                            domain in article_url.lower()
                            for domain in [
                                "youtube.com",
                                "facebook.com",
                                "twitter.com",
                                "linkedin.com",
                                "pinterest.com",
                                "amazon.com",
                                "ebay.com",
                                "reddit.com",
                                "wikipedia.org",
                            ]
                        ):
                            continue

                        # Извлекаем заголовок
                        title_elem = div.find("h3")
                        result_title = title_elem.get_text(strip=True) if title_elem else title

                        # Извлекаем PDF URL
                        pdf_url = article_url if article_url.lower().endswith(".pdf") else None

                        # Извлекаем DOI и arXiv ID
                        snippet_text = div.get_text()
                        doi = self._extract_doi_from_text(snippet_text + " " + article_url)
                        arxiv_id = self._extract_arxiv_id_from_text(snippet_text + " " + article_url)

                        results.append(
                            Metadata(
                                title=result_title,
                                article_url=article_url,
                                pdf_url=pdf_url,
                                doi=doi,
                                arxiv_id=arxiv_id,
                                source_name=self.name,
                            )
                        )

                        if len(results) >= max_results:
                            break

                    except Exception as e:
                        logger.debug(f"Ошибка при парсинге результата Google: {e}")
                        continue

            except TimeoutException:
                logger.warning(f"Timeout при загрузке Google для запроса {query_idx}")
                continue
            except WebDriverException as e:
                logger.warning(f"WebDriver ошибка для запроса {query_idx}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Ошибка при поиске в Google для запроса {query_idx}: {e}")
                continue

        logger.info(
            f"Google поиск для '{title[:60]}...' вернул {len(results)} релевантных результатов"
        )
        return results

    def _extract_doi_from_text(self, text: str) -> str | None:
        """Извлекает DOI из текста."""
        if not text:
            return None

        # Паттерн для DOI
        match = re.search(r"\b(10\.\d{4,}/[^\s]+)", text, re.IGNORECASE)
        if match:
            doi = match.group(1).rstrip(".,;)")
            # Базовая валидация
            if len(doi) > 7 and "/" in doi:
                return doi
        return None

    def _extract_arxiv_id_from_text(self, text: str) -> str | None:
        """Извлекает arXiv ID из текста."""
        if not text:
            return None

        # Паттерн для arXiv ID
        match = re.search(
            r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,})", text, re.IGNORECASE
        )
        if match:
            return match.group(1)

        # Альтернативный паттерн (без URL)
        match = re.search(r"\b(\d{4}\.\d{4,}(?:v\d+)?)\b", text)
        if match:
            arxiv_id = match.group(1)
            # Проверяем, что это похоже на arXiv ID (год >= 1991)
            year = int(arxiv_id[:4])
            if 1991 <= year <= 2030:
                return arxiv_id.split("v")[0]  # Убираем версию

        return None

    def close(self):
        """Закрывает WebDriver."""
        if self._driver:
            try:
                self._driver.quit()
                self._driver = None
                logger.debug("WebDriver закрыт")
            except Exception as e:
                logger.warning(f"Ошибка при закрытии WebDriver: {e}")

    def __del__(self):
        """Деструктор для закрытия WebDriver."""
        self.close()

    # Методы для совместимости с SourcePort (не используются)
    def get_by_doi(self, doi: str) -> Metadata | None:
        """Не поддерживается для Selenium."""
        return None

    def get_by_arxiv_id(self, arxiv_id: str) -> Metadata | None:
        """Не поддерживается для Selenium."""
        return None

    def get_pdf_candidates(
        self, doi: str | None = None, title: str | None = None
    ) -> list[dict[str, Any]]:
        """Не поддерживается для Selenium."""
        return []

