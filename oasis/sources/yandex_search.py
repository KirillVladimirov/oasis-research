"""Yandex поиск через Selenium для обхода Google CAPTCHA."""

import random
import re
import time
from typing import Any

from bs4 import BeautifulSoup
from loguru import logger
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from oasis.enrichment.cascade.matchers import strict_title_match
from oasis.sources.base import Metadata, SourcePort


class YandexSearchWrapper(SourcePort):
    """Источник данных через Yandex поиск (альтернатива Google)."""

    def __init__(
        self,
        delay_seconds: float = 5.0,
        headless: bool = True,
        timeout: int = 10,
    ):
        """Инициализация Yandex wrapper.

        Args:
            delay_seconds: Задержка между запросами (по умолчанию 5 сек)
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
        return "yandex"

    def search_by_title(
        self,
        title: str,
        year: int | None = None,
        authors: list[str] | None = None,
    ) -> list[Metadata]:
        """Ищет публикации по названию через Yandex.

        Args:
            title: Название публикации
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
            author_names = authors[:2] if len(authors) > 2 else authors
            author_str = " ".join(author_names)
            queries.append(f'"{title}" {year} {author_str} pdf')

        logger.debug(f"Yandex поиск с {len(queries)} вариантами: {title[:60]}...")

        results = []

        # Пробуем все варианты запроса
        for query_idx, query in enumerate(queries, 1):
            # Если уже нашли результат с PDF, не продолжаем
            if results and any(r.pdf_url for r in results):
                logger.debug("  PDF найден, пропускаем оставшиеся запросы")
                break

            try:
                self._rate_limit()
                driver = self._get_driver()

                # Формируем URL для Yandex
                query_encoded = query.replace(" ", "+").replace('"', "%22")
                url = f"https://yandex.ru/search/?text={query_encoded}"

                logger.debug(f"  Попытка {query_idx}/{len(queries)}: {query[:70]}...")
                driver.get(url)
                time.sleep(2)  # Даем странице загрузиться

                page_source = driver.page_source

                # Проверка CAPTCHA
                if self._check_for_captcha(page_source):
                    logger.warning(f"Yandex показал CAPTCHA для запроса {query_idx}, пропускаем")
                    continue

                soup = BeautifulSoup(page_source, "html.parser")

                # Парсим результаты Yandex
                # Yandex использует класс 'OrganicTitle-LinkText' для заголовков
                result_divs = soup.find_all("li", class_="serp-item")
                if not result_divs:
                    # Альтернативный селектор
                    result_divs = soup.find_all("div", class_="organic")

                logger.debug(f"  Yandex нашел {len(result_divs)} результатов для запроса {query_idx}")

                for div in result_divs[:10]:  # Берем больше для фильтрации
                    try:
                        # Извлекаем ссылку
                        link_elem = div.find("a", class_="link")
                        if not link_elem:
                            link_elem = div.find("a")
                        
                        if not link_elem or "href" not in link_elem.attrs:
                            continue

                        article_url = link_elem["href"]

                        # Пропускаем неакадемические домены
                        if any(
                            domain in article_url.lower()
                            for domain in [
                                "youtube.com",
                                "vk.com",
                                "ok.ru",
                                "dzen.ru",
                                "rutube.ru",
                                "amazon.com",
                                "ebay.com",
                            ]
                        ):
                            continue

                        # Извлекаем заголовок (приоритет: метаданные страницы)
                        result_title = None
                        
                        # 1. Пробуем найти заголовок в метаданных результата
                        title_elem = div.find("h2", class_=re.compile(r"title|heading"))
                        if not title_elem:
                            title_elem = div.find("a", class_=re.compile(r"link|title"))
                        if title_elem:
                            result_title = title_elem.get_text(strip=True)
                        
                        # 2. Если заголовок не найден или это PDF URL - пропускаем
                        if not result_title:
                            # Для PDF файлов не используем URL как заголовок
                            if article_url.lower().endswith(".pdf"):
                                logger.debug(f"Пропускаем PDF без метаданных заголовка: {article_url[:80]}...")
                                continue
                            # Для других ссылок используем оригинальный запрос
                            result_title = title
                        
                        # 3. Нормализуем заголовок (убираем лишнее)
                        # Убираем расширения файлов из заголовка
                        if result_title.lower().endswith(('.pdf', '.doc', '.docx')):
                            result_title = result_title.rsplit('.', 1)[0]
                        
                        # Убираем служебные префиксы
                        result_title = re.sub(r'^\[PDF\]\s*', '', result_title, flags=re.IGNORECASE)
                        result_title = re.sub(r'^\[DOC\]\s*', '', result_title, flags=re.IGNORECASE)
                        result_title = result_title.strip()

                        # Извлекаем PDF URL
                        pdf_url = article_url if article_url.lower().endswith(".pdf") else None

                        # Извлекаем DOI и arXiv ID
                        snippet_text = div.get_text()
                        doi = self._extract_doi_from_text(snippet_text + " " + article_url)
                        arxiv_id = self._extract_arxiv_id_from_text(snippet_text + " " + article_url)

                        # Валидация: если есть PDF, проверяем что заголовок нормальный
                        if pdf_url and result_title:
                            # Пропускаем если заголовок похож на URL или имя файла
                            if len(result_title) < 10 or result_title.lower().replace(' ', '').replace('-', '').replace('_', '') == article_url.lower().split('/')[-1].replace('.pdf', '').replace(' ', '').replace('-', '').replace('_', ''):
                                logger.debug(f"Пропускаем результат с заголовком из URL: {result_title[:60]}...")
                                continue
                            
                            # Проверяем базовое совпадение заголовка (не строго, только чтобы не было полной ерунды)
                            from oasis.enrichment.cascade.matchers import strict_title_match
                            if not strict_title_match(title, result_title, threshold=0.50):
                                logger.debug(f"Пропускаем результат: заголовок не совпадает с запросом '{title[:60]}...' vs '{result_title[:60]}...'")
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

                        if len(results) >= 3:
                            break

                    except Exception as e:
                        logger.debug(f"Ошибка при парсинге результата Yandex: {e}")
                        continue

            except TimeoutException:
                logger.warning(f"Timeout при загрузке Yandex для запроса {query_idx}")
                continue
            except WebDriverException as e:
                logger.warning(f"WebDriver ошибка для запроса {query_idx}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Ошибка при поиске в Yandex для запроса {query_idx}: {e}")
                continue

        logger.info(
            f"Yandex поиск для '{title[:60]}...' вернул {len(results)} релевантных результатов"
        )
        return results

    def _get_driver(self) -> webdriver.Chrome:
        """Получает или создает WebDriver."""
        if self._driver is None:
            try:
                chrome_options = Options()
                if self.headless:
                    chrome_options.add_argument("--headless=new")

                # Базовые опции
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument("--disable-gpu")

                # Обход детектирования автоматизации
                chrome_options.add_argument("--disable-blink-features=AutomationControlled")
                chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
                chrome_options.add_experimental_option("useAutomationExtension", False)

                # Дополнительные опции
                chrome_options.add_argument("--disable-web-security")
                chrome_options.add_argument("--disable-infobars")
                chrome_options.add_argument("--disable-notifications")
                chrome_options.add_argument("--start-maximized")
                chrome_options.add_argument("--window-size=1920,1080")

                # Ротация User-Agent
                user_agent = self._user_agents[self._current_ua_index]
                chrome_options.add_argument(f"user-agent={user_agent}")
                self._current_ua_index = (self._current_ua_index + 1) % len(self._user_agents)

                # Preferences
                prefs = {
                    "profile.default_content_setting_values.notifications": 2,
                }
                chrome_options.add_experimental_option("prefs", prefs)

                service = Service(ChromeDriverManager().install())
                self._driver = webdriver.Chrome(service=service, options=chrome_options)
                self._driver.set_page_load_timeout(self.timeout)

                # Маскируем WebDriver
                self._driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                    "source": """
                        Object.defineProperty(navigator, 'webdriver', {
                            get: () => undefined
                        });
                    """
                })

                logger.debug("WebDriver для Yandex создан успешно")
            except Exception as e:
                logger.error(f"Ошибка при создании WebDriver: {e}")
                raise

        return self._driver

    def _rate_limit(self):
        """Применяет rate limiting между запросами."""
        elapsed = time.time() - self._last_request_time
        random_delay = self.delay_seconds + random.uniform(-1.0, 1.0)
        random_delay = max(2.0, random_delay)

        if elapsed < random_delay:
            time.sleep(random_delay - elapsed)
        self._last_request_time = time.time()

    def _check_for_captcha(self, page_source: str) -> bool:
        """Проверяет наличие CAPTCHA на странице."""
        captcha_indicators = [
            "smartcaptcha",
            "я не робот",
            "подтвердите, что запросы отправляли вы",
            "captcha",
        ]
        page_lower = page_source.lower()
        return any(indicator in page_lower for indicator in captcha_indicators)

    def _extract_doi_from_text(self, text: str) -> str | None:
        """Извлекает DOI из текста."""
        if not text:
            return None

        match = re.search(r"\b(10\.\d{4,}/[^\s]+)", text, re.IGNORECASE)
        if match:
            doi = match.group(1).rstrip(".,;)")
            if len(doi) > 7 and "/" in doi:
                return doi
        return None

    def _extract_arxiv_id_from_text(self, text: str) -> str | None:
        """Извлекает arXiv ID из текста."""
        if not text:
            return None

        match = re.search(
            r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,})", text, re.IGNORECASE
        )
        if match:
            return match.group(1)

        match = re.search(r"\b(\d{4}\.\d{4,}(?:v\d+)?)\b", text)
        if match:
            arxiv_id = match.group(1)
            year = int(arxiv_id[:4])
            if 1991 <= year <= 2030:
                return arxiv_id.split("v")[0]

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
        """Не реализовано для Yandex."""
        return None

    def get_pdf_candidates(self, metadata: Metadata) -> list:
        """Не реализовано для Yandex."""
        return []

