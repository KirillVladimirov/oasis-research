"""Google Search источник для поиска статей."""

import re
import time
from typing import Any

import requests
from bs4 import BeautifulSoup
from loguru import logger

from oasis.enrichment.cascade.matchers import strict_title_match
from oasis.sources.base import Metadata, SourcePort


class GoogleSearchWrapper(SourcePort):
    """Источник данных через Google поиск."""

    def __init__(
        self,
        delay_seconds: float = 3.0,
        max_retries: int = 3,
        user_agent: str | None = None,
    ):
        """Инициализация Google Search wrapper.

        Args:
            delay_seconds: Задержка между запросами (по умолчанию 3 сек)
            max_retries: Максимальное количество повторных попыток
            user_agent: User-Agent для запросов (опционально)
        """
        self.delay_seconds = delay_seconds
        self.max_retries = max_retries
        self.user_agent = user_agent or (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        self._last_request_time = 0.0

    @property
    def name(self) -> str:
        """Имя источника."""
        return "google"

    def _rate_limit(self):
        """Применяет rate limiting между запросами."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)
        self._last_request_time = time.time()

    def _search_google(self, query: str, num_results: int = 10) -> list[str]:
        """Выполняет поиск через Google и возвращает список URL.

        Args:
            query: Поисковый запрос
            num_results: Максимальное количество результатов

        Returns:
            Список URL результатов поиска
        """
        try:
            from googlesearch import search

            self._rate_limit()
            
            # Выполняем поиск
            results = []
            for url in search(query, num_results=num_results, lang="en", sleep_interval=self.delay_seconds):
                results.append(url)
                if len(results) >= num_results:
                    break
            
            logger.debug(f"Google поиск по запросу '{query[:50]}...' вернул {len(results)} результатов")
            return results

        except Exception as e:
            logger.warning(f"Ошибка при поиске через Google: {e}")
            return []

    def _fetch_page(self, url: str) -> str:
        """Загружает содержимое страницы.

        Args:
            url: URL страницы

        Returns:
            HTML содержимое страницы
        """
        try:
            self._rate_limit()
            response = self.session.get(url, timeout=10, allow_redirects=True)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            logger.debug(f"Ошибка при загрузке страницы {url}: {e}")
        return ""

    def _extract_page_title(self, html: str) -> str:
        """Извлекает заголовок страницы из HTML.

        Args:
            html: HTML содержимое

        Returns:
            Извлеченный заголовок
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            # Приоритет 1: meta tag citation_title
            meta_title = soup.find("meta", attrs={"name": "citation_title"})
            if meta_title and meta_title.get("content"):
                return meta_title["content"].strip()
            
            # Приоритет 2: meta tag og:title
            og_title = soup.find("meta", attrs={"property": "og:title"})
            if og_title and og_title.get("content"):
                return og_title["content"].strip()
            
            # Приоритет 3: title tag
            title_tag = soup.find("title")
            if title_tag:
                title_text = title_tag.get_text().strip()
                # Очищаем от типичных суффиксов сайтов
                title_text = re.sub(r"\s*[\|\-]\s*(IEEE|ACM|Springer|arXiv).*$", "", title_text, flags=re.IGNORECASE)
                return title_text
            
            # Приоритет 4: h1 tag
            h1_tag = soup.find("h1")
            if h1_tag:
                return h1_tag.get_text().strip()
                
        except Exception as e:
            logger.debug(f"Ошибка при извлечении заголовка: {e}")
        
        return ""

    def _extract_doi_from_page(self, html: str) -> str:
        """Извлекает DOI из HTML страницы.

        Args:
            html: HTML содержимое

        Returns:
            Извлеченный DOI (или пустая строка)
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            # Приоритет 1: meta tag citation_doi
            meta_doi = soup.find("meta", attrs={"name": "citation_doi"})
            if meta_doi and meta_doi.get("content"):
                return meta_doi["content"].strip()
            
            # Приоритет 2: поиск в тексте по regex
            doi_match = re.search(r"10\.\d{4,}/[^\s\"<>]+", html)
            if doi_match:
                return doi_match.group(0)
                
        except Exception as e:
            logger.debug(f"Ошибка при извлечении DOI: {e}")
        
        return ""

    def _extract_arxiv_id_from_page(self, url: str, html: str) -> str:
        """Извлекает arXiv ID из URL или HTML.

        Args:
            url: URL страницы
            html: HTML содержимое

        Returns:
            Извлеченный arXiv ID (или пустая строка)
        """
        try:
            # Проверяем URL
            arxiv_match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,})", url)
            if arxiv_match:
                return arxiv_match.group(1)
            
            soup = BeautifulSoup(html, "html.parser")
            
            # meta tag citation_arxiv_id
            meta_arxiv = soup.find("meta", attrs={"name": "citation_arxiv_id"})
            if meta_arxiv and meta_arxiv.get("content"):
                return meta_arxiv["content"].strip()
            
            # Поиск в тексте
            arxiv_match = re.search(r"arxiv[:\s]+(\d{4}\.\d{4,})", html, re.IGNORECASE)
            if arxiv_match:
                return arxiv_match.group(1)
                
        except Exception as e:
            logger.debug(f"Ошибка при извлечении arXiv ID: {e}")
        
        return ""

    def _find_pdf_link_on_page(self, url: str, html: str) -> str:
        """Ищет ссылку на PDF на странице.

        Args:
            url: URL страницы
            html: HTML содержимое

        Returns:
            URL PDF файла (или пустая строка)
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            # Приоритет 1: meta tag citation_pdf_url
            meta_pdf = soup.find("meta", attrs={"name": "citation_pdf_url"})
            if meta_pdf and meta_pdf.get("content"):
                return meta_pdf["content"].strip()
            
            # Приоритет 2: ссылки с .pdf в href
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if ".pdf" in href.lower() or "pdf" in href.lower():
                    # Конвертируем относительные ссылки в абсолютные
                    if href.startswith("/"):
                        from urllib.parse import urljoin
                        href = urljoin(url, href)
                    if href.startswith("http"):
                        return href
                        
        except Exception as e:
            logger.debug(f"Ошибка при поиске PDF ссылки: {e}")
        
        return ""

    def _extract_metadata(
        self, url: str, html: str, expected_title: str
    ) -> Metadata | None:
        """Извлекает метаданные со страницы.

        Args:
            url: URL страницы
            html: HTML содержимое
            expected_title: Ожидаемый заголовок для валидации

        Returns:
            Metadata объект или None если не прошло валидацию
        """
        if not html:
            return None
        
        # Извлекаем заголовок
        page_title = self._extract_page_title(html)
        if not page_title:
            logger.debug(f"Не удалось извлечь заголовок со страницы {url}")
            return None
        
        # Строгая валидация заголовка (порог 0.80 для Google результатов)
        if not strict_title_match(expected_title, page_title, threshold=0.80):
            logger.debug(
                f"Заголовок страницы не совпадает с ожидаемым: "
                f"'{page_title[:50]}...' vs '{expected_title[:50]}...'"
            )
            return None
        
        # Извлекаем метаданные
        doi = self._extract_doi_from_page(html)
        arxiv_id = self._extract_arxiv_id_from_page(url, html)
        pdf_url = self._find_pdf_link_on_page(url, html)
        
        logger.debug(
            f"Извлечены метаданные для '{page_title[:50]}...': "
            f"DOI={doi}, arXiv={arxiv_id}, PDF={bool(pdf_url)}"
        )
        
        return Metadata(
            title=page_title,
            doi=doi or "",
            arxiv_id=arxiv_id or "",
            article_url=url,
            pdf_url=pdf_url or "",
            source_name=self.name,
        )

    def search_by_title(
        self, title: str, max_results: int = 10
    ) -> list[Metadata]:
        """Ищет статью через Google.

        Args:
            title: Название статьи
            max_results: Максимальное количество результатов

        Returns:
            Список найденных метаданных
        """
        if not title:
            return []
        
        # Формируем поисковый запрос
        query = f'"{title}" (pdf OR doi OR arxiv)'
        logger.debug(f"Google поиск: {query[:80]}...")
        
        # Выполняем поиск
        urls = self._search_google(query, num_results=max_results)
        
        # Парсим каждую страницу
        results = []
        for url in urls:
            # Пропускаем некоторые нерелевантные домены
            if any(
                skip_domain in url.lower()
                for skip_domain in ["youtube.com", "twitter.com", "facebook.com"]
            ):
                continue
            
            html = self._fetch_page(url)
            if html:
                metadata = self._extract_metadata(url, html, title)
                if metadata:
                    results.append(metadata)
        
        logger.info(f"Google поиск для '{title[:50]}...' вернул {len(results)} релевантных результатов")
        return results

    def get_by_doi(self, doi: str) -> Metadata | None:
        """Не поддерживается для Google поиска."""
        return None

    def get_by_title(self, title: str) -> Metadata | None:
        """Возвращает первый результат поиска по заголовку."""
        results = self.search_by_title(title, max_results=5)
        return results[0] if results else None

