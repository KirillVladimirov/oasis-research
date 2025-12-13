"""Обертка для поиска на ResearchGate через веб-скрапинг."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
from loguru import logger

from oasis.sources.base import Metadata, PdfCandidate, SourcePort

if TYPE_CHECKING:
    from oasis.models.reference import Reference


class ResearchGateWrapper(SourcePort):
    """Обертка для поиска публикаций на ResearchGate."""

    def __init__(self, cache=None, delay_seconds: float = 1.0):
        """Инициализация обертки.

        Args:
            cache: Кэш для API ответов (пока не используется)
            delay_seconds: Задержка между запросами для избежания блокировки
        """
        self.cache = cache
        self.delay_seconds = delay_seconds
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    @property
    def name(self) -> str:
        """Имя источника."""
        return "researchgate_wrapper"

    def _search_publications(self, query: str, max_results: int = 5) -> list[dict]:
        """Поиск публикаций на ResearchGate.

        Args:
            query: Поисковый запрос
            max_results: Максимальное количество результатов

        Returns:
            Список словарей с метаданными публикаций
        """
        try:
            # Кодируем запрос для URL
            encoded_query = quote(query)

            # URL для поиска публикаций
            search_url = f"https://www.researchgate.net/search/publication?q={encoded_query}"

            logger.debug(f"ResearchGate поиск: {search_url}")

            # Делаем запрос с задержкой
            time.sleep(self.delay_seconds)
            response = self.session.get(search_url, timeout=30)

            if response.status_code != 200:
                logger.warning(f"ResearchGate вернул статус {response.status_code}")
                return []

            soup = BeautifulSoup(response.content, 'html.parser')

            results = []

            # Ищем публикации в результатах поиска
            # ResearchGate имеет динамическую загрузку, поэтому ищем статические элементы
            publication_items = soup.find_all('div', class_='nova-legacy-v-publication-item__stack')

            if not publication_items:
                # Альтернативный селектор
                publication_items = soup.find_all('div', {'data-testid': 'publication-item'})

            logger.debug(f"Найдено элементов публикаций: {len(publication_items)}")

            for item in publication_items[:max_results]:
                try:
                    publication_data = self._extract_publication_data(item)
                    if publication_data:
                        results.append(publication_data)
                except Exception as e:
                    logger.debug(f"Ошибка при извлечении данных публикации: {e}")
                    continue

            return results

        except Exception as e:
            logger.warning(f"Ошибка при поиске на ResearchGate: {e}")
            return []

    def _extract_publication_data(self, item) -> dict | None:
        """Извлекает данные публикации из HTML элемента.

        Args:
            item: BeautifulSoup элемент публикации

        Returns:
            Словарь с метаданными или None
        """
        try:
            # Ищем заголовок
            title_elem = item.find('h5') or item.find('h4') or item.find('a', class_='nova-legacy-v-publication-item__title')
            if not title_elem:
                return None

            title = title_elem.get_text(strip=True)
            if not title:
                return None

            # Ищем ссылку на публикацию
            link_elem = title_elem.find_parent('a') if title_elem.name != 'a' else title_elem
            article_url = link_elem.get('href') if link_elem else None
            if article_url and not article_url.startswith('http'):
                article_url = f"https://www.researchgate.net{article_url}"

            # Ищем авторов
            authors_elem = item.find('div', class_='nova-legacy-v-publication-item__authors')
            authors = ""
            if authors_elem:
                author_links = authors_elem.find_all('a')
                authors = "; ".join([a.get_text(strip=True) for a in author_links[:5]])

            # Ищем год
            year = None
            year_elem = item.find('span', class_='nova-legacy-v-publication-item__meta-data-item')
            if year_elem:
                year_text = year_elem.get_text(strip=True)
                # Ищем год в формате 4 цифр
                import re
                year_match = re.search(r'\b(20\d{2}|19\d{2})\b', year_text)
                if year_match:
                    year = int(year_match.group(1))

            # Ищем DOI
            doi = None
            doi_elem = item.find('span', string=lambda text: text and 'DOI:' in text)
            if doi_elem:
                doi_text = doi_elem.get_text(strip=True)
                doi_match = re.search(r'DOI:\s*([^\s]+)', doi_text)
                if doi_match:
                    doi = doi_match.group(1)

            return {
                'title': title,
                'authors': authors,
                'year': year,
                'doi': doi,
                'article_url': article_url,
                'source': 'researchgate',
            }

        except Exception as e:
            logger.debug(f"Ошибка при извлечении данных публикации: {e}")
            return None

    def get_by_doi(self, doi: str) -> Metadata | None:
        """ResearchGate не поддерживает прямой поиск по DOI."""
        return None

    def search_by_title(self, title: str, max_results: int = 5) -> list[Metadata]:
        """Ищет публикации по названию на ResearchGate.

        Args:
            title: Название публикации
            max_results: Максимальное количество результатов

        Returns:
            Список Metadata объектов
        """
        if not title:
            return []

        logger.debug(f"Поиск на ResearchGate: '{title}'")

        # Ищем публикации
        publications = self._search_publications(title, max_results)

        results = []
        for pub in publications:
            # Проверяем релевантность по названию
            pub_title = pub.get('title', '').lower()
            search_title = title.lower()

            # Простая проверка схожести (можно улучшить)
            if any(word in pub_title for word in search_title.split()[:3]):
                metadata = Metadata(
                    title=pub.get('title', ''),
                    authors=pub.get('authors', ''),
                    year=pub.get('year'),
                    doi=pub.get('doi'),
                    article_url=pub.get('article_url'),
                    source_name=self.name,
                    raw_data=pub,
                )
                results.append(metadata)

        logger.debug(f"Найдено на ResearchGate: {len(results)} результатов")
        return results

    def get_pdf_candidates(self, ref: dict[str, Any] | Reference) -> list[PdfCandidate]:
        """ResearchGate обычно не предоставляет прямые PDF ссылки."""
        return []


class IEEEWrapper(SourcePort):
    """Обертка для поиска публикаций на IEEE Xplore."""

    def __init__(self, cache=None, delay_seconds: float = 1.0):
        """Инициализация обертки.

        Args:
            cache: Кэш для API ответов (пока не используется)
            delay_seconds: Задержка между запросами
        """
        self.cache = cache
        self.delay_seconds = delay_seconds
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    @property
    def name(self) -> str:
        """Имя источника."""
        return "ieee_wrapper"

    def _search_publications(self, query: str, max_results: int = 5) -> list[dict]:
        """Поиск публикаций на IEEE Xplore.

        Args:
            query: Поисковый запрос
            max_results: Максимальное количество результатов

        Returns:
            Список словарей с метаданными публикаций
        """
        try:
            # URL для поиска
            search_url = f"https://ieeexplore.ieee.org/search/searchresult.jsp?queryText={quote(query)}"

            logger.debug(f"IEEE Xplore поиск: {search_url}")

            # Делаем запрос с задержкой
            time.sleep(self.delay_seconds)
            response = self.session.get(search_url, timeout=30)

            if response.status_code != 200:
                logger.warning(f"IEEE Xplore вернул статус {response.status_code}")
                return []

            soup = BeautifulSoup(response.content, 'html.parser')

            results = []

            # Ищем результаты поиска
            publication_items = soup.find_all('div', class_='result-item')

            if not publication_items:
                # Альтернативный селектор
                publication_items = soup.find_all('div', class_='List-results-items')

            logger.debug(f"Найдено элементов публикаций на IEEE: {len(publication_items)}")

            for item in publication_items[:max_results]:
                try:
                    publication_data = self._extract_ieee_publication_data(item)
                    if publication_data:
                        results.append(publication_data)
                except Exception as e:
                    logger.debug(f"Ошибка при извлечении IEEE данных: {e}")
                    continue

            return results

        except Exception as e:
            logger.warning(f"Ошибка при поиске на IEEE Xplore: {e}")
            return []

    def _extract_ieee_publication_data(self, item) -> dict | None:
        """Извлекает данные публикации из IEEE Xplore.

        Args:
            item: BeautifulSoup элемент публикации

        Returns:
            Словарь с метаданными или None
        """
        try:
            # Ищем заголовок
            title_elem = item.find('h2') or item.find('h3') or item.find('a', class_='title')
            if not title_elem:
                return None

            title = title_elem.get_text(strip=True)
            if not title:
                return None

            # Ищем ссылку
            link_elem = title_elem.find_parent('a') if title_elem.name != 'a' else title_elem
            article_url = link_elem.get('href') if link_elem else None
            if article_url and not article_url.startswith('http'):
                article_url = f"https://ieeexplore.ieee.org{article_url}"

            # Ищем авторов
            authors_elem = item.find('div', class_='authors') or item.find('p', class_='author')
            authors = ""
            if authors_elem:
                author_links = authors_elem.find_all('a')
                authors = "; ".join([a.get_text(strip=True) for a in author_links[:5]])

            # Ищем год и DOI
            year = None
            doi = None

            meta_elem = item.find('div', class_='publisher-info') or item.find('div', class_='pub')
            if meta_elem:
                meta_text = meta_elem.get_text(strip=True)

                # Ищем год
                import re
                year_match = re.search(r'\b(20\d{2}|19\d{2})\b', meta_text)
                if year_match:
                    year = int(year_match.group(1))

                # Ищем DOI
                doi_match = re.search(r'DOI:\s*([^\s]+)', meta_text)
                if doi_match:
                    doi = doi_match.group(1)

            return {
                'title': title,
                'authors': authors,
                'year': year,
                'doi': doi,
                'article_url': article_url,
                'source': 'ieee',
            }

        except Exception as e:
            logger.debug(f"Ошибка при извлечении IEEE данных: {e}")
            return None

    def get_by_doi(self, doi: str) -> Metadata | None:
        """IEEE Xplore не поддерживает прямой поиск по DOI."""
        return None

    def search_by_title(self, title: str, max_results: int = 5) -> list[Metadata]:
        """Ищет публикации по названию на IEEE Xplore.

        Args:
            title: Название публикации
            max_results: Максимальное количество результатов

        Returns:
            Список Metadata объектов
        """
        if not title:
            return []

        logger.debug(f"Поиск на IEEE Xplore: '{title}'")

        # Ищем публикации
        publications = self._search_publications(title, max_results)

        results = []
        for pub in publications:
            # Проверяем релевантность
            pub_title = pub.get('title', '').lower()
            search_title = title.lower()

            if any(word in pub_title for word in search_title.split()[:3]):
                metadata = Metadata(
                    title=pub.get('title', ''),
                    authors=pub.get('authors', ''),
                    year=pub.get('year'),
                    doi=pub.get('doi'),
                    article_url=pub.get('article_url'),
                    source_name=self.name,
                    raw_data=pub,
                )
                results.append(metadata)

        logger.debug(f"Найдено на IEEE Xplore: {len(results)} результатов")
        return results

    def get_pdf_candidates(self, ref: dict[str, Any] | Reference) -> list[PdfCandidate]:
        """IEEE Xplore может предоставлять PDF для платных пользователей."""
        return []

