"""Парсеры для академических сайтов."""

import re
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from loguru import logger


class BaseAcademicParser:
    """Базовый класс для парсеров академических сайтов."""

    def extract(self, url: str, html: str) -> dict[str, Any]:
        """Извлекает метаданные со страницы.

        Args:
            url: URL страницы
            html: HTML содержимое

        Returns:
            Словарь с метаданными: title, doi, pdf_url, authors
        """
        raise NotImplementedError


class IEEEParser(BaseAcademicParser):
    """Парсер для IEEE Xplore."""

    def extract(self, url: str, html: str) -> dict[str, Any]:
        """Извлекает метаданные с IEEE Xplore."""
        soup = BeautifulSoup(html, "html.parser")
        metadata = {}

        # Заголовок
        title_meta = soup.find("meta", attrs={"property": "og:title"})
        if title_meta:
            metadata["title"] = title_meta.get("content", "").strip()
        else:
            h1 = soup.find("h1", class_=re.compile("document-title"))
            if h1:
                metadata["title"] = h1.get_text().strip()

        # DOI
        doi_meta = soup.find("meta", attrs={"name": "citation_doi"})
        if doi_meta:
            metadata["doi"] = doi_meta.get("content", "").strip()

        # PDF URL - IEEE требует аутентификацию, но можем найти ссылку
        pdf_meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
        if pdf_meta:
            metadata["pdf_url"] = pdf_meta.get("content", "").strip()

        # Авторы
        author_metas = soup.find_all("meta", attrs={"name": "citation_author"})
        if author_metas:
            authors = [meta.get("content", "").strip() for meta in author_metas]
            metadata["authors"] = "; ".join(authors)

        logger.debug(f"IEEE Parser извлек: {metadata.get('title', '')[:50]}...")
        return metadata


class ACMParser(BaseAcademicParser):
    """Парсер для ACM Digital Library."""

    def extract(self, url: str, html: str) -> dict[str, Any]:
        """Извлекает метаданные с ACM DL."""
        soup = BeautifulSoup(html, "html.parser")
        metadata = {}

        # Заголовок
        title_meta = soup.find("meta", attrs={"name": "citation_title"})
        if title_meta:
            metadata["title"] = title_meta.get("content", "").strip()

        # DOI
        doi_meta = soup.find("meta", attrs={"name": "citation_doi"})
        if doi_meta:
            metadata["doi"] = doi_meta.get("content", "").strip()

        # PDF URL
        pdf_meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
        if pdf_meta:
            metadata["pdf_url"] = pdf_meta.get("content", "").strip()

        # Авторы
        author_metas = soup.find_all("meta", attrs={"name": "citation_author"})
        if author_metas:
            authors = [meta.get("content", "").strip() for meta in author_metas]
            metadata["authors"] = "; ".join(authors)

        logger.debug(f"ACM Parser извлек: {metadata.get('title', '')[:50]}...")
        return metadata


class SpringerParser(BaseAcademicParser):
    """Парсер для Springer Link."""

    def extract(self, url: str, html: str) -> dict[str, Any]:
        """Извлекает метаданные со Springer Link."""
        soup = BeautifulSoup(html, "html.parser")
        metadata = {}

        # Заголовок
        title_meta = soup.find("meta", attrs={"name": "citation_title"})
        if title_meta:
            metadata["title"] = title_meta.get("content", "").strip()
        else:
            h1 = soup.find("h1", class_=re.compile("c-article"))
            if h1:
                metadata["title"] = h1.get_text().strip()

        # DOI
        doi_meta = soup.find("meta", attrs={"name": "citation_doi"})
        if doi_meta:
            metadata["doi"] = doi_meta.get("content", "").strip()

        # PDF URL
        pdf_meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
        if pdf_meta:
            metadata["pdf_url"] = pdf_meta.get("content", "").strip()

        # Авторы
        author_metas = soup.find_all("meta", attrs={"name": "citation_author"})
        if author_metas:
            authors = [meta.get("content", "").strip() for meta in author_metas]
            metadata["authors"] = "; ".join(authors)

        logger.debug(f"Springer Parser извлек: {metadata.get('title', '')[:50]}...")
        return metadata


class OpenReviewParser(BaseAcademicParser):
    """Парсер для OpenReview."""

    def extract(self, url: str, html: str) -> dict[str, Any]:
        """Извлекает метаданные с OpenReview."""
        soup = BeautifulSoup(html, "html.parser")
        metadata = {}

        # Заголовок
        title_meta = soup.find("meta", attrs={"property": "og:title"})
        if title_meta:
            metadata["title"] = title_meta.get("content", "").strip()
        else:
            h2 = soup.find("h2", class_=re.compile("citation_title"))
            if h2:
                metadata["title"] = h2.get_text().strip()

        # PDF URL - OpenReview обычно имеет прямые ссылки на PDF
        pdf_link = soup.find("a", class_=re.compile("pdf-link"))
        if pdf_link and pdf_link.get("href"):
            pdf_url = pdf_link["href"]
            if pdf_url.startswith("/"):
                pdf_url = urljoin("https://openreview.net", pdf_url)
            metadata["pdf_url"] = pdf_url

        logger.debug(f"OpenReview Parser извлек: {metadata.get('title', '')[:50]}...")
        return metadata


class ECVAParser(BaseAcademicParser):
    """Парсер для ECVA (европейская конференция по компьютерному зрению)."""

    def extract(self, url: str, html: str) -> dict[str, Any]:
        """Извлекает метаданные с ECVA."""
        soup = BeautifulSoup(html, "html.parser")
        metadata = {}

        # Заголовок
        title_meta = soup.find("meta", attrs={"name": "citation_title"})
        if title_meta:
            metadata["title"] = title_meta.get("content", "").strip()

        # PDF URL - ECVA обычно имеет прямые ссылки
        pdf_meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
        if pdf_meta:
            metadata["pdf_url"] = pdf_meta.get("content", "").strip()
        else:
            # Ищем ссылку на PDF
            pdf_link = soup.find("a", href=re.compile(r"\.pdf$", re.IGNORECASE))
            if pdf_link:
                pdf_url = pdf_link["href"]
                if pdf_url.startswith("/"):
                    pdf_url = urljoin(url, pdf_url)
                metadata["pdf_url"] = pdf_url

        # Авторы
        author_metas = soup.find_all("meta", attrs={"name": "citation_author"})
        if author_metas:
            authors = [meta.get("content", "").strip() for meta in author_metas]
            metadata["authors"] = "; ".join(authors)

        logger.debug(f"ECVA Parser извлек: {metadata.get('title', '')[:50]}...")
        return metadata


class CVFParser(BaseAcademicParser):
    """Парсер для CVF (Computer Vision Foundation) Open Access."""

    def extract(self, url: str, html: str) -> dict[str, Any]:
        """Извлекает метаданные с CVF Open Access."""
        soup = BeautifulSoup(html, "html.parser")
        metadata = {}

        # Заголовок
        title_tag = soup.find("div", id="papertitle")
        if title_tag:
            metadata["title"] = title_tag.get_text().strip()
        else:
            title_meta = soup.find("meta", attrs={"name": "citation_title"})
            if title_meta:
                metadata["title"] = title_meta.get("content", "").strip()

        # PDF URL - CVF всегда имеет открытые PDF
        pdf_meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
        if pdf_meta:
            metadata["pdf_url"] = pdf_meta.get("content", "").strip()
        else:
            # Ищем ссылку на PDF в контенте
            for link in soup.find_all("a", href=True):
                if "_paper.pdf" in link["href"].lower():
                    pdf_url = link["href"]
                    if pdf_url.startswith("/"):
                        pdf_url = urljoin("https://openaccess.thecvf.com", pdf_url)
                    metadata["pdf_url"] = pdf_url
                    break

        # Авторы
        authors_tag = soup.find("div", id="authors")
        if authors_tag:
            authors_text = authors_tag.get_text().strip()
            # Разделяем по запятым
            authors = [a.strip() for a in authors_text.split(",")]
            metadata["authors"] = "; ".join(authors)

        logger.debug(f"CVF Parser извлек: {metadata.get('title', '')[:50]}...")
        return metadata


class GenericParser(BaseAcademicParser):
    """Универсальный парсер для неизвестных сайтов."""

    def extract(self, url: str, html: str) -> dict[str, Any]:
        """Извлекает метаданные с помощью общих методов."""
        soup = BeautifulSoup(html, "html.parser")
        metadata = {}

        # Заголовок - пробуем разные meta tags
        for meta_name in ["citation_title", "og:title", "twitter:title"]:
            title_meta = soup.find("meta", attrs={"name": meta_name}) or soup.find(
                "meta", attrs={"property": meta_name}
            )
            if title_meta and title_meta.get("content"):
                metadata["title"] = title_meta["content"].strip()
                break

        # Если не нашли в meta, берем из title tag
        if not metadata.get("title"):
            title_tag = soup.find("title")
            if title_tag:
                metadata["title"] = title_tag.get_text().strip()

        # DOI
        doi_meta = soup.find("meta", attrs={"name": "citation_doi"})
        if doi_meta:
            metadata["doi"] = doi_meta.get("content", "").strip()
        else:
            # Ищем в тексте
            doi_match = re.search(r"10\.\d{4,}/[^\s\"<>]+", html)
            if doi_match:
                metadata["doi"] = doi_match.group(0)

        # PDF URL
        pdf_meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
        if pdf_meta:
            metadata["pdf_url"] = pdf_meta.get("content", "").strip()
        else:
            # Ищем ссылки с .pdf
            for link in soup.find_all("a", href=True):
                if ".pdf" in link["href"].lower():
                    pdf_url = link["href"]
                    if pdf_url.startswith("/"):
                        pdf_url = urljoin(url, pdf_url)
                    if pdf_url.startswith("http"):
                        metadata["pdf_url"] = pdf_url
                        break

        # Авторы
        author_metas = soup.find_all("meta", attrs={"name": "citation_author"})
        if author_metas:
            authors = [meta.get("content", "").strip() for meta in author_metas]
            metadata["authors"] = "; ".join(authors)

        logger.debug(f"Generic Parser извлек: {metadata.get('title', '')[:50]}...")
        return metadata


class AcademicPageParser:
    """Парсит академические сайты для извлечения метаданных."""

    PARSERS = {
        "ieeexplore.ieee.org": IEEEParser,
        "dl.acm.org": ACMParser,
        "link.springer.com": SpringerParser,
        "openreview.net": OpenReviewParser,
        "ecva.net": ECVAParser,
        "openaccess.thecvf.com": CVFParser,
    }

    def parse(self, url: str, html: str) -> dict[str, Any]:
        """Парсит страницу в зависимости от домена.

        Args:
            url: URL страницы
            html: HTML содержимое

        Returns:
            Словарь с метаданными
        """
        # Выбираем парсер по домену
        for domain, parser_class in self.PARSERS.items():
            if domain in url:
                parser = parser_class()
                return parser.extract(url, html)

        # Если не нашли специализированный парсер, используем универсальный
        parser = GenericParser()
        return parser.extract(url, html)

