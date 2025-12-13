"""GROBID API клиент."""

from typing import Any

import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from oasis.parsing.grobid.parser import parse_grobid_response


class GrobidClient:
    """Клиент для работы с GROBID API."""

    def __init__(self, grobid_url: str = "http://localhost:8070"):
        """Инициализация клиента.

        Args:
            grobid_url: URL GROBID сервиса
        """
        self.grobid_url = grobid_url

    def is_alive(self, timeout: int = 5) -> bool:
        """Проверяет доступность GROBID сервиса.

        Args:
            timeout: Таймаут проверки

        Returns:
            True если сервис доступен
        """
        try:
            response = requests.get(f"{self.grobid_url}/api/isalive", timeout=timeout)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def process_references(self, pdf_path: str, timeout: int = 120) -> list[dict[str, Any]]:
        """Извлечь ссылки из PDF через GROBID processReferences.

        Args:
            pdf_path: Путь к PDF файлу
            timeout: Таймаут запроса

        Returns:
            Список словарей с полями ссылок
        """
        with open(pdf_path, "rb") as f:
            response = requests.post(
                f"{self.grobid_url}/api/processReferences",
                files={"input": f},
                timeout=timeout,
            )
        response.raise_for_status()
        return parse_grobid_response(response.text)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def process_fulltext(self, pdf_path: str, timeout: int = 300) -> list[dict[str, Any]]:
        """Извлечь все ссылки из PDF через GROBID processFulltextDocument.

        Args:
            pdf_path: Путь к PDF файлу
            timeout: Таймаут запроса

        Returns:
            Список словарей с полями ссылок
        """
        with open(pdf_path, "rb") as f:
            response = requests.post(
                f"{self.grobid_url}/api/processFulltextDocument",
                files={"input": f},
                timeout=timeout,
            )
        response.raise_for_status()
        return parse_grobid_response(response.text)

