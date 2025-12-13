"""Network utilities for OASIS."""

import requests
from loguru import logger


def check_http_service(url: str, timeout: int = 10) -> bool:
    """Check if an HTTP service is healthy by making a GET request.

    Args:
        url: Service URL to check
        timeout: Request timeout in seconds

    Returns:
        True if service responds with 2xx status code, False otherwise
    """
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code in [200, 201, 202]
    except requests.exceptions.RequestException as e:
        logger.debug(f"HTTP check failed for {url}: {e}")
        return False
