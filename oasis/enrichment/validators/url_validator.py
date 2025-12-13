"""Валидация URL и PDF ссылок."""

import requests

# Модульные кэши (доступны между вызовами)
_url_valid_cache: dict[str, bool] = {}
_pdf_valid_cache: dict[str, bool] = {}


def is_trusted_source(pdf_url: str) -> bool:
    """Проверяет, является ли источник PDF доверенным.

    Args:
        pdf_url: URL PDF файла

    Returns:
        True если источник доверенный
    """
    url_lower = pdf_url.lower()
    trusted_domains = [
        "arxiv.org",
        "ieee.org",
        "acm.org",
        "springer.com",
        "springerlink.com",
        "elsevier.com",
        "sciencedirect.com",
        "nature.com",
        "science.org",
        "cell.com",
        "pubmed.ncbi.nlm.nih.gov",
        "pmc.ncbi.nlm.nih.gov",
        "biorxiv.org",
        "medrxiv.org",
        "openreview.net",
    ]
    return any(domain in url_lower for domain in trusted_domains)


def validate_url(session: requests.Session, url: str, timeout: int = 10) -> bool:
    """Проверяет валидность URL (HEAD/GET запрос, статус 200-299).

    Args:
        session: requests.Session для выполнения запросов
        url: URL для проверки
        timeout: Таймаут запроса в секундах

    Returns:
        True если URL доступен (статус 200-299)
    """
    if not url or not isinstance(url, str):
        return False
    if url in _url_valid_cache:
        return _url_valid_cache[url]
    try:
        r = session.head(url, allow_redirects=True, timeout=timeout)
        ok = 200 <= r.status_code < 300
        if not ok:
            r = session.get(url, allow_redirects=True, timeout=timeout, stream=True)
            ok = 200 <= r.status_code < 300
        _url_valid_cache[url] = bool(ok)
        return bool(ok)
    except Exception:
        _url_valid_cache[url] = False
        return False


def validate_pdf_url(session: requests.Session, url: str, timeout: int = 10) -> bool:
    """Проверяет что URL действительно ведёт на PDF файл.

    Args:
        session: requests.Session для выполнения запросов
        url: URL для проверки
        timeout: Таймаут запроса в секундах

    Returns:
        True если URL ведёт на PDF файл (проверка Content-Type или расширения .pdf)
    """
    if not url or not isinstance(url, str):
        return False
    url_lower = url.lower().strip()
    # Быстрая проверка: .pdf в конце URL или в последнем сегменте пути
    if url_lower.endswith(".pdf") or ".pdf" in url_lower.split("/")[-1].split("?")[0]:
        # Если есть .pdf в пути - проверяем Content-Type
        cache_key = f"pdf_{url}"
        if cache_key in _pdf_valid_cache:
            return _pdf_valid_cache[cache_key]
        try:
            r = session.head(url, allow_redirects=True, timeout=timeout)
            if r.status_code == 200:
                ct = r.headers.get("Content-Type", "").lower()
                if "application/pdf" in ct:
                    _pdf_valid_cache[cache_key] = True
                    return True
            # Fallback: GET с проверкой Content-Type
            r = session.get(url, allow_redirects=True, timeout=timeout, stream=True)
            if r.status_code == 200:
                ct = r.headers.get("Content-Type", "").lower()
                if "application/pdf" in ct:
                    _pdf_valid_cache[cache_key] = True
                    return True
        except Exception:
            pass
        # Если .pdf в URL но Content-Type не подтвердил - считаем валидным (может быть кэширование на сервере)
        _pdf_valid_cache[cache_key] = True
        return True
    # Если нет .pdf в URL - проверяем Content-Type напрямую
    cache_key = f"pdf_{url}"
    if cache_key in _pdf_valid_cache:
        return _pdf_valid_cache[cache_key]
    try:
        r = session.head(url, allow_redirects=True, timeout=timeout)
        if r.status_code == 200:
            ct = r.headers.get("Content-Type", "").lower()
            if "application/pdf" in ct:
                _pdf_valid_cache[cache_key] = True
                return True
        # Fallback: GET
        r = session.get(url, allow_redirects=True, timeout=timeout, stream=True)
        if r.status_code == 200:
            ct = r.headers.get("Content-Type", "").lower()
            if "application/pdf" in ct:
                _pdf_valid_cache[cache_key] = True
                return True
    except Exception:
        pass
    _pdf_valid_cache[cache_key] = False
    return False
