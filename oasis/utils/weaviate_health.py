# Проверка доступности Weaviate.

from __future__ import annotations

from urllib import error, request


def check_weaviate_health(weaviate_url: str, timeout: int = 10) -> bool:
    try:
        health_url = weaviate_url.rstrip("/") + "/v1/.well-known/ready"
        req = request.Request(health_url, method="GET")
        with request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (error.HTTPError, error.URLError):
        return False
