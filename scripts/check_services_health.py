#!/usr/bin/env python3
"""Проверка доступности сервисов OASIS."""

import sys

from loguru import logger

from oasis.utils.network import check_http_service


def main() -> int:
    logger.info("Проверка состояния сервисов OASIS")

    services = {
        "Weaviate": "http://localhost:8081/v1/.well-known/ready",
        "Fuseki": "http://localhost:3031/",
        "GROBID": "http://localhost:8070/api/isalive",
    }

    all_healthy = True
    for name, url in services.items():
        if check_http_service(url):
            logger.info(f"{name}: доступен")
        else:
            logger.error(f"{name}: отсутствует ответ")
            all_healthy = False

    if all_healthy:
        logger.success("Все сервисы доступны")
        return 0
    else:
        logger.error("Некоторые сервисы недоступны")
        return 1


if __name__ == "__main__":
    sys.exit(main())
