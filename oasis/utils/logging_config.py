"""Глобальная конфигурация логирования для OASIS."""

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger


def setup_logging(log_dir: str | Path = "logs", level: str = "DEBUG") -> None:
    """Настраивает глобальное логирование для всего приложения.
    
    Args:
        log_dir: Директория для логов
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Удаляем дефолтный handler (stdout)
    logger.remove()
    
    # 1. Консоль: INFO и выше (для UI)
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True,
    )
    
    # 2. Файл: DEBUG и выше (все детали)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"oasis_{timestamp}.log"
    
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        level=level,
        rotation="100 MB",
        retention="30 days",
        encoding="utf-8",
        backtrace=True,  # Полный traceback для ошибок
        diagnose=True,   # Детальная диагностика
    )
    
    # 3. Файл ошибок: WARNING и выше
    error_file = log_dir / f"oasis_errors_{timestamp}.log"
    logger.add(
        error_file,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        level="WARNING",
        rotation="50 MB",
        retention="60 days",
        encoding="utf-8",
        backtrace=True,
        diagnose=True,
    )
    
    logger.info(f"Логирование настроено: {log_file}")
    logger.info(f"Логи ошибок: {error_file}")
    logger.debug(f"Уровень логирования: {level}")


def get_logger():
    """Возвращает настроенный logger (для совместимости)."""
    return logger

