"""Кэш для валидации PDF."""

import json
from pathlib import Path
from typing import Any

# Путь к кэшу валидации PDF
_pdf_validate_cache_path = Path("data/.emb_cache/pdf_validate.json")

# Глобальный кэш соответствий PDF статьям
_pdf_match_cache: dict[str, dict[str, Any]] = {}

# Загрузка кэша из файла при импорте модуля
try:
    _pdf_validate_cache_path.parent.mkdir(parents=True, exist_ok=True)
    if _pdf_validate_cache_path.exists():
        _pdf_match_cache = json.loads(
            _pdf_validate_cache_path.read_text(encoding="utf-8")
        )
except Exception:
    _pdf_match_cache = {}


def save_pdf_validate_cache() -> None:
    """Сохранение кэша валидации PDF в файл."""
    try:
        _pdf_validate_cache_path.write_text(
            json.dumps(_pdf_match_cache, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass


def get_from_cache(cache_key: str) -> dict[str, Any] | None:
    """Получить результат валидации из кэша.

    Args:
        cache_key: Ключ кэша

    Returns:
        Результат валидации или None если нет в кэше
    """
    return _pdf_match_cache.get(cache_key)


def put_in_cache(cache_key: str, result: dict[str, Any]) -> None:
    """Сохранить результат валидации в кэш.

    Args:
        cache_key: Ключ кэша
        result: Результат валидации
    """
    _pdf_match_cache[cache_key] = result
    save_pdf_validate_cache()
