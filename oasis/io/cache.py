"""Кэширование API ответов и HTML контента."""

import json
from pathlib import Path
from typing import Any

from loguru import logger


class ApiCache:
    """Кэш для API ответов (JSON файл + память).

    Простая реализация с JSON файлом для персистентности и LRU в памяти.
    """

    def __init__(self, cache_path: str | Path = "data/.emb_cache/enrich_api_cache.json"):
        """Инициализация кэша.

        Args:
            cache_path: Путь к JSON файлу кэша
        """
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Загружает кэш с диска."""
        try:
            if self.cache_path.exists():
                self._cache_data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Не удалось загрузить кэш из {self.cache_path}: {e}")
            self._cache_data = {}

    def get(self, key: str) -> Any | None:
        """Получает значение из кэша.

        Args:
            key: Ключ кэша

        Returns:
            Значение из кэша или None
        """
        return self._cache_data.get(key)

    def set(self, key: str, value: Any) -> None:
        """Устанавливает значение в кэш.

        Args:
            key: Ключ кэша
            value: Значение для сохранения
        """
        self._cache_data[key] = value

    def save(self) -> None:
        """Сохраняет кэш на диск."""
        try:
            self.cache_path.write_text(
                json.dumps(self._cache_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"Не удалось сохранить кэш в {self.cache_path}: {e}")

    def clear(self) -> None:
        """Очищает кэш."""
        self._cache_data = {}

    def __len__(self) -> int:
        """Возвращает количество элементов в кэше."""
        return len(self._cache_data)


# Глобальный экземпляр кэша для обратной совместимости
_global_cache: ApiCache | None = None


def get_cache(cache_path: str | Path | None = None) -> ApiCache:
    """Получает глобальный экземпляр кэша.

    Args:
        cache_path: Путь к файлу кэша (используется только при первом вызове)

    Returns:
        Экземпляр ApiCache
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = ApiCache(cache_path or "data/.emb_cache/enrich_api_cache.json")
    return _global_cache


def set_cache(cache_instance: ApiCache) -> None:
    """Устанавливает глобальный экземпляр кэша.

    Args:
        cache_instance: Экземпляр ApiCache для использования
    """
    global _global_cache
    _global_cache = cache_instance

