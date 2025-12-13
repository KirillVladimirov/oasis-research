"""Реестр источников данных с возможностью включения/выключения."""

from typing import Any

from oasis.sources.base import SourcePort


class SourceRegistry:
    """Реестр источников данных."""

    def __init__(self):
        """Инициализация реестра."""
        self._sources: dict[str, SourcePort] = {}
        self._enabled: dict[str, bool] = {}

    def register(self, name: str, source: SourcePort, enabled: bool = True) -> None:
        """Регистрирует источник.

        Args:
            name: Имя источника
            source: Экземпляр SourcePort
            enabled: Включен ли источник по умолчанию
        """
        self._sources[name] = source
        self._enabled[name] = enabled

    def get(self, name: str) -> SourcePort | None:
        """Получает источник по имени.

        Args:
            name: Имя источника

        Returns:
            SourcePort или None если не найден
        """
        return self._sources.get(name)

    def is_enabled(self, name: str) -> bool:
        """Проверяет включен ли источник.

        Args:
            name: Имя источника

        Returns:
            True если источник включен
        """
        return self._enabled.get(name, False)

    def enable(self, name: str) -> None:
        """Включает источник.

        Args:
            name: Имя источника
        """
        if name in self._sources:
            self._enabled[name] = True

    def disable(self, name: str) -> None:
        """Выключает источник.

        Args:
            name: Имя источника
        """
        if name in self._sources:
            self._enabled[name] = False

    def get_enabled_sources(self) -> list[SourcePort]:
        """Возвращает список включенных источников.

        Returns:
            Список SourcePort объектов
        """
        return [src for name, src in self._sources.items() if self._enabled.get(name, False)]

    def list_all(self) -> list[str]:
        """Возвращает список всех зарегистрированных источников.

        Returns:
            Список имен источников
        """
        return list(self._sources.keys())


# Глобальный реестр
_global_registry: SourceRegistry | None = None


def get_registry() -> SourceRegistry:
    """Получает глобальный реестр источников.

    Returns:
        Экземпляр SourceRegistry
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = SourceRegistry()
    return _global_registry

