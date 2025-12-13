"""Объединение метаданных из разных источников."""

from typing import Any, Callable

from oasis.parsing.normalization import normalize_authors
from oasis.utils.text import clean_text_value


def merge_info(
    base: dict[str, Any],
    src: dict[str, Any],
    validate_url_fn: Callable[[str], bool] | None = None,
    validate_pdf_url_fn: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Объединение информации из двух словарей ссылок.

    Аккуратно сливает поля; IDs/URLs в приоритете, authors не перезаписывает если заполнен.

    Args:
        base: Базовый словарь, который будет обновлён
        src: Источник данных для обновления
        validate_url_fn: Функция для валидации URL (опционально)
        validate_pdf_url_fn: Функция для валидации PDF URL (опционально)

    Returns:
        Обновлённый словарь base
    """

    def _merge_urls_and_ids():
        """Объединение ID и URL."""
        for key in ("doi", "arxiv_id", "article_url", "pdf_url"):
            v = src.get(key)
            if v and not base.get(key):
                v_clean = clean_text_value(v)
                if key == "article_url":
                    if validate_url_fn and validate_url_fn(v_clean):
                        base[key] = v_clean
                    elif not validate_url_fn:
                        base[key] = v_clean
                elif key == "pdf_url":
                    if validate_pdf_url_fn and validate_pdf_url_fn(v_clean):
                        base[key] = v_clean
                    elif not validate_pdf_url_fn:
                        base[key] = v_clean
                    else:
                        # Если это не PDF - перемещаем в дополнительные ссылки
                        if v_clean:
                            existing_add = base.get("additional_urls") or []
                            if not isinstance(existing_add, list):
                                existing_add = []
                            if v_clean not in existing_add:
                                existing_add.append(v_clean)
                            base["additional_urls"] = existing_add
                else:
                    base[key] = v_clean

    def _merge_additional_urls():
        """Объединение дополнительных ссылок."""
        add_src = src.get("additional_urls") or []
        if add_src:
            if not isinstance(add_src, list):
                add_src = [add_src]
            valid = []
            for u in add_src:
                u_str = clean_text_value(u)
                if u_str and (not validate_url_fn or validate_url_fn(u_str)):
                    valid.append(u_str)
            existing = base.get("additional_urls") or []
            if not isinstance(existing, list):
                existing = []
            # union while preserving order
            for u in valid:
                if u not in existing:
                    existing.append(u)
            if existing:
                base["additional_urls"] = existing

    def _merge_metadata():
        """Объединение метаданных (title, venue, year, authors)."""
        # title
        if not base.get("title") and src.get("title"):
            base["title"] = clean_text_value(src.get("title"))

        # venue
        if not base.get("venue") and src.get("venue"):
            base["venue"] = clean_text_value(src.get("venue"))

        # year
        if not base.get("year") and src.get("year") is not None:
            v = src.get("year")
            try:
                base["year"] = int(v)
            except Exception:
                try:
                    base["year"] = int(str(v)[:4])
                except Exception:
                    pass

        # authors — не перезаписываем, если уже есть
        src_auth = src.get("authors")
        if src_auth:
            src_auth_str = normalize_authors(src_auth)
            base_auth_str = normalize_authors(base.get("authors")) if base.get("authors") else ""
            if not base_auth_str:
                base["authors"] = src_auth_str

    _merge_urls_and_ids()
    _merge_additional_urls()
    _merge_metadata()
    return base

