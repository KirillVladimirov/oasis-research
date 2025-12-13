"""Функции валидации URL, PDF и соответствия PDF статьям (обёртка для обратной совместимости)."""

from oasis.enrichment.validators import (
    get_from_cache,
    is_trusted_source,
    normalize_title_for_match,
    put_in_cache,
    save_pdf_validate_cache,
    validate_pdf_matches_reference,
    validate_pdf_url,
    validate_url,
)

# Реэкспорт для обратной совместимости
__all__ = [
    "normalize_title_for_match",
    "is_trusted_source",
    "validate_url",
    "validate_pdf_url",
    "validate_pdf_matches_reference",
    "get_from_cache",
    "put_in_cache",
    "save_pdf_validate_cache",
]
