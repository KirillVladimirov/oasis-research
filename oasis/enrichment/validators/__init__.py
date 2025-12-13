"""Валидаторы для обогащения ссылок."""

from oasis.enrichment.validators.cache import (
    get_from_cache,
    put_in_cache,
    save_pdf_validate_cache,
)
from oasis.enrichment.validators.pdf_validator import validate_pdf_matches_reference
from oasis.enrichment.validators.title_matcher import normalize_title_for_match
from oasis.enrichment.validators.url_validator import (
    is_trusted_source,
    validate_pdf_url,
    validate_url,
)

__all__ = [
    "normalize_title_for_match",
    "validate_url",
    "validate_pdf_url",
    "is_trusted_source",
    "validate_pdf_matches_reference",
    "get_from_cache",
    "put_in_cache",
    "save_pdf_validate_cache",
]
