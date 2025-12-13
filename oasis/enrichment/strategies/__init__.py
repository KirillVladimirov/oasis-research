"""Стратегии обогащения ссылок."""

from oasis.enrichment.strategies.doi_search import search_by_doi
from oasis.enrichment.strategies.unpaywall_search import search_unpaywall_pdf
from oasis.enrichment.strategies.cascade_strategy import execute_cascade_search
from oasis.enrichment.strategies.title_search import search_by_title_fallback
from oasis.enrichment.strategies.pdf_search import search_pdf_candidates
from oasis.enrichment.strategies.pdf_validation import validate_pdf_candidate

__all__ = [
    "search_by_doi",
    "search_unpaywall_pdf",
    "execute_cascade_search",
    "search_by_title_fallback",
    "search_pdf_candidates",
    "validate_pdf_candidate",
]

