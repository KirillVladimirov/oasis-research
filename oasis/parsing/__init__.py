"""Parsing модули для OASIS: извлечение ссылок из PDF, нормализация, guardrails."""

from oasis.parsing.guardrails import filter_phantom_references, validate_ref_number
from oasis.parsing.normalization import normalize_authors, normalize_reference
from oasis.parsing.pdf_refs import (
    GrobidClient,
    extract_references_from_pdf_text,
    parse_bibl_struct,
    parse_grobid_fulltext,
    parse_grobid_references,
)
from oasis.parsing.regexes import find_arxiv_id, find_doi, find_urls, parse_references_regex

__all__ = [
    "parse_grobid_references",
    "parse_grobid_fulltext",
    "parse_bibl_struct",
    "extract_references_from_pdf_text",
    "GrobidClient",
    "parse_references_regex",
    "find_doi",
    "find_arxiv_id",
    "find_urls",
    "normalize_reference",
    "normalize_authors",
    "filter_phantom_references",
    "validate_ref_number",
]

