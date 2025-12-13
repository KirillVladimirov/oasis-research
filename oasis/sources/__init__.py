"""Sources модули для OASIS: адаптеры внешних источников данных."""

from oasis.sources.arxiv import ArxivSource
from oasis.sources.base import Metadata, PdfCandidate, SourcePort
from oasis.sources.crossref import CrossrefSource
from oasis.sources.openalex import OpenAlexSource
from oasis.sources.semanticscholar import SemanticScholarSource
from oasis.sources.unpaywall import UnpaywallSource

__all__ = [
    "SourcePort",
    "Metadata",
    "PdfCandidate",
    "OpenAlexSource",
    "CrossrefSource",
    "SemanticScholarSource",
    "ArxivSource",
    "UnpaywallSource",
]

