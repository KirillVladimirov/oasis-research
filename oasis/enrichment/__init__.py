"""Enrichment modules for OASIS: validation, metrics, merging."""

from oasis.enrichment.dedupe import dedupe_references
from oasis.enrichment.merge import merge_info
from oasis.enrichment.metrics import calculate_reference_metrics
from oasis.enrichment.pipeline import setup_enrichment_sources
from oasis.enrichment.sources import SourceAggregator, merge_metadata
from oasis.enrichment.validate import (
    validate_url,
    validate_pdf_url,
    validate_pdf_matches_reference,
)
from oasis.enrichment.workflow import EnrichmentWorkflow

__all__ = [
    "validate_url",
    "validate_pdf_url",
    "validate_pdf_matches_reference",
    "calculate_reference_metrics",
    "SourceAggregator",
    "merge_metadata",
    "setup_enrichment_sources",
    "EnrichmentWorkflow",  # НОВОЕ: используйте вместо enrich_single_reference
    "dedupe_references",
    "merge_info",
]

