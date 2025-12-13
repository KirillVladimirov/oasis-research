"""Модули для работы с источниками данных в процессе обогащения."""

from oasis.enrichment.sources.aggregator import SourceAggregator
from oasis.enrichment.sources.merger import merge_metadata

__all__ = [
    "SourceAggregator",
    "merge_metadata",
]

