"""I/O модули для OASIS: загрузка, сохранение, кэширование."""

from oasis.io.cache import ApiCache, get_cache, set_cache
from oasis.io.loaders import load_references, load_references_raw, load_review_pdf
from oasis.io.registry import SourceRegistry, get_registry
from oasis.io.writers import export_metrics_report, save_references, save_references_raw

__all__ = [
    "load_review_pdf",
    "load_references_raw",
    "load_references",
    "save_references_raw",
    "save_references",
    "export_metrics_report",
    "ApiCache",
    "get_cache",
    "set_cache",
    "SourceRegistry",
    "get_registry",
]

