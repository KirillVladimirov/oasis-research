"""Pipelines для OASIS: стадии обработки ссылок."""

from oasis.pipelines.stage1_extract import extract_reference_titles
from oasis.pipelines.stage1_extract_topics import extract_topics_from_survey_papers
from oasis.pipelines.stage1_cluster_topics import cluster_topics_from_raw
from oasis.pipelines.stage1_canonical_topics import generate_canonical_topics
from oasis.pipelines.stage2_enrich import enrich_references

__all__ = [
    "extract_reference_titles",
    "extract_topics_from_survey_papers",
    "cluster_topics_from_raw",
    "generate_canonical_topics",
    "enrich_references",
]

