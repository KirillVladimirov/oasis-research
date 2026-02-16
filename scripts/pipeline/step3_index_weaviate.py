#!/usr/bin/env python3
"""Шаг 3: индексация датасетов в Weaviate. Логика в oasis.pipelines.index_weaviate."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oasis.pipelines.index_weaviate import run_all
from scripts.pipeline._config import ROOT, get_datasets, get_data_root, load_pipeline_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Шаг 3: индексация в Weaviate")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--weaviate-url", type=str, default=None)
    parser.add_argument("--embedding-model", type=str, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    data_root = ROOT / get_data_root(config)
    datasets = args.datasets or get_datasets(config)
    if not datasets:
        return 0
    weaviate_url = args.weaviate_url or config.get("weaviate_url", "http://localhost:8081")
    embedding_model = args.embedding_model or config.get("embedding_model", "models/bge-m3")
    chunk_size = args.chunk_size or config.get("chunk_size", 512)
    chunk_overlap = args.chunk_overlap or config.get("chunk_overlap", 50)
    return run_all(
        data_root=data_root,
        dataset_names=datasets,
        weaviate_url=weaviate_url,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


if __name__ == "__main__":
    sys.exit(main())
