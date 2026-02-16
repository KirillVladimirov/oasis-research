#!/usr/bin/env python3
# Шаг 8: scoring CQ, grounding в Weaviate, построение онтологии. Логика в oasis.pipelines.score_ground_ontology.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oasis.pipelines.score_ground_ontology import run_all
from scripts.pipeline._config import ROOT, get_datasets, load_pipeline_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Шаг 8: scoring, grounding, онтология")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--run-id", type=str, default="run1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-scoring", action="store_true")
    parser.add_argument("--skip-grounding", action="store_true")
    parser.add_argument("--skip-ontology", action="store_true")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    config_path = args.config or (ROOT / "configs" / "pipeline.yaml")
    artifacts_root = ROOT / config.get("artifacts_root", "artifacts")
    datasets = args.datasets or get_datasets(config)
    if not datasets:
        return 0
    return run_all(
        root_dir=ROOT,
        artifacts_root=artifacts_root,
        dataset_names=datasets,
        config_path=config_path,
        run_id=args.run_id,
        workers=args.workers,
        skip_scoring=args.skip_scoring,
        skip_grounding=args.skip_grounding,
        skip_ontology=args.skip_ontology,
    )


if __name__ == "__main__":
    sys.exit(main())
