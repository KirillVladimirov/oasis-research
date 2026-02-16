#!/usr/bin/env python3
# Шаг 9: оценка метрик по cqs_grounded и ontology. Логика в oasis.pipelines.eval_run.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oasis.pipelines.eval_run import run_all
from scripts.pipeline._config import ROOT, get_datasets, load_pipeline_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Шаг 9: оценка метрик")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--eval-config", type=Path, default=None)
    parser.add_argument("--reasoner", type=str, default="pellet")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    artifacts_root = ROOT / config.get("artifacts_root", "artifacts")
    datasets = args.datasets or get_datasets(config)
    eval_config = args.eval_config or (ROOT / "configs" / "compute_main_metrics.yaml")
    if not eval_config.exists():
        eval_config = None
    if not datasets:
        return 0
    return run_all(
        root_dir=ROOT,
        artifacts_root=artifacts_root,
        dataset_names=datasets,
        eval_config_path=eval_config,
        reasoner=args.reasoner,
    )


if __name__ == "__main__":
    sys.exit(main())
