#!/usr/bin/env python3
# Шаг 7: генерация CQs из тем BigARTM. Логика в oasis.pipelines.generate_cqs.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oasis.pipelines.generate_cqs import run_all
from scripts.pipeline._config import (
    ROOT,
    get_dataset_k,
    get_dataset_k_from_selection,
    get_datasets,
    load_pipeline_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Шаг 7: генерация CQs из BigARTM-тем")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--run-id", type=str, default="run1")
    parser.add_argument("--n-target", type=int, default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--strict-budget", action="store_true", default=True)
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    config_path = args.config or (ROOT / "configs" / "pipeline.yaml")
    artifacts_root = ROOT / config.get("artifacts_root", "artifacts")
    prompts_dir = ROOT / "prompts" / "s1"
    datasets = args.datasets or get_datasets(config)
    dataset_k = {**get_dataset_k(config), **get_dataset_k_from_selection(artifacts_root)}
    if not datasets:
        return 0
    n_target = args.n_target or config.get("experiment", {}).get("n_target_cqs", 200)
    return run_all(
        root_dir=ROOT,
        artifacts_root=artifacts_root,
        dataset_names=datasets,
        dataset_k=dataset_k,
        config_path=config_path,
        prompts_dir=prompts_dir,
        run_id=args.run_id,
        n_target=n_target,
        workers=args.workers,
        strict_budget=args.strict_budget,
    )


if __name__ == "__main__":
    sys.exit(main())
