#!/usr/bin/env python3
# Шаг 6: BigARTM — обучение и LLM-валидация по датасетам. Логика в oasis.pipelines.bigartm.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oasis.pipelines.bigartm import run_train_and_llm
from scripts.pipeline._config import (
    ROOT,
    get_datasets,
    get_dataset_k,
    get_dataset_k_from_selection,
    load_pipeline_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Шаг 6: BigARTM train + LLM по датасетам")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=15)
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    config_path = args.config or (ROOT / "configs" / "pipeline.yaml")
    artifacts_root = ROOT / config.get("artifacts_root", "artifacts")
    datasets = args.datasets or get_datasets(config)
    # K из результата step5 (k_selection_all.json) с подстановкой из config.dataset_k при отсутствии
    dataset_k = {**get_dataset_k(config), **get_dataset_k_from_selection(artifacts_root)}
    if not datasets:
        return 0
    return run_train_and_llm(
        artifacts_root=artifacts_root,
        dataset_names=datasets,
        dataset_k=dataset_k,
        config_path=config_path,
        workers=args.workers,
    )


if __name__ == "__main__":
    sys.exit(main())
