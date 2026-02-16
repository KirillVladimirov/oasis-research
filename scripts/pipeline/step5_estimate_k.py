#!/usr/bin/env python3
# Шаг 5: оценка числа тем K по датасетам (oasis.pipelines.estimate_k).

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oasis.pipelines.estimate_k import run_all
from scripts.pipeline._config import ROOT, get_data_root, get_datasets, load_pipeline_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Шаг 5: оценка числа тем K")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    artifacts_root = Path(config.get("artifacts_root", "artifacts"))
    out_path = args.out or (ROOT / artifacts_root / "topic_counting" / "k_selection_all.json")
    data_root = ROOT / get_data_root(config)
    dataset_names = get_datasets(config)
    return run_all(data_root=data_root, out_path=out_path, dataset_names=dataset_names)


if __name__ == "__main__":
    sys.exit(main())
