#!/usr/bin/env python3
# Шаг 4: подготовка к BigARTM (VW + словарь). Логика в oasis.pipelines.prepare_bigartm.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oasis.pipelines.prepare_bigartm import run_all
from scripts.pipeline._config import ROOT, get_datasets, get_data_root, load_pipeline_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Шаг 4: подготовка к BigARTM (prepare, dictionary)")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    data_root = ROOT / get_data_root(config)
    artifacts_root = ROOT / config.get("artifacts_root", "artifacts")
    datasets = args.datasets or get_datasets(config)
    if not datasets:
        return 0
    code = run_all(root_dir=ROOT, data_root=data_root, dataset_names=datasets, artifacts_root=artifacts_root)
    if code == 0:
        print("step4: prepare+dictionary datasets=%d ok" % len(datasets))
    else:
        print("step4: prepare+dictionary failed")
    return code


if __name__ == "__main__":
    sys.exit(main())
