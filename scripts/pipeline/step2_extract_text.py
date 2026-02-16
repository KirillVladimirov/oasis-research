#!/usr/bin/env python3
# Шаг 2 пайплайна: извлечение текста из PDF. Вход: paper_pdfs. Выход: topics/, rag/.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.pipeline._config import get_data_root, get_datasets, load_pipeline_config
from oasis.pipelines.extract_text import process_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Шаг 2: извлечение текста из PDF")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    data_root = Path(args.data_root) if args.data_root else Path(get_data_root(config))
    datasets = args.datasets or get_datasets(config)
    if not datasets:
        return 0

    total_ok, total_err, num_processed = 0, 0, 0
    for dataset in datasets:
        pdf_dir = data_root / dataset / "paper_pdfs"
        output_dir = data_root / dataset
        if not pdf_dir.exists():
            continue
        ok, err = process_dataset(pdf_dir, output_dir)
        total_ok += ok
        total_err += err
        num_processed += 1

    print("step2: datasets=%d ok=%d errors=%d" % (num_processed, total_ok, total_err))
    return 1 if total_err else 0


if __name__ == "__main__":
    sys.exit(main())
