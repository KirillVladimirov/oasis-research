#!/usr/bin/env python3
# Шаг 1 пайплайна: скачивание PDF датасетов из GitHub Awesome.
# Результат: data_root/<dataset>/paper_pdfs/

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.pipeline._config import get_data_root, get_datasets, load_pipeline_config
from oasis.pipelines.download_papers import download_topic, get_topic_config_by_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Шаг 1: скачивание датасетов (PDF из GitHub)")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    data_root = Path(args.data_root) if args.data_root else Path(get_data_root(config))
    datasets = args.datasets or get_datasets(config)
    if not datasets:
        return 0

    total = {"found": 0, "downloaded": 0, "skipped": 0, "errors": 0}
    for topic_id in datasets:
        cfg = get_topic_config_by_id(topic_id)
        if not cfg:
            continue
        s = download_topic(cfg, data_root=data_root)
        for k in total:
            total[k] += s[k]

    print("step1: found=%d downloaded=%d skipped=%d errors=%d" % (total["found"], total["downloaded"], total["skipped"], total["errors"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
