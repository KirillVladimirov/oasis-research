#!/usr/bin/env python3
# Шаг 10: материализация KG из ontology.owl и cqs_grounded.jsonl.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oasis.pipelines.kg_materialize import run_kg_materialize
from oasis.utils.weaviate_health import check_weaviate_health
from scripts.pipeline._config import ROOT, get_data_root, load_pipeline_config


def _discover_datasets(artifacts_root: Path) -> list[str]:
    out = []
    for d in artifacts_root.iterdir():
        if not d.is_dir():
            continue
        if (d / "ontology.owl").exists() and (d / "cqs_grounded.jsonl").exists():
            out.append(d.name)
    return sorted(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Шаг 10: материализация KG")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default=None, help="Один датасет; при --all не используется")
    parser.add_argument("--all", action="store_true", help="Запустить для всех датасетов (ontology.owl + cqs_grounded.jsonl)")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    artifacts_root = ROOT / config.get("artifacts_root", "artifacts")
    data_root = ROOT / get_data_root(config)
    weaviate_url = config.get("weaviate_url") or (config.get("grounding") or {}).get("weaviate_url", "http://localhost:8081")
    if not check_weaviate_health(weaviate_url):
        print("step10: Weaviate недоступен по %s (продолжаем)" % weaviate_url)

    if args.all:
        datasets = _discover_datasets(artifacts_root)
        if not datasets:
            print("step10: нет датасетов с ontology.owl и cqs_grounded.jsonl в", artifacts_root)
            return 1
        print("step10: датасетов для запуска:", len(datasets))
        last_code = 0
        for ds in datasets:
            print("\n--- dataset:", ds, "---")
            last_code = run_kg_materialize(
                artifacts_dir=artifacts_root,
                dataset=ds,
                config=config,
                data_root=data_root,
            )
            if last_code != 0:
                print("step10: ошибка для датасета", ds)
        return last_code

    dataset = args.dataset or "deep_active_learning"
    return run_kg_materialize(
        artifacts_dir=artifacts_root,
        dataset=dataset,
        config=config,
        data_root=data_root,
    )


if __name__ == "__main__":
    sys.exit(main())
