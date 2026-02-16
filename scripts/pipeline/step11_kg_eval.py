#!/usr/bin/env python3
# Шаг 11: замер качества графов знаний (план 11.x — единый CSV, манифест или автодискавери).

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oasis.pipelines.kg_eval import (
    load_datasets_manifest,
    run_kg_eval,
    write_kg_metrics_all_csv,
)
from scripts.pipeline._config import ROOT, load_pipeline_config


def _discover_kg_datasets(artifacts_root: Path) -> list[dict[str, str]]:
    """11.1 Вариант Б: автодискавери по artifacts/*/kg/kg.ttl, dataset_id = имя папки."""
    out = []
    for d in artifacts_root.iterdir():
        if not d.is_dir():
            continue
        kg_ttl = d / "kg" / "kg.ttl"
        if kg_ttl.exists():
            prov = d / "kg" / "kg_provenance.csv"
            if not prov.exists():
                prov = d / "kg" / "kg_provenance.parquet"
            out.append({
                "dataset_id": d.name,
                "kg_path": str(kg_ttl.resolve()),
                "stats_path": str((d / "kg" / "kg_stats.json").resolve()) if (d / "kg" / "kg_stats.json").exists() else "",
                "prov_path": str(prov.resolve()) if prov.exists() else "",
            })
    return sorted(out, key=lambda x: x["dataset_id"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Шаг 11: оценка качества KG (единый CSV)")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", type=str, default=None, help="Один датасет")
    parser.add_argument("--all", action="store_true", help="Все датасеты → kg_metrics_all.csv")
    parser.add_argument("--no-shacl", action="store_true", help="Не запускать SHACL")
    parser.add_argument("--manifest", type=Path, default=None, help="Путь к datasets_manifest.csv (11.1 Вариант А)")
    args = parser.parse_args()

    config = load_pipeline_config(args.config)
    artifacts_root = ROOT / config.get("artifacts_root", "artifacts")
    base_iri = (config.get("ontology") or {}).get("base_iri", "http://example.org")
    kg_eval_dir = artifacts_root / "kg_eval"

    if args.all:
        manifest_path = args.manifest or kg_eval_dir / "datasets_manifest.csv"
        if manifest_path.exists():
            datasets = load_datasets_manifest(manifest_path)
            if datasets:
                print("step11: загружен манифест", manifest_path, "→", len(datasets), "датасетов")
        else:
            datasets = _discover_kg_datasets(artifacts_root)
            print("step11: автодискавери →", len(datasets), "датасетов")
        if not datasets:
            print("step11: нет датасетов (манифест или artifacts/*/kg/kg.ttl)")
            return 1

        ontology_paths: dict[str, Path] = {}
        for ds in datasets:
            did = ds["dataset_id"]
            kg_path = Path(ds["kg_path"])
            dataset_dir = kg_path.parent.parent
            ontology_paths[did] = dataset_dir / "ontology.owl"

        rows = []
        for ds in datasets:
            did = ds["dataset_id"]
            kg_path = Path(ds["kg_path"])
            out_dir = kg_path.parent.parent / "kg_eval"
            prov_path = Path(ds["prov_path"]) if ds.get("prov_path") else None
            if not prov_path and kg_path.parent / "kg_provenance.parquet".exists():
                prov_path = kg_path.parent / "kg_provenance.parquet"
            if not prov_path and kg_path.parent / "kg_provenance.csv".exists():
                prov_path = kg_path.parent / "kg_provenance.csv"

            result = run_kg_eval(
                kg_path=kg_path,
                ontology_path=ontology_paths.get(did),
                base_iri=base_iri,
                out_dir=out_dir,
                prov_path=prov_path,
                run_shacl_validation=not args.no_shacl,
            )
            row = result["row"]
            row["dataset_id"] = did
            rows.append(row)

        kg_eval_dir.mkdir(parents=True, exist_ok=True)
        write_kg_metrics_all_csv(
            rows,
            kg_eval_dir / "kg_metrics_all.csv",
            sorted_path=kg_eval_dir / "kg_metrics_all_sorted.csv",
        )
        print("step11: записаны", kg_eval_dir / "kg_metrics_all.csv", "и kg_metrics_all_sorted.csv")
        return 0

    dataset = args.dataset or "deep_active_learning"
    kg_path = artifacts_root / dataset / "kg" / "kg.ttl"
    if not kg_path.exists():
        print("step11: не найден", kg_path)
        return 1
    ontology_path = artifacts_root / dataset / "ontology.owl"
    out_dir = artifacts_root / dataset / "kg_eval"
    prov_path = artifacts_root / dataset / "kg" / "kg_provenance.parquet"
    if not prov_path.exists():
        prov_path = artifacts_root / dataset / "kg" / "kg_provenance.csv"
    result = run_kg_eval(
        kg_path=kg_path,
        ontology_path=ontology_path,
        base_iri=base_iri,
        out_dir=out_dir,
        prov_path=prov_path if prov_path.exists() else None,
        run_shacl_validation=not args.no_shacl,
    )
    print("step11: результаты в", out_dir, "| status =", result["row"].get("status"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
