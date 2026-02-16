"""Загрузка конфига пайплайна (configs/pipeline.yaml)."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "pipeline.yaml"


def load_pipeline_config(config_path: Path | None = None) -> dict:
    path = config_path or DEFAULT_CONFIG
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def get_datasets(config: dict | None = None) -> list[str]:
    cfg = config or load_pipeline_config()
    return list(cfg.get("datasets", []))


def get_data_root(config: dict | None = None) -> str:
    cfg = config or load_pipeline_config()
    return str(cfg.get("data_root", "data"))


def get_dataset_k(config: dict | None = None) -> dict[str, int]:
    cfg = config or load_pipeline_config()
    return dict(cfg.get("dataset_k", {}))


def get_dataset_k_from_selection(artifacts_root: Path) -> dict[str, int]:
    """K по датасетам из результата step5 (artifacts/topic_counting/k_selection_all.json)."""
    path = artifacts_root / "topic_counting" / "k_selection_all.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        results = data.get("results") or []
        out = {}
        for r in results:
            k = r.get("k_best")
            input_dir = r.get("input_dir")
            if k is None or not input_dir:
                continue
            # input_dir вида "data/deep_active_learning/topics" -> dataset = deep_active_learning
            name = Path(input_dir).parent.name
            if name:
                out[name] = int(k)
        return out
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
