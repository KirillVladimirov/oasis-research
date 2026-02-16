# Оценка метрик по cqs_grounded и ontology. Оркестрация по датасетам.

from __future__ import annotations

from pathlib import Path

from oasis.pipelines.eval.run import run_eval


def run_all(
    root_dir: Path,
    artifacts_root: Path,
    dataset_names: list[str],
    eval_config_path: Path | None,
    reasoner: str = "pellet",
) -> int:
    """По каждому датасету запускает run_eval (oasis.pipelines.eval). Возвращает 0 если все ок, иначе 1."""
    failed = 0
    for dataset in dataset_names:
        cqs_path = artifacts_root / dataset / "cqs_grounded.jsonl"
        ontology_path = artifacts_root / dataset / "ontology.owl"
        eval_dir = artifacts_root / dataset / "eval"
        if not cqs_path.exists() or not ontology_path.exists():
            continue
        code = run_eval(
            cqs_path=cqs_path,
            ontology_path=ontology_path,
            out_dir=eval_dir,
            config_path=eval_config_path,
            reasoner=reasoner,
        )
        if code != 0:
            failed += 1
    return 1 if failed else 0
