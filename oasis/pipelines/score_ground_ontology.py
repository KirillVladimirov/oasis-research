# Scoring CQ, grounding в Weaviate, построение онтологии. Оркестрация по датасетам.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_all(
    root_dir: Path,
    artifacts_root: Path,
    dataset_names: list[str],
    config_path: Path,
    run_id: str = "run1",
    workers: int = 8,
    skip_scoring: bool = False,
    skip_grounding: bool = False,
    skip_ontology: bool = False,
) -> int:
    """По каждому датасету: scoring -> grounding -> ontology. Возвращает 0 если все ок, иначе 1."""
    scripts = root_dir / "scripts" / "experiments"
    scoring_script = scripts / "02_cq_semantic_scoring.py"
    grounding_script = scripts / "03_cq_grounding_weaviate.py"
    ontology_script = scripts / "04_build_ontology_from_cqs.py"
    if not scoring_script.exists() or not grounding_script.exists() or not ontology_script.exists():
        return 1
    cwd = str(root_dir.resolve())
    failed = 0
    for dataset in dataset_names:
        base = artifacts_root / dataset
        cqs = base / "cqs.jsonl"
        cqs_scored = base / "cqs_scored.jsonl"
        cqs_grounded = base / "cqs_grounded.jsonl"
        ontology = base / "ontology.owl"
        base.mkdir(parents=True, exist_ok=True)
        if not cqs.exists():
            continue
        inp = cqs
        if not skip_scoring:
            r = subprocess.run(
                [
                    sys.executable, str(scoring_script),
                    "--input", str(cqs), "--output", str(cqs_scored),
                    "--config", str(config_path),
                ],
                cwd=cwd,
            )
            if r.returncode != 0:
                failed += 1
                continue
            inp = cqs_scored
        if not skip_grounding:
            r = subprocess.run(
                [
                    sys.executable, str(grounding_script),
                    "--input", str(inp), "--output", str(cqs_grounded),
                    "--config", str(config_path), "--dataset", dataset,
                ],
                cwd=cwd,
            )
            if r.returncode != 0:
                failed += 1
                continue
            cqs_for_onto = cqs_grounded
        else:
            cqs_for_onto = cqs_grounded if cqs_grounded.exists() else inp
        if not skip_ontology and cqs_for_onto.exists():
            r = subprocess.run(
                [
                    sys.executable, str(ontology_script),
                    "--cqs", str(cqs_for_onto), "--out", str(ontology),
                    "--config", str(config_path), "--mode", "both", "--strategy", "S1",
                    "--run-id", run_id, "--workers", str(workers),
                ],
                cwd=cwd,
            )
            if r.returncode != 0:
                failed += 1
    return 1 if failed else 0
