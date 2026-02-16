# Генерация CQs из тем BigARTM. Оркестрация по датасетам.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_all(
    root_dir: Path,
    artifacts_root: Path,
    dataset_names: list[str],
    dataset_k: dict[str, int],
    config_path: Path,
    prompts_dir: Path,
    run_id: str = "run1",
    n_target: int = 200,
    workers: int = 12,
    strict_budget: bool = True,
) -> int:
    """По каждому датасету с заданным K генерирует cqs.jsonl. Возвращает 0 если все ок, иначе 1."""
    script = root_dir / "scripts" / "experiments" / "s1" / "01_generate_cqs_from_topics_bigartm.py"
    if not script.exists():
        return 1
    cwd = str(root_dir.resolve())
    failed = 0
    for dataset in dataset_names:
        k = dataset_k.get(dataset)
        if k is None:
            continue
        topics_path = artifacts_root / dataset / ("k%d" % k) / ("%s--k%d--topics_llm.jsonl" % (dataset, k))
        if not topics_path.exists():
            continue
        out_path = artifacts_root / dataset / "cqs.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(script),
            "--topics", str(topics_path),
            "--out", str(out_path),
            "--config", str(config_path),
            "--prompts-dir", str(prompts_dir),
            "--run-id", run_id,
            "--n-target", str(n_target),
            "--workers", str(workers),
        ]
        if strict_budget:
            cmd.append("--strict-budget")
        r = subprocess.run(cmd, cwd=cwd)
        if r.returncode != 0:
            failed += 1
    return 1 if failed else 0
