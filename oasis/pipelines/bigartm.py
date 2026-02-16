# BigARTM: обучение и LLM-валидация по датасетам. Без вызова скриптов.

from __future__ import annotations

from pathlib import Path

from oasis.pipelines.artm_llm_validation import run_llm_validation
from oasis.pipelines.artm_train import train_artm


def run_train_and_llm(
    artifacts_root: Path,
    dataset_names: list[str],
    dataset_k: dict[str, int],
    config_path: Path,
    workers: int = 15,
    modality_weights: str = "2:1",
) -> int:
    """Для каждого датасета с заданным K выполняет train_artm и run_llm_validation. Без subprocess."""
    for name in dataset_names:
        k = dataset_k.get(name)
        if k is None:
            continue
        art_root = Path(artifacts_root)
        if train_artm(
            artifacts_root=art_root,
            experiment=name,
            k_domain=k,
            modality_weights=modality_weights,
        ) != 0:
            return 1
        if run_llm_validation(
            artifacts_root=art_root,
            experiment=name,
            k_domain=k,
            config_path=config_path,
            workers=workers,
        ) != 0:
            return 1
    return 0
