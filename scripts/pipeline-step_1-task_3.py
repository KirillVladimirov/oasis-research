"""Запуск задачи 3 шага 1: нормализация кластеров тем с помощью LLM.

Скрипт считывает результаты кластеризации (topics_clustered.jsonl), формирует канонические темы
для каждого кластера и фильтрует noise-темы. Результаты сохраняются в canonical_topics.jsonl.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Подключаем проект
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from oasis.pipelines.stage1_canonical_topics import generate_canonical_topics

# ============================================================================
# ПАРАМЕТРЫ
# ============================================================================

TOPIC = "deep_active_learning"
CLUSTERED_TOPICS_PATH: Path | None = None  # если None -> outputs/{TOPIC}/topics_clustered.jsonl
OUTPUT_CANONICAL_PATH: Path | None = None  # если None -> outputs/{TOPIC}/canonical_topics.jsonl
NOISE_BATCH_SIZE = 10
TRACE_ID = "step_1_task_3"
RESUME_MODE = True  # продолжать с существующего canonical_topics.jsonl, если он уже есть
LLM_LOG_DIR: Path | None = None  # если None -> outputs/{TOPIC}/llm_logs/step_1_task_3
RESUME_FROM_CLUSTER: str | None = "CLUSTER_0277"  # например, "CLUSTER_0275"
LLM_TIMEOUT_SECONDS: float | None = 240.0  # увеличить ожидание ответов LLM
RESUME_NOISE = False  # переобрабатывать noise-темы даже при resume


def main() -> int:
    """Запускает нормализацию кластеров тем."""

    logger.info("=" * 80)
    logger.info("ШАГ 1, ЗАДАЧА 3: НОРМАЛИЗАЦИЯ КЛАСТЕРОВ ТЕМ")
    logger.info("=" * 80)

    project_root = Path(__file__).parent.parent
    clustered_path = CLUSTERED_TOPICS_PATH or (
        project_root / "outputs" / TOPIC / "topics_clustered.jsonl"
    )
    output_path = OUTPUT_CANONICAL_PATH or (
        project_root / "outputs" / TOPIC / "canonical_topics.jsonl"
    )

    log_dir = LLM_LOG_DIR or (project_root / "outputs" / TOPIC / "llm_logs" / "step_1_task_3")

    logger.info(f"Входной файл кластеров: {clustered_path}")
    logger.info(f"Файл для канонических тем: {output_path}")
    logger.info(f"Каталог логов LLM: {log_dir}")
    logger.info(f"Размер батча для noise-тем: {NOISE_BATCH_SIZE}")
    logger.info(f"Trace ID: {TRACE_ID}")
    logger.info(f"Возобновление: {RESUME_MODE}")
    logger.info(f"Возобновление с кластера: {RESUME_FROM_CLUSTER}")
    logger.info(f"Таймаут LLM (с): {LLM_TIMEOUT_SECONDS}")
    logger.info(f"Обработка шумовых тем при возобновлении: {RESUME_NOISE}")

    if not clustered_path.exists():
        logger.error(f"Файл {clustered_path} не найден. Сначала выполните задачу 2.")
        return 1

    try:
        stats = generate_canonical_topics(
            clustered_jsonl=clustered_path,
            output_jsonl=output_path,
            trace_id=TRACE_ID,
            noise_batch_size=NOISE_BATCH_SIZE,
            log_dir=log_dir,
            resume=RESUME_MODE,
            resume_from_cluster=RESUME_FROM_CLUSTER,
            llm_timeout=LLM_TIMEOUT_SECONDS,
            resume_noise=RESUME_NOISE,
        )
    except Exception as exc:
        logger.exception("Ошибка при формировании канонических тем")
        return 1

    logger.info("\nИтоговые метрики:")
    for key, value in stats.items():
        logger.info(f"{key}: {value}")

    logger.info("\nНормализация завершена успешно")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
