"""Шаг 1 задача 2: векторизация и кластеризация тем.

Загружает локальные темы, строит эмбеддинги и группирует их, сохраняя результаты
в topics_clustered.jsonl.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from oasis.pipelines.stage1_cluster_topics import cluster_topics_from_raw

TOPIC = "deep_active_learning"

INPUT_JSONL = None

OUTPUT_JSONL = None

EMBEDDING_MODEL = "models/bge-m3"
EMBEDDING_DEVICE = "cpu"

CLUSTERING_METHOD = "hdbscan"

CLUSTERING_PARAMS = {
    "min_cluster_size": 2,
    "min_samples": 1,
}

TRACE_ID = "step_1_task_2"


def main():
    """Запускает векторизацию и кластеризацию тем."""
    logger.info("=" * 80)
    logger.info("ШАГ 1, ЗАДАЧА 2: ВЕКТОРИЗАЦИЯ И КЛАСТЕРИЗАЦИЯ ТЕМ")
    logger.info("=" * 80)

    project_root = Path(__file__).parent.parent

    if INPUT_JSONL is None:
        input_jsonl = project_root / "outputs" / TOPIC / "topics_raw.jsonl"
    else:
        input_jsonl = Path(INPUT_JSONL)

    if OUTPUT_JSONL is None:
        output_jsonl = project_root / "outputs" / TOPIC / "topics_clustered.jsonl"
    else:
        output_jsonl = Path(OUTPUT_JSONL)

    logger.info(f"Тема: {TOPIC}")
    logger.info(f"Входной файл: {input_jsonl}")
    logger.info(f"Выходной файл: {output_jsonl}")
    logger.info(f"Модель эмбеддингов: {EMBEDDING_MODEL}")
    logger.info(f"Устройство эмбеддингов: {EMBEDDING_DEVICE}")
    logger.info(f"Метод кластеризации: {CLUSTERING_METHOD}")
    logger.info(f"Параметры кластеризации: {CLUSTERING_PARAMS}")
    logger.info(f"Trace ID: {TRACE_ID}")

    if not input_jsonl.exists():
        logger.error(f"Файл с локальными темами не найден: {input_jsonl}")
        logger.error("Убедитесь, что задача 1 шага 1 выполнена и создан topics_raw.jsonl")
        return 1

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    logger.info("\nЗапуск векторизации и кластеризации тем")

    try:
        stats = cluster_topics_from_raw(
            input_jsonl=input_jsonl,
            output_jsonl=output_jsonl,
            embedding_model=EMBEDDING_MODEL,
            embedding_device=EMBEDDING_DEVICE,
            clustering_method=CLUSTERING_METHOD,
            clustering_params=CLUSTERING_PARAMS,
            trace_id=TRACE_ID,
        )

        logger.info("\n" + "=" * 80)
        logger.info(" КЛАСТЕРИЗАЦИЯ ТЕМ ЗАВЕРШЕНА")
        logger.info("=" * 80)
        logger.info(f"Результаты сохранены: {output_jsonl}")

        logger.info("\nИТОГОВАЯ СТАТИСТИКА:")
        logger.info(f"Всего тем: {stats.get('total_topics', 0)}")
        logger.info(f"Количество кластеров: {stats.get('n_clusters', 0)}")
        logger.info(f"Точек шума: {stats.get('n_noise', 0)}")
        logger.info(f"Средний размер кластера: {stats.get('avg_cluster_size', 0):.2f}")
        logger.info(f"Средняя внутрикластерная дистанция: {stats.get('avg_intra_cluster_distance', 0):.4f}")
        logger.info(f"Длительность: {stats.get('duration_seconds', 0):.1f} секунд")

        if stats.get("n_clusters", 0) == 0:
            logger.warning("\nНе создано ни одного кластера. Проверьте параметры кластеризации.")
        else:
            total_topics = stats.get("total_topics", 0)
            n_clusters = stats.get("n_clusters", 0)
            n_noise = stats.get("n_noise", 0)
            clustered_topics = total_topics - n_noise

            if total_topics > 0:
                merge_ratio = (1 - n_clusters / clustered_topics) * 100 if clustered_topics > 0 else 0.0
                logger.info(f"\nДоля объединения тем: {merge_ratio:.1f}% ({clustered_topics} тем в {n_clusters} кластерах)")

            logger.info("\nКластеризация выполнена успешно")

        return 0

    except Exception as e:
        logger.error(f"Ошибка при кластеризации тем: {e}")
        logger.exception("Детали ошибки:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
