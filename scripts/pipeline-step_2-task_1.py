"""Шаг 2 задачи 1-6: создание RAG индекса корпуса."""

import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from oasis.pipelines.stage2_indexing import index_corpus
from oasis.utils.network import check_http_service

TOPIC = "deep_active_learning"
PDF_DIR = Path("data/deep_active_learning/paper_pdfs")
REFERENCES_CSV = Path("data/deep_active_learning/references.csv")
WEAVIATE_URL = "http://localhost:8081"
TRACE_ID = "step_2_task_1"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300
BATCH_SIZE = 100
EMBEDDING_BATCH_SIZE = 8
MODEL_PATH = "models/bge-m3"
DELETE_EXISTING = True


def main() -> int:
    """Запускает индексацию корпуса."""
    logger.info("=" * 80)
    logger.info("ШАГ 2, ЗАДАЧИ 1-6: Создание RAG индекса корпуса")
    logger.info("=" * 80)

    weaviate_health_url = f"{WEAVIATE_URL}/v1/.well-known/ready"
    logger.info(f"Проверка доступности Weaviate: {weaviate_health_url}")
    if not check_http_service(weaviate_health_url):
        logger.error(f"Weaviate недоступен по адресу {WEAVIATE_URL}")
        logger.error("Убедитесь, что Weaviate запущен: docker-compose up -d weaviate")
        return 1
    logger.success("Weaviate доступен")

    if not PDF_DIR.exists():
        logger.error(f"Директория с PDF не найдена: {PDF_DIR}")
        return 1

    if not REFERENCES_CSV.exists():
        logger.error(f"Файл метаданных не найден: {REFERENCES_CSV}")
        return 1

    try:
        stats = index_corpus(
            pdf_dir=PDF_DIR,
            references_csv=REFERENCES_CSV,
            weaviate_url=WEAVIATE_URL,
            trace_id=TRACE_ID,
            batch_size=BATCH_SIZE,
            embedding_batch_size=EMBEDDING_BATCH_SIZE,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            model_path=MODEL_PATH,
            delete_existing=DELETE_EXISTING,
        )

        logger.info("=" * 80)
        logger.info("СТАТИСТИКА ИНДЕКСАЦИИ")
        logger.info("=" * 80)
        logger.info(f"Обработано статей: {stats['papers_processed']}")
        logger.info(f"Обработано чанков: {stats['chunks_processed']}")
        logger.info(f"Пропущено статей: {stats['papers_failed']}")

        if stats["failed_files"]:
            logger.warning("Проблемные файлы:")
            for failed_file in stats["failed_files"]:
                logger.warning(f"  - {failed_file}")

        logger.success("Индексация завершена успешно")
        return 0

    except Exception as e:
        logger.exception(f"Ошибка при индексации: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
