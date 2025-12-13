"""Шаг 1 задача 1: извлечение тем из обзорных статей через LLM.

Скрипт для каждого раздела обзора вызывает LLM, собирает ключевые темы и сохраняет
результаты в topics_raw.jsonl.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Добавляем путь к проекту
sys.path.insert(0, str(Path(__file__).parent.parent))

from oasis.pipelines.stage1_extract_topics import extract_topics_from_survey_papers

TOPIC = "deep_active_learning"

SURVEY_PAPERS_JSON = None

OUTPUT_JSONL = None

TRACE_ID = "step_1_task_1"


def main():
    """Запускает извлечение тем из обзорных статей."""
    logger.info("=" * 80)
    logger.info("ШАГ 1, ЗАДАЧА 1: ИЗВЛЕЧЕНИЕ ТЕМ ИЗ ОБЗОРНЫХ СТАТЕЙ")
    logger.info("=" * 80)

    project_root = Path(__file__).parent.parent

    if SURVEY_PAPERS_JSON is None:
        survey_papers_json = project_root / "outputs" / TOPIC / "survey_papers.json"
    else:
        survey_papers_json = Path(SURVEY_PAPERS_JSON)

    if OUTPUT_JSONL is None:
        output_jsonl = project_root / "outputs" / TOPIC / "topics_raw.jsonl"
    else:
        output_jsonl = Path(OUTPUT_JSONL)

    logger.info(f"Тема: {TOPIC}")
    logger.info(f"Входной файл: {survey_papers_json}")
    logger.info(f"Выходной файл: {output_jsonl}")
    logger.info(f"Trace ID: {TRACE_ID}")

    if not survey_papers_json.exists():
        logger.error(f"Файл со списком обзорных статей не найден: {survey_papers_json}")
        logger.error("Убедитесь, что файл survey_papers.json создан согласно RFC-0001")
        return 1

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    logger.info("\nЗапуск извлечения тем из обзорных статей")

    try:
        stats = extract_topics_from_survey_papers(
            survey_papers_json=survey_papers_json,
            output_jsonl=output_jsonl,
            trace_id=TRACE_ID,
        )

        logger.info("\n" + "=" * 80)
        logger.info(" ИЗВЛЕЧЕНИЕ ТЕМ ЗАВЕРШЕНО")
        logger.info("=" * 80)
        logger.info(f"Результаты сохранены: {output_jsonl}")

        logger.info("\nИТОГОВАЯ СТАТИСТИКА:")
        logger.info(f"Всего обзорных статей: {stats.get('total_papers', 0)}")
        logger.info(f"Обработано статей: {stats.get('processed_papers', 0)}")
        logger.info(f"Всего разделов: {stats.get('total_sections', 0)}")
        logger.info(f"Обработано разделов: {stats.get('processed_sections', 0)}")
        logger.info(f"Успешных извлечений: {stats.get('successful_extractions', 0)}")
        logger.info(f"Ошибок: {stats.get('failed_extractions', 0)}")
        logger.info(f"Длительность: {stats.get('duration_seconds', 0):.1f} секунд")

        if stats.get("failed_extractions", 0) == 0:
            logger.info("\nВсе извлечения выполнены успешно")
        else:
            logger.warning(
                f"\nОбнаружены ошибки: {stats.get('failed_extractions', 0)} "
                f"из {stats.get('total_sections', 0)} разделов"
            )

        return 0

    except Exception as e:
        logger.error(f"Ошибка при извлечении тем: {e}")
        logger.exception("Детали ошибки:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
