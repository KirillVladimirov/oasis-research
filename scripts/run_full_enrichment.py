"""Запуск полного обогащения всех статей."""

import sys
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from oasis.pipelines.stage2_enrich import enrich_references


def main():
    """Запускает полное обогащение всех статей."""
    logger.info("=" * 80)
    logger.info("ПОЛНОЕ ОБОГАЩЕНИЕ ВСЕХ СТАТЕЙ")
    logger.info("=" * 80)
    
    csv_path = Path("data/deep_active_learning/references_raw.csv")
    topic = "deep_active_learning"
    trace_id = "full_enrichment"
    
    if not csv_path.exists():
        logger.error(f"Файл {csv_path} не найден")
        return
    
    df = pd.read_csv(csv_path)
    logger.info(f"Загружено {len(df)} ссылок из {csv_path}")
    
    pdf_before = df[df["pdf_url"].notna() & (df["pdf_url"] != "")].shape[0]
    logger.info(f"PDF до обогащения: {pdf_before}/{len(df)} ({pdf_before/len(df)*100:.1f}%)")
    
    # Преобразуем в список словарей
    raw_refs = df.to_dict("records")
    
    logger.info("\nЗапуск Stage 2: обогащение")
    
    def progress_callback(stage: str, progress: float, message: str):
        logger.info(f"[{stage}] {progress*100:.1f}% - {message}")
    
    try:
        output_csv, enriched_refs, metrics = enrich_references(
            raw_refs,
            topic=topic,
            trace_id=trace_id,
            progress_callback=progress_callback,
        )
        
        logger.info("\n" + "=" * 80)
        logger.info(" ОБОГАЩЕНИЕ ЗАВЕРШЕНО")
        logger.info("=" * 80)
        logger.info(f"Результаты сохранены: {output_csv}")
        
        # Итоговая статистика
        logger.info("\n ИТОГОВАЯ СТАТИСТИКА:")
        logger.info(f"Всего статей: {metrics.get('total_references', 0)}")
        logger.info(f"С PDF: {metrics.get('pdf_count', 0)} ({metrics.get('pdf_percentage', 0):.1f}%)")
        logger.info(f"С DOI: {metrics.get('doi_count', 0)} ({metrics.get('doi_percentage', 0):.1f}%)")
        logger.info(f"С arXiv ID: {metrics.get('arxiv_count', 0)} ({metrics.get('arxiv_percentage', 0):.1f}%)")
        
        # Сравнение с целью
        target = 185  # 90% от 205
        pdf_count = metrics.get('pdf_count', 0)
        logger.info(f"\n ЦЕЛЕВАЯ МЕТРИКА: {target}+ PDF (90%)")
        if pdf_count >= target:
            logger.info(f" ЦЕЛЬ ДОСТИГНУТА! ({pdf_count}/{len(df)} = {pdf_count/len(df)*100:.1f}%)")
        else:
            logger.info(f"️  Цель не достигнута ({pdf_count}/{len(df)} = {pdf_count/len(df)*100:.1f}%)")
            logger.info(f"   Не хватает: {target - pdf_count} PDF")
        
    except Exception as e:
        logger.error(f"Ошибка при обогащении: {e}")
        raise


if __name__ == "__main__":
    main()
