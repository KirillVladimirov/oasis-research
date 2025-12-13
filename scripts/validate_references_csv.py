"""Скрипт для валидации references.csv - проверка соответствия названий статей и PDF."""

import sys
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

# Добавляем корень проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from oasis.enrichment.cascade.matchers import strict_title_match


def extract_title_from_pdf_text(pdf_content: bytes, max_lines: int = 30) -> str | None:
    """Извлекает заголовок из первых строк PDF."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=pdf_content, filetype="pdf")
        if not doc or doc.page_count == 0:
            return None

        first_page = doc[0]
        text = first_page.get_text("text")
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        # Ищем первую строку длиной > 10 символов (вероятно заголовок)
        for line in lines[:max_lines]:
            if len(line) > 10 and not line.startswith(("http", "www", "doi:", "arXiv:")):
                return line

        return None
    except Exception as e:
        logger.debug(f"Ошибка при извлечении заголовка из PDF: {e}")
        return None


def validate_pdf_title(pdf_url: str, expected_title: str, session: requests.Session) -> dict:
    """Проверяет соответствие заголовка в PDF ожидаемому."""
    try:
        logger.debug(f"Проверка PDF: {pdf_url[:80]}...")
        response = session.get(pdf_url, timeout=10, allow_redirects=True)
        response.raise_for_status()

        if "application/pdf" not in response.headers.get("Content-Type", ""):
            return {"valid": False, "reason": "Not a PDF"}

        pdf_content = response.content
        if len(pdf_content) < 1000:
            return {"valid": False, "reason": "PDF too small"}

        # Извлекаем заголовок из PDF
        pdf_title = extract_title_from_pdf_text(pdf_content)
        if not pdf_title:
            return {"valid": False, "reason": "Cannot extract title from PDF"}

        # Проверяем совпадение
        if strict_title_match(expected_title, pdf_title, threshold=0.90):
            return {
                "valid": True,
                "pdf_title": pdf_title,
                "match_score": 1.0,
            }
        else:
            return {
                "valid": False,
                "reason": "Title mismatch",
                "expected_title": expected_title,
                "pdf_title": pdf_title,
            }

    except requests.exceptions.Timeout:
        return {"valid": False, "reason": "Timeout"}
    except requests.exceptions.HTTPError as e:
        return {"valid": False, "reason": f"HTTP {e.response.status_code}"}
    except Exception as e:
        return {"valid": False, "reason": f"Error: {str(e)[:100]}"}


def main():
    """Основная функция."""
    csv_path = Path("data/deep_active_learning/references.csv")
    
    if not csv_path.exists():
        logger.error(f"Файл не найден: {csv_path}")
        return

    logger.info(f"Загружаем CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Создаем сессию
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    })
    
    # Статистика
    total = len(df)
    with_pdf = df["pdf_url"].notna().sum()
    without_pdf = total - with_pdf
    
    logger.info(f"Всего статей: {total}")
    logger.info(f"С PDF URL: {with_pdf}")
    logger.info(f"Без PDF URL: {without_pdf}")
    
    # Проверяем статьи с PDF
    logger.info("\n" + "=" * 80)
    logger.info("Проверка соответствия заголовков и PDF")
    logger.info("=" * 80 + "\n")
    
    valid_count = 0
    invalid_count = 0
    error_count = 0
    
    mismatches = []
    
    for idx, row in df.iterrows():
        if pd.isna(row.get("pdf_url")):
            continue
        
        ref_number = row.get("ref_number", idx + 1)
        title = row.get("title", "")
        pdf_url = row.get("pdf_url", "")
        
        logger.info(f"\n[{ref_number}] {title[:60]}...")
        logger.info(f"  PDF URL: {pdf_url[:80]}...")
        
        result = validate_pdf_title(pdf_url, title, session)
        
        if result["valid"]:
            logger.success(f"   OK - заголовок совпадает")
            valid_count += 1
        else:
            reason = result.get("reason", "Unknown")
            logger.warning(f"   FAIL - {reason}")
            
            if reason == "Title mismatch":
                logger.warning(f"    Ожидалось: {result['expected_title'][:60]}...")
                logger.warning(f"    В PDF:     {result['pdf_title'][:60]}...")
                mismatches.append({
                    "ref_number": ref_number,
                    "expected_title": title,
                    "pdf_title": result.get("pdf_title", ""),
                    "pdf_url": pdf_url,
                })
                invalid_count += 1
            else:
                error_count += 1
    
    # Итоговая статистика
    logger.info("\n" + "=" * 80)
    logger.info("ИТОГОВАЯ СТАТИСТИКА")
    logger.info("=" * 80)
    logger.info(f"Всего статей: {total}")
    logger.info(f"С PDF URL: {with_pdf}")
    logger.info(f"Проверено PDF: {valid_count + invalid_count + error_count}")
    logger.info(f"   Валидные (заголовок совпадает): {valid_count}")
    logger.info(f"   Невалидные (заголовок НЕ совпадает): {invalid_count}")
    logger.info(f"  ️  Ошибки загрузки/парсинга: {error_count}")
    
    if with_pdf > 0:
        accuracy = (valid_count / with_pdf) * 100
        logger.info(f"\n Точность: {accuracy:.1f}% ({valid_count}/{with_pdf})")
    
    # Список несоответствий
    if mismatches:
        logger.warning(f"\n Найдено {len(mismatches)} несоответствий:")
        for m in mismatches:
            logger.warning(f"\n  [{m['ref_number']}] {m['expected_title'][:60]}...")
            logger.warning(f"    PDF содержит: {m['pdf_title'][:60]}...")
            logger.warning(f"    URL: {m['pdf_url'][:80]}...")
    
    # Сохраняем отчет
    report_path = Path("data/diagnostics/validation_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("ОТЧЕТ ВАЛИДАЦИИ REFERENCES.CSV\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Всего статей: {total}\n")
        f.write(f"С PDF URL: {with_pdf}\n")
        f.write(f"Проверено PDF: {valid_count + invalid_count + error_count}\n")
        f.write(f"   Валидные: {valid_count}\n")
        f.write(f"   Невалидные: {invalid_count}\n")
        f.write(f"  ️  Ошибки: {error_count}\n")
        if with_pdf > 0:
            f.write(f"\n Точность: {accuracy:.1f}% ({valid_count}/{with_pdf})\n")
        
        if mismatches:
            f.write(f"\n\n НЕСООТВЕТСТВИЯ ({len(mismatches)}):\n")
            f.write("=" * 80 + "\n\n")
            for m in mismatches:
                f.write(f"[{m['ref_number']}] {m['expected_title']}\n")
                f.write(f"  PDF содержит: {m['pdf_title']}\n")
                f.write(f"  URL: {m['pdf_url']}\n\n")
    
    logger.info(f"\nОтчет сохранен: {report_path}")


if __name__ == "__main__":
    main()

