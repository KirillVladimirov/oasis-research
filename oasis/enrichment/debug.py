"""Утилиты для анализа ошибок обогащения ссылок."""

import json
from pathlib import Path
from typing import Any

from loguru import logger


def analyze_enrichment_logs(enriched_refs: list[dict[str, Any]], output_path: str | Path | None = None) -> dict[str, Any]:
    """Анализирует логи обогащения и генерирует отчет.

    Args:
        enriched_refs: Список обогащенных ссылок с _enrichment_log
        output_path: Путь для сохранения отчета (опционально)

    Returns:
        Словарь с анализом ошибок
    """
    analysis = {
        "total_refs": len(enriched_refs),
        "refs_with_errors": 0,
        "refs_with_warnings": 0,
        "refs_without_log": 0,
        "error_types": {},
        "warning_types": {},
        "step_statistics": {},
        "failed_refs": [],
    }

    for ref in enriched_refs:
        log = ref.get("_enrichment_log")
        if not log:
            analysis["refs_without_log"] += 1
            continue

        has_errors = len(log.get("errors", [])) > 0
        has_warnings = len(log.get("warnings", [])) > 0

        if has_errors:
            analysis["refs_with_errors"] += 1
            analysis["failed_refs"].append({
                "ref_number": ref.get("ref_number", ""),
                "title": ref.get("title", "")[:100],
                "errors": log.get("errors", []),
                "warnings": log.get("warnings", []),
            })

            # Подсчет типов ошибок
            for error in log.get("errors", []):
                error_key = error.split(":")[0] if ":" in error else error[:50]
                analysis["error_types"][error_key] = analysis["error_types"].get(error_key, 0) + 1

        if has_warnings:
            analysis["refs_with_warnings"] += 1
            for warning in log.get("warnings", []):
                warning_key = warning.split(":")[0] if ":" in warning else warning[:50]
                analysis["warning_types"][warning_key] = analysis["warning_types"].get(warning_key, 0) + 1

        # Статистика по шагам
        for step in log.get("steps", []):
            step_name = step.get("step", "unknown")
            if step_name not in analysis["step_statistics"]:
                analysis["step_statistics"][step_name] = {
                    "total": 0,
                    "success": 0,
                    "not_found": 0,
                    "failed": 0,
                }
            analysis["step_statistics"][step_name]["total"] += 1
            result = step.get("result", "unknown")
            if result in analysis["step_statistics"][step_name]:
                analysis["step_statistics"][step_name][result] += 1

    # Сохранение отчета
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        logger.info(f"Отчет анализа сохранен в {output_path}")

    return analysis


def print_enrichment_analysis(analysis: dict[str, Any]) -> None:
    """Выводит анализ обогащения в консоль.

    Args:
        analysis: Результат analyze_enrichment_logs
    """
    print("\n" + "=" * 80)
    print("АНАЛИЗ ОБОГАЩЕНИЯ ССЫЛОК")
    print("=" * 80)
    print(f"Всего ссылок: {analysis['total_refs']}")
    
    total_refs = analysis['total_refs']
    if total_refs > 0:
        print(f"С ошибками: {analysis['refs_with_errors']} ({analysis['refs_with_errors']/total_refs*100:.1f}%)")
        print(f"С предупреждениями: {analysis['refs_with_warnings']} ({analysis['refs_with_warnings']/total_refs*100:.1f}%)")
    else:
        print(f"С ошибками: {analysis['refs_with_errors']}")
        print(f"С предупреждениями: {analysis['refs_with_warnings']}")
    print(f"Без логов: {analysis['refs_without_log']}")

    print("\n--- Статистика по шагам ---")
    for step_name, stats in analysis["step_statistics"].items():
        print(f"\n{step_name}:")
        print(f"  Всего: {stats['total']}")
        print(f"  Успешно: {stats.get('success', 0)} ({stats.get('success', 0)/stats['total']*100:.1f}%)")
        print(f"  Не найдено: {stats.get('not_found', 0)} ({stats.get('not_found', 0)/stats['total']*100:.1f}%)")
        print(f"  Провалено: {stats.get('failed', 0)} ({stats.get('failed', 0)/stats['total']*100:.1f}%)")

    if analysis["error_types"]:
        print("\n--- Типы ошибок (топ-10) ---")
        sorted_errors = sorted(analysis["error_types"].items(), key=lambda x: x[1], reverse=True)
        for error_type, count in sorted_errors[:10]:
            print(f"  {error_type}: {count}")

    if analysis["warning_types"]:
        print("\n--- Типы предупреждений (топ-10) ---")
        sorted_warnings = sorted(analysis["warning_types"].items(), key=lambda x: x[1], reverse=True)
        for warning_type, count in sorted_warnings[:10]:
            print(f"  {warning_type}: {count}")

    if analysis["failed_refs"]:
        print(f"\n--- Проблемные ссылки (первые 10) ---")
        for ref_info in analysis["failed_refs"][:10]:
            print(f"\n  [{ref_info['ref_number']}] {ref_info['title']}")
            for error in ref_info["errors"][:3]:
                print(f"     {error}")

    print("\n" + "=" * 80)

