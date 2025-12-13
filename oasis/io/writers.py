"""Сохранение данных в файлы."""

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from oasis.models.metrics import EnrichmentMetrics
from oasis.models.reference import Reference


def save_references_raw(refs: list[Reference], csv_path: str | Path) -> None:
    """Сохраняет сырые ссылки в CSV файл.

    Args:
        refs: Список Reference объектов
        csv_path: Путь к CSV файлу для сохранения
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    csv_data = []
    for ref in refs:
        ref_dict = ref.to_dict()
        csv_data.append(
            {
                "ref_number": str(ref_dict.get("ref_number", "")),
                "title": ref_dict.get("title", ""),
                "raw_text": ref_dict.get("raw_text", ""),
                "doi": ref_dict.get("doi", ""),
                "arxiv_id": ref_dict.get("arxiv_id", ""),
                "year": ref_dict.get("year") if ref_dict.get("year") else "",
                "authors": ref_dict.get("authors", ""),
                "venue": ref_dict.get("venue", ""),
            }
        )

    df = pd.DataFrame(csv_data)
    df.to_csv(
        path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
        lineterminator="\n",
        escapechar="\\",
        doublequote=True,
        na_rep="",
    )


def save_references(refs: list[Reference], csv_path: str | Path) -> None:
    """Сохраняет обогащённые ссылки в CSV файл.

    Args:
        refs: Список Reference объектов
        csv_path: Путь к CSV файлу для сохранения
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    csv_data = []
    for ref in refs:
        ref_dict = ref.to_dict()
        csv_data.append(
            {
                "ref_id": ref_dict.get("ref_id", ""),
                "ref_number": ref_dict.get("ref_number", ""),
                "title": ref_dict.get("title", ""),
                "year": ref_dict.get("year") if ref_dict.get("year") else "",
                "doi": ref_dict.get("doi", ""),
                "arxiv_id": ref_dict.get("arxiv_id", ""),
                "venue": ref_dict.get("venue", ""),
                "first_author": ref_dict.get("first_author", ""),
                "authors": ref_dict.get("authors", ""),
                "article_url": ref_dict.get("article_url", ""),
                "pdf_url": ref_dict.get("pdf_url", ""),
                "additional_urls": json.dumps(
                    [str(url) for url in (ref_dict.get("additional_urls") or []) if isinstance(url, (str, int, float))],
                    ensure_ascii=False,
                ),
            }
        )

    df = pd.DataFrame(csv_data)
    df.to_csv(path, index=False, encoding="utf-8")


def export_metrics_report(metrics: EnrichmentMetrics, output_path: str | Path) -> None:
    """Экспортирует метрики в JSON файл.

    Args:
        metrics: Объект EnrichmentMetrics
        output_path: Путь к файлу для сохранения (JSON)
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metrics_dict = metrics.to_dict()
    path.write_text(json.dumps(metrics_dict, indent=2, ensure_ascii=False), encoding="utf-8")

