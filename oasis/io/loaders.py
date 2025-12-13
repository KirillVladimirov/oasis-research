"""Загрузка данных из файлов."""

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from oasis.models.reference import Reference


def load_review_pdf(pdf_path: str | Path) -> bytes:
    """Загружает PDF файл обзора.

    Args:
        pdf_path: Путь к PDF файлу

    Returns:
        Содержимое PDF файла как bytes

    Raises:
        FileNotFoundError: Если файл не найден
        IOError: Если не удалось прочитать файл
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF файл не найден: {pdf_path}")
    return path.read_bytes()


def load_references_raw(csv_path: str | Path) -> list[Reference]:
    """Загружает сырые ссылки из CSV файла.

    Args:
        csv_path: Путь к CSV файлу (references_raw.csv)

    Returns:
        Список Reference объектов

    Raises:
        FileNotFoundError: Если файл не найден
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV файл не найден: {csv_path}")

    df = pd.read_csv(path, encoding="utf-8")
    references = []

    for _, row in df.iterrows():
        ref_dict: dict[str, Any] = {}
        for col in df.columns:
            value = row.get(col)
            # Преобразуем NaN в None или пустую строку
            if pd.isna(value):
                if col == "year":
                    ref_dict[col] = None
                else:
                    ref_dict[col] = ""
            else:
                ref_dict[col] = value

        # Преобразуем ref_number в строку если нужно
        if "ref_number" in ref_dict:
            ref_dict["ref_number"] = str(ref_dict["ref_number"])

        references.append(Reference.from_dict(ref_dict))

    return references


def load_references(csv_path: str | Path) -> list[Reference]:
    """Загружает обогащённые ссылки из CSV файла.

    Args:
        csv_path: Путь к CSV файлу (references.csv)

    Returns:
        Список Reference объектов

    Raises:
        FileNotFoundError: Если файл не найден
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV файл не найден: {csv_path}")

    df = pd.read_csv(path, encoding="utf-8")
    references = []

    for _, row in df.iterrows():
        ref_dict: dict[str, Any] = {}
        for col in df.columns:
            value = row.get(col)

            # Специальная обработка для additional_urls (JSON строка)
            if col == "additional_urls":
                if pd.isna(value) or not value:
                    ref_dict[col] = []
                else:
                    try:
                        import json

                        ref_dict[col] = json.loads(value) if isinstance(value, str) else value
                    except Exception:
                        ref_dict[col] = []
                continue

            # Преобразуем NaN в None или пустую строку
            if pd.isna(value):
                if col == "year":
                    ref_dict[col] = None
                else:
                    ref_dict[col] = ""
            else:
                ref_dict[col] = value

        # Преобразуем ref_number в строку если нужно
        if "ref_number" in ref_dict:
            ref_dict["ref_number"] = str(ref_dict["ref_number"])

        references.append(Reference.from_dict(ref_dict))

    return references


def load_survey_papers(json_path: str | Path) -> list[dict[str, Any]]:
    """Загружает канонический список обзорных статей для генерации seed вопросов.

    Args:
        json_path: Путь к JSON файлу (survey_papers.json)

    Returns:
        Список словарей с метаданными обзорных статей

    Raises:
        FileNotFoundError: Если файл не найден
        ValueError: Если структура JSON некорректна
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON файл не найден: {json_path}")

    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Ошибка парсинга JSON файла {json_path}: {e}") from e

    # Валидация структуры
    if not isinstance(data, dict):
        raise ValueError(f"JSON файл должен содержать объект, получен {type(data)}")

    if "papers" not in data:
        raise ValueError("JSON файл должен содержать поле 'papers'")

    if not isinstance(data["papers"], list):
        raise ValueError("Поле 'papers' должно быть списком")

    # Валидация структуры каждой статьи
    for idx, paper in enumerate(data["papers"]):
        if not isinstance(paper, dict):
            raise ValueError(f"Статья #{idx} должна быть объектом, получен {type(paper)}")

        required_fields = ["paper_id", "title", "pdf_path"]
        for field in required_fields:
            if field not in paper:
                raise ValueError(f"Статья #{idx} должна содержать поле '{field}'")

    return data["papers"]

