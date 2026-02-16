#!/usr/bin/env python3
"""
Скрипт для проверки индексации всех датасетов в Weaviate.

Проверяет наличие классов Document_<dataset> и Chunk_<dataset> для каждого датасета
и выводит статистику по количеству документов и чанков.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import weaviate
except ImportError:
    print("Error: weaviate-client is not installed. Install it with: pip install 'weaviate-client>=3.26.7,<4.0.0'")
    sys.exit(1)


# Ожидаемые датасеты
EXPECTED_DATASETS = [
    "ml_systems",
    "deep_learning_neural_networks",
    "diffusion_models",
    "generative_models_in_pathology",
    "graph_nn_research",
    "graph_nn_systems",
    "healthcare_foundation_models",
    "ml_systems_distributed_training",
    "ml_systems_inference_system",
    "ml_systems_mixture_of_experts_moe",
    "model_quantization",
    "multimodal_ai",
    "speech_recognition_synthesis",
    "deep_active_learning",
    "causal_inference",
    "automl",
]


def get_schema(client: weaviate.Client) -> dict:
    """Получает схему Weaviate."""
    try:
        return client.schema.get()
    except Exception as e:
        print(f"Error getting schema: {e}")
        return {}


def count_objects(client: weaviate.Client, class_name: str) -> int:
    """Подсчитывает количество объектов в классе."""
    try:
        result = client.query.aggregate(class_name).with_meta_count().do()
        if "data" in result and "Aggregate" in result["data"]:
            agg = result["data"]["Aggregate"]
            if agg and class_name in agg:
                return agg[class_name][0]["meta"]["count"]
        return 0
    except Exception as e:
        print(f"  Warning: Could not count objects in {class_name}: {e}")
        return -1


def extract_datasets_from_classes(classes: list[dict]) -> set[str]:
    """Извлекает имена датасетов из имен классов."""
    datasets = set()
    for cls in classes:
        class_name = cls.get("class", "")
        if class_name.startswith("Document_"):
            dataset = class_name.replace("Document_", "")
            datasets.add(dataset)
        elif class_name.startswith("Chunk_"):
            dataset = class_name.replace("Chunk_", "")
            datasets.add(dataset)
    return datasets


def check_datasets(weaviate_url: str) -> int:
    """Проверяет индексацию всех датасетов."""
    print(f"Проверка Weaviate по адресу: {weaviate_url}\n")

    # Подключение к Weaviate
    try:
        client = weaviate.Client(weaviate_url)
    except Exception as e:
        print(f"Ошибка подключения к Weaviate: {e}")
        return 1

    # Получаем схему
    schema = get_schema(client)
    if not schema or "classes" not in schema:
        print("Ошибка: не удалось получить схему Weaviate")
        return 1

    classes = schema["classes"]
    print(f"Всего классов в Weaviate: {len(classes)}\n")

    # Извлекаем датасеты из классов
    found_datasets = extract_datasets_from_classes(classes)
    
    # Проверяем каждый ожидаемый датасет
    print("=" * 80)
    print(f"{'Датасет':<40} {'Document':<15} {'Chunk':<15} {'Статус'}")
    print("=" * 80)

    missing_datasets = []
    incomplete_datasets = []
    complete_datasets = []

    for dataset in EXPECTED_DATASETS:
        doc_class = f"Document_{dataset}"
        chunk_class = f"Chunk_{dataset}"

        doc_exists = any(cls.get("class") == doc_class for cls in classes)
        chunk_exists = any(cls.get("class") == chunk_class for cls in classes)

        if not doc_exists and not chunk_exists:
            print(f"{dataset:<40} {'НЕТ':<15} {'НЕТ':<15} [ERROR] ОТСУТСТВУЕТ")
            missing_datasets.append(dataset)
        elif doc_exists and chunk_exists:
            doc_count = count_objects(client, doc_class)
            chunk_count = count_objects(client, chunk_class)
            doc_str = f"{doc_count}" if doc_count >= 0 else "?"
            chunk_str = f"{chunk_count}" if chunk_count >= 0 else "?"
            print(f"{dataset:<40} {doc_str:<15} {chunk_str:<15} [OK] ИНДЕКСИРОВАН")
            complete_datasets.append((dataset, doc_count, chunk_count))
        else:
            status = "[WARN] ЧАСТИЧНО"
            if doc_exists:
                doc_count = count_objects(client, doc_class)
                doc_str = f"{doc_count}" if doc_count >= 0 else "?"
                chunk_str = "НЕТ"
            else:
                doc_str = "НЕТ"
                chunk_count = count_objects(client, chunk_class)
                chunk_str = f"{chunk_count}" if chunk_count >= 0 else "?"
            print(f"{dataset:<40} {doc_str:<15} {chunk_str:<15} {status}")
            incomplete_datasets.append(dataset)

    print("=" * 80)
    print()

    # Сводка
    print("СВОДКА:")
    print(f"  [OK] Полностью проиндексировано: {len(complete_datasets)}/{len(EXPECTED_DATASETS)}")
    print(f"  [WARN] Частично проиндексировано: {len(incomplete_datasets)}")
    print(f"  [ERROR] Отсутствует: {len(missing_datasets)}")

    if complete_datasets:
        print("\nДетальная статистика по проиндексированным датасетам:")
        print(f"{'Датасет':<40} {'Документов':<15} {'Чанков':<15}")
        print("-" * 70)
        for dataset, doc_count, chunk_count in sorted(complete_datasets):
            print(f"{dataset:<40} {doc_count:<15} {chunk_count:<15}")

    if incomplete_datasets:
        print(f"\n[WARN] Частично проиндексированные датасеты: {', '.join(incomplete_datasets)}")

    if missing_datasets:
        print(f"\n[ERROR] Отсутствующие датасеты: {', '.join(missing_datasets)}")
        print("\nДля индексации используйте:")
        print("  ./scripts/index_all_datasets.sh")
        print("или")
        print("  python3 scripts/pipeline/step3_index_weaviate.py --datasets <dataset_name>")

    # Проверяем лишние классы (не из ожидаемых датасетов)
    extra_datasets = found_datasets - set(EXPECTED_DATASETS)
    if extra_datasets:
        print(f"\n[WARN] Найдены дополнительные датасеты (не в списке ожидаемых): {', '.join(sorted(extra_datasets))}")

    return 0 if len(complete_datasets) == len(EXPECTED_DATASETS) else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Проверка индексации всех датасетов в Weaviate"
    )
    parser.add_argument(
        "--weaviate-url",
        type=str,
        default="http://localhost:8081",
        help="URL Weaviate сервера (по умолчанию: http://localhost:8081)",
    )
    args = parser.parse_args()

    return check_datasets(args.weaviate_url)


if __name__ == "__main__":
    sys.exit(main())
