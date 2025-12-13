"""Векторизация и кластеризация тем из обзорных статей (Шаг 1, задача 2 конвейера)."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
from tqdm import tqdm

try:
    import hdbscan
except ImportError:
    hdbscan = None


def load_local_topics(jsonl_path: str | Path) -> list[dict[str, Any]]:
    """Загружает все локальные темы из topics_raw.jsonl.

    Args:
        jsonl_path: Путь к JSONL файлу с локальными темами

    Returns:
        Список словарей с полями: article_id, section_id, local_topic_id,
        title, description, text_span, source_record_index

    Raises:
        FileNotFoundError: Если файл не найден
        json.JSONDecodeError: Если файл содержит невалидный JSON
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл с локальными темами не найден: {jsonl_path}")

    topics = []
    source_record_index = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Ошибка парсинга JSON на строке {line_num}: {e}")
                continue

            article_id = record.get("article_id", "")
            article_title = record.get("article_title", "")
            section_id = record.get("section_id", "")
            section_heading = record.get("section_heading", "")
            topics_list = record.get("topics", [])

            # Извлекаем каждую тему из списка
            for topic in topics_list:
                topic_entry = {
                    "article_id": article_id,
                    "article_title": article_title,
                    "section_id": section_id,
                    "section_heading": section_heading,
                    "local_topic_id": topic.get("local_topic_id", ""),
                    "title": topic.get("title", ""),
                    "description": topic.get("description", ""),
                    "text_span": topic.get("text_span", ""),
                    "source_record_index": source_record_index,
                }
                topics.append(topic_entry)

            source_record_index += 1

    logger.info(f"Загружено {len(topics)} локальных тем из {source_record_index} записей")
    return topics


def create_topic_texts(topics: list[dict[str, Any]]) -> list[str]:
    """Создаёт текстовые представления тем для векторизации.

    Формат: "{title}. {description}"

    Args:
        topics: Список словарей с темами (должны содержать поля title и description)

    Returns:
        Список строк для векторизации
    """
    texts = []
    for topic in topics:
        title = topic.get("title", "").strip()
        description = topic.get("description", "").strip()

        # Объединяем title и description
        if title and description:
            text = f"{title}. {description}"
        elif title:
            text = title
        elif description:
            text = description
        else:
            text = ""

        texts.append(text)

    return texts


def embed_topics(
    topic_texts: list[str],
    model_name: str | Path = "models/bge-m3",
    batch_size: int = 32,
    normalize: bool = True,
    device: str = "cpu",
    trace_id: str | None = None,
) -> np.ndarray:
    """Вычисляет эмбеддинги для списка текстов тем.

    Args:
        topic_texts: Список текстов для векторизации
        model_name: Путь к локальной модели или название модели SentenceTransformer
        batch_size: Размер батча для векторизации
        normalize: Нормализовать ли эмбеддинги (для косинусной близости)
        device: Устройство для вычислений ("cpu" или "cuda")
        trace_id: Идентификатор трейса для логирования

    Returns:
        Массив эмбеддингов shape (n_topics, embedding_dim)

    Raises:
        RuntimeError: Если модель не может быть загружена
    """
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""

    # Преобразуем путь к модели в абсолютный, если это локальный путь
    model_path = Path(model_name)
    if model_path.exists():
        model_name = str(model_path.absolute())
        logger.info(f"{trace_prefix}Использование локальной модели: {model_name}")
    else:
        logger.info(f"{trace_prefix}Загрузка модели из HuggingFace: {model_name}")

    logger.info(f"{trace_prefix}Загрузка модели эмбеддингов: {model_name} (device={device})")
    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception as e:
        raise RuntimeError(f"Не удалось загрузить модель {model_name}: {e}") from e

    logger.info(f"{trace_prefix}Векторизация {len(topic_texts)} тем (batch_size={batch_size})")

    # Векторизуем с прогресс-баром
    embeddings = model.encode(
        topic_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        device=device,
    )

    logger.info(f"{trace_prefix}Вычислено {len(embeddings)} эмбеддингов размерности {embeddings.shape[1]}")
    return embeddings


def cluster_topics(
    embeddings: np.ndarray,
    method: str = "hdbscan",
    n_clusters: int | None = None,
    min_cluster_size: int = 2,
    min_samples: int = 1,
    trace_id: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Кластеризует темы на основе их эмбеддингов.

    Args:
        embeddings: Массив эмбеддингов shape (n_topics, embedding_dim)
        method: Метод кластеризации ("hdbscan", "agglomerative", "kmeans")
        n_clusters: Количество кластеров (для agglomerative/kmeans)
        min_cluster_size: Минимальный размер кластера (для HDBSCAN)
        min_samples: Минимальное количество образцов в кластере (для HDBSCAN)
        trace_id: Идентификатор трейса для логирования

    Returns:
        Кортеж (labels, cluster_metadata):
        - labels: массив меток кластеров (-1 для шума в HDBSCAN)
        - cluster_metadata: словарь с метриками кластеров

    Raises:
        ValueError: Если указан неверный метод кластеризации
        RuntimeError: Если HDBSCAN запрошен, но не установлен
    """
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    n_topics = len(embeddings)

    logger.info(f"{trace_prefix}Кластеризация {n_topics} тем методом {method}")

    if method == "hdbscan":
        if hdbscan is None:
            raise RuntimeError("HDBSCAN не установлен. Установите: pip install hdbscan")

        # HDBSCAN не поддерживает 'cosine' напрямую, но для нормализованных эмбеддингов
        # евклидова метрика эквивалентна косинусной
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",  # Для нормализованных векторов эквивалентно cosine
        )
        labels = clusterer.fit_predict(embeddings)

        # Подсчитываем метрики
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels >= 0])  # Исключаем шум (-1)
        n_noise = np.sum(labels == -1)

        # Средний размер кластера
        cluster_sizes = [np.sum(labels == label) for label in unique_labels if label >= 0]
        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0.0

        # Средняя внутрикластерная дистанция
        intra_cluster_distances = []
        for label in unique_labels:
            if label >= 0:
                cluster_points = embeddings[labels == label]
                if len(cluster_points) > 1:
                    # Вычисляем попарные косинусные расстояния
                    centroid = np.mean(cluster_points, axis=0)
                    distances = 1 - np.dot(cluster_points, centroid)  # 1 - cosine similarity
                    intra_cluster_distances.extend(distances)

        avg_intra_cluster_distance = np.mean(intra_cluster_distances) if intra_cluster_distances else 0.0

        metadata = {
            "method": "hdbscan",
            "n_clusters": int(n_clusters),
            "n_noise": int(n_noise),
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "avg_cluster_size": float(avg_cluster_size),
            "avg_intra_cluster_distance": float(avg_intra_cluster_distance),
        }

    elif method == "agglomerative":
        if n_clusters is None:
            # Эвристика: примерно sqrt(n_topics / 2)
            n_clusters = max(2, int(np.sqrt(n_topics / 2)))
            logger.info(f"{trace_prefix}Автоматически выбрано количество кластеров: {n_clusters}")

        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="average",
            metric="cosine",
        )
        labels = clusterer.fit_predict(embeddings)

        # Подсчитываем метрики
        unique_labels = np.unique(labels)
        cluster_sizes = [np.sum(labels == label) for label in unique_labels]
        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0.0

        # Средняя внутрикластерная дистанция
        intra_cluster_distances = []
        for label in unique_labels:
            cluster_points = embeddings[labels == label]
            if len(cluster_points) > 1:
                centroid = np.mean(cluster_points, axis=0)
                distances = 1 - np.dot(cluster_points, centroid)
                intra_cluster_distances.extend(distances)

        avg_intra_cluster_distance = np.mean(intra_cluster_distances) if intra_cluster_distances else 0.0

        metadata = {
            "method": "agglomerative",
            "n_clusters": int(n_clusters),
            "n_noise": 0,
            "linkage": "average",
            "metric": "cosine",
            "avg_cluster_size": float(avg_cluster_size),
            "avg_intra_cluster_distance": float(avg_intra_cluster_distance),
        }

    elif method == "kmeans":
        if n_clusters is None:
            # Эвристика: примерно sqrt(n_topics / 2)
            n_clusters = max(2, int(np.sqrt(n_topics / 2)))
            logger.info(f"{trace_prefix}Автоматически выбрано количество кластеров: {n_clusters}")

        clusterer = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=10,
            random_state=42,
        )
        labels = clusterer.fit_predict(embeddings)

        # Подсчитываем метрики
        unique_labels = np.unique(labels)
        cluster_sizes = [np.sum(labels == label) for label in unique_labels]
        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0.0

        # Средняя внутрикластерная дистанция (расстояние до центроида)
        intra_cluster_distances = []
        for i, label in enumerate(unique_labels):
            cluster_points = embeddings[labels == label]
            if len(cluster_points) > 0:
                centroid = clusterer.cluster_centers_[i]
                distances = np.linalg.norm(cluster_points - centroid, axis=1)
                intra_cluster_distances.extend(distances)

        avg_intra_cluster_distance = np.mean(intra_cluster_distances) if intra_cluster_distances else 0.0

        metadata = {
            "method": "kmeans",
            "n_clusters": int(n_clusters),
            "n_noise": 0,
            "init": "k-means++",
            "avg_cluster_size": float(avg_cluster_size),
            "avg_intra_cluster_distance": float(avg_intra_cluster_distance),
        }

    else:
        raise ValueError(f"Неизвестный метод кластеризации: {method}. Доступны: hdbscan, agglomerative, kmeans")

    logger.info(
        f"{trace_prefix}Кластеризация завершена: {metadata['n_clusters']} кластеров, "
        f"шум: {metadata['n_noise']}, средний размер: {metadata['avg_cluster_size']:.2f}"
    )

    return labels, metadata


def cluster_topics_from_raw(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    embedding_model: str | Path = "models/bge-m3",
    embedding_device: str = "cpu",
    clustering_method: str = "hdbscan",
    clustering_params: dict[str, Any] | None = None,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Основная функция: загружает темы, векторизует и кластеризует.

    Args:
        input_jsonl: Путь к входному JSONL файлу с локальными темами
        output_jsonl: Путь к выходному JSONL файлу с кластерами
        embedding_model: Путь к локальной модели или название модели для эмбеддингов
        embedding_device: Устройство для вычисления эмбеддингов ("cpu" по умолчанию)
        clustering_method: Метод кластеризации ("hdbscan", "agglomerative", "kmeans")
        clustering_params: Дополнительные параметры для кластеризации
        trace_id: Идентификатор трейса для логирования

    Returns:
        Словарь со статистикой: total_topics, n_clusters, n_noise,
        avg_cluster_size, duration_seconds, started_at, completed_at
    """
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    started_at = datetime.now()

    logger.info(f"{trace_prefix}Начало кластеризации тем")
    logger.info(f"{trace_prefix}Входной файл: {input_jsonl}")
    logger.info(f"{trace_prefix}Выходной файл: {output_jsonl}")
    logger.info(f"{trace_prefix}Модель эмбеддингов: {embedding_model} (device={embedding_device})")
    logger.info(f"{trace_prefix}Метод кластеризации: {clustering_method}")

    # Загружаем локальные темы
    topics = load_local_topics(input_jsonl)
    total_topics = len(topics)

    if total_topics == 0:
        logger.warning(f"{trace_prefix}Не найдено тем для кластеризации")
        return {
            "total_topics": 0,
            "n_clusters": 0,
            "n_noise": 0,
            "avg_cluster_size": 0.0,
            "duration_seconds": 0.0,
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now().isoformat(),
        }

    # Создаём текстовые представления
    topic_texts = create_topic_texts(topics)

    # Векторизуем темы
    embeddings = embed_topics(
        topic_texts=topic_texts,
        model_name=embedding_model,
        batch_size=32,
        normalize=True,
        device=embedding_device,
        trace_id=trace_id,
    )

    # Подготавливаем параметры кластеризации
    if clustering_params is None:
        clustering_params = {}

    # Извлекаем параметры для выбранного метода
    if clustering_method == "hdbscan":
        min_cluster_size = clustering_params.get("min_cluster_size", 2)
        min_samples = clustering_params.get("min_samples", 1)
        n_clusters = None
    elif clustering_method in ("agglomerative", "kmeans"):
        n_clusters = clustering_params.get("n_clusters", None)
        min_cluster_size = 2
        min_samples = 1
    else:
        n_clusters = None
        min_cluster_size = 2
        min_samples = 1

    # Кластеризуем
    labels, cluster_metadata = cluster_topics(
        embeddings=embeddings,
        method=clustering_method,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        trace_id=trace_id,
    )

    # Сохраняем результаты
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"{trace_prefix}Сохранение результатов кластеризации в {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        # Группируем темы по кластерам
        unique_labels = np.unique(labels)
        cluster_id_counter = 0

        for label in unique_labels:
            if label == -1:
                # Точки шума - сохраняем отдельно
                noise_indices = np.where(labels == label)[0]
                for idx in noise_indices:
                    record = {
                        "cluster_id": None,
                        "topic_index": int(idx),
                        "topic_metadata": topics[idx],
                        "is_noise": True,
                        "clustered_at": datetime.now().isoformat(),
                        "trace_id": trace_id,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                # Кластер
                cluster_indices = np.where(labels == label)[0].tolist()
                cluster_id = f"CLUSTER_{cluster_id_counter:04d}"
                cluster_id_counter += 1

                # Метаданные тем в кластере
                topic_metadata_list = [topics[idx] for idx in cluster_indices]

                record = {
                    "cluster_id": cluster_id,
                    "topic_indices": cluster_indices,
                    "topic_metadata": topic_metadata_list,
                    "cluster_size": len(cluster_indices),
                    "is_noise": False,
                    "clustered_at": datetime.now().isoformat(),
                    "trace_id": trace_id,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    completed_at = datetime.now()
    duration_seconds = (completed_at - started_at).total_seconds()

    stats = {
        "total_topics": total_topics,
        "n_clusters": cluster_metadata["n_clusters"],
        "n_noise": cluster_metadata["n_noise"],
        "avg_cluster_size": cluster_metadata["avg_cluster_size"],
        "avg_intra_cluster_distance": cluster_metadata["avg_intra_cluster_distance"],
        "duration_seconds": duration_seconds,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
    }

    logger.info(f"{trace_prefix}Кластеризация завершена. Статистика: {stats}")
    return stats

