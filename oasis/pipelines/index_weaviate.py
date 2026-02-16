# Индексация датасетов в Weaviate. Вызывается из scripts/pipeline/step3_index_weaviate.

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import weaviate
from loguru import logger
from tqdm import tqdm


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _embed_texts(
    texts: list[str],
    model_name: str,
    batch_size: int = 32,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("sentence-transformers нужен для эмбеддингов. Установите: pip install sentence-transformers")
    model = SentenceTransformer(str(model_name), device="cpu")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings)


def _create_schema(client: weaviate.Client, dataset: str) -> tuple[str, str]:
    doc_class = f"Document_{dataset}"
    chunk_class = f"Chunk_{dataset}"
    if client.schema.exists(chunk_class):
        client.schema.delete_class(chunk_class)
    if client.schema.exists(doc_class):
        client.schema.delete_class(doc_class)

    client.schema.create_class({
        "class": doc_class,
        "description": f"Метаданные документов датасета {dataset}",
        "vectorizer": "none",
        "properties": [
            {"name": "paper_id", "dataType": ["text"]},
            {"name": "title", "dataType": ["text"]},
            {"name": "year", "dataType": ["int"]},
            {"name": "doi", "dataType": ["text"]},
            {"name": "arxiv_id", "dataType": ["text"]},
            {"name": "venue", "dataType": ["text"]},
            {"name": "source", "dataType": ["text"]},
        ],
    })
    client.schema.create_class({
        "class": chunk_class,
        "description": f"Чанки текста датасета {dataset}",
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {"distance": "cosine"},
        "invertedIndexConfig": {
            "bm25": {"b": 0.75, "k1": 1.2},
            "stopwords": {"preset": "en"},
        },
        "properties": [
            {"name": "chunk_id", "dataType": ["text"]},
            {"name": "paper_id", "dataType": ["text"]},
            {"name": "text", "dataType": ["text"], "indexInverted": True},
            {"name": "chunk_index", "dataType": ["int"]},
            {"name": "doi", "dataType": ["text"]},
            {"name": "year", "dataType": ["int"]},
            {"name": "source", "dataType": ["text"]},
        ],
    })
    client.schema.property.create(
        chunk_class,
        {"name": "parent", "dataType": [doc_class], "description": "Ссылка на документ"},
    )
    return doc_class, chunk_class


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    step = chunk_size - overlap
    chunks: list[str] = []
    index = 0
    while index < len(words):
        chunk_words = words[index : index + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        index += step
    return chunks or [text]


def _load_references(dataset_dir: Path) -> dict[str, dict[str, str | int | None]]:
    refs_path = dataset_dir / "references.csv"
    if not refs_path.exists():
        return {}
    refs: dict[str, dict[str, str | int | None]] = {}
    with refs_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paper_id = row.get("ref_id") or Path(row.get("pdf_url", "")).stem
            if not paper_id:
                continue
            refs[paper_id] = {
                "title": row.get("title", ""),
                "year": int(row["year"]) if row.get("year") else None,
                "doi": row.get("doi", ""),
                "arxiv_id": row.get("arxiv_id", ""),
                "venue": row.get("venue", ""),
                "source": dataset_dir.name,
            }
    return refs


def index_dataset(
    dataset: str,
    weaviate_url: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    data_root: Path,
) -> int:
    """Индексирует один датасет в Weaviate. Возвращает 0 при успехе, 1 при ошибке."""
    dataset_dir = data_root / dataset
    rag_dir = dataset_dir / "rag"
    if not rag_dir.exists():
        logger.warning("Нет директории rag для датасета {}: {}", dataset, rag_dir)
        return 1
    rag_files = sorted(rag_dir.glob("*.txt"))
    if not rag_files:
        logger.warning("Нет .txt файлов в {} для датасета {}", rag_dir, dataset)
        return 1

    references = _load_references(dataset_dir)
    client = weaviate.Client(weaviate_url)
    doc_class, chunk_class = _create_schema(client, dataset)
    doc_uuids: dict[str, str] = {}
    chunk_counts: dict[str, int] = {}

    for rag_file in tqdm(rag_files, desc="Индексация %s" % dataset, unit="док"):
        paper_id = rag_file.stem
        try:
            text = _read_text(rag_file)
        except Exception as e:
            logger.warning("Не удалось прочитать {}: {}", rag_file, e)
            continue
        meta = references.get(paper_id, {})

        with client.batch as batch:
            batch.batch_size = batch_size
            doc_uuid = batch.add_data_object(
                {
                    "paper_id": paper_id,
                    "title": meta.get("title", paper_id),
                    "year": meta.get("year"),
                    "doi": meta.get("doi", ""),
                    "arxiv_id": meta.get("arxiv_id", ""),
                    "venue": meta.get("venue", ""),
                    "source": meta.get("source", dataset),
                },
                doc_class,
            )
            batch.flush()
        if doc_uuid:
            doc_uuids[paper_id] = doc_uuid

        chunks = _chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
        if not chunks:
            chunk_counts[paper_id] = 0
            continue
        embeddings = _embed_texts(chunks, embedding_model, batch_size=32)
        with client.batch as batch:
            batch.batch_size = batch_size
            for chunk_index, (chunk_value, vector) in enumerate(zip(chunks, embeddings)):
                vec = vector.tolist() if hasattr(vector, "tolist") else list(vector)
                batch.add_data_object(
                    {
                        "chunk_id": f"{paper_id}::chunk_{chunk_index:03d}",
                        "paper_id": paper_id,
                        "text": chunk_value,
                        "chunk_index": chunk_index,
                        "doi": meta.get("doi", ""),
                        "year": meta.get("year"),
                        "source": meta.get("source", dataset),
                    },
                    chunk_class,
                    vector=vec,
                )
            batch.flush()
        chunk_counts[paper_id] = len(chunks)

    ingest_index = dataset_dir / "ingest_index.csv"
    ingest_index.parent.mkdir(parents=True, exist_ok=True)
    with ingest_index.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["paper_id", "document_uuid", "n_chunks", "dataset", "schema_version"])
        for pid, doc_uuid in doc_uuids.items():
            w.writerow([pid, doc_uuid, chunk_counts.get(pid, 0), dataset, "1.0"])

    logger.info("Готово {}: документов {}, чанков {}", dataset, len(doc_uuids), sum(chunk_counts.values()))
    return 0


def run_all(
    data_root: Path,
    dataset_names: list[str],
    weaviate_url: str,
    embedding_model: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    batch_size: int = 100,
) -> int:
    """Индексирует все переданные датасеты в Weaviate. Возвращает 0 если все ок, иначе 1."""
    failed = 0
    for name in dataset_names:
        if index_dataset(
            dataset=name,
            weaviate_url=weaviate_url,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
            data_root=data_root,
        ) != 0:
            failed += 1
    return 1 if failed else 0
