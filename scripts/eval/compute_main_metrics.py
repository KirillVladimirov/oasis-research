#!/usr/bin/env python3
"""Вычисляет основные метрики качества для baseline LLM/RAG и ontology+KG ответов.

Этот скрипт работает с выходными данными из step8_generate_answers.py:
- baseline_answers.jsonl
- ontology_answers.jsonl

Структура данных step8:
- answer_text: текст ответа
- context.stats: метрики контекста (top_k, context_tokens_est, n_triples_selected, etc.)
- timings_ms: тайминги этапов (retrieval, llm, mapping, extraction)

Также поддерживает старый формат данных для обратной совместимости.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from rdflib import Graph
from rdflib.namespace import OWL, RDF, RDFS

load_dotenv()

try:
    import fitz  # type: ignore
except ImportError as exc:  # pragma: no cover - окружение без PyMuPDF
    raise RuntimeError("PyMuPDF (fitz) is required to build the reference answers") from exc


DEFAULT_CONFIG_PATH = Path("configs/compute_main_metrics.yaml")


# ============================================================================
# Вспомогательные функции для вызова LLM
# ============================================================================


def _create_openai_client(timeout: float | None = None) -> OpenAI:
    """Создаёт клиента OpenAI-совместимого API (Gemini через aitunnel)."""

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE", "https://api.aitunnel.ru/v1/")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env variable is required")

    if timeout is None:
        timeout_env = os.getenv("OPENAI_TIMEOUT")
        if timeout_env:
            try:
                timeout = float(timeout_env)
            except ValueError:
                logger.warning("Invalid OPENAI_TIMEOUT value: %s", timeout_env)
                timeout = None

    client_kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url}
    if timeout is not None:
        client_kwargs["timeout"] = timeout

    return OpenAI(**client_kwargs)


def call_llm_json_with_retries(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    trace_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Выполняет chat.completions с требованием JSON-ответа и повторными попытками."""

    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )

            choice = response.choices[0]
            content = choice.message.content
            finish_reason = choice.finish_reason
            if not content:
                raise ValueError("Empty LLM response")

            parsed = json.loads(content)
            # Некоторые модели могут вернуть список вместо объекта, несмотря на json_object
            if isinstance(parsed, list):
                if len(parsed) == 0:
                    raise ValueError("LLM returned empty list instead of JSON object")
                # Если список содержит один элемент-словарь, используем его
                if len(parsed) == 1 and isinstance(parsed[0], dict):
                    logger.warning(
                        "%sLLM returned list with single dict, using it as parsed result",
                        trace_prefix,
                    )
                    parsed = parsed[0]
                else:
                    raise ValueError(
                        f"LLM returned list instead of JSON object: {parsed}. "
                        "Expected a single JSON object with fields like 'reference_answer_text'."
                    )
            if not isinstance(parsed, dict):
                raise ValueError(f"LLM returned non-dict, non-list JSON: {type(parsed)}")
            
            raw_payload = response.model_dump()
            raw_payload["finish_reason"] = finish_reason
            return parsed, raw_payload

        except Exception as exc:  # pragma: no cover - сетевые ошибки
            if attempt >= max_retries - 1:
                raise
            delay = retry_delay * (2**attempt)
            logger.warning(
                "%sLLM call failed (attempt %s/%s): %s. Retrying in %.1fs",
                trace_prefix,
                attempt + 1,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError("LLM retry loop terminated unexpectedly")  # pragma: no cover


# ============================================================================
# Загрузка входных данных и их выравнивание
# ============================================================================


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Загружает JSONL в список словарей."""

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL entry in {path}: {exc}") from exc
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    """Сохраняет записи в JSONL файл."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_cqs(path: Path) -> pd.DataFrame:
    """Загружает файл с каноническими вопросами."""

    records = read_jsonl(path)
    data = []
    for rec in records:
        cq_id = rec.get("cq_id")
        question = rec.get("text") or rec.get("question")
        if not cq_id or not question:
            raise ValueError(f"CQ record must contain cq_id and text/question: {rec}")
        data.append(
            {
                "cq_id": str(cq_id),
                "question": str(question).strip(),
            }
        )
    return pd.DataFrame(data)


def _extract_answer_text(record: dict[str, Any]) -> str:
    """Извлекает текст ответа независимо от структуры записи."""

    if "answer_text" in record and isinstance(record["answer_text"], str):
        return record["answer_text"].strip()

    for key in ("answer", "final_answer", "response", "model_answer"):
        value = record.get(key)
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            inner = value.get("text") or value.get("content")
            if isinstance(inner, str):
                return inner.strip()

    return (record.get("text") or "").strip()


def load_answers(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Загружает ответы и собирает базовую статистику."""

    records = read_jsonl(path)
    rows: list[dict[str, Any]] = []
    id_counter: Counter[str] = Counter()
    empty_ids: list[str] = []

    for rec in records:
        cq_id = str(rec.get("cq_id"))
        if not cq_id:
            raise ValueError(f"Answer record without cq_id in {path}: {rec}")
        text = _extract_answer_text(rec)
        if not text:
            empty_ids.append(cq_id)

        rows.append(
            {
                "cq_id": cq_id,
                "answer_text": text,
                "answer_record": rec,
            }
        )
        id_counter[cq_id] += 1

    duplicates = [cid for cid, count in id_counter.items() if count > 1]

    stats = {
        "total": len(rows),
        "unique": len(id_counter),
        "duplicates": sorted(duplicates),
        "empty_answer_ids": sorted(set(empty_ids)),
    }

    return pd.DataFrame(rows), stats


def align_answers(
    cqs_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    onto_df: pd.DataFrame,
    baseline_stats: dict[str, Any],
    onto_stats: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Выравнивает вопросы и ответы и формирует sanity-отчёт."""

    base_df = baseline_df.rename(
        columns={
            "answer_text": "baseline_answer_text",
            "answer_record": "baseline_answer_record",
        }
    )
    onto_df = onto_df.rename(
        columns={
            "answer_text": "ontology_answer_text",
            "answer_record": "ontology_answer_record",
        }
    )

    df = (
        cqs_df.merge(base_df, on="cq_id", how="left")
        .merge(onto_df, on="cq_id", how="left")
        .sort_values("cq_id")
        .reset_index(drop=True)
    )

    cq_ids = set(df["cq_id"])
    base_ids = set(base_df["cq_id"]) if not base_df.empty else set()
    onto_ids = set(onto_df["cq_id"]) if not onto_df.empty else set()

    sanity_report = {
        "total_cqs": len(df),
        "baseline_total": baseline_stats.get("total", 0),
        "ontology_total": onto_stats.get("total", 0),
        "baseline_missing_ids": sorted(cq_ids - base_ids),
        "ontology_missing_ids": sorted(cq_ids - onto_ids),
        "baseline_duplicates": baseline_stats.get("duplicates", []),
        "ontology_duplicates": onto_stats.get("duplicates", []),
        "baseline_empty_answer_ids": baseline_stats.get("empty_answer_ids", []),
        "ontology_empty_answer_ids": onto_stats.get("empty_answer_ids", []),
    }

    return df, sanity_report


# ============================================================================
# Построение эталонных ответов
# ============================================================================


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Извлекает текст из PDF обзора."""

    if not pdf_path.exists():
        raise FileNotFoundError(f"Review PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    try:
        pages = [page.get_text() for page in doc]
    finally:
        doc.close()

    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[dict[str, Any]]:
    """Разбивает текст на пересекающиеся чанки заданной длины."""

    cleaned = " ".join(text.split())
    chunks: list[dict[str, Any]] = []
    start = 0
    chunk_id = 1
    text_len = len(cleaned)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text_value = cleaned[start:end]
        chunks.append(
            {
                "chunk_id": f"review_chunk_{chunk_id:04d}",
                "text": chunk_text_value,
                "start_char": start,
                "end_char": end,
            }
        )
        chunk_id += 1
        if end == text_len:
            break
        start = max(end - overlap, 0)
    return chunks


class ReviewChunkIndexer:
    """Локальный индекс по чанкам обзора с BM25 и эмбеддингами."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        bm25_cfg: dict[str, Any],
        embedding_cfg: dict[str, Any],
    ) -> None:
        self.chunks = chunks
        self.k1 = float(bm25_cfg.get("k1", 1.5))
        self.b = float(bm25_cfg.get("b", 0.75))

        from sklearn.feature_extraction.text import CountVectorizer  # локальный импорт

        texts = [chunk["text"] for chunk in chunks]
        self.vectorizer = CountVectorizer(stop_words="english")
        self.doc_term_matrix = self.vectorizer.fit_transform(texts)
        self.doc_lengths = np.asarray(self.doc_term_matrix.sum(axis=1)).ravel()
        self.avg_doc_length = float(self.doc_lengths.mean()) if len(self.doc_lengths) else 1.0

        # Вычисляем статистики idf для BM25
        df_counts = np.asarray((self.doc_term_matrix > 0).sum(axis=0)).ravel()
        total_docs = len(chunks)
        self.idf = np.log((total_docs - df_counts + 0.5) / (df_counts + 0.5) + 1.0)

        self.embedding_enabled = bool(embedding_cfg.get("enabled", False))
        self.embedding_model_name = embedding_cfg.get("model_name")
        self.embedding_device = embedding_cfg.get("device", "cpu")

        self.embedding_model = None
        self.chunk_embeddings = None
        if self.embedding_enabled and self.embedding_model_name:
            self._load_embeddings(texts)
        else:
            self.embedding_enabled = False

    def _load_embeddings(self, texts: list[str]) -> None:
        """Создаёт эмбеддинги чанков с помощью SentenceTransformer."""

        try:
            from sentence_transformers import SentenceTransformer  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover - опциональная зависимость
            raise RuntimeError("sentence-transformers is required for embedding retrieval") from exc

        model = SentenceTransformer(self.embedding_model_name, device=self.embedding_device)
        self.chunk_embeddings = model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.embedding_model = model

    def _bm25_scores(self, query: str) -> np.ndarray:
        """Вычисляет BM25-скор для каждого чанка."""

        query_vec = self.vectorizer.transform([query])
        scores = np.zeros(len(self.chunks), dtype=float)
        for idx, term_index in enumerate(query_vec.indices):
            term_freqs = self.doc_term_matrix[:, term_index].toarray().ravel()
            idf_value = self.idf[term_index]
            numerator = term_freqs * (self.k1 + 1)
            denominator = term_freqs + self.k1 * (1 - self.b + self.b * (self.doc_lengths / self.avg_doc_length))
            scores += idf_value * (numerator / (denominator + 1e-9))
        return scores

    def _embedding_scores(self, query: str) -> np.ndarray:
        """Вычисляет косинусные сходства на основе эмбеддингов."""

        if not self.embedding_model or self.chunk_embeddings is None:
            return np.zeros(len(self.chunks), dtype=float)
        query_emb = self.embedding_model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        return self.chunk_embeddings @ query_emb

    @staticmethod
    def _normalize(scores: np.ndarray) -> np.ndarray:
        if not scores.size:
            return scores
        min_val = scores.min()
        max_val = scores.max()
        if math.isclose(max_val, min_val):
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

    def retrieve(self, query: str, top_k: int, fusion_alpha: float) -> list[dict[str, Any]]:
        """Возвращает top-k чанков, отсортированных по комбинированному скору."""

        bm25_scores = self._bm25_scores(query)
        embedding_scores = self._embedding_scores(query) if self.embedding_enabled else np.zeros_like(bm25_scores)

        bm25_norm = self._normalize(bm25_scores)
        embedding_norm = self._normalize(embedding_scores)

        combined = (1 - fusion_alpha) * bm25_norm + fusion_alpha * embedding_norm if self.embedding_enabled else bm25_norm
        ranked_idx = np.argsort(combined)[::-1][:top_k]

        return [
            {
                "chunk_id": self.chunks[i]["chunk_id"],
                "text": self.chunks[i]["text"],
                "start_char": self.chunks[i]["start_char"],
                "end_char": self.chunks[i]["end_char"],
                "bm25_score": float(bm25_scores[i]),
                "embedding_score": float(embedding_scores[i]),
                "fused_score": float(combined[i]),
            }
            for i in ranked_idx
        ]


@dataclass
class ReferenceBuilderConfig:
    review_pdf_path: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    fusion_alpha: float
    llm: dict[str, Any]
    bm25: dict[str, Any]
    embedding: dict[str, Any]
    max_retries: int


class ReferenceAnswerBuilder:
    """Генерирует эталонные ответы на основе review.pdf."""

    def __init__(self, config: ReferenceBuilderConfig) -> None:
        self.config = config
        self.client = _create_openai_client()
        self.chunks: list[dict[str, Any]] | None = None
        self.indexer: ReviewChunkIndexer | None = None

    def _ensure_index(self) -> None:
        if self.indexer:
            return
        logger.info("Initializing local review index")
        review_text = extract_text_from_pdf(self.config.review_pdf_path)
        self.chunks = chunk_text(
            review_text,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        if not self.chunks:
            raise RuntimeError("Failed to create review chunks")
        self.indexer = ReviewChunkIndexer(
            self.chunks,
            bm25_cfg=self.config.bm25,
            embedding_cfg=self.config.embedding,
        )

    def build_for_question(self, cq_id: str, question: str) -> dict[str, Any]:
        """Строит эталонный ответ для одного вопроса CQ."""

        self._ensure_index()
        assert self.indexer is not None  # защитная проверка для mypy

        retrieved_chunks = self.indexer.retrieve(
            question,
            top_k=self.config.top_k,
            fusion_alpha=self.config.fusion_alpha,
        )

        context_blocks = []
        for chunk in retrieved_chunks:
            context_blocks.append(
                f"[{chunk['chunk_id']}] score={chunk['fused_score']:.3f} chars={chunk['start_char']}-{chunk['end_char']}\n{chunk['text']}"
            )

        system_prompt = (
            "You are generating a reference answer based solely on the supplied review excerpt. "
            "Use only the provided chunks, never add new facts. "
            "Return JSON with key reference_answer_text (string) and optional notes."
        )

        user_prompt = (
            f"Question (cq_id={cq_id}): {question}\n\n"
            "Relevant review chunks:\n"
            + "\n\n".join(context_blocks)
        )

        parsed, raw = call_llm_json_with_retries(
            self.client,
            model=self.config.llm["model"],
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(self.config.llm.get("temperature", 0.0)),
            max_tokens=int(self.config.llm.get("max_tokens", 800)),
            max_retries=self.config.max_retries,
            retry_delay=float(self.config.llm.get("retry_delay", 2.0)),
            trace_id=f"reference::{cq_id}",
        )

        reference_answer = str(parsed.get("reference_answer_text") or "").strip()
        if not reference_answer:
            raise ValueError(f"Reference answer is empty for {cq_id}")

        return {
            "cq_id": cq_id,
            "question": question,
            "reference_answer_text": reference_answer,
            "review_evidence_ids": [chunk["chunk_id"] for chunk in retrieved_chunks],
            "review_retrieval_stats": {
                "top_k": len(retrieved_chunks),
                "mean_fused_score": float(np.mean([chunk["fused_score"] for chunk in retrieved_chunks])) if retrieved_chunks else 0.0,
                "bm25_mean": float(np.mean([chunk["bm25_score"] for chunk in retrieved_chunks])) if retrieved_chunks else 0.0,
                "embedding_mean": float(np.mean([chunk["embedding_score"] for chunk in retrieved_chunks])) if retrieved_chunks else 0.0,
                "chunks": retrieved_chunks,
            },
            "llm_model": self.config.llm["model"],
            "llm_raw": raw,
        }


# ============================================================================
# Оценка ответов LLM-судьёй
# ============================================================================


@dataclass
class JudgeConfig:
    model: str
    temperature: float
    max_tokens: int
    max_retries: int
    retry_delay: float
    scale: dict[str, Any]


class LLMJudge:
    """LLM-судья, сравнивающий baseline и ontology ответы относительно эталона."""

    def __init__(self, config: JudgeConfig) -> None:
        self.config = config
        self.client = _create_openai_client()

    def _build_prompt(
        self,
        cq_id: str,
        question: str,
        reference_answer: str,
        baseline_answer: str,
        onto_answer: str,
    ) -> tuple[str, str]:
        min_score = self.config.scale.get("min_score", 1)
        max_score = self.config.scale.get("max_score", 5)

        system_prompt = (
            "You are an answer quality judge. Score each answer independently against the reference. "
            "Return JSON with fields score_baseline, score_onto, winner, and optional notes. "
            f"Scores must be integers from {min_score} to {max_score}. winner in ['baseline','ontology','tie']."
        )

        def _format_answer(label: str, text: str) -> str:
            cleaned = text.strip() if text else ""
            return f"{label}:\n{cleaned or '[no answer provided]'}"

        user_prompt = (
            f"cq_id: {cq_id}\n"
            f"Question: {question}\n\n"
            f"Reference answer:\n{reference_answer}\n\n"
            f"{_format_answer('Baseline answer', baseline_answer)}\n\n"
            f"{_format_answer('Ontology answer', onto_answer)}\n\n"
            "Judge each answer strictly against the reference answer. "
            "If an answer is empty, assign the minimum score. "
            "Briefly justify the winner choice in notes."
        )

        return system_prompt, user_prompt

    def judge_row(
        self,
        cq_id: str,
        question: str,
        reference_answer: str,
        baseline_answer: str,
        onto_answer: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        system_prompt, user_prompt = self._build_prompt(
            cq_id,
            question,
            reference_answer,
            baseline_answer,
            onto_answer,
        )

        parsed, raw = call_llm_json_with_retries(
            self.client,
            model=self.config.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            max_retries=self.config.max_retries,
            retry_delay=self.config.retry_delay,
            trace_id=f"judge::{cq_id}",
        )

        return parsed, {"cq_id": cq_id, "request": {"question": question}, "response": raw}


# ============================================================================
# Метрики
# ============================================================================


def compute_quality_metrics(df: pd.DataFrame, good_threshold: float | None = None) -> dict[str, Any]:
    """Считает агрегаты по оценкам baseline/ontology."""

    def _stats(series: pd.Series) -> dict[str, Any]:
        values = series.dropna().astype(float).tolist()
        if not values:
            return {}
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=0)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    df = df.copy()
    df["delta"] = df["score_onto"] - df["score_baseline"]
    df["outcome"] = df["delta"].apply(lambda x: "win" if x > 0 else ("loss" if x < 0 else "tie"))

    quality = {
        "score_baseline": _stats(df["score_baseline"]),
        "score_onto": _stats(df["score_onto"]),
        "delta": _stats(df["delta"]),
        "win_rate": float((df["outcome"] == "win").mean()),
        "tie_rate": float((df["outcome"] == "tie").mean()),
        "loss_rate": float((df["outcome"] == "loss").mean()),
    }

    if good_threshold is not None:
        for column in ("score_baseline", "score_onto"):
            mask = df[column] >= good_threshold
            quality.setdefault("good_answer_rate", {})[column] = float(mask.mean()) if not mask.empty else math.nan

    return quality


def _extract_nested(value: dict[str, Any] | None, field: str | None) -> Any:
    if value is None or not field:
        return None
    current: Any = value
    for part in field.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def compute_rag_metrics(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Собирает per-CQ и агрегированные метрики RAG."""

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        payload = row.get("baseline_answer_record") or {}
        rag_topk = _extract_nested(payload, config.get("topk_field"))
        context_tokens = _extract_nested(payload, config.get("context_tokens_field"))
        latency_retrieval = _extract_nested(payload, "timings_ms.retrieval")
        latency_llm = _extract_nested(payload, "timings_ms.llm")
        
        # Суммируем latency если доступны оба компонента
        if latency_retrieval is not None and latency_llm is not None:
            latency = float(latency_retrieval) + float(latency_llm)
        else:
            latency = _extract_nested(payload, config.get("latency_field"))
            if latency is None and latency_retrieval is not None:
                latency = latency_retrieval

        # Fallback: если topk не найден, пытаемся извлечь из context.items
        if rag_topk is None and config.get("fallback_context_fields"):
            for field in config["fallback_context_fields"]:
                ctx = _extract_nested(payload, field)
                if isinstance(ctx, list):
                    rag_topk = len(ctx)
                    break
                # Также проверяем context.items напрямую
                if field == "context.items" and isinstance(ctx, list):
                    rag_topk = len(ctx)
                    break

        rows.append(
            {
                "cq_id": row["cq_id"],
                "rag_topk": rag_topk,
                "rag_context_tokens": context_tokens,
                "rag_latency_ms": latency,
            }
        )

    rag_df = pd.DataFrame(rows)
    summary = {}
    for column in ("rag_topk", "rag_context_tokens", "rag_latency_ms"):
        values = pd.to_numeric(rag_df[column], errors="coerce").dropna()
        if values.empty:
            continue
        summary[column] = {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    return rag_df, summary


def compute_onto_context_metrics(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Собирает per-CQ метрики использования онтологии и KG."""

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        payload = row.get("ontology_answer_record") or {}
        facts_count = _extract_nested(payload, config.get("facts_count_field"))
        tokens = _extract_nested(payload, config.get("tokens_field"))
        terms_count = _extract_nested(payload, config.get("ontology_terms_field"))
        
        # Суммируем latency из всех компонентов ontology pipeline
        latency_mapping = _extract_nested(payload, "timings_ms.mapping")
        latency_extraction = _extract_nested(payload, "timings_ms.extraction")
        latency_llm = _extract_nested(payload, "timings_ms.llm")
        
        if latency_mapping is not None and latency_extraction is not None and latency_llm is not None:
            latency = float(latency_mapping) + float(latency_extraction) + float(latency_llm)
        else:
            latency = _extract_nested(payload, config.get("latency_field"))
            if latency is None and latency_mapping is not None:
                latency = latency_mapping

        # Fallback: если facts_count не найден, пытаемся извлечь из context.items.facts
        if facts_count is None and config.get("facts_list_fields"):
            for field in config["facts_list_fields"]:
                ctx = _extract_nested(payload, field)
                if isinstance(ctx, list):
                    facts_count = len(ctx)
                    break

        rows.append(
            {
                "cq_id": row["cq_id"],
                "n_kg_facts_in_context": facts_count,
                "kg_context_tokens": tokens,
                "n_ontology_terms_in_context": terms_count,
                "onto_retrieval_latency_ms": latency,
            }
        )

    onto_df = pd.DataFrame(rows)
    summary: dict[str, Any] = {}

    for column in (
        "n_kg_facts_in_context",
        "kg_context_tokens",
        "n_ontology_terms_in_context",
        "onto_retrieval_latency_ms",
    ):
        values = pd.to_numeric(onto_df[column], errors="coerce").dropna()
        if values.empty:
            continue
        summary[column] = {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    if not onto_df.empty:
        facts_numeric = pd.to_numeric(onto_df["n_kg_facts_in_context"], errors="coerce").fillna(0)
        summary["kg_context_non_empty_rate"] = float((facts_numeric > 0).mean())

    return onto_df, summary


def compute_structural_metrics(tbox_path: Path, kg_path: Path) -> dict[str, Any]:
    """Вычисляет структурные метрики онтологии и графа знаний."""

    if not tbox_path.exists():
        raise FileNotFoundError(f"TBox file not found: {tbox_path}")
    if not kg_path.exists():
        raise FileNotFoundError(f"KG file not found: {kg_path}")

    tbox_graph = Graph()
    tbox_graph.parse(tbox_path)

    kg_graph = Graph()
    kg_graph.parse(kg_path)

    class_nodes = list(tbox_graph.subjects(RDF.type, OWL.Class))
    classes = {str(node) for node in class_nodes}
    object_props = {str(s) for s in tbox_graph.subjects(RDF.type, OWL.ObjectProperty)}
    data_props = {str(s) for s in tbox_graph.subjects(RDF.type, OWL.DatatypeProperty)}

    classes_with_labels = {
        str(node)
        for node in class_nodes
        if any(tbox_graph.objects(node, RDFS.label))
    }

    tbox_metrics = {
        "n_classes": len(classes),
        "n_object_properties": len(object_props),
        "n_data_properties": len(data_props),
        "n_classes_with_label": len(classes_with_labels),
    }

    tbox_entities = {str(cls) for cls in tbox_graph.subjects()}
    tbox_entities.update(classes)
    tbox_entities.update(object_props)
    tbox_entities.update(data_props)

    individuals = {str(s) for s in kg_graph.subjects(RDF.type, OWL.NamedIndividual)}
    if not individuals:
        individuals = {
            str(s)
            for s, _, _ in kg_graph.triples((None, RDF.type, None))
            if str(s) not in tbox_entities
        }

    abox_triples = [
        (s, p, o)
        for s, p, o in kg_graph
        if str(s) not in tbox_entities
    ]

    used_classes = {
        str(o)
        for s, p, o in kg_graph.triples((None, RDF.type, None))
        if str(s) not in tbox_entities
    }

    used_props = {str(p) for _, p, _ in abox_triples}

    kg_metrics = {
        "n_individuals": len(individuals),
        "n_assertion_triples": len(abox_triples),
        "n_used_classes_in_kb": len(used_classes),
        "n_used_properties_in_kb": len(used_props),
    }

    return {"ontology_tbox": tbox_metrics, "knowledge_graph": kg_metrics}


# ============================================================================
# Формирование отчётов
# ============================================================================


def render_quality_markdown(quality: dict[str, Any]) -> str:
    lines = ["# Aggregate Answer Quality Metrics", "", "| Metric | Baseline | Ontology |", "| --- | --- | --- |"]
    for metric in ("mean", "median", "std"):
        baseline_value = quality.get("score_baseline", {}).get(metric)
        onto_value = quality.get("score_onto", {}).get(metric)
        lines.append(
            f"| {metric} | {baseline_value:.3f} | {onto_value:.3f} |"
            if baseline_value is not None and onto_value is not None
            else f"| {metric} | NA | NA |"
        )

    lines.append("")
    lines.append(
        f"- Win/Tie/Loss: {quality.get('win_rate', 0):.3f} / {quality.get('tie_rate', 0):.3f} / {quality.get('loss_rate', 0):.3f}"
    )
    if "good_answer_rate" in quality:
        gar = quality["good_answer_rate"]
        baseline_good = gar.get("score_baseline")
        onto_good = gar.get("score_onto")
        baseline_str = f"{baseline_good:.3f}" if baseline_good is not None else "NA"
        onto_str = f"{onto_good:.3f}" if onto_good is not None else "NA"
        lines.append(f"- P(score >= T): baseline={baseline_str}, ontology={onto_str}")
    return "\n".join(lines)


def render_summary_markdown(
    sanity_report: dict[str, Any],
    quality: dict[str, Any],
    rag_summary: dict[str, Any],
    onto_context_summary: dict[str, Any],
    structural_metrics: dict[str, Any],
) -> str:
    lines = ["# Main Metrics Summary", ""]
    lines.append("## Coverage sanity")
    lines.append(f"- Total CQs: {sanity_report.get('total_cqs', 0)}")
    lines.append(f"- Baseline missing: {len(sanity_report.get('baseline_missing_ids', []))}")
    lines.append(f"- Ontology missing: {len(sanity_report.get('ontology_missing_ids', []))}")
    lines.append("")
    lines.append("## Quality scores")
    lines.append(
        f"- Baseline mean score: {quality.get('score_baseline', {}).get('mean'):.3f}"
        if quality.get("score_baseline", {}).get("mean") is not None
        else "- Baseline mean score: NA"
    )
    lines.append(
        f"- Ontology mean score: {quality.get('score_onto', {}).get('mean'):.3f}"
        if quality.get("score_onto", {}).get("mean") is not None
        else "- Ontology mean score: NA"
    )
    lines.append(
        f"- Win/Tie/Loss: {quality.get('win_rate', 0):.3f}/{quality.get('tie_rate', 0):.3f}/{quality.get('loss_rate', 0):.3f}"
    )
    lines.append("")
    lines.append("## Baseline RAG telemetry")
    if rag_summary:
        for key, stats in rag_summary.items():
            lines.append(f"- {key}: mean={stats['mean']:.2f}, median={stats['median']:.2f}")
    else:
        lines.append("- no telemetry found")
    lines.append("")
    lines.append("## Ontology context usage")
    if onto_context_summary:
        for key, stats in onto_context_summary.items():
            if isinstance(stats, dict):
                lines.append(f"- {key}: mean={stats['mean']:.2f}, median={stats['median']:.2f}")
            else:
                lines.append(f"- {key}: {stats:.3f}")
    else:
        lines.append("- no ontology context telemetry")
    lines.append("")
    lines.append("## Structural stats")
    lines.append(f"- Ontology classes: {structural_metrics.get('ontology_tbox', {}).get('n_classes', 0)}")
    lines.append(f"- KG individuals: {structural_metrics.get('knowledge_graph', {}).get('n_individuals', 0)}")
    return "\n".join(lines)


# ============================================================================
# Точка входа
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute baseline vs ontology+KG metrics for CQ answers.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    inputs_cfg = config["inputs"]
    outputs_cfg = config["outputs"]

    cqs_path = Path(inputs_cfg["cqs_path"])
    baseline_path = Path(inputs_cfg["answers_baseline_path"])
    onto_path = Path(inputs_cfg["answers_onto_path"])
    review_pdf_path = Path(inputs_cfg["review_pdf_path"])
    tbox_path = Path(inputs_cfg["ontology_tbox_path"])
    kg_path = Path(inputs_cfg["kg_owl_path"])

    logger.info("Loading CQs and answer sets")
    cqs_df = load_cqs(cqs_path)
    baseline_df, baseline_stats = load_answers(baseline_path)
    onto_df, onto_stats = load_answers(onto_path)

    merged_df, sanity_report = align_answers(cqs_df, baseline_df, onto_df, baseline_stats, onto_stats)
    write_jsonl([sanity_report], Path(outputs_cfg["sanity_report_path"]))

    logger.info("Building reference answers from review.pdf")
    reference_path = Path(outputs_cfg["reference_answers_path"])
    reference_records: list[dict[str, Any]]
    if reference_path.exists() and config.get("reference_builder", {}).get("reuse_cached", True):
        reference_records = read_jsonl(reference_path)
    else:
        ref_builder_cfg = ReferenceBuilderConfig(
            review_pdf_path=review_pdf_path,
            chunk_size=int(config["reference_builder"]["chunk_size"]),
            chunk_overlap=int(config["reference_builder"]["chunk_overlap"]),
            top_k=int(config["reference_builder"]["top_k"]),
            fusion_alpha=float(config["reference_builder"].get("fusion_alpha", 0.5)),
            llm=config["reference_builder"]["llm"],
            bm25=config["reference_builder"].get("bm25", {}),
            embedding=config["reference_builder"].get("embedding", {}),
            max_retries=int(config["reference_builder"].get("max_retries", 3)),
        )
        builder = ReferenceAnswerBuilder(ref_builder_cfg)
        reference_records = []
        for _, row in merged_df.iterrows():
            reference_records.append(builder.build_for_question(row["cq_id"], row["question"]))
        write_jsonl(reference_records, reference_path)

    reference_df = pd.DataFrame(reference_records)[["cq_id", "reference_answer_text", "review_evidence_ids", "review_retrieval_stats"]]
    merged_df = merged_df.merge(reference_df, on="cq_id", how="left")

    logger.info("Running gemini-3-flash-preview judge")
    judge_cfg = JudgeConfig(
        model=config["judge"]["model"],
        temperature=float(config["judge"].get("temperature", 0.0)),
        max_tokens=int(config["judge"].get("max_tokens", 600)),
        max_retries=int(config["judge"].get("max_retries", 3)),
        retry_delay=float(config["judge"].get("retry_delay", 2.0)),
        scale=config["judge"].get("scale", {}),
    )
    judge = LLMJudge(judge_cfg)
    judge_records: list[dict[str, Any]] = []
    judge_raw_records: list[dict[str, Any]] = []
    for _, row in merged_df.iterrows():
        parsed, raw = judge.judge_row(
            row["cq_id"],
            row["question"],
            row.get("reference_answer_text", ""),
            row.get("baseline_answer_text", "") or "No answer provided",
            row.get("ontology_answer_text", "") or "No answer provided",
        )

        def _to_float(key: str) -> float:
            value = parsed.get(key)
            if value is None:
                raise ValueError(f"Judge response missing {key} for {row['cq_id']}")
            return float(value)

        judge_records.append(
            {
                "cq_id": row["cq_id"],
                "score_baseline": _to_float("score_baseline"),
                "score_onto": _to_float("score_onto"),
                "winner": parsed.get("winner"),
                "notes": parsed.get("notes"),
            }
        )
        judge_raw_records.append(raw)

    write_jsonl(judge_records, Path(outputs_cfg["judge_scores_path"]))
    write_jsonl(judge_raw_records, Path(outputs_cfg["judge_raw_path"]))

    judge_df = pd.DataFrame(judge_records)
    merged_df = merged_df.merge(judge_df, on="cq_id", how="left")

    good_threshold = config.get("quality", {}).get("good_answer_threshold")
    quality_metrics = compute_quality_metrics(merged_df, good_threshold=good_threshold)
    aggregate_json_path = Path(outputs_cfg["aggregate_json_path"])
    aggregate_json_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_json_path.write_text(json.dumps(quality_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    aggregate_md_path = Path(outputs_cfg["aggregate_md_path"])
    aggregate_md_path.write_text(render_quality_markdown(quality_metrics), encoding="utf-8")

    rag_df, rag_summary = compute_rag_metrics(merged_df, config.get("rag_metrics", {}))
    onto_context_df, onto_context_summary = compute_onto_context_metrics(merged_df, config.get("onto_context_metrics", {}))

    structural_metrics = compute_structural_metrics(tbox_path, kg_path)
    structural_path = Path(outputs_cfg["ontology_structural_metrics_path"])
    structural_path.parent.mkdir(parents=True, exist_ok=True)
    structural_path.write_text(json.dumps(structural_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    per_cq_df = (
        merged_df[
            [
                "cq_id",
                "question",
                "reference_answer_text",
                "baseline_answer_text",
                "ontology_answer_text",
                "score_baseline",
                "score_onto",
            ]
        ]
        .merge(rag_df, on="cq_id", how="left")
        .merge(onto_context_df, on="cq_id", how="left")
    )
    per_cq_df["delta"] = per_cq_df["score_onto"] - per_cq_df["score_baseline"]
    per_cq_df["outcome"] = per_cq_df["delta"].apply(lambda x: "win" if x > 0 else ("loss" if x < 0 else "tie"))

    per_cq_path = Path(outputs_cfg["per_cq_scores_path"])
    per_cq_path.parent.mkdir(parents=True, exist_ok=True)
    per_cq_df.to_csv(per_cq_path, index=False)

    onto_context_path = Path(outputs_cfg["onto_context_usage_path"])
    onto_context_df.to_csv(onto_context_path, index=False)

    summary_metrics = {
        "sanity": sanity_report,
        "quality": quality_metrics,
        "rag": rag_summary,
        "ontology_context": onto_context_summary,
        "structural": structural_metrics,
    }
    summary_json_path = Path(outputs_cfg["summary_metrics_json_path"])
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(json.dumps(summary_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_md_path = Path(outputs_cfg["summary_metrics_md_path"])
    summary_md_path.write_text(
        render_summary_markdown(sanity_report, quality_metrics, rag_summary, onto_context_summary, structural_metrics),
        encoding="utf-8",
    )

    logger.info("Done. Reports saved under configured paths")


if __name__ == "__main__":
    main()
