# Метрики CQ (корректность, дубликаты). Вызывается из oasis.pipelines.eval.run.

from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from loguru import logger
from tqdm import tqdm

from oasis.pipelines.eval.cq_duplicates import exact_duplicate_stats, near_duplicate_pairs
from oasis.pipelines.eval.cq_normalize import normalize_cq, tokenize

TEXT_KEYS = ("cq", "question", "text")
INTERROGATIVE_STARTS = {
    "what", "which", "how", "why", "when", "where", "is", "are", "can",
    "do", "does", "did", "could", "would", "should", "who", "whom", "whose",
}
PLACEHOLDER_RE = re.compile(r"\b(xxx|tbd|todo|something|etc)\b|\[mask\]", re.IGNORECASE)
BAD_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
CONJUNCTION_RE = re.compile(r"\b(and|or)\b", re.IGNORECASE)
COMPLEX_INTERROGATIVE_RE = re.compile(
    r"\bwhat\b.*\band\b.*\b(how|why|when|where|who|which)\b", re.IGNORECASE
)


def percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    if p <= 0:
        return float(values_sorted[0])
    if p >= 100:
        return float(values_sorted[-1])
    k = (len(values_sorted) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(values_sorted[int(k)])
    return float(values_sorted[int(f)] * (c - k) + values_sorted[int(c)] * (k - f))


def pick_text(obj: dict) -> tuple[str, str]:
    for key in TEXT_KEYS:
        if key in obj and isinstance(obj[key], str):
            return obj[key], key
    return "", ""


def load_jsonl(path: Path, *, total: int | None = None) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, total=total, desc="Reading CQs"):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def run(
    input_path: Path,
    output_path: Path,
    config_path: Path | None = None,
    log_level: str = "INFO",
    semantic_dedup: bool = False,
    semantic_thresholds: str = "0.92,0.95",
    top_n: int = 20,
    jaccard_threshold: float = 0.9,
    shingle_size: int = 3,
) -> None:
    """Считает метрики CQ и пишет output_path (JSON)."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config: dict = {}
    if config_path and config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cqs_cfg = config.get("cqs", {})
    interrogatives = set(cqs_cfg.get("interrogatives", [])) or INTERROGATIVE_STARTS
    jaccard_threshold = cqs_cfg.get("jaccard_threshold", jaccard_threshold)
    shingle_size = cqs_cfg.get("shingle_size", shingle_size)
    semantic_cfg = cqs_cfg.get("semantic", {})
    if semantic_cfg.get("enabled", False):
        semantic_dedup = True
    semantic_model = semantic_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    thresholds_str = semantic_cfg.get("thresholds")
    if thresholds_str is not None:
        semantic_thresholds = ",".join(str(x) for x in thresholds_str)
    if semantic_dedup:
        thresholds_list = [float(t.strip()) for t in semantic_thresholds.split(",") if t.strip()]
    else:
        thresholds_list = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_lines = count_lines(input_path)
    metrics_meta: dict = {
        "config_path": str(config_path) if config_path and config_path.exists() else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger.info("Loaded input {} with {} lines", input_path, total_lines)

    total = 0
    missing_text = 0
    ends_with_qmark = 0
    starts_with_interrogative = 0
    contains_placeholder = 0
    multiple_questions = 0
    trailing_period_without_qmark = 0
    has_conjunction = 0
    has_semicolon_or_colon = 0
    total_commas = 0
    non_atomic = 0
    bad_chars_total = 0
    chars_total = 0
    lengths: list[int] = []
    normalized_texts: list[str] = []
    ids: list[str] = []
    token_lists: list[list[str]] = []
    raw_texts: list[str] = []
    examples: dict = {
        "missing_qmark": [],
        "not_interrogative": [],
        "placeholder": [],
        "multiple_questions": [],
        "non_atomic": [],
    }

    for obj in load_jsonl(input_path, total=total_lines):
        total += 1
        text, _ = pick_text(obj)
        if not text:
            missing_text += 1
            continue
        cq_id = obj.get("id") or obj.get("cq_id") or f"row_{total}"
        normalized, removed_trailing_period = normalize_cq(text, remove_trailing_period=True)
        if removed_trailing_period:
            trailing_period_without_qmark += 1
        tokens = tokenize(normalized)
        lengths.append(len(tokens))
        normalized_texts.append(normalized)
        ids.append(cq_id)
        token_lists.append(tokens)
        raw_texts.append(text)
        trimmed = text.strip()
        if trimmed.endswith("?"):
            ends_with_qmark += 1
        elif len(examples["missing_qmark"]) < top_n:
            examples["missing_qmark"].append({"id": cq_id, "text": text})
        first_token = tokens[0].lower() if tokens else ""
        if first_token in interrogatives:
            starts_with_interrogative += 1
        elif len(examples["not_interrogative"]) < top_n:
            examples["not_interrogative"].append({"id": cq_id, "text": text})
        if PLACEHOLDER_RE.search(text):
            contains_placeholder += 1
            if len(examples["placeholder"]) < top_n:
                examples["placeholder"].append({"id": cq_id, "text": text})
        if trimmed.count("?") > 1:
            multiple_questions += 1
            if len(examples["multiple_questions"]) < top_n:
                examples["multiple_questions"].append({"id": cq_id, "text": text})
        comma_count = trimmed.count(",")
        total_commas += comma_count
        has_conj = bool(CONJUNCTION_RE.search(trimmed))
        has_conjunction += int(has_conj)
        has_semicolon_or_colon += int(";" in trimmed or ":" in trimmed)
        is_non_atomic = (
            (has_conj and ("," in trimmed or bool(COMPLEX_INTERROGATIVE_RE.search(trimmed))))
            or ";" in trimmed or ":" in trimmed or comma_count >= 2
        )
        if is_non_atomic:
            non_atomic += 1
            if len(examples["non_atomic"]) < top_n:
                examples["non_atomic"].append({"id": cq_id, "text": text})
        bad_chars_total += len(BAD_CHAR_RE.findall(text))
        chars_total += len(text)

    duplicates_sorted, exact_duplicates_count, unique_ratio = exact_duplicate_stats(normalized_texts)
    logger.info("Computing near-duplicate pairs with Jaccard")
    near_dup_pairs, total_pairs = near_duplicate_pairs(
        token_lists,
        ids,
        normalized_texts,
        threshold=jaccard_threshold,
        shingle_size=shingle_size,
        progress=tqdm,
    )
    near_dup_pairs_sorted = sorted(near_dup_pairs, key=lambda x: x["score"], reverse=True)
    near_dup_ratio = (len(near_dup_pairs) / total_pairs) if total_pairs else 0.0

    semantic_metrics: dict = {}
    if semantic_dedup and normalized_texts and thresholds_list:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformer model {}", semantic_model)
            model = SentenceTransformer(semantic_model)
            logger.info("Encoding {} CQs", len(normalized_texts))
            embeddings = model.encode(
                normalized_texts, show_progress_bar=True, convert_to_numpy=True
            ).astype(np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms
            union_finds = {thr: list(range(len(embeddings))) for thr in thresholds_list}

            def find(parent: list[int], x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(parent: list[int], a: int, b: int) -> None:
                ra, rb = find(parent, a), find(parent, b)
                if ra != rb:
                    parent[rb] = ra

            for i in tqdm(range(len(embeddings)), desc="Semantic pairs"):
                sims = embeddings[i] @ embeddings[i + 1:].T
                for offset, score in enumerate(sims, start=i + 1):
                    for thr in thresholds_list:
                        if score >= thr:
                            union(union_finds[thr], i, offset)
            for thr in thresholds_list:
                parent = union_finds[thr]
                clusters = defaultdict(list)
                for idx in range(len(embeddings)):
                    root = find(parent, idx)
                    clusters[root].append(idx)
                cluster_sizes = [len(items) for items in clusters.values() if len(items) > 1]
                total_in_clusters = sum(cluster_sizes)
                semantic_metrics[str(thr)] = {
                    "semantic_dup_ratio": (
                        (total_in_clusters / len(embeddings)) if embeddings.size else 0.0
                    ),
                    "semantic_clusters_count": sum(1 for s in cluster_sizes if s > 1),
                    "largest_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
                }
        except Exception as e:
            logger.warning("Semantic dedup failed: {}", e)

    metrics = {
        "meta": metrics_meta,
        "total": total,
        "missing_text": missing_text,
        "well_formedness": {
            "ends_with_qmark_ratio": (ends_with_qmark / total) if total else 0.0,
            "starts_with_interrogative_ratio": (starts_with_interrogative / total) if total else 0.0,
            "contains_placeholder_ratio": (contains_placeholder / total) if total else 0.0,
            "multiple_questions_ratio": (multiple_questions / total) if total else 0.0,
            "trailing_period_without_qmark_ratio": (
                (trailing_period_without_qmark / total) if total else 0.0
            ),
            "bad_charset_ratio": (bad_chars_total / chars_total) if chars_total else 0.0,
        },
        "atomicity": {
            "has_conjunction_ratio": (has_conjunction / total) if total else 0.0,
            "has_semicolon_or_colon_ratio": (has_semicolon_or_colon / total) if total else 0.0,
            "avg_commas": (total_commas / total) if total else 0.0,
            "non_atomic_ratio": (non_atomic / total) if total else 0.0,
            "atomicity_proxy_rate": (1.0 - (non_atomic / total)) if total else 0.0,
        },
        "length_tokens": {
            "min": min(lengths) if lengths else 0,
            "avg": (sum(lengths) / len(lengths)) if lengths else 0.0,
            "p95": percentile(lengths, 95),
            "max": max(lengths) if lengths else 0,
        },
        "duplicates": {
            "exact_duplicates_count": exact_duplicates_count,
            "unique_ratio": unique_ratio,
            "top_duplicates": duplicates_sorted[:top_n],
            "near_duplicates_jaccard": {
                "threshold": jaccard_threshold,
                "shingle_size": shingle_size,
                "near_dup_pairs": len(near_dup_pairs),
                "near_dup_ratio": near_dup_ratio,
                "top_pairs": near_dup_pairs_sorted[:top_n],
            },
        },
        "semantic_duplicates": semantic_metrics,
        "examples": examples,
    }
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote metrics to {}", output_path)
