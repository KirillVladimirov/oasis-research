#!/usr/bin/env python3
"""Генерация ответов для двух пайплайнов: baseline RAG и ontology+KG.

Этот скрипт генерирует ответы на CQ для двух пайплайнов с одинаковыми LLM параметрами
для обеспечения корректного сравнения.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import weaviate
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF, RDFS
from rapidfuzz import fuzz
from tqdm import tqdm

# Добавляем путь к проекту для импорта модулей
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from oasis.pipelines.stage1_extract_topics import _create_openai_client
from oasis.pipelines.stage2_indexing import hybrid_search

load_dotenv()

# ============================================================================
# Константы и конфигурация
# ============================================================================

PROMPT_VERSION = "step8_v1"
DEFAULT_EMBEDDING_MODEL = "models/bge-m3"

# Промпты для генерации ответов
BASELINE_SYSTEM_PROMPT = """You are an expert assistant answering questions about Deep Active Learning based solely on the provided context.
Your task is to generate a clear, accurate answer using ONLY the information from the context.
Do not use any external knowledge beyond what is provided in the context.

Answer format:
- Provide a direct answer to the question
- If the context contains relevant information, cite it naturally
- If the context is insufficient, state that clearly
- Be concise but complete"""

ONTOLOGY_SYSTEM_PROMPT = """You are an expert assistant answering questions about Deep Active Learning using structured knowledge from an ontology and knowledge graph.
Your task is to generate a clear, accurate answer using ONLY the provided structured context (ontology terms and knowledge graph facts).
Do not use any external knowledge beyond what is provided in the context.

Answer format:
- Provide a direct answer to the question
- Use the ontology terms and facts from the knowledge graph to support your answer
- If the context is insufficient, state that clearly
- Be concise but complete"""


# ============================================================================
# Модели данных
# ============================================================================


@dataclass
class CQRecord:
    """Запись CQ из входного JSONL."""

    cq_id: str
    question: str
    raw_data: dict[str, Any]


@dataclass
class AnswerRecord:
    """Запись ответа для сохранения в JSONL."""

    run_id: str
    pipeline: str
    cq_id: str
    question: str
    context: dict[str, Any]
    llm: dict[str, Any]
    answer_text: str
    postprocess: dict[str, Any]
    timings_ms: dict[str, int]
    error: str | None = None


# ============================================================================
# Загрузка входных данных
# ============================================================================


def load_cqs(cqs_path: Path) -> list[CQRecord]:
    """Загружает CQ из JSONL файла."""
    cqs = []
    with open(cqs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cq_id = data.get("cq_id") or data.get("id", "")
            question = data.get("question") or data.get("text", "")
            if not cq_id or not question:
                logger.warning(f"Пропущена запись без cq_id или question: {data}")
                continue
            cqs.append(CQRecord(cq_id=cq_id, question=question, raw_data=data))
    logger.info(f"Загружено {len(cqs)} CQ из {cqs_path}")
    return cqs


def load_processed_cq_ids(output_dir: Path, pipeline: str) -> set[str]:
    """Загружает множество уже обработанных cq_id из существующего JSONL."""
    processed = set()
    output_file = output_dir / f"{pipeline}_answers.jsonl"
    if not output_file.exists():
        return processed

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                cq_id = data.get("cq_id")
                if cq_id:
                    processed.add(cq_id)
            except json.JSONDecodeError:
                continue

    logger.info(f"Найдено {len(processed)} уже обработанных CQ для {pipeline}")
    return processed


# ============================================================================
# Pipeline A: Baseline RAG
# ============================================================================


def estimate_tokens(text: str) -> int:
    """Грубая оценка количества токенов (приблизительно 4 символа на токен)."""
    return len(text) // 4


def truncate_context(text: str, max_tokens: int) -> str:
    """Обрезает контекст до максимального количества токенов."""
    tokens = estimate_tokens(text)
    if tokens <= max_tokens:
        return text

    # Обрезаем по символам (приблизительно)
    max_chars = max_tokens * 4
    return text[:max_chars]


def pipeline_baseline_rag(
    cq: CQRecord,
    weaviate_client: weaviate.Client,
    llm_client: OpenAI,
    *,
    top_k: int = 8,
    alpha: float = 0.5,
    max_context_tokens: int = 4000,
    max_output_tokens: int = 800,
    temperature: float = 0.2,
    model: str,
    trace_id: str | None = None,
) -> AnswerRecord:
    """Pipeline A: Baseline RAG с гибридным поиском в Weaviate."""
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    start_time = time.time()

    # A1. Ретрив
    retrieval_start = time.time()
    try:
        chunks = hybrid_search(
            query=cq.question,
            weaviate_client=weaviate_client,
            top_k=top_k,
            alpha=alpha,
            filters=None,
            trace_id=trace_id,
            model_path=DEFAULT_EMBEDDING_MODEL,
        )
    except Exception as e:
        logger.error(f"{trace_prefix}Ошибка поиска для {cq.cq_id}: {e}")
        return AnswerRecord(
            run_id="",
            pipeline="baseline_rag",
            cq_id=cq.cq_id,
            question=cq.question,
            context={"type": "rag_text", "items": [], "stats": {}},
            llm={"model": model, "temperature": temperature, "max_output_tokens": max_output_tokens, "prompt_id": PROMPT_VERSION},
            answer_text="",
            postprocess={"format_ok": False},
            timings_ms={"retrieval": 0, "llm": 0},
            error=str(e),
        )

    retrieval_time_ms = int((time.time() - retrieval_start) * 1000)

    # Собираем текстовый контекст
    context_texts = []
    for chunk in chunks:
        text = chunk.get("text", "")
        if text and isinstance(text, str):
            context_texts.append(text)
    
    context_text = "\n\n".join(context_texts)
    context_text = truncate_context(context_text, max_context_tokens)

    # Метрики RAG - преобразуем score в float
    scores = []
    for chunk in chunks:
        if "score" in chunk:
            score = chunk.get("score", 0.0)
            try:
                score_float = float(score) if score is not None else 0.0
                scores.append(score_float)
            except (ValueError, TypeError):
                continue
    
    context_stats = {
        "top_k": top_k,
        "chunks_retrieved": len(chunks),
        "context_chars": len(context_text),
        "context_tokens_est": estimate_tokens(context_text),
        "scores_summary": {
            "mean": sum(scores) / len(scores) if scores else 0.0,
            "max": max(scores) if scores else 0.0,
            "min": min(scores) if scores else 0.0,
        },
    }

    # Формируем items с безопасным преобразованием типов
    context_items = []
    for chunk in chunks:
        score = chunk.get("score", 0.0)
        try:
            score_float = float(score) if score is not None else 0.0
        except (ValueError, TypeError):
            score_float = 0.0
        
        context_items.append({
            "chunk_id": str(chunk.get("chunk_id", "")),
            "paper_id": str(chunk.get("paper_id", "")),
            "text": str(chunk.get("text", "")),
            "score": score_float,
        })
    
    context = {
        "type": "rag_text",
        "items": context_items,
        "stats": context_stats,
    }

    # A2. Генерация ответа LLM
    llm_start = time.time()
    user_prompt = f"""Question: {cq.question}

Context from research papers:
{context_text}

Based ONLY on the provided context, answer the question. If the context is insufficient, state that clearly."""

    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": BASELINE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        answer_text = response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"{trace_prefix}Ошибка LLM для {cq.cq_id}: {e}")
        answer_text = ""
        error_msg = str(e)
    else:
        error_msg = None

    llm_time_ms = int((time.time() - llm_start) * 1000)

    # A3. Постобработка
    format_ok = bool(answer_text and len(answer_text.strip()) > 10)

    return AnswerRecord(
        run_id="",  # Будет установлен позже
        pipeline="baseline_rag",
        cq_id=cq.cq_id,
        question=cq.question,
        context=context,
        llm={"model": model, "temperature": temperature, "max_output_tokens": max_output_tokens, "prompt_id": PROMPT_VERSION},
        answer_text=answer_text,
        postprocess={"format_ok": format_ok},
        timings_ms={"retrieval": retrieval_time_ms, "llm": llm_time_ms},
        error=error_msg,
    )


# ============================================================================
# Pipeline B: Ontology + KG
# ============================================================================


class OntologyLexicon:
    """Лексикон онтологии для быстрого сопоставления терминов."""

    def __init__(self, tbox_graph: Graph, namespace: Namespace):
        self.tbox_graph = tbox_graph
        self.namespace = namespace
        self.classes: dict[str, dict[str, Any]] = {}
        self.properties: dict[str, dict[str, Any]] = {}
        self._build_lexicon()

    def _build_lexicon(self):
        """Строит лексикон классов и свойств из TBox."""
        # Классы
        for class_iri in self.tbox_graph.subjects(RDF.type, None):
            if isinstance(class_iri, URIRef):
                localname = self._extract_localname(str(class_iri))
                label = self._get_label(class_iri)
                normalized = self._normalize_text(label or localname)
                self.classes[normalized] = {
                    "iri": str(class_iri),
                    "localname": localname,
                    "label": label,
                    "normalized": normalized,
                    "kind": "class",
                }

        # Свойства (object properties)
        for prop_iri in self.tbox_graph.subjects(RDF.type, None):
            if isinstance(prop_iri, URIRef):
                localname = self._extract_localname(str(prop_iri))
                label = self._get_label(prop_iri)
                normalized = self._normalize_text(label or localname)
                if normalized not in self.classes:  # Избегаем дубликатов
                    self.properties[normalized] = {
                        "iri": str(prop_iri),
                        "localname": localname,
                        "label": label,
                        "normalized": normalized,
                        "kind": "property",
                    }

        logger.info(f"Построен лексикон: {len(self.classes)} классов, {len(self.properties)} свойств")

    def _extract_localname(self, iri: str) -> str:
        """Извлекает локальное имя из IRI."""
        if "#" in iri:
            return iri.split("#")[-1]
        if "/" in iri:
            return iri.split("/")[-1]
        return iri

    def _get_label(self, iri: URIRef) -> str | None:
        """Получает rdfs:label для IRI."""
        labels = list(self.tbox_graph.objects(iri, RDFS.label))
        if labels:
            return str(labels[0])
        return None

    def _normalize_text(self, text: str) -> str:
        """Нормализует текст для сопоставления."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def match_terms(self, question: str, top_n: int = 10) -> list[dict[str, Any]]:
        """Сопоставляет термины из вопроса с элементами онтологии."""
        question_normalized = self._normalize_text(question)
        question_tokens = question_normalized.split()

        # Ищем совпадения в классах и свойствах
        all_terms = {**self.classes, **self.properties}
        matches = []

        for norm_key, term_info in all_terms.items():
            # Простое сопоставление по токенам
            score = fuzz.partial_ratio(question_normalized, norm_key)
            if score > 30:  # Порог минимального совпадения
                matches.append(
                    {
                        "iri": term_info["iri"],
                        "label": term_info["label"],
                        "localname": term_info["localname"],
                        "score": score,
                        "kind": term_info["kind"],
                    }
                )

        # Сортируем по score и возвращаем top-N
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[:top_n]


class KGIndex:
    """Индекс для быстрого извлечения фактов из KG."""

    def __init__(self, kg_graph: Graph, namespace: Namespace):
        self.kg_graph = kg_graph
        self.namespace = namespace
        self.entity_to_outgoing: dict[str, list[tuple[URIRef, URIRef]]] = defaultdict(list)
        self.entity_to_incoming: dict[str, list[tuple[URIRef, URIRef]]] = defaultdict(list)
        self.entity_labels: dict[str, str] = {}
        self._build_index()

    def _build_index(self):
        """Строит индексы для быстрого доступа к триплетам."""
        for subj, pred, obj in self.kg_graph:
            if isinstance(subj, URIRef) and isinstance(obj, URIRef):
                subj_str = str(subj)
                obj_str = str(obj)
                self.entity_to_outgoing[subj_str].append((pred, obj))
                self.entity_to_incoming[obj_str].append((pred, subj))

                # Сохраняем labels
                if subj_str not in self.entity_labels:
                    labels = list(self.kg_graph.objects(subj, RDFS.label))
                    if labels:
                        self.entity_labels[subj_str] = str(labels[0])
                if obj_str not in self.entity_labels:
                    labels = list(self.kg_graph.objects(obj, RDFS.label))
                    if labels:
                        self.entity_labels[obj_str] = str(labels[0])

        logger.info(f"Построен индекс KG: {len(self.entity_to_outgoing)} сущностей с исходящими связями")

    def extract_subgraph(self, entity_iris: list[str], max_depth: int = 2, max_triples: int = 150) -> list[dict[str, Any]]:
        """Извлекает подграф вокруг указанных сущностей."""
        visited: set[tuple[str, str, str]] = set()
        facts: list[dict[str, Any]] = []
        queue: list[tuple[str, int]] = [(iri, 0) for iri in entity_iris]  # (iri, depth)

        while queue and len(facts) < max_triples:
            current_iri, depth = queue.pop(0)
            if depth > max_depth:
                continue

            # Исходящие связи
            for pred, obj in self.entity_to_outgoing.get(current_iri, []):
                triple_key = (current_iri, str(pred), str(obj))
                if triple_key not in visited:
                    visited.add(triple_key)
                    facts.append(
                        {
                            "subject": self.entity_labels.get(current_iri, self._extract_localname(current_iri)),
                            "predicate": self._extract_localname(str(pred)),
                            "object": self.entity_labels.get(str(obj), self._extract_localname(str(obj))),
                            "subject_iri": current_iri,
                            "predicate_iri": str(pred),
                            "object_iri": str(obj),
                        }
                    )
                    if depth < max_depth and str(obj) not in [q[0] for q in queue]:
                        queue.append((str(obj), depth + 1))

            # Входящие связи
            for pred, subj in self.entity_to_incoming.get(current_iri, []):
                triple_key = (str(subj), str(pred), current_iri)
                if triple_key not in visited:
                    visited.add(triple_key)
                    facts.append(
                        {
                            "subject": self.entity_labels.get(str(subj), self._extract_localname(str(subj))),
                            "predicate": self._extract_localname(str(pred)),
                            "object": self.entity_labels.get(current_iri, self._extract_localname(current_iri)),
                            "subject_iri": str(subj),
                            "predicate_iri": str(pred),
                            "object_iri": current_iri,
                        }
                    )
                    if depth < max_depth and str(subj) not in [q[0] for q in queue]:
                        queue.append((str(subj), depth + 1))

        return facts[:max_triples]

    def _extract_localname(self, iri: str) -> str:
        """Извлекает локальное имя из IRI."""
        if "#" in iri:
            return iri.split("#")[-1]
        if "/" in iri:
            return iri.split("/")[-1]
        return iri


def pipeline_ontology_kg(
    cq: CQRecord,
    ontology_lexicon: OntologyLexicon,
    kg_index: KGIndex,
    llm_client: OpenAI,
    *,
    max_context_tokens: int = 4000,
    max_output_tokens: int = 800,
    temperature: float = 0.2,
    model: str,
    trace_id: str | None = None,
) -> AnswerRecord:
    """Pipeline B: Ontology + KG с структурированным контекстом."""
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    start_time = time.time()

    # B1. Mapping CQ → ontology elements
    mapping_start = time.time()
    matched_terms = ontology_lexicon.match_terms(cq.question, top_n=10)
    mapping_time_ms = int((time.time() - mapping_start) * 1000)

    # B2. Извлечение фактов из KG
    extraction_start = time.time()
    entity_iris = [term["iri"] for term in matched_terms[:5]]  # Берем top-5 для извлечения фактов
    facts = kg_index.extract_subgraph(entity_iris, max_depth=2, max_triples=150)
    extraction_time_ms = int((time.time() - extraction_start) * 1000)

    # Форматируем структурированный контекст
    facts_text = "\n".join([f"- {f['subject']} --{f['predicate']}--> {f['object']}" for f in facts[:50]])
    terms_text = "\n".join([f"- {t['label'] or t['localname']} ({t['kind']})" for t in matched_terms[:10]])

    structured_context = f"""Ontology Terms:
{terms_text}

Knowledge Graph Facts:
{facts_text}"""

    structured_context = truncate_context(structured_context, max_context_tokens)

    context_stats = {
        "n_matched_terms": len(matched_terms),
        "matched_terms_top": [{"iri": t["iri"], "label": t["label"], "score": t["score"], "kind": t["kind"]} for t in matched_terms[:10]],
        "n_triples_selected": len(facts),
        "n_unique_entities": len(set([f["subject_iri"] for f in facts] + [f["object_iri"] for f in facts])),
        "kg_context_tokens_est": estimate_tokens(structured_context),
    }

    context = {
        "type": "ontology_kg_structured",
        "items": {
            "matched_terms": matched_terms[:10],
            "facts": facts[:50],
        },
        "stats": context_stats,
    }

    # B3. Генерация ответа LLM
    llm_start = time.time()
    user_prompt = f"""Question: {cq.question}

Structured Context (Ontology Terms and Knowledge Graph Facts):
{structured_context}

Based ONLY on the provided structured context, answer the question. If the context is insufficient, state that clearly."""

    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ONTOLOGY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        answer_text = response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"{trace_prefix}Ошибка LLM для {cq.cq_id}: {e}")
        answer_text = ""
        error_msg = str(e)
    else:
        error_msg = None

    llm_time_ms = int((time.time() - llm_start) * 1000)

    # B4. Постобработка
    format_ok = bool(answer_text and len(answer_text.strip()) > 10)

    return AnswerRecord(
        run_id="",  # Будет установлен позже
        pipeline="ontology_kg",
        cq_id=cq.cq_id,
        question=cq.question,
        context=context,
        llm={"model": model, "temperature": temperature, "max_output_tokens": max_output_tokens, "prompt_id": PROMPT_VERSION},
        answer_text=answer_text,
        postprocess={"format_ok": format_ok},
        timings_ms={"mapping": mapping_time_ms, "extraction": extraction_time_ms, "llm": llm_time_ms},
        error=error_msg,
    )


# ============================================================================
# Сохранение результатов
# ============================================================================


def save_answer_record(record: AnswerRecord, output_file: Path):
    """Сохраняет запись ответа в JSONL файл."""
    record_dict = {
        "run_id": record.run_id,
        "pipeline": record.pipeline,
        "cq_id": record.cq_id,
        "question": record.question,
        "context": record.context,
        "llm": record.llm,
        "answer_text": record.answer_text,
        "postprocess": record.postprocess,
        "timings_ms": record.timings_ms,
        "error": record.error,
    }
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record_dict, ensure_ascii=False) + "\n")


def save_run_manifest(output_dir: Path, config: dict[str, Any]):
    """Сохраняет манифест прогона."""
    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info(f"Сохранен манифест прогона: {manifest_path}")


# ============================================================================
# Главная функция
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Генерация ответов для baseline RAG и ontology+KG пайплайнов")
    parser.add_argument("--cqs", type=Path, required=True, help="Путь к файлу cq_canonical.jsonl")
    parser.add_argument("--pipeline", choices=["baseline_rag", "ontology_kg", "both"], default="both", help="Какой пайплайн запустить")
    parser.add_argument("--weaviate_url", type=str, default="http://localhost:8081", help="URL Weaviate")
    parser.add_argument("--weaviate_collection", type=str, default="Chunk", help="Название коллекции в Weaviate")
    parser.add_argument("--tbox", type=Path, help="Путь к TBox OWL файлу (для ontology_kg)")
    parser.add_argument("--kg", type=Path, help="Путь к KG OWL файлу (для ontology_kg)")
    parser.add_argument("--llm_base_url", type=str, help="Base URL для LLM API")
    parser.add_argument("--llm_model", type=str, default="gemini-2.5-flash", help="Модель LLM")
    parser.add_argument("--out_dir", type=Path, required=True, help="Выходная директория")
    parser.add_argument("--top_k", type=int, default=8, help="Количество результатов для RAG")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha для hybrid поиска")
    parser.add_argument("--max_context_tokens", type=int, default=4000, help="Максимум токенов в контексте")
    parser.add_argument("--max_output_tokens", type=int, default=800, help="Максимум токенов в ответе")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature для LLM")
    parser.add_argument("--resume", action="store_true", help="Пропускать уже обработанные CQ")

    args = parser.parse_args()

    # Создаем выходную директорию
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Генерируем run_id
    run_id = datetime.now(timezone.utc).isoformat()

    # Загружаем CQ
    cqs = load_cqs(args.cqs)
    if not cqs:
        logger.error("Не найдено ни одного CQ для обработки")
        return 1

    # Инициализация клиентов
    weaviate_client = None
    if args.pipeline in ["baseline_rag", "both"]:
        try:
            weaviate_client = weaviate.Client(url=args.weaviate_url)
            logger.info(f"Подключен к Weaviate: {args.weaviate_url}")
        except Exception as e:
            logger.error(f"Ошибка подключения к Weaviate: {e}")
            return 1

    llm_client = _create_openai_client()
    if args.llm_base_url:
        llm_client.base_url = args.llm_base_url
    logger.info(f"Инициализирован LLM клиент: {args.llm_model}")

    # Инициализация онтологии и KG
    ontology_lexicon = None
    kg_index = None
    if args.pipeline in ["ontology_kg", "both"]:
        if not args.tbox or not args.kg:
            logger.error("Для ontology_kg пайплайна требуются --tbox и --kg")
            return 1

        try:
            # Загружаем TBox
            tbox_graph = Graph()
            tbox_graph.parse(str(args.tbox))
            namespace = Namespace("http://example.org/deep_active_learning#")
            ontology_lexicon = OntologyLexicon(tbox_graph, namespace)
            logger.info(f"Загружен TBox: {len(tbox_graph)} триплетов")

            # Загружаем KG
            kg_graph = Graph()
            kg_graph.parse(str(args.kg))
            kg_index = KGIndex(kg_graph, namespace)
            logger.info(f"Загружен KG: {len(kg_graph)} триплетов")
        except Exception as e:
            logger.error(f"Ошибка загрузки онтологии/KG: {e}")
            return 1

    # Обработка CQ
    processed_baseline = set()
    processed_ontology = set()

    if args.resume:
        if args.pipeline in ["baseline_rag", "both"]:
            processed_baseline = load_processed_cq_ids(args.out_dir, "baseline_rag")
        if args.pipeline in ["ontology_kg", "both"]:
            processed_ontology = load_processed_cq_ids(args.out_dir, "ontology_kg")

    # Файлы для сохранения
    baseline_file = args.out_dir / "baseline_answers.jsonl"
    ontology_file = args.out_dir / "ontology_answers.jsonl"

    # Сохраняем манифест
    manifest_config = {
        "run_id": run_id,
        "pipeline": args.pipeline,
        "cqs_path": str(args.cqs),
        "llm_model": args.llm_model,
        "llm_base_url": args.llm_base_url or os.getenv("OPENAI_API_BASE"),
        "temperature": args.temperature,
        "max_context_tokens": args.max_context_tokens,
        "max_output_tokens": args.max_output_tokens,
        "top_k": args.top_k,
        "alpha": args.alpha,
        "prompt_version": PROMPT_VERSION,
        "created_at": run_id,
    }
    save_run_manifest(args.out_dir, manifest_config)

    # Обработка
    for cq in tqdm(cqs, desc="Processing CQs"):
        # Baseline RAG
        if args.pipeline in ["baseline_rag", "both"]:
            if cq.cq_id not in processed_baseline:
                try:
                    record = pipeline_baseline_rag(
                        cq,
                        weaviate_client,
                        llm_client,
                        top_k=args.top_k,
                        alpha=args.alpha,
                        max_context_tokens=args.max_context_tokens,
                        max_output_tokens=args.max_output_tokens,
                        temperature=args.temperature,
                        model=args.llm_model,
                        trace_id=cq.cq_id,
                    )
                    record.run_id = run_id
                    save_answer_record(record, baseline_file)
                except Exception as e:
                    logger.error(f"Ошибка обработки baseline для {cq.cq_id}: {e}")
                    error_record = AnswerRecord(
                        run_id=run_id,
                        pipeline="baseline_rag",
                        cq_id=cq.cq_id,
                        question=cq.question,
                        context={"type": "rag_text", "items": [], "stats": {}},
                        llm={"model": args.llm_model, "temperature": args.temperature, "max_output_tokens": args.max_output_tokens, "prompt_id": PROMPT_VERSION},
                        answer_text="",
                        postprocess={"format_ok": False},
                        timings_ms={"retrieval": 0, "llm": 0},
                        error=str(e),
                    )
                    save_answer_record(error_record, baseline_file)

        # Ontology + KG
        if args.pipeline in ["ontology_kg", "both"]:
            if cq.cq_id not in processed_ontology:
                try:
                    record = pipeline_ontology_kg(
                        cq,
                        ontology_lexicon,
                        kg_index,
                        llm_client,
                        max_context_tokens=args.max_context_tokens,
                        max_output_tokens=args.max_output_tokens,
                        temperature=args.temperature,
                        model=args.llm_model,
                        trace_id=cq.cq_id,
                    )
                    record.run_id = run_id
                    save_answer_record(record, ontology_file)
                except Exception as e:
                    logger.error(f"Ошибка обработки ontology для {cq.cq_id}: {e}")
                    error_record = AnswerRecord(
                        run_id=run_id,
                        pipeline="ontology_kg",
                        cq_id=cq.cq_id,
                        question=cq.question,
                        context={"type": "ontology_kg_structured", "items": {}, "stats": {}},
                        llm={"model": args.llm_model, "temperature": args.temperature, "max_output_tokens": args.max_output_tokens, "prompt_id": PROMPT_VERSION},
                        answer_text="",
                        postprocess={"format_ok": False},
                        timings_ms={"mapping": 0, "extraction": 0, "llm": 0},
                        error=str(e),
                    )
                    save_answer_record(error_record, ontology_file)

    logger.success(f"Обработка завершена. Результаты сохранены в {args.out_dir}")
    return 0


if __name__ == "__main__":
    exit(main())

