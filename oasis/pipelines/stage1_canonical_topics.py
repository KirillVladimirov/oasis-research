"""Нормализация кластеров тем с помощью LLM (Шаг 1, задача 3)."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from loguru import logger
from tqdm import tqdm
from openai import OpenAI

from oasis.pipelines.stage1_extract_topics import _create_openai_client

CLUSTER_FLUSH_INTERVAL = 5
NOISE_FLUSH_INTERVAL = 2


CANONICAL_CLUSTER_SYSTEM_PROMPT = """
You are an expert in Deep Active Learning (DAL) and scientific text analysis.

You receive a cluster of local topics extracted from review articles. Each topic includes a title, description,
and source metadata. The cluster may be:
- clearly about Deep Active Learning,
- partially related (mixed DAL + generic ML),
- or off-domain / noisy.

Your tasks:

1) ASSESS RELEVANCE OF THE CLUSTER TO DEEP ACTIVE LEARNING
- Consider the target domain: Deep Active Learning (pool-based and stream-based active learning, query strategies,
  acquisition functions, uncertainty estimation, diversity-based and hybrid selection, label/annotation budget,
  stopping criteria, DAL benchmarks and datasets, DAL for vision/NLP, noisy labels in AL, theoretical analysis of AL, etc.).
- The cluster is "meaningful" if the majority of its local topics concern DAL-specific concepts, methods, or
  experimental setups (not just generic deep learning).
- If the cluster is mostly generic ML (e.g., general architectures, loss functions, optimization) without a clear
  DAL context, treat it as NOT relevant.

2) IF THE CLUSTER IS MEANINGFUL, CREATE A CANONICAL TOPIC
For a relevant DAL cluster, produce a canonical topic with:
- name: concise English title (3–10 words) that clearly reflects the DAL topic;
- description: 3–6 sentences summarizing the recurring ideas, nuances, and variations across the local topics.
  Highlight what is specific to Deep Active Learning (e.g., type of acquisition strategy, budget handling, etc.);
- sources: list of distinct source citations; each source must use an existing article_id + section_id + section_heading
  taken from the input metadata (do not invent new articles or sections);
- local_topic_ids: all local topic ids from this cluster that you actually used to construct the canonical topic.

If some local topics in the cluster are only weakly related or slightly off, you may still keep the cluster,
but focus the name/description on the DAL-relevant core and still list all used local_topic_ids.

3) IF THE CLUSTER IS NOT RELEVANT / NOISY, REJECT IT
- Mark the cluster as rejected and provide a short reason (e.g. "generic deep learning without DAL context",
  "too vague and heterogeneous", "off-domain topic").
- In this case, canonical_topic MUST be null.

GENERAL RULES:
- Use ONLY the provided local topics and their metadata; do not invent new sources or new ids.
- Keep the writing in English, with a neutral scientific tone.
- Be conservative with DAL relevance: if DAL-specific connection is unclear or extremely weak, prefer decision="drop".

OUTPUT FORMAT:
- Output a SINGLE valid JSON object with the following schema:

{
  "cluster_id": "<cluster id from input>",
  "decision": "keep" | "drop",
  "reason": "<short reason if decision='drop'; can be an empty string if decision='keep'>",
  "canonical_topic": {
    "name": "...",
    "description": "...",
    "sources": [
      {"article_id": "...", "section_id": "...", "section_heading": "..."}
    ],
    "local_topic_ids": ["...", "..."]
  }
}

- When decision = "drop", set canonical_topic to null.
- Do NOT hallucinate article ids, section ids, or section headings: always copy them from the input.
- Do NOT add any extra fields or commentary outside this JSON object.
"""


NOISE_BATCH_SYSTEM_PROMPT = """
You are an expert in Deep Active Learning (DAL).

You receive a batch of local topics that were not assigned to any cluster. Many of them may be noisy,
redundant, or off-domain. Your goal is to process and keep only the clearly DAL-relevant topics.

For each topic you decide to keep:
- You can rewrite/improve the title (3-10 words) and description (3-6 sentences) to make them more canonical
- You can merge similar topics into one canonical topic
- You must return the canonical representation with new name and description

Output format (strictly valid JSON):
{
  "canonical_topics": [
    {
      "local_topic_id": "T001",  // Original ID from input (required for tracking)
      "name": "Improved canonical title",  // Your rewritten/improved title
      "description": "Improved canonical description with 3-6 sentences"  // Your rewritten/improved description
    },
    ...
  ]
}

Rules:
- Return at most 5 canonical topics from the input batch
- If none of the topics are DAL-related, return empty array: {"canonical_topics": []}
- You can merge multiple input topics into one canonical topic (use the first local_topic_id as reference)
- Be conservative: when in doubt, exclude the topic
- Always improve/rewrite titles and descriptions to be more canonical and clear
"""



@dataclass
class LocalTopic:
    """Локальная тема раздела обзора."""

    local_topic_id: str
    title: str
    description: str
    article_id: str
    article_title: str
    section_id: str
    section_heading: str
    source_record_index: int
    cluster_id: str | None = None

    def to_source_dict(self) -> dict[str, str]:
        """Возвращает словарь с источником для JSON."""

        return {
            "article_id": self.article_id,
            "article_title": self.article_title,
            "section_id": self.section_id,
            "section_heading": self.section_heading,
        }

    def to_payload(self) -> dict[str, Any]:
        """Словарь с сокращённой информацией для LLM."""

        return {
            "local_topic_id": self.local_topic_id,
            "title": self.title,
            "description": self.description,
            "article_id": self.article_id,
            "article_title": self.article_title,
            "section_id": self.section_id,
            "section_heading": self.section_heading,
        }


@dataclass
class ClusterGroup:
    """Группа локальных тем в кластере."""

    cluster_id: str
    topics: list[LocalTopic] = field(default_factory=list)


@dataclass
class CanonicalTopic:
    """Результат нормализации темы."""

    canonical_topic_id: str
    name: str
    description: str
    sources: list[dict[str, Any]]
    local_topics: list[dict[str, Any]]
    cluster_id: str | None
    is_noise: bool
    trace_id: str | None
    generated_at: str
    drop_reason: str | None = None

    def to_json(self) -> str:
        """Возвращает JSON-строку."""

        payload = {
            "canonical_topic_id": self.canonical_topic_id,
            "cluster_id": self.cluster_id,
            "name": self.name,
            "description": self.description,
            "sources": self.sources,
            "local_topics": self.local_topics,
            "is_noise": self.is_noise,
            "generated_at": self.generated_at,
            "trace_id": self.trace_id,
        }
        if self.drop_reason:
            payload["drop_reason"] = self.drop_reason
        return json.dumps(payload, ensure_ascii=False)


def load_topics_clustered(
    clustered_jsonl: str | Path, trace_id: str | None = None
) -> tuple[list[ClusterGroup], list[LocalTopic]]:
    """Загружает результаты кластеризации и разделяет их на кластеры и noise.

    Args:
        clustered_jsonl: Путь к файлу topics_clustered.jsonl
        trace_id: Идентификатор трейса

    Returns:
        Кортеж (clusters, noise_topics)
    """

    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    path = Path(clustered_jsonl)
    if not path.exists():
        raise FileNotFoundError(f"Файл с кластерами не найден: {clustered_jsonl}")

    clusters: list[ClusterGroup] = []
    noise_topics: list[LocalTopic] = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(f"{trace_prefix}Строка {line_num} содержит невалидный JSON: {exc}")
                continue

            cluster_id = entry.get("cluster_id")
            is_noise = entry.get("is_noise", False) or cluster_id is None
            metadata = entry.get("topic_metadata", [])
            metadata_list = metadata if isinstance(metadata, list) else [metadata]

            topics: list[LocalTopic] = []
            for topic_data in metadata_list:
                try:
                    topics.append(
                        LocalTopic(
                            local_topic_id=topic_data.get("local_topic_id", ""),
                            title=topic_data.get("title", ""),
                            description=topic_data.get("description", ""),
                            article_id=topic_data.get("article_id", ""),
                            article_title=topic_data.get("article_title", ""),
                            section_id=topic_data.get("section_id", ""),
                            section_heading=topic_data.get("section_heading", ""),
                            source_record_index=topic_data.get("source_record_index", -1),
                            cluster_id=None if is_noise else cluster_id,
                        )
                    )
                except Exception as exc:  # pragma: no cover - защитный код
                    logger.warning(f"{trace_prefix}Пропуск темы в строке {line_num}: {exc}")

            if not topics:
                continue

            if is_noise:
                noise_topics.extend(topics)
            else:
                clusters.append(ClusterGroup(cluster_id=cluster_id, topics=topics))

    logger.info(
        f"{trace_prefix}Загружено {len(clusters)} кластеров и {len(noise_topics)} noise-тем из {path}"
    )
    return clusters, noise_topics


def _call_llm_json(
    client: OpenAI,
    *,
    system_prompt: str,
    user_payload: dict[str, Any],
    trace_id: str | None = None,
    max_tokens: int = 2000,
    temperature: float = 0.2,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> dict[str, Any]:
    """Вызывает LLM и возвращает распарсенный JSON."""

    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

    for attempt in range(max_retries):
        try:
            logger.debug(f"{trace_prefix}LLM вызов (попытка {attempt + 1}/{max_retries})")
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_CANONICAL_MODEL", "gpt-5.1"),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Пустой ответ от LLM")
            try:
                parsed = json.loads(content)
                return parsed
            except json.JSONDecodeError as json_exc:
                # Логируем сырой ответ для отладки
                content_preview = content[:500] if len(content) > 500 else content
                logger.error(
                    f"{trace_prefix}Ошибка парсинга JSON ответа LLM (попытка {attempt + 1}): {json_exc}\n"
                    f"Начало ответа: {content_preview}..."
                )
                # Сохраняем полный ответ в лог для отладки
                if attempt == max_retries - 1:
                    logger.error(f"{trace_prefix}Полный невалидный JSON ответ LLM:\n{content}")
                raise json_exc
        except Exception as exc:
            if attempt < max_retries - 1:
                delay = retry_delay * (2**attempt)
                logger.warning(
                    f"{trace_prefix}Ошибка LLM (попытка {attempt + 1}): {exc}. "
                    f"Повтор через {delay:.1f}с"
                )
                time.sleep(delay)
            else:
                logger.error(f"{trace_prefix}LLM не ответил после {max_retries} попыток: {exc}")
                raise RuntimeError(f"Ошибка вызова LLM: {exc}") from exc

    raise RuntimeError("Неожиданное завершение LLM вызова")


def _request_cluster_canonical_topic(
    client: OpenAI, cluster: ClusterGroup, trace_id: str | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Вызывает LLM для нормализации одного кластера и возвращает payload + ответ."""

    payload = {
        "cluster_id": cluster.cluster_id,
        "local_topics": [topic.to_payload() for topic in cluster.topics],
    }
    response = _call_llm_json(
        client,
        system_prompt=CANONICAL_CLUSTER_SYSTEM_PROMPT,
        user_payload=payload,
        trace_id=trace_id,
    )
    return payload, response


def _request_noise_selection(
    client: OpenAI, batch_topics: Sequence[LocalTopic], batch_index: int, trace_id: str | None
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Отправляет партию noise-тем в LLM и возвращает payload, переработанные темы и ответ.
    
    Returns:
        Кортеж (payload, canonical_topics, response), где:
        - payload: исходные данные, отправленные в LLM
        - canonical_topics: список переработанных тем от LLM с полями local_topic_id, name, description
        - response: полный ответ LLM
    """

    payload = {
        "batch_id": batch_index,
        "topics": [
            {
                "local_topic_id": t.local_topic_id,
                "title": t.title,
                "description": t.description,
            }
            for t in batch_topics
        ],
    }
    response = _call_llm_json(
        client,
        system_prompt=NOISE_BATCH_SYSTEM_PROMPT,
        user_payload=payload,
        trace_id=trace_id,
        max_tokens=6000,  # Увеличено для переработанных тем с длинными описаниями
    )
    canonical_topics = response.get("canonical_topics", [])
    if not isinstance(canonical_topics, list):
        logger.warning("LLM вернул некорректный формат для noise batch, ожидается список canonical_topics")
        return payload, [], response
    
    # Валидация и нормализация переработанных тем
    normalized_topics = []
    for topic in canonical_topics:
        if not isinstance(topic, dict):
            continue
        local_topic_id = topic.get("local_topic_id")
        name = topic.get("name")
        description = topic.get("description")
        
        if not local_topic_id or not name or not description:
            logger.warning(f"Пропущена тема с неполными данными: {topic}")
            continue
        
        normalized_topics.append({
            "local_topic_id": str(local_topic_id).strip(),
            "name": str(name).strip(),
            "description": str(description).strip(),
        })
    
    return payload, normalized_topics, response


def _build_canonical_topic_from_cluster(
    cluster: ClusterGroup, response: dict[str, Any], trace_id: str | None
) -> CanonicalTopic | None:
    """Формирует CanonicalTopic из ответа LLM по кластеру."""

    decision = response.get("decision")
    if decision not in {"keep", "drop"}:
        logger.warning(
            f"[trace_id={trace_id}] Ответ LLM для {cluster.cluster_id} без корректного decision: {decision}"
        )
        return None

    if decision == "drop":
        reason = response.get("reason")
        if reason:
            logger.info(f"[trace_id={trace_id}] Кластер {cluster.cluster_id} отклонён: {reason}")
        else:
            logger.info(f"[trace_id={trace_id}] Кластер {cluster.cluster_id} отклонён без причины")
        return None

    canonical = response.get("canonical_topic")
    if not canonical:
        logger.warning(
            f"[trace_id={trace_id}] Ответ LLM для {cluster.cluster_id} не содержит canonical_topic"
        )
        return None

    name = canonical.get("name", "").strip()
    description = canonical.get("description", "").strip()
    if not name or not description:
        logger.warning(
            f"[trace_id={trace_id}] Ответ LLM для {cluster.cluster_id} содержит пустые поля name/description"
        )
        return None

    sources = canonical.get("sources") or []
    if not isinstance(sources, list) or not sources:
        # Если источники не указаны, генерируем их из локальных тем
        sources = _aggregate_sources(cluster.topics)
    else:
        sources = _normalize_sources(sources)

    local_ids = canonical.get("local_topic_ids") or [topic.local_topic_id for topic in cluster.topics]
    local_ids = [lid for lid in local_ids if isinstance(lid, str)]
    if not local_ids:
        local_ids = [topic.local_topic_id for topic in cluster.topics]

    local_topic_entries = [
        {
            "local_topic_id": topic.local_topic_id,
            "article_id": topic.article_id,
            "section_id": topic.section_id,
            "title": topic.title,
        }
        for topic in cluster.topics
        if topic.local_topic_id in local_ids
    ]

    return CanonicalTopic(
        canonical_topic_id=cluster.cluster_id,
        cluster_id=cluster.cluster_id,
        name=name,
        description=description,
        sources=sources,
        local_topics=local_topic_entries,
        is_noise=False,
        generated_at=datetime.now().isoformat(),
        trace_id=trace_id,
    )


def _aggregate_sources(topics: Iterable[LocalTopic]) -> list[dict[str, Any]]:
    """Собирает уникальные источники из локальных тем."""

    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    for topic in topics:
        key = (topic.article_id, topic.section_id)
        if key in seen:
            continue
        seen.add(key)
        result.append(topic.to_source_dict())
    return result


def _normalize_sources(sources: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Проверяет и нормализует список источников из ответа LLM."""

    normalized: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        article_id = str(source.get("article_id", "")).strip()
        section_id = str(source.get("section_id", "")).strip()
        if not article_id:
            continue
        normalized.append(
            {
                "article_id": article_id,
                "article_title": source.get("article_title", ""),
                "section_id": section_id,
                "section_heading": source.get("section_heading", ""),
            }
        )
    return normalized or []


def _load_existing_canonical_topics(
    output_path: Path,
) -> tuple[set[str], set[str], int]:
    """Загружает уже сохранённые канонические темы (для режима resume)."""

    processed_clusters: set[str] = set()
    processed_noise_local_ids: set[str] = set()
    total_topics = 0

    if not output_path.exists():
        return processed_clusters, processed_noise_local_ids, total_topics

    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_topics += 1
            cluster_id = entry.get("cluster_id")
            if cluster_id:
                processed_clusters.add(cluster_id)
            if entry.get("is_noise"):
                for local_topic in entry.get("local_topics", []) or []:
                    if not isinstance(local_topic, dict):
                        continue
                    lid = local_topic.get("local_topic_id")
                    if isinstance(lid, str) and lid:
                        processed_noise_local_ids.add(lid)

    return processed_clusters, processed_noise_local_ids, total_topics


def _flush_handle(file_handle) -> None:
    """Безопасно сбрасывает буфер на диск."""

    file_handle.flush()
    try:
        os.fsync(file_handle.fileno())
    except OSError as exc:  # pragma: no cover - зависит от FS
        logger.warning(f"Не удалось выполнить fsync файла {file_handle.name}: {exc}")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Дописывает запись в JSONL с немедленным fsync."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        _flush_handle(handle)


def _log_llm_result(
    log_path: Path,
    *,
    kind: str,
    trace_id: str | None,
    payload: dict[str, Any],
    response: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Сохраняет исходный payload/ответ LLM."""

    entry = {
        "timestamp": datetime.now().isoformat(),
        "kind": kind,
        "trace_id": trace_id,
        "payload": payload,
        "response": response,
    }
    entry.update(metadata)
    _append_jsonl(log_path, entry)


def _flush_topics(
    output_path: Path,
    buffer: list[CanonicalTopic],
    *,
    trace_prefix: str,
    reason: str,
) -> None:
    """Сбрасывает накопленные канонические темы на диск."""

    if not buffer:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"{trace_prefix}Сбрасываю {len(buffer)} тем (причина: {reason}) в файл {output_path}"
    )
    with output_path.open("a", encoding="utf-8") as handle:
        for topic in buffer:
            handle.write(topic.to_json() + "\n")
        _flush_handle(handle)
    logger.info(
        f"{trace_prefix}Сброс {len(buffer)} тем завершён, данные гарантировано записаны на диск"
    )
    buffer.clear()


def _initialize_canonical_topics_generation(
    clustered_jsonl: str | Path | None,
    output_jsonl: str | Path | None,
    trace_id: str | None,
    log_dir: str | Path | None,
    resume: bool,
    resume_noise: bool,
    llm_timeout: float | None,
) -> tuple[
    Path,  # output_path
    Path,  # clusters_log_path
    Path,  # noise_log_path
    list[ClusterGroup],  # clusters
    list[LocalTopic],  # noise_topics
    OpenAI,  # client
    set[str],  # processed_clusters
    set[str],  # processed_noise_topic_ids
    int,  # canonical_topics_total
    int,  # initial_clusters_processed
    int,  # initial_noise_processed
]:
    """Инициализирует генерацию канонических тем: пути, resume, загрузка данных."""
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    project_root = Path(__file__).parent.parent.parent

    if clustered_jsonl is None:
        clustered_jsonl = project_root / "outputs" / "deep_active_learning" / "topics_clustered.jsonl"
    if output_jsonl is None:
        output_jsonl = project_root / "outputs" / "deep_active_learning" / "canonical_topics.jsonl"

    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if resume:
        (
            processed_clusters,
            processed_noise_topic_ids,
            canonical_topics_total,
        ) = _load_existing_canonical_topics(output_path)
        initial_clusters_processed = len(processed_clusters)
        initial_noise_processed = len(processed_noise_topic_ids)
        if output_path.exists():
            logger.info(
                f"{trace_prefix}Режим resume: найдено {canonical_topics_total} тем "
                f"(кластеров={initial_clusters_processed}, noise-тем={initial_noise_processed}). Продолжаем запись."
            )
        else:
            output_path.touch()
            logger.info(f"{trace_prefix}Режим resume: файл не найден, начинаем с нуля.")
        if not resume_noise and processed_noise_topic_ids:
            logger.info(
                f"{trace_prefix}Resume noise отключён — переобрабатываем {len(processed_noise_topic_ids)} noise-тем."
            )
            processed_noise_topic_ids = set()
    else:
        output_path.open("w", encoding="utf-8").close()
        processed_clusters = set()
        processed_noise_topic_ids = set()
        canonical_topics_total = 0
        initial_clusters_processed = 0
        initial_noise_processed = 0

    if log_dir is None:
        log_dir_path = output_path.parent / "llm_logs" / "step_1_task_3"
    else:
        log_dir_path = Path(log_dir)
    clusters_log_path = log_dir_path / "clusters.jsonl"
    noise_log_path = log_dir_path / "noise_batches.jsonl"
    logger.info(f"{trace_prefix}Каталог логов LLM: {log_dir_path}")

    clusters, noise_topics = load_topics_clustered(clustered_jsonl, trace_id=trace_id)
    client = _create_openai_client(timeout=llm_timeout)

    return (
        output_path,
        clusters_log_path,
        noise_log_path,
        clusters,
        noise_topics,
        client,
        processed_clusters,
        processed_noise_topic_ids,
        canonical_topics_total,
        initial_clusters_processed,
        initial_noise_processed,
    )


def _process_clusters(
    clusters: list[ClusterGroup],
    client: OpenAI,
    output_path: Path,
    clusters_log_path: Path,
    processed_clusters: set[str],
    pending_topics: list[CanonicalTopic],
    resume_from_cluster: str | None,
    trace_id: str | None,
) -> tuple[int, int, dict[str, Any]]:
    """Обрабатывает кластеры тем через LLM и формирует канонические темы."""
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    accepted_clusters = 0
    dropped_clusters = 0
    cluster_requests_since_flush = 0
    resume_skip_info = {
        "target_cluster": resume_from_cluster,
        "skipped_clusters": 0,
        "resumed_from_cluster": None,
    }

    for idx, cluster in enumerate(
        tqdm(clusters, desc="LLM для кластеров", unit="cluster"), start=1
    ):
        # Пропускаем только уже обработанные кластеры
        if cluster.cluster_id in processed_clusters:
            resume_skip_info["skipped_clusters"] += 1
            logger.info(
                f"{trace_prefix}Кластер {cluster.cluster_id} уже обработан ранее, пропускаю (resume)."
            )
            continue
        
        # Логируем начало обработки при resume (только для первого необработанного кластера)
        if resume_from_cluster and resume_skip_info["resumed_from_cluster"] is None:
            resume_skip_info["resumed_from_cluster"] = cluster.cluster_id
            logger.info(
                f"{trace_prefix}Resume: начинаю обработку с {cluster.cluster_id} (пропущено {resume_skip_info['skipped_clusters']} уже обработанных кластеров)"
            )
        payload, response = _request_cluster_canonical_topic(client, cluster, trace_id)
        _log_llm_result(
            clusters_log_path,
            kind="cluster",
            trace_id=trace_id,
            payload=payload,
            response=response,
            metadata={"cluster_id": cluster.cluster_id},
        )
        canonical = _build_canonical_topic_from_cluster(cluster, response, trace_id)
        if canonical:
            pending_topics.append(canonical)
            accepted_clusters += 1
            processed_clusters.add(cluster.cluster_id)
        else:
            dropped_clusters += 1
        cluster_requests_since_flush += 1
        if cluster_requests_since_flush >= CLUSTER_FLUSH_INTERVAL:
            _flush_topics(
                output_path,
                pending_topics,
                trace_prefix=trace_prefix,
                reason="CLUSTER_FLUSH_INTERVAL",
            )
            cluster_requests_since_flush = 0
        if idx % 25 == 0:
            logger.info(
                f"{trace_prefix}Обработано {idx}/{len(clusters)} кластеров "
                f"(принято={accepted_clusters}, отклонено={dropped_clusters})"
            )

    # Гарантированно сохраняем всё, что набралось за этап кластеров
    _flush_topics(
        output_path,
        pending_topics,
        trace_prefix=trace_prefix,
        reason="CLUSTERS_DONE",
    )

    logger.info(
        f"{trace_prefix}Кластеров обработано: {len(clusters)} (принято {accepted_clusters}, отброшено {dropped_clusters})"
    )

    return accepted_clusters, dropped_clusters, resume_skip_info


def _process_noise_topics(
    noise_topics: list[LocalTopic],
    client: OpenAI,
    output_path: Path,
    noise_log_path: Path,
    pending_topics: list[CanonicalTopic],
    noise_batch_size: int,
    trace_id: str | None,
) -> tuple[int, int]:
    """Обрабатывает noise-темы через LLM и формирует канонические темы.
    
    Для каждого батча отправляет темы в LLM, который выбирает 5-10 значимых тем.
    Все выбранные темы записываются в canonical_topics.jsonl без дедубликации.
    
    Returns:
        Кортеж (noise_selected, noise_added), где:
        - noise_selected: количество выбранных LLM тем
        - noise_added: количество тем, добавленных в pending_topics
    """
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    noise_selected = 0
    noise_added = 0
    noise_total = len(noise_topics)
    noise_requests_since_flush = 0

    for batch_index in tqdm(
        range(0, noise_total, noise_batch_size),
        desc="Фильтрация noise",
        unit="batch",
    ):
        batch = noise_topics[batch_index : batch_index + noise_batch_size]
        if not batch:
            continue
        batch_number = batch_index // noise_batch_size
        try:
            payload, canonical_topics_from_llm, response = _request_noise_selection(
                client, batch, batch_number, trace_id
            )
        except RuntimeError as exc:
            logger.error(
                f"{trace_prefix}Ошибка при обработке noise-батча {batch_number + 1}: {exc}. "
                f"Пропускаю батч и продолжаю."
            )
            # Логируем ошибку, но продолжаем обработку следующих батчей
            _log_llm_result(
                noise_log_path,
                kind="noise_batch_error",
                trace_id=trace_id,
                payload={"batch_id": batch_number, "topics": [{"local_topic_id": t.local_topic_id} for t in batch]},
                response={"error": str(exc)},
                metadata={"batch_index": batch_number, "topics_in_batch": len(batch), "error": True},
            )
            continue
        
        _log_llm_result(
            noise_log_path,
            kind="noise_batch",
            trace_id=trace_id,
            payload=payload,
            response=response,
            metadata={"batch_index": batch_number, "topics_in_batch": len(batch)},
        )
        if not canonical_topics_from_llm:
            noise_requests_since_flush += 1
            if noise_requests_since_flush >= NOISE_FLUSH_INTERVAL:
                _flush_topics(
                    output_path,
                    pending_topics,
                    trace_prefix=trace_prefix,
                    reason="NOISE_FLUSH_INTERVAL_NO_SELECTION",
                )
                noise_requests_since_flush = 0
            continue
        logger.info(
            f"{trace_prefix}Noise-батч {batch_number + 1}: "
            f"получено {len(canonical_topics_from_llm)} переработанных тем из {len(batch)}"
        )
        
        # Создаём словарь для быстрого поиска оригинальных тем по local_topic_id
        batch_by_id = {topic.local_topic_id: topic for topic in batch}
        
        # Сохраняем все переработанные темы от LLM
        for canonical_topic_data in canonical_topics_from_llm:
            local_topic_id = canonical_topic_data["local_topic_id"]
            original_topic = batch_by_id.get(local_topic_id)
            
            if not original_topic:
                logger.warning(
                    f"{trace_prefix}LLM вернул local_topic_id={local_topic_id}, "
                    f"которого нет в батче. Пропускаю."
                )
                continue
            
            # Используем переработанные name и description от LLM
            pending_topics.append(
                CanonicalTopic(
                    canonical_topic_id=f"NOISE_{uuid.uuid4().hex}",
                    cluster_id=None,
                    name=canonical_topic_data["name"],  # От LLM
                    description=canonical_topic_data["description"],  # От LLM
                    sources=[original_topic.to_source_dict()],  # Источники из оригинальной темы
                    local_topics=[
                        {
                            "local_topic_id": local_topic_id,
                            "article_id": original_topic.article_id,
                            "section_id": original_topic.section_id,
                            "title": original_topic.title,  # Оригинальный title для справки
                        }
                    ],
                    is_noise=True,
                    generated_at=datetime.now().isoformat(),
                    trace_id=trace_id,
                )
            )
            noise_selected += 1
            noise_added += 1
        noise_requests_since_flush += 1
        if noise_requests_since_flush >= NOISE_FLUSH_INTERVAL:
            _flush_topics(
                output_path,
                pending_topics,
                trace_prefix=trace_prefix,
                reason="NOISE_FLUSH_INTERVAL",
            )
            noise_requests_since_flush = 0

    logger.info(
        f"{trace_prefix}Noise-тем: всего {noise_total}, отобрано {noise_selected}, отброшено {noise_total - noise_selected}"
    )

    # После завершения обработки noise гарантированно сохраняем буфер
    _flush_topics(
        output_path,
        pending_topics,
        trace_prefix=trace_prefix,
        reason="NOISE_DONE",
    )

    return noise_selected, noise_added


def generate_canonical_topics(
    clustered_jsonl: str | Path | None = None,
    output_jsonl: str | Path | None = None,
    trace_id: str | None = None,
    noise_batch_size: int = 20,
    log_dir: str | Path | None = None,
    resume: bool = False,
    resume_from_cluster: str | None = None,
    llm_timeout: float | None = None,
    resume_noise: bool = True,
) -> dict[str, Any]:
    """Формирует канонические темы на основе кластеров локальных тем."""
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    started_at = datetime.now().isoformat()

    # Инициализация
    (
        output_path,
        clusters_log_path,
        noise_log_path,
        clusters,
        noise_topics,
        client,
        processed_clusters,
        processed_noise_topic_ids,
        canonical_topics_total,
        initial_clusters_processed,
        initial_noise_processed,
    ) = _initialize_canonical_topics_generation(
        clustered_jsonl,
        output_jsonl,
        trace_id,
        log_dir,
        resume,
        resume_noise,
        llm_timeout,
    )

    pending_topics: list[CanonicalTopic] = []

    # Обработка кластеров
    accepted_clusters, dropped_clusters, resume_skip_info = _process_clusters(
        clusters,
        client,
        output_path,
        clusters_log_path,
        processed_clusters,
        pending_topics,
        resume_from_cluster,
        trace_id,
    )
    canonical_topics_total += accepted_clusters

    # Обработка noise-тем
    noise_selected, noise_added = _process_noise_topics(
        noise_topics,
        client,
        output_path,
        noise_log_path,
        pending_topics,
        noise_batch_size,
        trace_id,
    )
    canonical_topics_total += noise_added

    # Финальный сброс накопленных тем
    _flush_topics(
        output_path,
        pending_topics,
        trace_prefix=trace_prefix,
        reason="FINAL_FLUSH",
    )

    # Формирование статистики
    finished_at = datetime.now().isoformat()
    log_dir_path = clusters_log_path.parent
    noise_total = len(noise_topics)
    stats = {
        "clusters_total": len(clusters),
        "clusters_kept": accepted_clusters,
        "clusters_dropped": dropped_clusters,
        "noise_total": noise_total,
        "noise_selected": noise_selected,
        "noise_added": noise_added,
        "noise_dropped": noise_total - noise_selected,
        "canonical_topics_total": canonical_topics_total,
        "output_path": str(output_path),
        "logs_dir": str(log_dir_path),
        "started_at": started_at,
        "finished_at": finished_at,
        "resume_used": resume,
        "clusters_skipped": initial_clusters_processed,
        "noise_topics_skipped": initial_noise_processed,
        "resume_from_cluster": resume_from_cluster,
        "manual_resume_skipped": resume_skip_info["skipped_clusters"],
        "manual_resume_started_at": resume_skip_info["resumed_from_cluster"],
    }
    logger.info(f"{trace_prefix}Нормализация завершена. Итоговых канонических тем: {canonical_topics_total}")
    return stats


__all__ = [
    "generate_canonical_topics",
    "load_topics_clustered",
]

