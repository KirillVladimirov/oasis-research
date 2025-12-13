# %% [markdown]
"""
# Демонстрация Шага 7 (Pipeline): Построение графа знаний из статей

Этот ноутбук реализует полный пайплайн построения графа знаний (ABox) напрямую из полных текстов статей ("MetaGraph-style").

**Основные этапы:**
1.  **Corpus Processing:** Инкрементальная обработка PDF (выбираем новые статьи, которые еще не были обработаны).
2.  **LLM Extraction:** Извлечение сущностей и отношений.
    *   Текст разбивается на чанки (5000 символов, перекрытие 100).
    *   Для каждого чанка выполняются два типа запросов:
        *   **Simple Entities:** Стратегии, Метрики, Ограничения и т.д.
        *   **Complex (TDM):** Task-Dataset-Model и связи между ними.
    *   Результаты агрегируются и дедублицируются.
    *   Сохранение в `extractions/{paper_id}.json`.
3.  **Entity Resolution:** Кластеризация синонимов.
4.  **Ontology Alignment:** Привязка к TBox.
5.  **Materialization:** Создание графа `deep_active_learning_kg.owl`.
"""

# %%

import sys
from pathlib import Path
import json
import os
import re
from typing import Any, Optional, List, Dict, Set, Tuple
from collections import defaultdict, Counter
import random
import shutil
import hashlib
import datetime as dt

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm
from owlready2 import *
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

# Добавляем путь к проекту
sys.path.insert(0, str(Path().absolute().parent))

# Загрузка переменных окружения (по образцу Step 6A)
env_path = Path().absolute().parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info(f" Загрузили .env: {env_path}")
else:
    load_dotenv()
    logger.warning(" Файл .env не найден в корне проекта, используем системные переменные")

from oasis.pipelines.stage1_extract_topics import _create_openai_client
# Импортируем функцию очистки текста из stage2 (или дублируем её, если импорт сложен)
try:
    from oasis.pipelines.stage2_indexing import clean_page_text
except ImportError:
    # Fallback implementation if import fails (e.g. unexpected path)
    def clean_page_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# Настройки отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', None)

logger.info(" Импорты и настройки загружены")

# %%

# КОНФИГУРАЦИЯ

TOPIC = "deep_active_learning"
BASE_DIR = Path(f"../outputs/{TOPIC}")

# Входные данные
PDF_DIR = Path("../data/deep_active_learning/paper_pdfs")
REFERENCES_CSV = Path("../data/deep_active_learning/references.csv")
INPUT_OWL = BASE_DIR / "step6c/deep_active_learning_enriched.owl"

# Выходные данные
OUTPUT_DIR = BASE_DIR / "step7"
EXTRACTIONS_DIR = OUTPUT_DIR / "extractions"
EXTRACTIONS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_KG = OUTPUT_DIR / "deep_active_learning_kg.owl"

# Параметры инкрементальной обработки
BATCH_SIZE = int(os.getenv("STEP7_BATCH_SIZE", 200))
TARGET_TOTAL_PAPERS = int(os.getenv("STEP7_TARGET_TOTAL", 189))  # Если 0, то без лимита
SELECTION_SEED = int(os.getenv("STEP7_SELECTION_SEED", 2025))

# Chunking
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 100

# Версионирование промптов и ноутбука
PROMPT_VERSION = "step7_v1.1"
NOTEBOOK_VERSION = "2025-11-25"

# LLM & Embeddings
# LLM_MODEL = os.getenv("OPENAI_CANONICAL_MODEL", "gpt-5.1")
LLM_MODEL = os.getenv("OPENAI_CANONICAL_MODEL", "gemini-2.5-flash")

LLM_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", 200.0))
EMBEDDING_MODEL_PATH = "../models/bge-m3"  # Local path preference
if not Path(f"../{EMBEDDING_MODEL_PATH}").exists():
    EMBEDDING_MODEL_PATH = "BAAI/bge-m3"

LOG_SIMPLE_RAW = bool(int(os.getenv("STEP7_LOG_SIMPLE_RAW", "0")))
LOG_TDM_RAW = bool(int(os.getenv("STEP7_LOG_TDM_RAW", "0")))
SANITY_DUMP_RAW_JSON = bool(int(os.getenv("STEP7_SANITY_DUMP_RAW", "0")))


# Инициализация
try:
    openai_client = _create_openai_client(timeout=LLM_TIMEOUT)
    logger.success(f" OpenAI клиент инициализирован: {LLM_MODEL}")
except Exception as e:
    logger.error(f" Ошибка инициализации OpenAI: {e}")
    openai_client = None

logger.info(f"  PDF Dir: {PDF_DIR}")
logger.info(f"  Extractions Cache: {EXTRACTIONS_DIR}")

# %%

import PyPDF2

def get_papers_to_process(pdf_dir: Path, extractions_dir: Path, refs_csv: Path, batch_size: int) -> List[dict]:
    # 1. Scan PDF directory
    if not pdf_dir.exists():
        logger.error(f"PDF directory not found: {pdf_dir}")
        return []
        
    all_pdfs = sorted(pdf_dir.glob("*.pdf"))
    logger.info(f"Найдено всего PDF файлов: {len(all_pdfs)}")
    
    # 2. Scan processed IDs
    processed_ids = {f.stem for f in extractions_dir.glob("*.json")}
    logger.info(f"Уже обработано (найдено JSON): {len(processed_ids)}")
    
    # 3. Ограничиваем корпус и перетасовываем один раз
    upper_bound = TARGET_TOTAL_PAPERS if TARGET_TOTAL_PAPERS > 0 else len(all_pdfs)
    pdf_candidates = list(all_pdfs[:upper_bound])
    rng = random.Random(SELECTION_SEED)
    rng.shuffle(pdf_candidates)
    logger.info(f"Используем SELECTION_SEED={SELECTION_SEED} для шифрования порядка статей")
    if pdf_candidates:
        logger.debug(f"Примеры статей после шафла: {[p.name for p in pdf_candidates[:5]]}")
    
    to_process = []
    for pdf_path in tqdm(pdf_candidates, desc="Scanning PDFs"):
        paper_id = pdf_path.stem
        if paper_id in processed_ids:
            continue
        to_process.append({
            "paper_id": paper_id,
            "pdf_path": pdf_path,
            "title": paper_id  # Fallback title
        })
        if len(to_process) >= batch_size:
            break
    
    if len(to_process) < batch_size:
        logger.warning(f"Найдено только {len(to_process)} новых статей (batch_size={batch_size})")
    
    return to_process

def parse_pdf_text(pdf_path: Path) -> str:
    try:
        reader = PyPDF2.PdfReader(str(pdf_path))
        text_parts = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            text_parts.append(clean_page_text(txt))
        return "\n".join(text_parts)
    except Exception as e:
        logger.warning(f"Error parsing {pdf_path}: {e}")
        return ""

# Run selection
print(PDF_DIR, EXTRACTIONS_DIR, REFERENCES_CSV, BATCH_SIZE)
papers_batch = get_papers_to_process(PDF_DIR, EXTRACTIONS_DIR, REFERENCES_CSV, BATCH_SIZE)
logger.success(f"Отобрано для обработки: {len(papers_batch)} статей")

if papers_batch:
    logger.info("Отобранные статьи:")
    for p in papers_batch:
        logger.info(f" - {p['paper_id']}")
else:
    logger.info("Нет новых статей для обработки (или все уже обработаны).")

# %%
# --- МОДЕЛИ ДЛЯ LLM ---

LLM_ENTITY_TYPES = [
    "AcquisitionStrategy",
    "LabelingRegime",
    "BudgetConstraint",
    "PerformanceMetric",
    "NoiseType",
    "Tool",
    "StoppingCriterion",
    "AnnotationSource",
]


class SimpleEntity(BaseModel):
    span: str
    entity_type: str = Field(..., description=", ".join(LLM_ENTITY_TYPES))
    short_description: Optional[str] = None
    evidence_span: Optional[str] = None

class SimpleExtractionResult(BaseModel):
    entities: List[SimpleEntity] = Field(default_factory=list)

class TDMRelation(BaseModel):
    relation_type: Optional[str] = Field(
        default=None,
        description="evaluatedOn | usedFor | outperforms"
    )
    target: Optional[str] = None
    evidence_span: Optional[str] = None

class TDMModel(BaseModel):
    model_name: str
    role: Optional[str] = Field(
        default=None,
        description="'main' or 'baseline'"
    )
    evidence_span: Optional[str] = None
    relations: List[TDMRelation] = Field(default_factory=list)

class TDMDataset(BaseModel):
    dataset_name: str
    evidence_span: Optional[str] = None
    models: List[TDMModel] = Field(default_factory=list)

class TDMTask(BaseModel):
    task_name: str
    task_type: Optional[str] = None
    evidence_span: Optional[str] = None
    datasets: List[TDMDataset] = Field(default_factory=list)

class TDMExtractionResult(BaseModel):
    tasks: List[TDMTask] = Field(default_factory=list)


# --- PROMPTS (из @step-7.md) ---

SYSTEM_PROMPT_SIMPLE = """You are an expert in Deep Active Learning and scientific information extraction.
You build a Knowledge Graph directly from research papers and must stay faithful to evidence while keeping recall high.

Goal:
- capture every mention of the target entity types when there is textual support
- prefer concise spans taken verbatim from the text
- never fabricate entities not grounded in the segment

Target entity catalog (use these exact labels):
- AcquisitionStrategy (e.g. BALD, entropy sampling, CoreSet, Query-By-Committee)
- LabelingRegime (e.g. pool-based, stream-based, membership query synthesis)
- BudgetConstraint (e.g. annotation budget, time budget, number of queries, annotation cost)
- PerformanceMetric (e.g. accuracy, F1, AUROC, error rate, IoU)
- NoiseType (e.g. label noise, annotator disagreement, class imbalance)
- Tool (e.g. PyTorch, TensorFlow, HuggingFace Transformers, scikit-learn)

Extraction rules:
1. Aim for high recall: if the mention is plausible and supported, extract it with evidence.
2. Use exact spans from the segment and keep them short.
3. Every entity must include a short_description (1-2 sentences) explaining its role in context.
4. Provide an evidence_span quoting text that proves the entity exists.
5. Always return JSON with the key "entities" (use an empty list when nothing is found).
6. Output strictly valid JSON-no explanations, comments, or trailing commas.
"""

SYSTEM_PROMPT_TDM = """You are an expert in Deep Active Learning tasked with extracting Task-Dataset-Model structures from research papers.
Organize information into a hierarchy Task -> Dataset -> Model, capturing as many grounded facts as possible.

Definitions:
- Task: ML/NLP problems such as image classification, semantic segmentation, named entity recognition, sentiment analysis.
- Dataset: named benchmarks or corpora (CIFAR-10, SVHN, ImageNet, AG News, etc.).
- Model: concrete methods or algorithms (ResNet-50 with BALD, BERT with entropy sampling, proposed strategy, baselines).

Rules:
1. Work at high recall while remaining faithful to the text; do not invent tasks/datasets/models.
2. For each task, list every dataset mentioned; for each dataset, list every evaluated or compared model.
3. Mark model.role as "main" when the paper proposes/improves it, otherwise "baseline" (omit only if impossible to tell).
4. Capture explicit relations when present: evaluatedOn, usedFor, outperforms.
5. Every mention (task/dataset/model/relation) must have an evidence_span quoting the supporting text.
6. Always return a JSON object with the key "tasks" (use an empty list if nothing is found).
"""

# %%
GLOBAL_LLM_CALLS = {"simple": 0, "tdm": 0}


def get_text_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start += chunk_size - overlap
    return chunks


def retry_llm_call(func, *args, retries=3, delay=2.0, **kwargs):
    last_exception = None
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            # loguru-стиль форматирования
            logger.warning(
                "LLM attempt {}/{} failed: {}...",
                attempt + 1,
                retries,
                str(e)[:200],
            )
            time.sleep(delay * (attempt + 1))
    # После всех попыток - выбрасываем последнюю ошибку
    raise last_exception



def dump_raw_completion(label: str, message) -> None:
    """Логируем сырое содержимое ответа LLM.

    Используем loguru-форматирование ({}), без f-строк,
    чтобы избежать проблем с переносами строк и спецсимволами.
    """
    try:
        logger.debug(
            "{} RAW COMPLETION:\n{}",
            label,
            json.dumps(message.model_dump(), ensure_ascii=False, indent=2),
        )
    except Exception:
        # Фоллбек, если у message нет .model_dump() или json.dumps упал
        logger.debug(
            "{} RAW COMPLETION (repr): {!r}",
            label,
            message,
        )



def preview_raw_json(messages: List[Dict[str, str]], client, *, max_tokens: int = 800) -> Optional[str]:
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            timeout=LLM_TIMEOUT,
        )
        raw = response.choices[0].message.content or ""
        return raw.strip()
    except Exception as err:
        logger.warning(f"Не удалось получить сырой ответ LLM: {err}")
        return None


def build_simple_messages(text: str) -> List[Dict[str, str]]:
    user_prompt = f"""Text segment:
'''{text}'''

Task:
Identify every mention of the target entity types inside this segment. Err on the side of capturing all grounded mentions.

Return JSON exactly in this schema:
{{
  "entities": [
    {{
      "span": "<verbatim text span>",
      "entity_type": "AcquisitionStrategy | LabelingRegime | BudgetConstraint | PerformanceMetric | NoiseType | Tool",
      "short_description": "<1-2 sentence summary of the entity's role>",
      "evidence_span": "<quote that proves the entity appears in the text>"
    }}
  ]
}}
The key "entities" must always be present (use an empty list when nothing is found).
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_SIMPLE},
        {"role": "user", "content": user_prompt},
    ]


def build_tdm_messages(text: str) -> List[Dict[str, str]]:
    user_prompt = f"""Text segment:
'''{text}'''

Task:
1. Enumerate all Deep Active Learning tasks described in this segment.
2. For each task, list datasets and, under each dataset, list the models tied to it.
3. For each model, capture its role (main/baseline) when possible and include any explicit relations (evaluatedOn, usedFor, outperforms).

Return JSON exactly in this schema:

{{
  "tasks": [
    {{
      "task_name": "...",
      "task_type": "classification | segmentation | generation | other",
      "evidence_span": "...",
      "datasets": [
        {{
          "dataset_name": "...",
          "evidence_span": "...",
          "models": [
            {{
              "model_name": "...",
              "role": "main | baseline | null",
              "evidence_span": "...",
              "relations": [
                {{
                  "relation_type": "evaluatedOn | usedFor | outperforms",
                  "target": "...",
                  "evidence_span": "..."
                }}
              ]
            }}
          ]
        }}
      ]
    }}
  ]
}}

The key "tasks" must always be present (use an empty list when nothing is found).
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_TDM},
        {"role": "user", "content": user_prompt},
    ]


def extract_chunk_simple(text: str, client, stats: Dict[str, int], *, debug_raw: bool = False) -> List[SimpleEntity]:
    def _call():
        try:
            response = client.beta.chat.completions.parse(
                model=LLM_MODEL,
                messages=build_simple_messages(text),
                response_format=SimpleExtractionResult,
                timeout=60.0,
            )
            stats["simple_calls"] += 1
            GLOBAL_LLM_CALLS["simple"] += 1
            message = response.choices[0].message
            if debug_raw or LOG_SIMPLE_RAW:
                dump_raw_completion("SIMPLE", message)
            parsed = getattr(message, "parsed", None)
            entities = parsed.entities if parsed else []
            if (debug_raw or LOG_SIMPLE_RAW) and not entities:
                logger.warning("LLM вернул пустой список entities для текущего чанка.")
            return entities
        except Exception as parse_err:
            err_txt = str(parse_err)
            if '"entities"' in err_txt or "'entities'" in err_txt:
                logger.warning(
                    "LLM schema response missing 'entities'. Treating as empty list for this chunk."
                )
                return []
            raise

    try:
        return retry_llm_call(_call)
    except Exception as e:
        logger.error(f"CRITICAL: Simple extraction completely failed for chunk. Error: {e}")
        raise e


def extract_chunk_tdm(text: str, client, stats: Dict[str, int], *, debug_raw: bool = False) -> List[TDMTask]:
    def _call():
        try:
            response = client.beta.chat.completions.parse(
                model=LLM_MODEL,
                messages=build_tdm_messages(text),
                response_format=TDMExtractionResult,
                timeout=80.0,
            )
            stats["tdm_calls"] += 1
            GLOBAL_LLM_CALLS["tdm"] += 1
            message = response.choices[0].message
            if debug_raw or LOG_TDM_RAW:
                dump_raw_completion("TDM", message)
            parsed = getattr(message, "parsed", None)
            tasks = parsed.tasks if parsed else []
            if (debug_raw or LOG_TDM_RAW) and not tasks:
                logger.warning("LLM вернул пустой список tasks для текущего чанка.")
            return tasks
        except Exception as parse_err:
            err_txt = str(parse_err)
            if '"tasks"' in err_txt or "'tasks'" in err_txt:
                logger.warning(
                    "LLM schema response missing 'tasks'. Treating as empty list for this chunk."
                )
                return []
            raise

    try:
        return retry_llm_call(_call)
    except Exception as e:
        logger.error(f"CRITICAL: TDM extraction completely failed for chunk. Error: {e}")
        raise e


def process_paper_full(paper: dict, client) -> Optional[Dict[str, Any]]:
    full_text = parse_pdf_text(paper["pdf_path"])
    if not full_text:
        logger.warning(f"Empty text for paper {paper['paper_id']}")
        return None

    chunks = get_text_chunks(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
    logger.info(f"Paper {paper['paper_id']}: {len(chunks)} chunks")

    paper_stats = {"simple_calls": 0, "tdm_calls": 0}
    all_simple: List[SimpleEntity] = []
    all_tasks: List[TDMTask] = []

    for i, chunk in tqdm(enumerate(chunks), desc="Processing Chanks"):
        try:
            simples = extract_chunk_simple(chunk, client, paper_stats)
            all_simple.extend(simples)

            tasks = extract_chunk_tdm(chunk, client, paper_stats)
            all_tasks.extend(tasks)
        except Exception as e:
            logger.error(
                f" STOPPING paper processing due to error in chunk {i}/{len(chunks)}: {e}"
            )
            return None

    metadata = {
        "prompt_version": PROMPT_VERSION,
        "notebook_version": NOTEBOOK_VERSION,
        "llm_model": LLM_MODEL,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunks_processed": len(chunks),
        "llm_requests_simple": paper_stats["simple_calls"],
        "llm_requests_tdm": paper_stats["tdm_calls"],
        "estimated_prompt_tokens": len(chunks) * (CHUNK_SIZE // 4),
        "estimated_total_requests": paper_stats["simple_calls"] + paper_stats["tdm_calls"],
    }

    return {
        "paper_id": paper["paper_id"],
        "title": paper["title"],
        "simple_entities": [e.model_dump() for e in all_simple],
        "tdm_structures": [t.model_dump() for t in all_tasks],
        "chunks_count": len(chunks),
        "metadata": metadata,
    }

# %%
# Санити-тест извлечения на одном чанке
RUN_CHUNK_TEST = False  # переключите на True и перезапустите ячейку для одиночного теста
TEST_PAPER_ID: Optional[str] = None  # например, "61-2022.findings-acl.172" (без .pdf)
TEST_CHUNK_INDEX = 0  # по умолчанию берём первый чанк
CHUNK_PREVIEW_CHARS = 600
SANITY_DUMP_RAW_JSON = True

    
if RUN_CHUNK_TEST:
    if not openai_client:
        logger.error("Невозможно запустить тест: OpenAI клиент не инициализирован.")
    else:
        logger.info("\n=== САНИТИ-ТЕСТ LLM НА ОДНОМ ЧАНКЕ ===")
        sample_paper: Optional[Dict[str, Any]] = None
        
        if TEST_PAPER_ID:
            candidate_pdf = PDF_DIR / f"{TEST_PAPER_ID}.pdf"
            if candidate_pdf.exists():
                sample_paper = {
                    "paper_id": TEST_PAPER_ID,
                    "pdf_path": candidate_pdf,
                    "title": TEST_PAPER_ID,
                }
            else:
                logger.warning(
                    f"Файл {candidate_pdf} не найден. "
                    "Будет выбран первый доступный из текущей партии."
                )

        if sample_paper is None:
            fallback_candidates = (
                papers_batch
                or get_papers_to_process(
                    PDF_DIR, EXTRACTIONS_DIR, REFERENCES_CSV, batch_size=1
                )
            )
            if fallback_candidates:
                sample_paper = fallback_candidates[0]

        if sample_paper is None:
            logger.error(
                "Не удалось подобрать статью для теста. "
                "Убедитесь, что PDF-файлы доступны."
            )
        else:
            logger.info(f"Используем статью {sample_paper['paper_id']}")
            raw_text = parse_pdf_text(sample_paper["pdf_path"])
            test_chunks = get_text_chunks(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)

            if not test_chunks:
                logger.error(
                    f"Парсер вернул пустой текст для {sample_paper['paper_id']}. "
                    "Проверьте parse_pdf_text()."
                )
            else:
                chunk_idx = max(0, min(TEST_CHUNK_INDEX, len(test_chunks) - 1))
                first_chunk = test_chunks[chunk_idx]
                chunk_len = len(first_chunk)

                if SANITY_DUMP_RAW_JSON:
                    print("SANITY_DUMP_RAW_JSON")
                    raw_simple = preview_raw_json(build_simple_messages(first_chunk), openai_client, max_tokens=800)
                    if raw_simple:
                        logger.info(f"RAW SIMPLE JSON:\n{raw_simple}")
                    raw_tdm = preview_raw_json(build_tdm_messages(first_chunk), openai_client, max_tokens=1200)
                    if raw_tdm:
                        logger.info(f"RAW TDM JSON:\n{raw_tdm}")
                
                logger.info(f"Чанк #{chunk_idx} длиной {chunk_len} символов")
                if chunk_len < 400:
                    logger.warning(
                        "Выбранный чанк короче 400 символов - "
                        "LLM может не увидеть целевые сущности."
                    )

                preview = first_chunk[:CHUNK_PREVIEW_CHARS]
                logger.info(
                    f"Фрагмент чанка (первые {len(preview)} символов):\n"
                    f"{preview}{'...' if chunk_len > CHUNK_PREVIEW_CHARS else ''}"
                )
        
                if SANITY_DUMP_RAW_JSON:
                    raw_simple = preview_raw_json(
                        build_simple_messages(first_chunk),
                        openai_client,
                        max_tokens=800,
                    )
                    if raw_simple:
                        logger.info(f"RAW SIMPLE JSON:\n{raw_simple}")

                    raw_tdm = preview_raw_json(
                        build_tdm_messages(first_chunk),
                        openai_client,
                        max_tokens=1200,
                    )
                    if raw_tdm:
                        logger.info(f"RAW TDM JSON:\n{raw_tdm}")

                test_stats = {"simple_calls": 0, "tdm_calls": 0}
                try:
                    simple_preview = extract_chunk_simple(
                        first_chunk, openai_client, test_stats, debug_raw=True
                    )
                    tdm_preview = extract_chunk_tdm(
                        first_chunk, openai_client, test_stats, debug_raw=True
                    )
                    logger.info(
                        f" Simple entities: {len(simple_preview)} | "
                        f"TDM tasks: {len(tdm_preview)}"
                    )
                    if not simple_preview and not tdm_preview:
                        logger.warning(
                            "LLM вернул пустые entities/tasks для тестового чанка. "
                            "Проверьте промпты и содержимое текста."
                        )
                    if simple_preview:
                        logger.info(
                            f"Пример сущности: {simple_preview[0].model_dump()}"
                        )
                    if tdm_preview:
                        logger.info(
                            f"Пример задачи: {tdm_preview[0].model_dump()}"
                        )
                except Exception as preview_err:
                    logger.error(f"Санити-тест завершился ошибкой: {preview_err}")

# %%
# Execution Loop
if papers_batch and openai_client:
    logger.info(" Запуск цикла обработки (Chunks + LLM)...")

    for p in tqdm(papers_batch, desc="Processing Papers"):
        out_file = EXTRACTIONS_DIR / f"{p['paper_id']}.json"
        if out_file.exists():
            continue

        try:
            result = process_paper_full(p, openai_client)
            if result:
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {p['paper_id']} ({result['chunks_count']} chunks)")
            else:
                logger.warning(f" Пропущена статья {p['paper_id']} из-за ошибок извлечения")
        except Exception as e:
            logger.error(f"Unexpected error processing {p['paper_id']}: {e}")

    logger.success(" Обработка завершена.")

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from loguru import logger

# --- Общий справочник типов сущностей (для консистентности с шагом 2) ---

LLM_ENTITY_TYPES = [
    "AcquisitionStrategy",
    "LabelingRegime",
    "BudgetConstraint",
    "PerformanceMetric",
    "NoiseType",
    "Tool",
    "Task",
    "Dataset",
    "Model",
]

# Pydantic-модели для валидации извлечений


class ExtractionMetadata(BaseModel):
    """Служебная информация об одном прогоне LLM-экстракции по статье."""

    prompt_version: Optional[str] = None
    notebook_version: Optional[str] = None
    llm_model: Optional[str] = None
    created_at_utc: Optional[str] = None

    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    chunks_processed: Optional[int] = None

    llm_requests_simple: int = 0
    llm_requests_tdm: int = 0

    estimated_prompt_tokens: Optional[int] = None
    estimated_total_requests: Optional[int] = None


class SimpleEntityRecord(BaseModel):
    """Результат Simple-экстракции одной сущности из текста."""

    span: str
    entity_type: str = Field(
        ...,
        description=(
            "Высокоуровневый тип сущности. "
            "Ожидается одно из: " + ", ".join(LLM_ENTITY_TYPES)
        ),
    )
    short_description: Optional[str] = None
    evidence_span: Optional[str] = None


class TDMRelationRecord(BaseModel):
    """Связь модели с другой сущностью (dataset/task/model)."""

    relation_type: Optional[str] = Field(
        default=None,
        description="evaluatedOn | usedFor | outperforms",
    )
    target: Optional[str] = None
    evidence_span: Optional[str] = None


class TDMModelRecord(BaseModel):
    """Модель в контексте конкретного датасета и задачи."""

    model_name: str
    role: Optional[str] = Field(
        default=None,
        description="'main' (proposed) | 'baseline' | None",
    )
    evidence_span: Optional[str] = None
    relations: List[TDMRelationRecord] = Field(default_factory=list)


class TDMDatasetRecord(BaseModel):
    """Датасет в рамках конкретной задачи."""

    dataset_name: str
    evidence_span: Optional[str] = None
    models: List[TDMModelRecord] = Field(default_factory=list)


class TDMTaskRecord(BaseModel):
    """Задача (Task) верхнего уровня в структуре Task-Dataset-Model."""

    task_name: str
    task_type: Optional[str] = Field(
        default=None,
        description="classification | segmentation | generation | other | None",
    )
    evidence_span: Optional[str] = None
    datasets: List[TDMDatasetRecord] = Field(default_factory=list)


class PaperExtractionRecord(BaseModel):
    """Полный результат LLM-экстракции по одной статье."""

    paper_id: str
    title: str
    simple_entities: List[SimpleEntityRecord] = Field(default_factory=list)
    tdm_structures: List[TDMTaskRecord] = Field(default_factory=list)
    chunks_count: int
    metadata: ExtractionMetadata


# Загрузка и валидация JSON-файлов с экстракциями


def load_validated_extractions(
    dir_path: Path,
    *,
    fail_fast: bool = False,
) -> List[PaperExtractionRecord]:
    """
    Загрузить все *.json из каталога и провалидировать их через Pydantic.

    :param dir_path: каталог с extractions/*.json
    :param fail_fast: если True - при первой ошибке валидации бросаем исключение,
                      если False - логируем ошибки и продолжаем.
    :return: список валидных PaperExtractionRecord
    """
    records: List[PaperExtractionRecord] = []
    failures: List[Tuple[Path, Exception]] = []

    json_files = sorted(dir_path.glob("*.json"))
    if not json_files:
        logger.warning(f" В каталоге {dir_path} не найдено ни одного *.json файла")

    for json_path in json_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            record = PaperExtractionRecord.model_validate(raw)
            records.append(record)
        except ValidationError as ve:
            failures.append((json_path, ve))
            msg = f"Ошибка валидации файла {json_path.name}: {ve}"
            if fail_fast:
                logger.error(msg)
                raise
            else:
                logger.warning(msg)
        except Exception as exc:
            failures.append((json_path, exc))
            msg = f"Ошибка чтения/парсинга файла {json_path.name}: {exc}"
            if fail_fast:
                logger.error(msg)
                raise
            else:
                logger.warning(msg)

    if failures:
        logger.warning(
            f" Не удалось корректно распарсить {len(failures)} файлов из "
            f"{len(json_files)} в {dir_path}"
        )
        for path, err in failures[:5]:
            logger.warning(f"  - {path.name}: {err}")

    logger.info(f" Успешно загружено {len(records)} файлов с извлечениями")
    return records


# Сводная статистика по извлечениям


def summarize_extractions(records: List[PaperExtractionRecord]) -> None:
    """Вывести в лог базовую статистику по извлечениям и показать примеры."""
    if not records:
        logger.warning("Нет валидных извлечений для суммаризации.")
        return

    total_simple_entities = sum(len(rec.simple_entities) for rec in records)
    total_tasks = sum(len(rec.tdm_structures) for rec in records)
    total_llm_requests = sum(
        (rec.metadata.llm_requests_simple + rec.metadata.llm_requests_tdm)
        for rec in records
    )

    logger.info(
        f"Сводка по извлечениям: "
        f"simple_entities={total_simple_entities}, "
        f"tdm_tasks={total_tasks}, "
        f"llm_requests{total_llm_requests}"
    )

    # Распределение по типам сущностей
    entity_type_counts = {}
    for rec in records:
        for e in rec.simple_entities:
            entity_type_counts[e.entity_type] = entity_type_counts.get(e.entity_type, 0) + 1

    if entity_type_counts:
        logger.info("Распределение simple_entities по типам:")
        for etype, cnt in sorted(entity_type_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  - {etype}: {cnt}")

    # Пример первых N статей в табличном виде
    sample_rows = []
    for rec in records[:5]:
        sample_rows.append(
            {
                "paper_id": rec.paper_id,
                "simple_entities": len(rec.simple_entities),
                "tdm_tasks": len(rec.tdm_structures),
                "llm_calls": rec.metadata.llm_requests_simple + rec.metadata.llm_requests_tdm,
                "prompt_version": rec.metadata.prompt_version,
                "llm_model": rec.metadata.llm_model,
            }
        )
    if sample_rows:
        display(pd.DataFrame(sample_rows))


# Вызов шага 3

validated_extractions = load_validated_extractions(EXTRACTIONS_DIR, fail_fast=False)
summarize_extractions(validated_extractions)

# %%
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import re
from pathlib import Path

import numpy as np
from loguru import logger
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

# ----------------------------------------------------------------------------
# Конфигурация
# ----------------------------------------------------------------------------

CANONICAL_THRESHOLD = float(os.getenv("STEP7_CANONICAL_THRESHOLD", 0.18))
CANONICAL_OUTPUT = OUTPUT_DIR / "canonical_entities.jsonl"

# Допустимые типы сущностей (из шага извлечения)
VALID_ENTITY_TYPES = {
    "AcquisitionStrategy",
    "LabelingRegime",
    "BudgetConstraint",
    "PerformanceMetric",
    "NoiseType",
    "Tool",
    "Task",
    "Dataset",
    "Model",
}

# Простая эвристика для починки кривых entity_type
ENTITY_TYPE_FALLBACKS = {
    "AcquisitionStrategy  ": "AcquisitionStrategy",
    "PerfoAcquisitionStrategyrmanceMetric": "PerformanceMetric",
    "L": "Other",
    "Other": "Other",
}


# ----------------------------------------------------------------------------
# Модели (можно брать те же, что в Step 3; здесь оставлю для самодостаточности)
# ----------------------------------------------------------------------------

class ExtractionMetadata(BaseModel):
    prompt_version: Optional[str] = None
    notebook_version: Optional[str] = None
    llm_model: Optional[str] = None
    created_at_utc: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    chunks_processed: Optional[int] = None
    llm_requests_simple: int = 0
    llm_requests_tdm: int = 0
    estimated_prompt_tokens: Optional[int] = None
    estimated_total_requests: Optional[int] = None


class SimpleEntityRecord(BaseModel):
    span: str
    entity_type: str
    short_description: Optional[str] = None
    evidence_span: Optional[str] = None


class TDMRelationRecord(BaseModel):
    relation_type: Optional[str] = None
    target: Optional[str] = None
    evidence_span: Optional[str] = None


class TDMModelRecord(BaseModel):
    model_name: str
    role: Optional[str] = None
    evidence_span: Optional[str] = None
    relations: List[TDMRelationRecord] = []


class TDMDatasetRecord(BaseModel):
    dataset_name: str
    evidence_span: Optional[str] = None
    models: List[TDMModelRecord] = []


class TDMTaskRecord(BaseModel):
    task_name: str
    task_type: Optional[str] = None
    evidence_span: Optional[str] = None
    datasets: List[TDMDatasetRecord] = []


class PaperExtractionRecord(BaseModel):
    paper_id: str
    title: str
    simple_entities: List[SimpleEntityRecord]
    tdm_structures: List[TDMTaskRecord]
    chunks_count: int
    metadata: ExtractionMetadata


# ----------------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------------

def normalize_label(text: str) -> str:
    """Нормализация текстовой метки для сравнения (без учёта регистра и лишних пробелов)."""
    return re.sub(r"\s+", " ", text.strip().lower())


def normalize_entity_type(raw_type: str) -> str:
    """Нормализуем entity_type: режем пробелы, маппим кривые значения."""
    t = (raw_type or "").strip()
    if t in ENTITY_TYPE_FALLBACKS:
        return ENTITY_TYPE_FALLBACKS[t]
    if t in VALID_ENTITY_TYPES:
        return t
    # всё, что мы не узнали - логируем и сваливаем в 'Other'
    if t:
        logger.debug(f"Неизвестный entity_type='{t}', маппим в 'Other'")
    return "Other"


def collect_entity_mentions(records: List[PaperExtractionRecord]) -> List[Dict[str, Any]]:
    """Собираем сырые упоминания сущностей из simple_entities и TDM-структур."""
    mentions: List[Dict[str, Any]] = []
    for rec in records:
        paper_id = rec.paper_id

        # simple_entities
        for ent in rec.simple_entities:
            label = (ent.span or "").strip()
            if not label:
                continue
            etype = normalize_entity_type(ent.entity_type)
            mentions.append(
                {
                    "entity_type": etype,
                    "label": label,
                    "normalized_label": normalize_label(label),
                    "paper_id": paper_id,
                }
            )

        # TDM: Task / Dataset / Model
        for task in rec.tdm_structures:
            task_label = (task.task_name or "").strip()
            if task_label:
                mentions.append(
                    {
                        "entity_type": "Task",
                        "label": task_label,
                        "normalized_label": normalize_label(task_label),
                        "paper_id": paper_id,
                    }
                )

            for dataset in task.datasets:
                ds_label = (dataset.dataset_name or "").strip()
                if ds_label:
                    mentions.append(
                        {
                            "entity_type": "Dataset",
                            "label": ds_label,
                            "normalized_label": normalize_label(ds_label),
                            "paper_id": paper_id,
                        }
                    )

                for model in dataset.models:
                    m_label = (model.model_name or "").strip()
                    if m_label:
                        mentions.append(
                            {
                                "entity_type": "Model",
                                "label": m_label,
                                "normalized_label": normalize_label(m_label),
                                "paper_id": paper_id,
                            }
                        )

    logger.info(f"Собрано {len(mentions)} упоминаний сущностей из {len(records)} статей")
    return mentions


def canonicalize_entities(
    records: List[PaperExtractionRecord],
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str], str]]:
    """Канонизируем сущности по типам с помощью эмбеддингов и кластеризации.

    Возвращаем:
    - список canonical_entities
    - словарь alias_to_canonical[(entity_type, normalized_label)] -> canonical_id
    """
    mentions = collect_entity_mentions(records)

    # Группируем по (entity_type, normalized_label) - сначала аггрегируем точные совпадения
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for mention in mentions:
        etype = mention["entity_type"]
        norm = mention["normalized_label"]
        if not norm:
            continue

        bucket = grouped[etype].setdefault(
            norm,
            {
                "label": mention["label"],
                "normalized_label": norm,
                "examples": [],
                "count": 0,
            },
        )
        bucket["examples"].append(
            {"paper_id": mention["paper_id"], "label": mention["label"]}
        )
        bucket["count"] += 1

    logger.info(f"Группировка по типам: найдено {len(grouped)} entity_types")

    embedder = SentenceTransformer(EMBEDDING_MODEL_PATH)
    canonical_entities: List[Dict[str, Any]] = []
    alias_to_canonical: Dict[Tuple[str, str], str] = {}

    # Канонизация отдельно по каждому типу
    for entity_type, entry_map in grouped.items():
        entry_list = list(entry_map.values())
        if not entry_list:
            continue

        logger.info(
            f"Канонизация типа '{entity_type}': {len(entry_list)} уникальных нормализованных меток"
        )

        texts = [entry["label"] for entry in entry_list]
        if len(texts) == 1:
            # Тривиальный случай: только одна метка
            canonical_id = f"{entity_type[:3].upper()}_0001"
            canonical_entities.append(
                {
                    "canonical_id": canonical_id,
                    "entity_type": entity_type,
                    "canonical_label": entry_list[0]["label"],
                    "aliases": [entry_list[0]["label"]],
                    "source_mentions": entry_list[0]["examples"],
                    "frequency": entry_list[0]["count"],
                }
            )
            alias_to_canonical[(entity_type, entry_list[0]["normalized_label"])] = canonical_id
            continue

        embeddings = embedder.encode(texts, show_progress_bar=False)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=CANONICAL_THRESHOLD,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)

        cluster_buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for idx, lbl in enumerate(labels):
            cluster_buckets[int(lbl)].append(entry_list[idx])

        logger.info(
            f"Тип '{entity_type}': сформировано {len(cluster_buckets)} кластеров (threshold={CANONICAL_THRESHOLD})"
        )

        for cluster_idx, cluster_entries in cluster_buckets.items():
            canonical_id = f"{entity_type[:3].upper()}_{cluster_idx:04d}"
            # canonical_label = самая частотная форма
            canonical_label = max(cluster_entries, key=lambda item: item["count"])["label"]
            aliases = [entry["label"] for entry in cluster_entries]

            mentions_accum: List[Dict[str, Any]] = []
            freq = 0
            for entry in cluster_entries:
                freq += entry["count"]
                mentions_accum.extend(entry["examples"])
                alias_to_canonical[(entity_type, entry["normalized_label"])] = canonical_id

            canonical_entities.append(
                {
                    "canonical_id": canonical_id,
                    "entity_type": entity_type,
                    "canonical_label": canonical_label,
                    "aliases": aliases,
                    "source_mentions": mentions_accum,
                    "frequency": freq,
                }
            )

    canonical_entities.sort(key=lambda item: (item["entity_type"], item["canonical_label"]))

    CANONICAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CANONICAL_OUTPUT, "w", encoding="utf-8") as f:
        for entity in canonical_entities:
            json.dump(entity, f, ensure_ascii=False)
            f.write("\n")

    logger.info(
        f" Канонизировано {len(canonical_entities)} сущностей (entity_types={len(grouped)})"
    )

    return canonical_entities, alias_to_canonical


# Запуск шага 4
canonical_entities, alias_to_canonical = canonicalize_entities(validated_extractions)
logger.info(f"Пример канонической сущности: {canonical_entities[0] if canonical_entities else 'NONE'}")

# %%
Шаг 5. Построить реестр EntityCandidate  индивиды онтологии

# %%
5.1. Модель данных EntityCandidate

# %%
class EntityCandidate(BaseModel):
    canonical_id: str               # ACQ_2435
    entity_type: str                # AcquisitionStrategy / Dataset / Task / ...
    canonical_label: str            # 'Content Distance'
    frequency: int                  # сколько раз встретилось в корпусе
    ontology_class_iri: Optional[str] = None  # IRI класса в TBox
    status: str = "auto_pending"    # auto_accepted / auto_rejected / needs_review
    source_mentions: List[Dict[str, str]] = []  # paper_id + raw label (из canonical_entities)

# %%
5.2. Маппинг entity_type  класс онтологии

# %%
ENTITY_CLASS_MAP = {
    "AcquisitionStrategy": "dal:AcquisitionStrategy",
    "LabelingRegime": "dal:LabelingRegime",
    "BudgetConstraint": "dal:Constraint",          # или более специфичный подкласс
    "PerformanceMetric": "dal:Metric",
    "NoiseType": "dal:NoiseType",
    "Tool": "dal:Tool",
    "Task": "dal:Task",
    "Dataset": "dal:Dataset",
    "Model": "dal:Model",
    "Other": None,
}

# %%
Step 5 - EntityCandidate: сделать EntityCandidate.jsonl с ontology_class_iri и статусами.

# %%
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

# КОНФИГ ШАГА 5

# Путь к каноническим сущностям (должен совпадать с тем, что ты использовал на шаге 4)
CANONICAL_ENTITIES_PATH: Path = CANONICAL_OUTPUT  # например: OUTPUT_DIR / "canonical_entities.jsonl"

# Куда кладём EntityCandidate
ENTITY_CANDIDATES_PATH: Path = OUTPUT_DIR / "EntityCandidate.jsonl"

# Базовый namespace онтологии (ПОДПРАВЬ под свой OWL, если нужно)
DAL_NS = "https://w3id.org/deep-active-learning#"

# Маппинг типов сущностей (из LLM) на классы онтологии
ENTITY_CLASS_MAP: Dict[str, Optional[str]] = {
    "AcquisitionStrategy": f"{DAL_NS}AcquisitionStrategy",
    "LabelingRegime": f"{DAL_NS}LabelingRegime",
    "BudgetConstraint": f"{DAL_NS}Constraint",          # или более специфичный класс, если есть
    "PerformanceMetric": f"{DAL_NS}Metric",
    "NoiseType": f"{DAL_NS}NoiseType",
    "Tool": f"{DAL_NS}Tool",
    "Task": f"{DAL_NS}Task",
    "Dataset": f"{DAL_NS}Dataset",
    "Model": f"{DAL_NS}Model",
    # Всё странное/шум - явно в "Other"
    "Other": None,
}

# Пороговые параметры для авто-акцепта
MIN_FREQ_ACCEPT: int = 2        # минимум вхождений в корпусе для auto_accepted
MAX_LABEL_LEN: int = 80         # максимум длины строки (символов) для auto_accepted
MAX_TOKENS_LABEL: int = 12      # максимум слов в метке, иначе считаем предложением


# МОДЕЛЬ EntityCandidate

class EntityCandidate(BaseModel):
    canonical_id: str
    entity_type: str              # AcquisitionStrategy / Dataset / Task / ...
    canonical_label: str
    ontology_class_iri: Optional[str] = None

    status: str                   # auto_accepted / auto_rejected / needs_review
    frequency: int

    aliases: List[str] = Field(default_factory=list)
    source_mentions: List[Dict[str, Any]] = Field(default_factory=list)


# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ

def load_canonical_entities(path: Path) -> List[Dict[str, Any]]:
    """Загружает canonical_entities.jsonl  список dict."""
    if not path.exists():
        raise FileNotFoundError(f"Файл с каноническими сущностями не найден: {path}")

    entities: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entities.append(json.loads(line))
    logger.info(f" Загружено {len(entities)} канонических сущностей из {path}")
    return entities


def looks_like_sentence(label: str) -> bool:
    """Грубая эвристика: длинный текст c пробелами и точкой/вопросительным знаком в конце."""
    stripped = label.strip()
    tokens = stripped.split()
    if len(tokens) > MAX_TOKENS_LABEL:
        return True
    if stripped.endswith(".") or stripped.endswith("?"):
        if len(tokens) > 6:
            return True
    return False


def normalize_entity_type(entity_type: str) -> str:
    """Нормализуем шумные/ошибочные типы в один 'Other' там, где нужно."""
    clean = entity_type.strip()
    if clean in ENTITY_CLASS_MAP:
        return clean

    # Явные артефакты из логов:
    noisy_aliases = {
        "PerfoAcquisitionStrategyrmanceMetric",
        "AcquisitionStrategy  ",
        "L",
    }
    if clean in noisy_aliases:
        return "Other"

    # Всё незнакомое отправляем в Other
    return "Other"


def decide_status(
    entity_type: str,
    label: str,
    frequency: int,
    ontology_class_iri: Optional[str],
) -> str:
    """Определяем статус кандидата: auto_accepted / needs_review / auto_rejected."""
    # Если не можем привязать к классу онтологии - сразу отбрасываем
    if ontology_class_iri is None:
        return "auto_rejected"

    # Слишком длинный текст - скорее всего, это не имя сущности
    if len(label) > MAX_LABEL_LEN or looks_like_sentence(label):
        return "needs_review"

    # Слишком редкие и странные сущности можно тоже отправить в review
    if frequency < MIN_FREQ_ACCEPT:
        return "needs_review"

    # Всё остальное считаем достаточно надёжным
    return "auto_accepted"


def build_entity_candidates(
    canonical_entities_list: List[Dict[str, Any]]
) -> List[EntityCandidate]:
    """Строим список EntityCandidate из canonical_entities."""
    candidates: List[EntityCandidate] = []

    for ent in canonical_entities_list:
        canonical_id: str = ent["canonical_id"]
        raw_entity_type: str = ent["entity_type"]
        entity_type: str = normalize_entity_type(raw_entity_type)

        canonical_label: str = ent["canonical_label"].strip()
        frequency: int = int(ent.get("frequency", 1))

        aliases: List[str] = ent.get("aliases", []) or []
        source_mentions: List[Dict[str, Any]] = ent.get("source_mentions", []) or []

        ontology_class_iri: Optional[str] = ENTITY_CLASS_MAP.get(entity_type)

        status: str = decide_status(
            entity_type=entity_type,
            label=canonical_label,
            frequency=frequency,
            ontology_class_iri=ontology_class_iri,
        )

        candidate = EntityCandidate(
            canonical_id=canonical_id,
            entity_type=entity_type,
            canonical_label=canonical_label,
            ontology_class_iri=ontology_class_iri,
            status=status,
            frequency=frequency,
            aliases=aliases,
            source_mentions=source_mentions,
        )
        candidates.append(candidate)

    return candidates


def save_entity_candidates(candidates: List[EntityCandidate], path: Path) -> None:
    """Сохраняем EntityCandidate.jsonl."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for cand in candidates:
            f.write(cand.model_dump_json(ensure_ascii=False))
            f.write("\n")
    logger.info(f" Сохранено {len(candidates)} EntityCandidate в {path}")


def summarize_entity_candidates(candidates: List[EntityCandidate]) -> None:
    """Краткая сводка по статусам и типам."""
    total = len(candidates)
    by_status = Counter(c.status for c in candidates)
    by_type = Counter(c.entity_type for c in candidates)

    logger.info(f"Сводка по EntityCandidate: всего={total}")
    logger.info("Распределение по статусам:")
    for st, cnt in sorted(by_status.items(), key=lambda x: x[0]):
        share = cnt / total if total else 0.0
        logger.info(f"  - {st}: {cnt} ({share:.1%})")

    logger.info("Топ-10 типов по количеству кандидатов:")
    for etype, cnt in by_type.most_common(10):
        logger.info(f"  - {etype}: {cnt}")


# ЗАПУСК ШАГА 5

# 1) Берём canonical_entities либо из памяти, либо с диска
try:
    canonical_entities_list: List[Dict[str, Any]] = canonical_entities  # из предыдущей ячейки
    logger.info("Используем canonical_entities из памяти Python.")
except NameError:
    canonical_entities_list = load_canonical_entities(CANONICAL_ENTITIES_PATH)

# 2) Строим EntityCandidate
entity_candidates: List[EntityCandidate] = build_entity_candidates(canonical_entities_list)

# 3) Сохраняем на диск
save_entity_candidates(entity_candidates, ENTITY_CANDIDATES_PATH)

# 4) Печатаем сводку
summarize_entity_candidates(entity_candidates)

# %%
Step 6 - RelationCandidate: собрать и нормализовать все TDM-отношения в RelationCandidate.jsonl + маппинг на object properties.

# %%
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from pydantic import BaseModel, Field

# КОНФИГ ШАГА 6 (RelationCandidate)

RELATION_CANDIDATES_PATH: Path = OUTPUT_DIR / "RelationCandidate.jsonl"

# Порог частоты для auto_accepted
MIN_RELATION_FREQ_ACCEPT: int = 2

# Разрешённые типы отношений (можешь расширить при необходимости)
ALLOWED_RELATION_TYPES = {"evaluatedOn", "usedFor", "outperforms"}


# Pydantic-модели RelationCandidate

class RelationEvidence(BaseModel):
    paper_id: str
    source: str = Field(..., description="tdm_implied | tdm_explicit")
    evidence_span: Optional[str] = None


class RelationCandidate(BaseModel):
    relation_id: str
    relation_type: str   # evaluatedOn | usedFor | outperforms

    subject_canonical_id: str
    object_canonical_id: str
    subject_type: str
    object_type: str

    status: str          # auto_accepted | needs_review | auto_rejected
    frequency: int

    evidences: List[RelationEvidence] = Field(default_factory=list)


# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ АЛИАСОВ И КАНОНИКАЛИЗАЦИИ

def normalize_label(text: str) -> str:
    """Та же нормализация, что и на шаге 4 (на всякий случай дублируем)."""
    return re.sub(r"\s+", " ", text.strip().lower())


def build_alias_index(canonical_entities_list: List[Dict[str, Any]]) -> Dict[Tuple[str, str], str]:
    """
    Строим индекс (entity_type, normalized_label) -> canonical_id
    по canonical_entities.jsonl.
    """
    index: Dict[Tuple[str, str], str] = {}

    for ent in canonical_entities_list:
        entity_type: str = ent["entity_type"]
        canonical_id: str = ent["canonical_id"]
        labels: List[str] = [ent["canonical_label"]] + list(ent.get("aliases") or [])

        for label in labels:
            norm = normalize_label(label)
            key = (entity_type, norm)
            # Если коллизия - оставляем первый, можно логировать при желании
            if key not in index:
                index[key] = canonical_id

    logger.info(f"Индекс алиасов построен: {len(index)} ключей (entity_type, normalized_label)")
    return index


def resolve_canonical_id(
    alias_index: Dict[Tuple[str, str], str],
    entity_type: str,
    label: Optional[str],
) -> Optional[str]:
    """Пытаемся найти canonical_id по (entity_type, label)."""
    if not label:
        return None
    norm = normalize_label(label)
    return alias_index.get((entity_type, norm))


# ИЗВЛЕЧЕНИЕ СВЯЗЕЙ ИЗ TDM-СТРУКТУР

def collect_relation_candidates(
    records: List[PaperExtractionRecord],
    canonical_entities_list: List[Dict[str, Any]],
) -> List[RelationCandidate]:
    """
    Строим RelationCandidate на основе TDM-структур и aliascanonical.
    Правила:
      - Dataset usedFor Task
      - Model usedFor Task
      - Model evaluatedOn Dataset
      - Model outperforms Model (из model.relations)
    """
    alias_index = build_alias_index(canonical_entities_list)

    # key = (relation_type, subj_id, obj_id, subj_type, obj_type)
    grouped_evidences: Dict[
        Tuple[str, str, str, str, str],
        List[RelationEvidence],
    ] = defaultdict(list)

    for rec in records:
        paper_id = rec.paper_id

        for task in rec.tdm_structures:
            task_label = task.task_name.strip() if task.task_name else ""
            task_cid = resolve_canonical_id(alias_index, "Task", task_label)

            for dataset in task.datasets:
                dataset_label = dataset.dataset_name.strip() if dataset.dataset_name else ""
                dataset_cid = resolve_canonical_id(alias_index, "Dataset", dataset_label)

                # 1) Dataset usedFor Task
                if task_cid and dataset_cid:
                    key = ("usedFor", dataset_cid, task_cid, "Dataset", "Task")
                    grouped_evidences[key].append(
                        RelationEvidence(
                            paper_id=paper_id,
                            source="tdm_implied",
                            evidence_span=dataset.evidence_span or task.evidence_span,
                        )
                    )

                # 2-4) Модели и их отношения
                for model in dataset.models:
                    model_label = model.model_name.strip() if model.model_name else ""
                    model_cid = resolve_canonical_id(alias_index, "Model", model_label)

                    if not model_cid:
                        continue

                    # 2) Model usedFor Task
                    if task_cid:
                        key = ("usedFor", model_cid, task_cid, "Model", "Task")
                        grouped_evidences[key].append(
                            RelationEvidence(
                                paper_id=paper_id,
                                source="tdm_implied",
                                evidence_span=model.evidence_span or task.evidence_span,
                            )
                        )

                    # 3) Model evaluatedOn Dataset
                    if dataset_cid:
                        key = ("evaluatedOn", model_cid, dataset_cid, "Model", "Dataset")
                        grouped_evidences[key].append(
                            RelationEvidence(
                                paper_id=paper_id,
                                source="tdm_implied",
                                evidence_span=model.evidence_span or dataset.evidence_span,
                            )
                        )

                    # 4) Явные отношения из model.relations (outperforms / evaluatedOn / usedFor)
                    for rel in model.relations or []:
                        rel_type = (rel.relation_type or "").strip()
                        if rel_type not in ALLOWED_RELATION_TYPES:
                            continue
                        target_label = (rel.target or "").strip()
                        if not target_label:
                            continue

                        if rel_type == "outperforms":
                            target_type = "Model"
                        elif rel_type == "evaluatedOn":
                            target_type = "Dataset"
                        elif rel_type == "usedFor":
                            target_type = "Task"
                        else:
                            continue

                        target_cid = resolve_canonical_id(alias_index, target_type, target_label)
                        if not target_cid:
                            continue

                        key = (rel_type, model_cid, target_cid, "Model", target_type)
                        grouped_evidences[key].append(
                            RelationEvidence(
                                paper_id=paper_id,
                                source="tdm_explicit",
                                evidence_span=rel.evidence_span or model.evidence_span,
                            )
                        )

    # Теперь собираем RelationCandidate
    relation_candidates: List[RelationCandidate] = []
    for idx, (key, evid_list) in enumerate(grouped_evidences.items(), start=1):
        relation_type, subj_id, obj_id, subj_type, obj_type = key
        freq = len(evid_list)

        status = "auto_accepted" if freq >= MIN_RELATION_FREQ_ACCEPT else "needs_review"

        relation_id = f"REL_{idx:05d}"

        candidate = RelationCandidate(
            relation_id=relation_id,
            relation_type=relation_type,
            subject_canonical_id=subj_id,
            object_canonical_id=obj_id,
            subject_type=subj_type,
            object_type=obj_type,
            status=status,
            frequency=freq,
            evidences=evid_list,
        )
        relation_candidates.append(candidate)

    logger.info(
        f"Сформировано {len(relation_candidates)} RelationCandidate "
        f"из {len(grouped_evidences)} уникальных ключей отношений"
    )
    return relation_candidates


def save_relation_candidates(
    relations: List[RelationCandidate],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rel in relations:
            f.write(rel.model_dump_json(ensure_ascii=False))
            f.write("\n")
    logger.info(f" Сохранено {len(relations)} RelationCandidate в {path}")


def summarize_relation_candidates(relations: List[RelationCandidate]) -> None:
    total = len(relations)
    by_type = Counter(rel.relation_type for rel in relations)
    by_status = Counter(rel.status for rel in relations)

    logger.info(f"Сводка по RelationCandidate: всего={total}")
    logger.info("По типам отношений:")
    for rtype, cnt in sorted(by_type.items(), key=lambda x: x[0]):
        share = cnt / total if total else 0.0
        logger.info(f"  - {rtype}: {cnt} ({share:.1%})")

    logger.info("По статусам:")
    for st, cnt in sorted(by_status.items(), key=lambda x: x[0]):
        share = cnt / total if total else 0.0
        logger.info(f"  - {st}: {cnt} ({share:.1%})")


# ЗАПУСК ШАГА 6

# 1) canonical_entities_list - либо из памяти, либо загрузим как на шаге 5
try:
    canonical_entities_list = canonical_entities
    logger.info("Для RelationCandidate используем canonical_entities из памяти Python.")
except NameError:
    canonical_entities_list = load_canonical_entities(CANONICAL_ENTITIES_PATH)

# 2) validated_extractions уже есть после шага 3
relation_candidates = collect_relation_candidates(
    records=validated_extractions,
    canonical_entities_list=canonical_entities_list,
)

# 3) Сохраняем и печатаем сводку
save_relation_candidates(relation_candidates, RELATION_CANDIDATES_PATH)
summarize_relation_candidates(relation_candidates)

# %%
Step 7 - ABox builder: генератор OWL/Turtle для индивидов и триплетов + reasoner/SHACL.

# %%
Шага 7A: построение ABox из EntityCandidate + RelationCandidate.

# %%
from pathlib import Path
import json
import re
from typing import Dict, Any, List, Tuple

from loguru import logger
from rdflib import Graph, Namespace, URIRef, RDF, RDFS, Literal

# Пути - подстрой под свой проект
BASE_DIR = Path("..").resolve()
ONTO_PATH = BASE_DIR / "outputs" / "deep_active_learning" / "step6c" / "deep_active_learning_enriched.owl"
STEP7_DIR = BASE_DIR / "outputs" / "deep_active_learning" / "step7"
ENTITY_CANDIDATES_PATH = STEP7_DIR / "EntityCandidate.jsonl"
RELATION_CANDIDATES_PATH = STEP7_DIR / "RelationCandidate.jsonl"

KG_OUTPUT_DIR = STEP7_DIR / "kg"
KG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KG_TURTLE_PATH = KG_OUTPUT_DIR / "deep_active_learning_kg.ttl"
ENTITY_IRI_MAP_PATH = KG_OUTPUT_DIR / "entity_iri_map.json"

# --------------------------------------------------------------------
# 1. Маппинг типов сущностей на классы онтологии
#    ЗДЕСЬ НУЖНО СИНХРОНИЗИРОВАТЬ С ТВОЕЙ TBOX
# --------------------------------------------------------------------
# Предполагаем, что в онтологии есть класс с таким фрагментом IRI
# (например, http://example.org/dal#Task и т.п.)
ENTITY_TYPE_TO_CLASS_LOCAL = {
    "Task": "Task",
    "Dataset": "Dataset",
    "Model": "Model",
    "AcquisitionStrategy": "AcquisitionStrategy",
    "PerformanceMetric": "PerformanceMetric",
    "BudgetConstraint": "BudgetConstraint",
    "LabelingRegime": "LabelingRegime",
    "NoiseType": "NoiseType",
    "Tool": "Tool",
    # резерв, если что-то будет попадать в "Other"
    "Other": "DAL_Entity",
}

# --------------------------------------------------------------------
# 2. Маппинг типов отношений на объектные свойства онтологии
#    Тоже нужно подстроить под реальные имена в TBox
# --------------------------------------------------------------------
RELATION_TYPE_TO_PROPERTY_LOCAL = {
    # модель / стратегия  задача
    "usedFor": "usedForTask",          # или "solvesTask" / "isUsedForTask" и т.п.
    # модель / стратегия  датасет
    "evaluatedOn": "evaluatedOnDataset",
    # модель / стратегия  другая модель / стратегия
    "outperforms": "outperformsMethod",
}

# --------------------------------------------------------------------
# Вспомогательные функции
# --------------------------------------------------------------------
def safe_local_name(text: str, fallback: str) -> str:
    """
    Делает из произвольной строки безопасный фрагмент IRI.
    """
    if not text:
        return fallback
    t = text.strip()
    # убираем кавычки
    t = t.replace('"', "").replace("'", "")
    # заменяем всё не [A-Za-z0-9_] на '_'
    t = re.sub(r"[^A-Za-z0-9_]+", "_", t)
    # не начинаем с цифры
    if re.match(r"^[0-9]", t):
        t = f"e_{t}"
    # если всё исчезло, используем fallback
    return t or fallback


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    logger.info(f"Загружено {len(rows)} строк из {path}")
    return rows

# %%
ONTO_PATH

# %%
# ---------------------------------------------------------
# Загрузка онтологии и базовый namespace
# ---------------------------------------------------------
onto_graph = Graph()
onto_graph.parse(str(ONTO_PATH))
logger.info(f"Онтология загружена: {ONTO_PATH}")

# Если в онтологии объявлен базовый namespace, можно считать его из графа.
# Для простоты - задаём руками; важно, чтобы он совпадал с тем, как ты
# создавал классы в шаге 6.
DAL = Namespace("http://example.org/deep_active_learning#")
onto_graph.bind("dal", DAL)

# %%
Шаг 7A: создание индивидов по EntityCandidate

# %%
from collections import defaultdict
from pathlib import Path
import json
import re
from typing import Dict, Any, List, Tuple

from loguru import logger
from rdflib import Graph, Namespace, URIRef, RDF, RDFS, Literal

# ---------------------------------------------------------
# Пути (если уже объявлены - этот блок можно пропустить)
# ---------------------------------------------------------
BASE_DIR = Path("..").resolve()
# ONTO_PATH = BASE_DIR / "data" / "ontology" / "deep_active_learning_enriched.owl"
STEP7_DIR = BASE_DIR / "outputs" / "deep_active_learning" / "step7"
ENTITY_CANDIDATES_PATH = STEP7_DIR / "EntityCandidate.jsonl"
RELATION_CANDIDATES_PATH = STEP7_DIR / "RelationCandidate.jsonl"

KG_OUTPUT_DIR = STEP7_DIR / "kg"
KG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KG_TURTLE_PATH = KG_OUTPUT_DIR / "deep_active_learning_kg.ttl"
ENTITY_IRI_MAP_PATH = KG_OUTPUT_DIR / "entity_iri_map.json"

# ---------------------------------------------------------
# Маппинги типов  классы / свойства онтологии
# (подстрой под реальные названия в TBox!)
# ---------------------------------------------------------
ENTITY_TYPE_TO_CLASS_LOCAL = {
    "Task": "Task",
    "Dataset": "Dataset",
    "Model": "Model",
    "AcquisitionStrategy": "AcquisitionStrategy",
    "PerformanceMetric": "PerformanceMetric",
    "BudgetConstraint": "BudgetConstraint",
    "LabelingRegime": "LabelingRegime",
    "NoiseType": "NoiseType",
    "Tool": "Tool",
    "Other": "DAL_Entity",
}

RELATION_TYPE_TO_PROPERTY_LOCAL = {
    "usedFor": "usedForTask",
    "evaluatedOn": "evaluatedOnDataset",
    "outperforms": "outperformsMethod",
}

# ---------------------------------------------------------
# Утилиты
# ---------------------------------------------------------
def safe_local_name(text: str, fallback: str) -> str:
    if not text:
        return fallback
    t = text.strip()
    t = t.replace('"', "").replace("'", "")
    t = re.sub(r"[^A-Za-z0-9_]+", "_", t)
    if re.match(r"^[0-9]", t):
        t = f"e_{t}"
    return t or fallback


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    logger.info(f"Загружено {len(rows)} строк из {path}")
    return rows


def filter_entity_candidates(
    rows: List[Dict[str, Any]],
    statuses: Tuple[str, ...] = ("auto_accepted",),
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        status = row.get("status", "needs_review")
        if status in statuses:
            filtered.append(row)
    logger.info(f"Отфильтровано {len(filtered)} EntityCandidate со статусами {list(statuses)}")
    return filtered


def filter_relation_candidates(
    rows: List[Dict[str, Any]],
    statuses: Tuple[str, ...] = ("auto_accepted",),
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        status = row.get("status", "needs_review")
        if status in statuses:
            filtered.append(row)
    logger.info(f"Отфильтровано {len(filtered)} RelationCandidate со статусами {list(statuses)}")
    return filtered

# ---------------------------------------------------------
# Загрузка онтологии
# ---------------------------------------------------------
onto_graph = Graph()
onto_graph.parse(str(ONTO_PATH))
logger.info(f"Онтология загружена из {ONTO_PATH}")

# Заменить IRI на твой реальный base из Protégé
DAL = Namespace("http://example.org/deep_active_learning#")
onto_graph.bind("dal", DAL)

# ---------------------------------------------------------
# Шаг 7A: загрузка кандидатов и создание IRI
# ---------------------------------------------------------
entity_rows_raw = load_jsonl(ENTITY_CANDIDATES_PATH)
relation_rows_raw = load_jsonl(RELATION_CANDIDATES_PATH)

entity_rows = filter_entity_candidates(entity_rows_raw, statuses=("auto_accepted",))
relation_rows = filter_relation_candidates(relation_rows_raw, statuses=("auto_accepted",))


def build_entity_iris(
    entities: List[Dict[str, Any]],
    base_ns: Namespace,
) -> Dict[str, URIRef]:
    entity_id_to_iri: Dict[str, URIRef] = {}
    for row in entities:
        entity_id = row.get("entity_id") or row.get("canonical_id")
        if not entity_id:
            continue

        entity_type = row.get("entity_type", "Other")
        label = row.get("canonical_label") or row.get("label") or entity_id

        local = safe_local_name(label, fallback=entity_id)
        local_name = f"{entity_type}_{local}"

        iri = base_ns[local_name]
        entity_id_to_iri[entity_id] = iri

    logger.info(f"Построено {len(entity_id_to_iri)} IRI для сущностей")
    return entity_id_to_iri


def materialize_entities_to_graph(
    g: Graph,
    base_ns: Namespace,
    entities: List[Dict[str, Any]],
    entity_id_to_iri: Dict[str, URIRef],
) -> None:
    created = 0
    missing_type = 0

    for row in entities:
        entity_id = row.get("entity_id") or row.get("canonical_id")
        if not entity_id:
            continue

        iri = entity_id_to_iri.get(entity_id)
        if iri is None:
            continue

        entity_type = row.get("entity_type", "Other")
        canonical_label = row.get("canonical_label") or row.get("label") or entity_id

        class_local = ENTITY_TYPE_TO_CLASS_LOCAL.get(entity_type)
        if not class_local:
            missing_type += 1
            logger.warning(
                f"Нет маппинга класса для entity_type='{entity_type}', entity_id='{entity_id}'"
            )
            continue

        class_iri = base_ns[class_local]

        g.add((iri, RDF.type, class_iri))
        g.add((iri, RDFS.label, Literal(canonical_label)))
        created += 1

    logger.info(
        f"В граф добавлено {created} индивидов (rdf:type + rdfs:label), "
        f"пропущено из-за отсутствия маппинга класса: {missing_type}"
    )


entity_id_to_iri = build_entity_iris(entity_rows, DAL)
materialize_entities_to_graph(onto_graph, DAL, entity_rows, entity_id_to_iri)

# сохраняем маппинг entity_id  IRI
entity_iri_map_serializable = {ent_id: str(iri) for ent_id, iri in entity_id_to_iri.items()}
with ENTITY_IRI_MAP_PATH.open("w", encoding="utf-8") as f:
    json.dump(entity_iri_map_serializable, f, ensure_ascii=False, indent=2)
logger.info(f" Маппинг entity_id  IRI сохранён в {ENTITY_IRI_MAP_PATH}")

# ---------------------------------------------------------
# Шаг 7B: объектные свойства из RelationCandidate
# ---------------------------------------------------------
def materialize_relations_to_graph(
    g: Graph,
    base_ns: Namespace,
    relations: List[Dict[str, Any]],
    entity_id_to_iri: Dict[str, URIRef],
) -> None:
    created = 0
    skipped_no_entities = 0
    skipped_no_prop = 0

    for row in relations:
        relation_type = row.get("relation_type")
        if not relation_type:
            continue

        prop_local = RELATION_TYPE_TO_PROPERTY_LOCAL.get(relation_type)
        if not prop_local:
            skipped_no_prop += 1
            continue

        prop_iri = base_ns[prop_local]

        src_id = row.get("source_entity_id")
        tgt_id = row.get("target_entity_id")
        if not src_id or not tgt_id:
            skipped_no_entities += 1
            continue

        subj = entity_id_to_iri.get(src_id)
        obj = entity_id_to_iri.get(tgt_id)
        if subj is None or obj is None:
            skipped_no_entities += 1
            continue

        g.add((subj, prop_iri, obj))
        created += 1

    logger.info(
        f"В граф добавлено {created} объектных утверждений, "
        f"пропущено (нет сущностей): {skipped_no_entities}, "
        f"пропущено (нет маппинга свойства): {skipped_no_prop}"
    )


materialize_relations_to_graph(onto_graph, DAL, relation_rows, entity_id_to_iri)

# ---------------------------------------------------------
# Шаг 7C: сериализация KG (TBox + ABox)
# ---------------------------------------------------------
onto_graph.serialize(destination=str(KG_TURTLE_PATH), format="turtle")
logger.info(f" KG (TBox + ABox) сохранён в {KG_TURTLE_PATH}")

# %%
# ---------------------------------------------------------
# Шаг 7A. Создание индивидов (NamedIndividuals) по EntityCandidate
# ---------------------------------------------------------

def build_entity_iris(
    entities: List[Dict[str, Any]],
    base_ns: Namespace,
) -> Dict[str, URIRef]:
    """
    Строит IRI для каждого entity_id.
    Возвращает маппинг entity_id -> URIRef.
    """
    entity_id_to_iri: Dict[str, URIRef] = {}

    for row in entities:
        entity_id = row.get("entity_id") or row.get("canonical_id")
        if not entity_id:
            # если по какой-то причине id нет - пропускаем
            continue

        entity_type = row.get("entity_type", "Other")
        label = row.get("canonical_label") or row.get("label") or entity_id

        # безопасное локальное имя: тип + id или по лейблу
        local = safe_local_name(label, fallback=entity_id)
        # чтобы избежать коллизий типов, добавим префикс типа
        local_name = f"{entity_type}_{local}"

        iri = base_ns[local_name]
        entity_id_to_iri[entity_id] = iri

    logger.info(f"Построено {len(entity_id_to_iri)} IRI для сущностей")
    return entity_id_to_iri


def materialize_entities_to_graph(
    g: Graph,
    base_ns: Namespace,
    entities: List[Dict[str, Any]],
    entity_id_to_iri: Dict[str, URIRef],
) -> None:
    """
    Добавляет в граф triples вида:
    - <entity> rdf:type <Class>
    - <entity> rdfs:label "Canonical label"
    """
    created = 0
    missing_type = 0

    for row in entities:
        entity_id = row.get("entity_id") or row.get("canonical_id")
        if not entity_id:
            continue

        iri = entity_id_to_iri.get(entity_id)
        if iri is None:
            continue

        entity_type = row.get("entity_type", "Other")
        canonical_label = row.get("canonical_label") or row.get("label") or entity_id

        class_local = ENTITY_TYPE_TO_CLASS_LOCAL.get(entity_type)
        if not class_local:
            missing_type += 1
            logger.warning(
                f"Нет маппинга класса для entity_type='{entity_type}', entity_id='{entity_id}'"
            )
            continue

        class_iri = base_ns[class_local]

        # rdf:type
        g.add((iri, RDF.type, class_iri))
        # rdfs:label
        g.add((iri, RDFS.label, Literal(canonical_label)))

        created += 1

    logger.info(
        f"В граф добавлено {created} индивидов (rdf:type + rdfs:label). "
        f"Пропущено из-за отсутствия маппинга класса: {missing_type}"
    )


# Создаём IRI и материализуем индивиды
entity_id_to_iri = build_entity_iris(entity_rows, DAL)
materialize_entities_to_graph(onto_graph, DAL, entity_rows, entity_id_to_iri)

# Сохраняем маппинг entity_id  IRI для дальнейших шагов
entity_iri_map_serializable = {
    ent_id: str(iri) for ent_id, iri in entity_id_to_iri.items()
}
with ENTITY_IRI_MAP_PATH.open("w", encoding="utf-8") as f:
    json.dump(entity_iri_map_serializable, f, ensure_ascii=False, indent=2)

logger.info(f" Маппинг entity_id  IRI сохранён в {ENTITY_IRI_MAP_PATH}")

# %%
Шаг 7B: добавление фактов по RelationCandidate

# %%
Шаг 7B: добавление фактов по RelationCandidate

# %%
Шаг 7C: сохранение объединённого графа (TBox + ABox)

# %%
# ---------------------------------------------------------
# Шаг 7C. Сериализация ABox + TBox в Turtle
# ---------------------------------------------------------

onto_graph.serialize(destination=str(KG_TURTLE_PATH), format="turtle")
logger.info(f" KG (TBox + ABox) сохранён в {KG_TURTLE_PATH}")
