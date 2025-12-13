# %% [markdown]
"""
# %%
# Дедупликация CQs через LLM

Этот ноутбук демонстрирует дедупликацию компетентностных вопросов (CQs) через LLM (ChatGPT), которая заменяет простую текстовую дедупликацию на семантическую.

## Преимущества LLM-подхода:
- **Семантическое понимание**: обнаружение перефразировок и синонимов
- **Контекстная оценка**: учет доменной специфики Deep Active Learning
- **Обнаружение near-duplicates**: вопросы, которые задают одно и то же, но сформулированы по-разному

## Входные данные:
- `cq_scored.jsonl` (только CQs с `quality_decision == 'keep'`)

## Выходные данные:
- `cq_final.jsonl` с дедуплицированными CQs и новыми стабильными ID
"""

# %%
import sys
from pathlib import Path
import json
import os
import time
from typing import List, Dict, Optional, Any
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# %%
# Добавляем путь к проекту
sys.path.insert(0, str(Path().absolute().parent))

# %%
# Загрузка переменных окружения из .env файла
env_path = Path().absolute().parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

from loguru import logger
from pydantic import BaseModel, Field
from IPython.display import display

from oasis.pipelines.stage1_extract_topics import _create_openai_client

# %%
# Настройки визуализации
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 200)
pd.set_option('display.width', None)

logger.info(" Импорты загружены")

# %%
# НАСТРОЙКИ ПУТЕЙ И ПАРАМЕТРОВ

# %%
# Пути к данным
TOPIC = "deep_active_learning"
CQ_SCORED_PATH = Path(f"../outputs/{TOPIC}/step3/cq_scored.jsonl")
OUTPUT_DIR = Path(f"../outputs/{TOPIC}/step3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Параметры LLM для дедупликации
LLM_MODEL = os.getenv("OPENAI_CANONICAL_MODEL", "gpt-5.1")
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 4000  # Достаточно для 306 CQs
LLM_MAX_RETRIES = 3
LLM_TIMEOUT = 180.0  # Таймаут в секундах (3 минуты)

# %%
# Trace ID для трассировки
TRACE_ID = "demo_step3_llm_deduplication"

logger.info(f" Настройки загружены")
logger.info(f"  CQ scored path: {CQ_SCORED_PATH}")
logger.info(f"  Output dir: {OUTPUT_DIR}")
logger.info(f"  LLM model: {LLM_MODEL}, timeout: {LLM_TIMEOUT}с, max_tokens: {LLM_MAX_TOKENS}")

# %%
# ИНИЦИАЛИЗАЦИЯ КЛИЕНТОВ

# %%
# Инициализация OpenAI клиента
try:
    openai_client = _create_openai_client(timeout=LLM_TIMEOUT)
    logger.success(f" OpenAI клиент инициализирован (timeout: {LLM_TIMEOUT}с)")
except Exception as e:
    logger.error(f" Ошибка инициализации OpenAI клиента: {e}")
    openai_client = None

# %%
# ЗАГРУЗКА ДАННЫХ

# %%
# Pydantic модели (переиспользование из основного ноутбука)
class EvidenceChunkRef(BaseModel):
    """Ссылка на evidence chunk."""
    chunk_id: str = Field(..., description="Идентификатор чанка")
    article_id: str = Field(..., description="Идентификатор статьи")

class CQCandidate(BaseModel):
    """Модель кандидата CQ."""
    cq_id: str = Field(..., description="Идентификатор CQ (ST_XX_QYY)")
    super_topic_id: str = Field(..., description="Идентификатор супертемы")
    text: str = Field(..., description="Текст вопроса")
    role: str = Field(..., description="Роль вопроса (class/relation/process/constraint/comparison)")
    answer: str = Field(..., description="Ответ на вопрос")
    answer_type: str = Field(..., description="Тип ответа (text/list/numeric/boolean/INSUFFICIENT_EVIDENCE)")
    evidence_chunks: List[EvidenceChunkRef] = Field(..., description="Список ссылок на evidence chunks")
    generation_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Метаданные генерации")

class QualityScores(BaseModel):
    """Оценки качества CQ."""
    relevance: float = Field(..., ge=0.0, le=1.0, description="DAL-специфичность (0-1)")
    clarity: float = Field(..., ge=0.0, le=1.0, description="Читаемость и однозначность (0-1)")
    groundedness: float = Field(..., ge=0.0, le=1.0, description="Насколько ответ поддержан чанками (0-1)")
    answerability: float = Field(..., ge=0.0, le=1.0, description="Ответ не INSUFFICIENT_EVIDENCE (0-1)")

class CQScored(CQCandidate):
    """Модель оцененного CQ (расширяет CQCandidate)."""
    quality_scores: QualityScores = Field(..., description="Оценки качества")
    quality_decision: str = Field(..., description="Решение: keep или drop")

class CQFinal(CQScored):
    """Финальный CQ с новым стабильным ID."""
    pass  # Все поля наследуются от CQScored

def load_cq_scored(path: Path) -> List[CQScored]:
    """Загружает CQs из JSONL файла и фильтрует только keep."""
    cqs = []
    
    if not path.exists():
        logger.error(f" Файл не найден: {path}")
        return cqs
    
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Преобразуем evidence_chunks в EvidenceChunkRef
                if 'evidence_chunks' in data:
                    evidence_refs = [
                        EvidenceChunkRef(**ec) if isinstance(ec, dict) else ec
                        for ec in data['evidence_chunks']
                    ]
                    data['evidence_chunks'] = evidence_refs
                
                # Преобразуем quality_scores в QualityScores
                if 'quality_scores' in data:
                    data['quality_scores'] = QualityScores(**data['quality_scores'])
                
                cq = CQScored(**data)
                cqs.append(cq)
            except Exception as e:
                logger.warning(f" Ошибка загрузки строки {line_num}: {e}")
                continue
    
    # Фильтруем только keep
    keep_cqs = [cq for cq in cqs if cq.quality_decision == 'keep']
    
    logger.info(f" Загружено {len(cqs)} CQs из {path}")
    logger.info(f"  Всего CQs: {len(cqs)}")
    logger.info(f"  Keep CQs: {len(keep_cqs)}")
    logger.info(f"  Drop CQs: {len(cqs) - len(keep_cqs)}")
    
    return keep_cqs

# %%
# Загрузка данных
cq_scored = load_cq_scored(CQ_SCORED_PATH)

# %%
# ПРОМПТЫ ДЛЯ LLM

DEDUP_SYSTEM_PROMPT = """
You are an expert in Deep Active Learning and scientific writing.

You receive a list of competency questions (CQs) that may contain duplicates
or near-duplicates.

Your task is to identify redundant questions and decide which ones to keep.

Definitions
-----------
- Two questions are duplicates if they ask essentially the same thing in the
  same Deep Active Learning context, even if wording is slightly different.
- Near-duplicates should be treated as duplicates if they differ only by
  minor rephrasing, synonym choice, or word order.
- If two questions overlap heavily but one is clearly more precise or better
  formulated, KEEP the better one and DROP the weaker one.

Decision rule
-------------
- For each question, you must output a binary decision:
  - 1 = keep (representative of its group of similar questions)
  - 0 = drop (duplicate, near-duplicate, or too weak/trivial compared to others)
- In each group of duplicates/near-duplicates, there should be at most ONE
  question with decision 1; all others in that group must be 0.
- It is acceptable that some unique, non-duplicated questions are also marked 0
  if they are clearly trivial, off-topic, or poorly formulated.

Important constraints
---------------------
- DO NOT rewrite, merge, or edit any questions.
- DO NOT invent new questions or ids.
- Base your decisions ONLY on the given question texts.
- Treat index values as positional identifiers: index i in the input list
  corresponds to position i in the output decisions list.

Output format
-------------
You must return a single JSON object with exactly one key "decisions":

{
  "decisions": [1, 0, 1, 0, ...]
}

- The "decisions" array MUST have the same length and order as the input list.
- Each element is either 0 or 1 (integer), no other values.
- Do NOT include explanations, comments, or any extra fields.
"""

def create_cq_dedup_user_prompt(cq_items: List[Dict[str, Any]]) -> str:
    """
    Создает user prompt для дедупликации CQs.
    
    Args:
        cq_items: список словарей с полями:
          - index: int
          - cq_id: str
          - text: str
          - super_topic_id: str (опционально)
          - role: str (опционально)
    """
    cq_list_text = json.dumps(cq_items, indent=2, ensure_ascii=False)
    
    prompt = f"""
You are given a JSON array of competency questions (CQs) for Deep Active Learning.

Each item has:
- "index": integer position in the list
- "cq_id": question identifier
- "text": question text
- "super_topic_id": super topic identifier (for context)
- "role": question role (class/relation/process/constraint/comparison)

Your task is to decide which questions to KEEP (1) and which to DROP (0)
based on semantic duplication and near-duplication.

Input CQs:

{cq_list_text}

Remember:
- If several questions ask essentially the same thing, you must keep at most ONE
  of them (the best one) and drop the others.
- The output MUST be a JSON object with a single key "decisions", whose value
  is an array of 0/1 decisions aligned with the input indices.

Return JSON:

{{
  "decisions": [1, 0, 1, 0, ...]
}}
"""
    return prompt

logger.info(" Промпты для LLM определены")

# %%
# ПОДГОТОВКА ДАННЫХ ДЛЯ ПРОМПТА

def prepare_cq_items_for_prompt(cqs: List[CQScored]) -> List[Dict[str, Any]]:
    """
    Подготавливает данные CQs для промпта.
    
    Args:
        cqs: список CQScored объектов
        
    Returns:
        список словарей с полями: index, cq_id, text, super_topic_id, role
    """
    items = []
    
    for idx, cq in enumerate(cqs):
        # Обрезаем text до 300 символов, если длиннее
        text = cq.text
        if len(text) > 300:
            text = text[:297] + "..."
        
        item = {
            "index": idx,
            "cq_id": cq.cq_id,
            "text": text,
            "super_topic_id": cq.super_topic_id,
            "role": cq.role
        }
        items.append(item)
    
    logger.info(f" Подготовлено {len(items)} CQ items для промпта")
    logger.info(f"  Средняя длина текста: {np.mean([len(item['text']) for item in items]):.1f} символов")
    
    return items

# %%
# Подготовка данных
cq_items = prepare_cq_items_for_prompt(cq_scored)

# %%
# ВЫЗОВ LLM ДЛЯ ДЕДУПЛИКАЦИИ

def deduplicate_cqs_llm(
    cqs: List[CQScored],
    openai_client,
    trace_id: str = None
) -> List[int]:
    """
    Вызывает LLM для дедупликации CQs.
    
    Args:
        cqs: список CQScored объектов
        openai_client: OpenAI клиент
        trace_id: идентификатор трассировки
        
    Returns:
        список решений [1, 0, 1, ...] той же длины, что и cqs
    """
    if not openai_client:
        logger.error(" OpenAI клиент не инициализирован")
        return [1] * len(cqs)  # Fallback: оставляем все
    
    # Подготовка данных
    cq_items = prepare_cq_items_for_prompt(cqs)
    user_prompt = create_cq_dedup_user_prompt(cq_items)
    
    max_tokens = LLM_MAX_TOKENS
    retries = LLM_MAX_RETRIES
    
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"[trace_id={trace_id}] LLM вызов для дедупликации (попытка {attempt}/{retries})")
            logger.debug(f"[trace_id={trace_id}] Количество CQs: {len(cqs)}, max_tokens: {max_tokens}")
            
            response = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": DEDUP_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=max_tokens,
                timeout=LLM_TIMEOUT
            )
            
            # Извлекаем текст ответа
            response_text = response.choices[0].message.content.strip()
            
            # Парсим JSON
            try:
                # Убираем markdown code blocks, если есть
                if response_text.startswith("```"):
                    # Ищем начало JSON
                    start_idx = response_text.find("{")
                    end_idx = response_text.rfind("}") + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        response_text = response_text[start_idx:end_idx]
                
                result = json.loads(response_text)
                
                if "decisions" not in result:
                    raise ValueError("Отсутствует ключ 'decisions' в ответе LLM")
                
                decisions = result["decisions"]
                
                # Валидация длины
                if len(decisions) != len(cqs):
                    raise ValueError(
                        f"Длина массива decisions ({len(decisions)}) не совпадает "
                        f"с количеством CQs ({len(cqs)})"
                    )
                
                # Валидация значений
                if not all(isinstance(d, (int, bool)) and d in (0, 1) for d in decisions):
                    raise ValueError("Массив decisions содержит недопустимые значения (должны быть 0 или 1)")
                
                # Преобразуем bool в int
                decisions = [int(d) for d in decisions]
                
                logger.success(f"[trace_id={trace_id}]  Получены решения от LLM: {sum(decisions)} keep, {len(decisions) - sum(decisions)} drop")
                return decisions
                
            except json.JSONDecodeError as e:
                logger.warning(f"[trace_id={trace_id}] Ошибка парсинга JSON (попытка {attempt}/{retries}): {e}")
                logger.debug(f"[trace_id={trace_id}] Начало ответа LLM (первые 500 символов):")
                logger.debug(response_text[:500])
                
                # Сохраняем полный ответ в файл для отладки
                error_file = OUTPUT_DIR / f"llm_error_response_{trace_id}_{int(time.time())}.txt"
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(response_text)
                logger.info(f"[trace_id={trace_id}] Полный ответ сохранен в: {error_file}")
                
                # Увеличиваем max_tokens для следующей попытки
                if attempt < retries:
                    max_tokens = min(max_tokens * 2, 16000)
                    logger.info(f"[trace_id={trace_id}] Увеличиваем max_tokens до {max_tokens} для следующей попытки")
                    time.sleep(1.0 * attempt)  # Экспоненциальная задержка
                    continue
                else:
                    raise
                    
        except Exception as e:
            logger.error(f"[trace_id={trace_id}] Ошибка LLM (попытка {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(2.0 * attempt)
                continue
            else:
                logger.error(f"[trace_id={trace_id}] Не удалось получить решения от LLM после {retries} попыток")
                # Fallback: оставляем все CQs
                return [1] * len(cqs)
    
    # Fallback (не должно сюда дойти)
    return [1] * len(cqs)

# %%
# Вызов LLM для дедупликации
if cq_scored and openai_client:
    logger.info(f" Дедупликация {len(cq_scored)} CQs через LLM...")
    decisions = deduplicate_cqs_llm(cq_scored, openai_client, trace_id=TRACE_ID)
else:
    logger.warning(" Нет данных для дедупликации или клиент не инициализирован")
    decisions = []

# %%
# ПРИМЕНЕНИЕ РЕШЕНИЙ И СОЗДАНИЕ ФИНАЛЬНЫХ CQs

if decisions and len(decisions) == len(cq_scored):
    # Фильтруем CQs с decision == 1
    unique_cqs = [cq for cq, decision in zip(cq_scored, decisions) if decision == 1]
    
    logger.info(f" После дедупликации: {len(unique_cqs)} CQs (удалено {len(cq_scored) - len(unique_cqs)})")
    
    # Сортируем по качеству: (relevance + clarity) / 2 (по убыванию)
    unique_cqs.sort(
        key=lambda cq: (
            cq.quality_scores.relevance + cq.quality_scores.clarity
        ) / 2,
        reverse=True
    )
    
    # Присваиваем новые стабильные ID
    final_cqs = []
    for i, cq in enumerate(unique_cqs, 1):
        cq_dict = cq.model_dump()
        cq_dict.pop('cq_id', None)  # Удаляем старый cq_id
        new_cq = CQFinal(
            **cq_dict,
            cq_id=f"CQ_{i:04d}"  # Новое стабильное ID
        )
        final_cqs.append(new_cq)
    
    logger.success(f" Создано {len(final_cqs)} финальных CQs")
    
    # Статистика
    logger.info(f"\n Статистика дедупликации:")
    logger.info(f"  До дедупликации: {len(cq_scored)} CQs")
    logger.info(f"  После дедупликации: {len(final_cqs)} CQs")
    logger.info(f"  Удалено дубликатов: {len(cq_scored) - len(final_cqs)}")
    logger.info(f"  Доля сохраненных: {len(final_cqs) / len(cq_scored):.2%}")
    
    # Распределение по super_topic_id
    super_topic_counts = Counter(cq.super_topic_id for cq in final_cqs)
    logger.info(f"\n  Распределение по супертемам: {len(super_topic_counts)} уникальных супертем")
    
    # Распределение по role
    role_counts = Counter(cq.role for cq in final_cqs)
    logger.info(f"  Распределение по ролям:")
    for role, count in role_counts.items():
        logger.info(f"    {role}: {count}")
else:
    logger.error(" Ошибка: решения LLM не соответствуют количеству CQs")
    final_cqs = []

# %%
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ

if final_cqs:
    final_path = OUTPUT_DIR / "cq_final.jsonl"
    
    with open(final_path, 'w', encoding='utf-8') as f:
        for cq in final_cqs:
            f.write(cq.model_dump_json() + '\n')
    
    logger.success(f" Сохранено {len(final_cqs)} финальных CQs в {final_path}")
else:
    logger.warning(" Нет финальных CQs для сохранения")

# %%
# ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ

if final_cqs:
    # DataFrame с финальными CQs
    df_final = pd.DataFrame([
        {
            "cq_id": cq.cq_id,
            "super_topic_id": cq.super_topic_id,
            "text": cq.text[:100] + "..." if len(cq.text) > 100 else cq.text,
            "role": cq.role,
            "relevance": cq.quality_scores.relevance,
            "clarity": cq.quality_scores.clarity,
            "avg_score": (cq.quality_scores.relevance + cq.quality_scores.clarity) / 2
        }
        for cq in final_cqs
    ])
    
    logger.info("\n Статистика финальных CQs:")
    logger.info(f"  Всего: {len(final_cqs)}")
    logger.info(f"  По супертемам: {df_final['super_topic_id'].nunique()}")
    logger.info(f"  Распределение по ролям:")
    role_counts = df_final['role'].value_counts()
    for role, count in role_counts.items():
        logger.info(f"    {role}: {count}")
    
    # Таблица с примерами
    logger.info("\n Примеры финальных CQs (первые 10):")
    display(df_final.head(10))
    
    # Графики
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Распределение по супертемам (топ-10)
    top_super_topics = df_final['super_topic_id'].value_counts().head(10)
    axes[0].barh(range(len(top_super_topics)), top_super_topics.values)
    axes[0].set_yticks(range(len(top_super_topics)))
    axes[0].set_yticklabels(top_super_topics.index)
    axes[0].set_xlabel("Количество CQs")
    axes[0].set_title("Топ-10 супертем по количеству CQs")
    axes[0].invert_yaxis()
    
    # Распределение по ролям (pie chart)
    role_counts = df_final['role'].value_counts()
    axes[1].pie(role_counts.values, labels=role_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1].set_title("Распределение CQs по ролям")
    
    plt.tight_layout()
    plt.show()
    
    # Распределение оценок качества
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(df_final['relevance'], bins=20, alpha=0.7, label='Relevance')
    axes[0].hist(df_final['clarity'], bins=20, alpha=0.7, label='Clarity')
    axes[0].set_xlabel("Оценка")
    axes[0].set_ylabel("Количество CQs")
    axes[0].set_title("Распределение оценок качества")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(df_final['relevance'], df_final['clarity'], alpha=0.5)
    axes[1].set_xlabel("Relevance")
    axes[1].set_ylabel("Clarity")
    axes[1].set_title("Relevance vs Clarity")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
else:
    logger.warning(" Нет данных для визуализации")

# %%
# СРАВНЕНИЕ С ПРОСТОЙ ТЕКСТОВОЙ ДЕДУПЛИКАЦИЕЙ

import re

def normalize_text_for_dedup(text: str) -> str:
    """Нормализует текст для дедупликации."""
    # Приводим к нижнему регистру, убираем пунктуацию
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# %%
# Простая текстовая дедупликация
if cq_scored:
    seen_texts = set()
    unique_cqs_text = []
    
    for cq in cq_scored:
        normalized = normalize_text_for_dedup(cq.text)
        if normalized not in seen_texts:
            seen_texts.add(normalized)
            unique_cqs_text.append(cq)
    
    logger.info(f"\n Сравнение методов дедупликации:")
    logger.info(f"  Исходное количество: {len(cq_scored)} CQs")
    logger.info(f"  LLM-дедупликация: {len(final_cqs) if final_cqs else 0} CQs")
    logger.info(f"  Текстовая дедупликация: {len(unique_cqs_text)} CQs")
    
    # Находим различия
    if final_cqs and unique_cqs_text:
        final_cq_ids = {cq.cq_id for cq in final_cqs}
        text_cq_ids = {cq.cq_id for cq in unique_cqs_text}
        
        only_llm = final_cq_ids - text_cq_ids
        only_text = text_cq_ids - final_cq_ids
        both = final_cq_ids & text_cq_ids
        
        logger.info(f"\n  Пересечение (в обоих): {len(both)} CQs")
        logger.info(f"  Только в LLM-результатах: {len(only_llm)} CQs")
        logger.info(f"  Только в текстовых результатах: {len(only_text)} CQs")
        
        # Примеры CQs, которые LLM удалил, а текстовая дедупликация оставила
        if only_text:
            logger.info(f"\n  Примеры CQs, удаленных LLM, но оставленных текстовой дедупликацией:")
            for cq_id in list(only_text)[:3]:
                cq = next((cq for cq in unique_cqs_text if cq.cq_id == cq_id), None)
                if cq:
                    logger.info(f"    - {cq_id}: {cq.text[:80]}...")
        
        # Примеры CQs, которые текстовая дедупликация удалила, а LLM оставил
        if only_llm:
            logger.info(f"\n  Примеры CQs, удаленных текстовой дедупликацией, но оставленных LLM:")
            for cq_id in list(only_llm)[:3]:
                cq = next((cq for cq in final_cqs if cq.cq_id == cq_id), None)
                if cq:
                    logger.info(f"    - {cq_id}: {cq.text[:80]}...")
else:
    logger.warning(" Нет данных для сравнения")
