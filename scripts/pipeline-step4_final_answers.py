# %% [markdown]
"""
# Шаг 4: Генерация финальных ответов и валидация доказательств

Этот ноутбук реализует шаг 4 из `pipeline_desc.md` (строки 472-522): для каждого CQ получить **окончательный ответ по всему корпусу**, тем самым проверив, отвечает ли корпус на этот вопрос и не является ли вопрос тривиальным или неразрешимым.

## Ключевые задачи:
1. **Независимый поиск по вопросу**: RAG-поиск по всему корпусу (не только по супертеме)
2. **Генерация финального ответа**: LLM-Answerer генерирует ответ на основе найденных чанков
3. **Валидация доказательств**: LLM-судья проверяет обоснованность ответа (опционально)

## Входные данные:
- `cq_final.jsonl` (306 CQs с локальными ответами из шага 3)

## Выходные данные:
- `cq_with_final_answers.jsonl` с полями:
  - `final_answer`: финальный ответ по всему корпусу
  - `validation`: результаты валидации через LLM-судью (опционально)
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

# Добавляем путь к проекту
sys.path.insert(0, str(Path().absolute().parent))

# Загрузка переменных окружения из .env файла (ADR-0006)
env_path = Path().absolute().parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

import weaviate
from loguru import logger
from pydantic import BaseModel, Field
from IPython.display import display
from tqdm import tqdm

from oasis.pipelines.stage2_indexing import hybrid_search
from oasis.pipelines.stage1_extract_topics import _create_openai_client

# Настройки визуализации
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 200)
pd.set_option('display.width', None)

logger.info(" Импорты загружены")

# %%
# НАСТРОЙКИ ПУТЕЙ И ПАРАМЕТРОВ

# Пути к данным
WEAVIATE_URL = "http://localhost:8081"
TOPIC = "deep_active_learning"
CQ_FINAL_PATH = Path(f"../outputs/{TOPIC}/step3/cq_final.jsonl")
OUTPUT_DIR = Path(f"../outputs/{TOPIC}/step4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Параметры RAG поиска
RAG_TOP_K = 25  # Количество чанков для поиска (20-30)
RAG_ALPHA = 0.5  # Вес векторного поиска (0.0 = только BM25, 1.0 = только векторный)

# Параметры LLM
LLM_MODEL = os.getenv("OPENAI_CANONICAL_MODEL", "gpt-5.1")
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 4000  # Максимальное количество токенов для ответа
LLM_MAX_RETRIES = 3
LLM_TIMEOUT = 300.0  # Таймаут в секундах (5 минут)

# Параметры валидации
USE_VALIDATION = True  # Использовать LLM-судью для валидации ответов

# Trace ID для трассировки
TRACE_ID = "demo_step4_final_answers"

logger.info(f" Настройки загружены")
logger.info(f"  CQ final path: {CQ_FINAL_PATH}")
logger.info(f"  Output dir: {OUTPUT_DIR}")
logger.info(f"  RAG top_k: {RAG_TOP_K}, alpha: {RAG_ALPHA}")
logger.info(f"  LLM model: {LLM_MODEL}, timeout: {LLM_TIMEOUT}с, max_tokens: {LLM_MAX_TOKENS}")
logger.info(f"  Use validation: {USE_VALIDATION}")

# %%
# ИНИЦИАЛИЗАЦИЯ КЛИЕНТОВ

# Инициализация Weaviate клиента
try:
    weaviate_client = weaviate.Client(WEAVIATE_URL, startup_period=10)
    is_live = weaviate_client.is_live()
    is_ready = weaviate_client.is_ready()
    logger.success(f" Подключено к Weaviate: {WEAVIATE_URL}")
    logger.info(f"  is_live: {is_live}, is_ready: {is_ready}")
except Exception as e:
    logger.error(f" Ошибка подключения к Weaviate: {e}")
    weaviate_client = None

# Инициализация OpenAI клиента
try:
    openai_client = _create_openai_client(timeout=LLM_TIMEOUT)
    logger.success(f" OpenAI клиент инициализирован (timeout: {LLM_TIMEOUT}с)")
except Exception as e:
    logger.error(f" Ошибка инициализации OpenAI клиента: {e}")
    openai_client = None

# %%
# PYDANTIC МОДЕЛИ

# Переиспользование моделей из основного ноутбука
class EvidenceChunkRef(BaseModel):
    """Ссылка на evidence chunk."""
    chunk_id: str = Field(..., description="Идентификатор чанка")
    article_id: str = Field(..., description="Идентификатор статьи")

class CQCandidate(BaseModel):
    """Модель кандидата CQ."""
    cq_id: str = Field(..., description="Идентификатор CQ")
    super_topic_id: str = Field(..., description="Идентификатор супертемы")
    text: str = Field(..., description="Текст вопроса")
    role: str = Field(..., description="Роль вопроса")
    answer: str = Field(..., description="Ответ на вопрос (локальный)")
    answer_type: str = Field(..., description="Тип ответа")
    evidence_chunks: List[EvidenceChunkRef] = Field(..., description="Список ссылок на evidence chunks")
    generation_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Метаданные генерации")

class QualityScores(BaseModel):
    """Оценки качества CQ."""
    relevance: float = Field(..., ge=0.0, le=1.0)
    clarity: float = Field(..., ge=0.0, le=1.0)
    groundedness: float = Field(..., ge=0.0, le=1.0)
    answerability: float = Field(..., ge=0.0, le=1.0)

class CQScored(CQCandidate):
    """Модель оцененного CQ."""
    quality_scores: QualityScores = Field(..., description="Оценки качества")
    quality_decision: str = Field(..., description="Решение: keep или drop")

class CQFinal(CQScored):
    """Финальный CQ с новым стабильным ID."""
    pass

# Новые модели для шага 4
class FinalAnswer(BaseModel):
    """Финальный ответ на CQ по всему корпусу."""
    text: str = Field(..., description="Сгенерированный финальный ответ")
    answer_type: str = Field(..., description="Тип ответа (text/list/numeric/boolean/INSUFFICIENT_EVIDENCE)")
    status: str = Field(..., description="Пометка качества ответа (OK или INSUFFICIENT_EVIDENCE)")
    evidence_chunks: List[str] = Field(..., description="Список chunk_id фрагментов, на которые опирается ответ")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Оценка уверенности модели в ответе (0-1)")

class Validation(BaseModel):
    """Результаты валидации ответа через LLM-судью."""
    justification_score: float = Field(..., ge=0.0, le=1.0, description="Насколько ответ следует из указанных фрагментов (0-1)")
    uses_external_knowledge: bool = Field(..., description="Использует ли ответ знания вне данных")
    comment: str = Field(..., description="Пояснение от судьи")
    revised_answer: Optional[str] = Field(default=None, description="Откорректированный ответ (если предложен)")

class CQWithFinalAnswer(CQFinal):
    """CQ с финальным ответом и валидацией."""
    final_answer: FinalAnswer = Field(..., description="Финальный ответ по всему корпусу")
    validation: Optional[Validation] = Field(default=None, description="Результаты валидации через LLM-судью")

logger.info(" Pydantic модели определены")

# %%
# ЗАГРУЗКА ДАННЫХ

def load_cq_final(path: Path) -> List[CQFinal]:
    """Загружает CQs из JSONL файла."""
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
                
                cq = CQFinal(**data)
                cqs.append(cq)
            except Exception as e:
                logger.warning(f" Ошибка загрузки строки {line_num}: {e}")
                continue
    
    logger.info(f" Загружено {len(cqs)} CQs из {path}")
    
    # Статистика
    if cqs:
        role_counts = Counter(cq.role for cq in cqs)
        super_topic_counts = Counter(cq.super_topic_id for cq in cqs)
        logger.info(f"  Распределение по ролям:")
        for role, count in role_counts.items():
            logger.info(f"    {role}: {count}")
        logger.info(f"  Уникальных супертем: {len(super_topic_counts)}")
    
    return cqs

# Загрузка данных
cq_final_list = load_cq_final(CQ_FINAL_PATH)

# %%
# НЕЗАВИСИМЫЙ ПОИСК ПО ВОПРОСУ

def search_evidence_for_cq(
    cq: CQFinal,
    weaviate_client,
    top_k: int = RAG_TOP_K,
    alpha: float = RAG_ALPHA,
    trace_id: str = None
) -> List[dict]:
    """
    Выполняет независимый поиск evidence chunks для CQ по всему корпусу.
    
    Args:
        cq: CQ объект
        weaviate_client: Weaviate клиент
        top_k: Количество чанков для поиска
        alpha: Вес векторного поиска
        trace_id: Идентификатор трассировки
        
    Returns:
        Список чанков с полями: chunk_id, text, paper_id, score
    """
    if not weaviate_client:
        logger.error(f"[trace_id={trace_id}] Weaviate клиент не инициализирован")
        return []
    
    try:
        # Используем текст вопроса как запрос для поиска по всему корпусу
        query = cq.text
        
        logger.debug(f"[trace_id={trace_id}] Поиск evidence для CQ {cq.cq_id}: {query[:100]}...")
        
        # Гибридный поиск по всему корпусу (без фильтров по супертеме)
        search_results = hybrid_search(
            query=query,
            weaviate_client=weaviate_client,
            top_k=top_k,
            alpha=alpha,
            filters=None,  # Без фильтров - поиск по всему корпусу
            trace_id=trace_id
        )
        
        logger.debug(f"[trace_id={trace_id}] Найдено {len(search_results)} чанков для CQ {cq.cq_id}")
        
        return search_results
        
    except Exception as e:
        logger.error(f"[trace_id={trace_id}] Ошибка поиска evidence для CQ {cq.cq_id}: {e}")
        return []

logger.info(" Функция поиска evidence определена")

# %%
# ПРОМПТЫ ДЛЯ LLM-ANSWERER

ANSWERER_SYSTEM_PROMPT = """
You are an expert in Deep Active Learning (DAL) and scientific reading comprehension.

Your role:
Given ONE competency question (CQ) and a set of text fragments ("evidence chunks")
from research papers, you must produce a final answer that is STRICTLY grounded in
those fragments.

GENERAL CONSTRAINTS
-------------------
- Use ONLY the provided fragments as sources of knowledge.
- Do NOT rely on external knowledge, memory, or general intuition.
- If the fragments do not support a precise and reliable answer, you MUST return
  INSUFFICIENT_EVIDENCE.
- Do NOT invent datasets, numbers, performance improvements, or methods that are
  not clearly present in the fragments.

HOW TO FORMULATE THE ANSWER
---------------------------
1. Read the CQ and all evidence fragments carefully.
2. If the fragments clearly and directly answer the question:
   - Synthesize a concise answer using your own wording,
     but do NOT go beyond what the fragments justify.
   - Make the answer specific to Deep Active Learning when applicable
     (e.g. acquisition strategy, uncertainty estimate, budget, stopping criterion).
3. If the fragments partially touch the topic but do NOT provide enough
   information to answer the CQ confidently:
   - Set text to "INSUFFICIENT_EVIDENCE".
   - You may add a short explanation AFTER this keyword, e.g.
     "INSUFFICIENT_EVIDENCE: the fragments describe X but do not state Y."
4. If the fragments are contradictory on the key point of the CQ:
   - Treat the situation as insufficient evidence.
   - Set answer_type to "INSUFFICIENT_EVIDENCE" and status to "INSUFFICIENT_EVIDENCE".
   - Optionally mention the contradiction briefly in the text field.

ANSWER TYPE SELECTION
---------------------
Set answer_type according to the most natural form of the answer:

- "text":
  A short textual answer (typically 2-4 sentences), when the CQ asks for an
  explanation, definition, or qualitative description.

- "list":
  A list of items explicitly enumerated in the evidence
  (e.g. a set of acquisition strategies, datasets, metrics).
  Use a textual list (e.g. separated by semicolons or line breaks).

- "numeric":
  A numeric value ONLY if it is explicitly and unambiguously given
  in the fragments (e.g. a specific performance number, budget, or threshold).

- "boolean":
  "yes" / "no" when the question can be answered unambiguously as a yes/no
  based on the fragments.

- "INSUFFICIENT_EVIDENCE":
  If the evidence is incomplete, ambiguous, or does not clearly support
  ANY concrete answer.

STATUS AND CONFIDENCE
---------------------
- status:
  - "OK"                  : when you provide a concrete answer
                            (answer_type in {"text", "list", "numeric", "boolean"}).
  - "INSUFFICIENT_EVIDENCE": when answer_type is "INSUFFICIENT_EVIDENCE".

- confidence (0.0-1.0):
  - Close to 1.0: the answer is explicitly and directly supported by multiple
                  fragments with no contradictions.
  - Around 0.5: the answer is only partially supported or somewhat indirect.
  - Close to 0.0: should normally not occur, because in such cases you must use
                  INSUFFICIENT_EVIDENCE instead of guessing.

EVIDENCE CHUNKS
---------------
- "evidence_chunks" must be a list of chunk_ids (strings) from the provided input.
- For status = "OK":
  - Select 1-3 chunk_ids that MOST directly support the answer.
  - Do NOT include chunks that are only loosely related.
- For status = "INSUFFICIENT_EVIDENCE":
  - You may return an empty list OR the chunk_ids that show why the evidence
    is insufficient or contradictory.

OUTPUT FORMAT
-------------
You MUST return a single JSON object with EXACTLY the following keys:

{
  "text": "Your answer text or INSUFFICIENT_EVIDENCE",
  "answer_type": "text|list|numeric|boolean|INSUFFICIENT_EVIDENCE",
  "status": "OK|INSUFFICIENT_EVIDENCE",
  "evidence_chunks": ["chunk_id1", "chunk_id2", ...],
  "confidence": 0.85
}

Additional requirements:
- Respond in English.
- Do NOT include any extra keys, comments, explanations, or markdown.
- Do NOT wrap the JSON object in code fences.
"""


def create_answerer_user_prompt(cq: CQFinal, evidence_chunks: List[dict]) -> str:
    """
    Создает user prompt для LLM-Answerer.
    
    Args:
        cq: CQ объект
        evidence_chunks: список чанков с полями chunk_id, text, paper_id, score
    """
    # Формируем текст evidence chunks
    chunks_text = []
    for i, chunk in enumerate(evidence_chunks, 1):
        chunk_id = chunk.get('chunk_id', f'chunk_{i}')
        text = chunk.get('text', '')
        score = chunk.get('score', 0.0)
        # Явно преобразуем score в float перед форматированием
        score_float = float(score) if score is not None else 0.0
        chunks_text.append(f"[Chunk {i} (ID: {chunk_id}, score: {score_float:.3f})]\n{text}\n")
    
    evidence_text = "\n".join(chunks_text)
    
    prompt = f"""
You are given a competency question (CQ) for Deep Active Learning and a set of text fragments from research papers.

Question (CQ ID: {cq.cq_id}):
{cq.text}

Text fragments from research papers:
{evidence_text}

Your task:
1. Answer the question based ONLY on the provided fragments.
2. If the fragments do not contain sufficient information, return "INSUFFICIENT_EVIDENCE".
3. Determine the answer_type:
   - "text": general textual answer
   - "list": enumeration of items
   - "numeric": a number or range
   - "boolean": yes/no answer
   - "INSUFFICIENT_EVIDENCE": insufficient information
4. Provide a confidence score (0.0-1.0).
5. List the chunk_ids that support your answer.

Return JSON:
{{
  "text": "...",
  "answer_type": "...",
  "status": "OK|INSUFFICIENT_EVIDENCE",
  "evidence_chunks": ["chunk_id1", "chunk_id2", ...],
  "confidence": 0.85
}}
"""
    return prompt

logger.info(" Промпты для LLM-Answerer определены")

# %%
# ГЕНЕРАЦИЯ ФИНАЛЬНОГО ОТВЕТА

def generate_final_answer(
    cq: CQFinal,
    evidence_chunks: List[dict],
    openai_client,
    trace_id: str = None
) -> FinalAnswer:
    """
    Генерирует финальный ответ на CQ через LLM-Answerer.
    
    Args:
        cq: CQ объект
        evidence_chunks: список чанков с полями chunk_id, text, paper_id, score
        openai_client: OpenAI клиент
        trace_id: Идентификатор трассировки
        
    Returns:
        FinalAnswer объект
    """
    if not openai_client:
        logger.error(f"[trace_id={trace_id}] OpenAI клиент не инициализирован")
        return FinalAnswer(
            text="INSUFFICIENT_EVIDENCE: OpenAI client not available",
            answer_type="INSUFFICIENT_EVIDENCE",
            status="INSUFFICIENT_EVIDENCE",
            evidence_chunks=[],
            confidence=0.0
        )
    
    if not evidence_chunks:
        logger.warning(f"[trace_id={trace_id}] Нет evidence chunks для CQ {cq.cq_id}")
        return FinalAnswer(
            text="INSUFFICIENT_EVIDENCE: No evidence chunks found",
            answer_type="INSUFFICIENT_EVIDENCE",
            status="INSUFFICIENT_EVIDENCE",
            evidence_chunks=[],
            confidence=0.0
        )
    
    user_prompt = create_answerer_user_prompt(cq, evidence_chunks)
    max_tokens = LLM_MAX_TOKENS
    retries = LLM_MAX_RETRIES
    
    for attempt in range(1, retries + 1):
        try:
            logger.debug(f"[trace_id={trace_id}] LLM вызов для генерации ответа (CQ {cq.cq_id}, попытка {attempt}/{retries})")
            
            response = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": ANSWERER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=max_tokens,
                timeout=LLM_TIMEOUT
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Парсим JSON
            try:
                # Убираем markdown code blocks, если есть
                if response_text.startswith("```"):
                    start_idx = response_text.find("{")
                    end_idx = response_text.rfind("}") + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        response_text = response_text[start_idx:end_idx]
                
                result = json.loads(response_text)
                
                # Преобразуем старое поле final_answer в text (для обратной совместимости)
                if "final_answer" in result and "text" not in result:
                    result["text"] = result.pop("final_answer")
                
                # Валидация через Pydantic
                final_answer = FinalAnswer(**result)
                
                logger.debug(f"[trace_id={trace_id}]  Сгенерирован ответ для CQ {cq.cq_id}: status={final_answer.status}, confidence={final_answer.confidence:.2f}")
                return final_answer
                
            except json.JSONDecodeError as e:
                logger.warning(f"[trace_id={trace_id}] Ошибка парсинга JSON (CQ {cq.cq_id}, попытка {attempt}/{retries}): {e}")
                logger.debug(f"[trace_id={trace_id}] Начало ответа LLM (первые 500 символов):")
                logger.debug(response_text[:500])
                
                # Сохраняем полный ответ в файл для отладки
                error_file = OUTPUT_DIR / f"llm_error_response_{trace_id}_{cq.cq_id}_{int(time.time())}.txt"
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(response_text)
                logger.info(f"[trace_id={trace_id}] Полный ответ сохранен в: {error_file}")
                
                # Увеличиваем max_tokens для следующей попытки
                if attempt < retries:
                    max_tokens = min(max_tokens * 2, 16000)
                    logger.info(f"[trace_id={trace_id}] Увеличиваем max_tokens до {max_tokens} для следующей попытки")
                    time.sleep(1.0 * attempt)
                    continue
                else:
                    raise
                    
        except Exception as e:
            logger.error(f"[trace_id={trace_id}] Ошибка LLM (CQ {cq.cq_id}, попытка {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(2.0 * attempt)
                continue
            else:
                logger.error(f"[trace_id={trace_id}] Не удалось сгенерировать ответ для CQ {cq.cq_id} после {retries} попыток")
                # Fallback
                return FinalAnswer(
                    text="INSUFFICIENT_EVIDENCE: LLM generation failed",
                    answer_type="INSUFFICIENT_EVIDENCE",
                    status="INSUFFICIENT_EVIDENCE",
                    evidence_chunks=[],
                    confidence=0.0
                )
    
    # Fallback (не должно сюда дойти)
    return FinalAnswer(
        text="INSUFFICIENT_EVIDENCE: Unexpected error",
        answer_type="INSUFFICIENT_EVIDENCE",
        status="INSUFFICIENT_EVIDENCE",
        evidence_chunks=[],
        confidence=0.0
    )

logger.info(" Функция генерации финального ответа определена")

# %%
# ПРОМПТЫ ДЛЯ LLM-СУДЬИ

VALIDATOR_SYSTEM_PROMPT = """
You are a critical validation judge for competency questions (CQs) in the domain
of Deep Active Learning (DAL).

INPUT
-----
For each evaluation you receive:
- ONE competency question (CQ)
- A proposed answer (final_answer) with an answer_type
- A list of evidence chunks, each with:
  - chunk_id
  - text (fragment from a research paper)

Your task is to decide how well the proposed answer is logically justified by
the evidence chunks, and whether it uses external knowledge beyond them.

KEY PRINCIPLES
--------------
- You MUST base your judgment ONLY on the provided evidence chunks.
- The domain is Deep Active Learning, but you MAY NOT add any knowledge that is
  not clearly supported or entailed by the fragments.
- If the answer introduces statements that are not present, not entailed, or
  contradict the fragments, this counts as using external knowledge and/or
  being unjustified.

JUSTIFICATION SCORE (0.0-1.0)
------------------------------
"justification_score" measures how strongly the proposed answer is supported
by the evidence chunks.

Use the following guidelines:
- 1.0:
  The answer is directly supported by the fragments. Key claims can be
  traced verbatim or via simple paraphrase to specific chunks.
  No important parts rely on guesses or external knowledge.
- ~0.7:
  The answer is mostly supported, but some minor parts require mild inference
  or are only indirectly stated. No major hallucinations.
- ~0.5:
  The answer is partially supported, but several important claims are not
  clearly present or are only weakly suggested by the fragments.
- ~0.2:
  The answer has substantial problems; most key claims are not in the evidence
  or are unclear.
- 0.0:
  The answer is not supported at all, contradicts the fragments, or is purely
  speculative.

USES EXTERNAL KNOWLEDGE
-----------------------
"uses_external_knowledge" is a boolean:

- true:
  The answer introduces information, explanations, numbers, comparisons, or
  domain-specific claims that cannot be found or safely inferred from the
  provided fragments. This includes generic DAL facts if they are not present
  in the fragments.

- false:
  All important claims in the answer can be found in, or are simple
  paraphrases/straightforward combinations of, the evidence fragments.

If justification_score is very high (e.g. >= 0.8), "uses_external_knowledge"
will typically be false. If you mark "uses_external_knowledge": true, the
justification_score should rarely be above ~0.7.

COMMENT
-------
Provide a short, precise explanation of your judgment, for example:

- Which parts of the answer are well supported (mentioning chunk_ids).
- Which claims are missing in the evidence or speculative.
- Whether there are contradictions between chunks and the answer.

The comment should be 1-3 sentences, not an essay.

REVISED ANSWER
--------------
"revised_answer" is OPTIONAL and should follow these rules:

- If the original answer is well supported and does not need changes:
  - Set "revised_answer": null.

- If the original answer is mostly correct but contains:
  - minor hallucinations,
  - unnecessary external information,
  - or could be made more faithful to the evidence,
  then:
  - Provide a corrected version that:
    * stays within what the fragments support,
    * removes unsupported claims,
    * remains concise (typically 2-4 sentences).

- If the original answer is essentially unjustified (justification_score very low):
  - You MAY set "revised_answer" to a better, evidence-based answer if possible,
    or to "INSUFFICIENT_EVIDENCE" if the fragments truly do not allow a
    reliable answer.

In ALL cases, the revised_answer MUST be grounded ONLY in the evidence fragments.

OUTPUT FORMAT
-------------
Return a single JSON object with EXACTLY the following keys:

{
  "justification_score": 0.85,
  "uses_external_knowledge": false,
  "comment": "The answer is well supported by chunk_id1 and chunk_id3; one minor claim about performance gain is not explicitly stated.",
  "revised_answer": null
}

Constraints:
- "justification_score" MUST be a number in [0.0, 1.0].
- "uses_external_knowledge" MUST be a boolean (true/false).
- "comment" MUST be a short string.
- "revised_answer" MUST be either:
  - null, or
  - a string with a revised answer.

Do NOT include any extra keys, explanations, or markdown.
Do NOT wrap the JSON in code fences.
"""


def create_validator_user_prompt(
    cq: CQFinal,
    final_answer: FinalAnswer,
    evidence_chunks: List[dict]
) -> str:
    """
    Создает user prompt для LLM-судьи.
    
    Args:
        cq: CQ объект
        final_answer: сгенерированный финальный ответ
        evidence_chunks: список чанков с полями chunk_id, text, paper_id, score
    """
    # Формируем текст evidence chunks
    chunks_text = []
    for i, chunk in enumerate(evidence_chunks, 1):
        chunk_id = chunk.get('chunk_id', f'chunk_{i}')
        text = chunk.get('text', '')
        chunks_text.append(f"[Chunk {i} (ID: {chunk_id})]\n{text}\n")
    
    evidence_text = "\n".join(chunks_text)
    
    prompt = f"""
You are evaluating a proposed answer to a competency question.

Question (CQ ID: {cq.cq_id}):
{cq.text}

Proposed answer:
{final_answer.text}

Answer type: {final_answer.answer_type}
Status: {final_answer.status}
Confidence: {final_answer.confidence:.2f}
Evidence chunks used: {', '.join(final_answer.evidence_chunks)}

Text fragments from research papers:
{evidence_text}

Your task:
1. Evaluate whether the proposed answer is logically justified by the fragments.
2. Check if the answer uses knowledge outside the provided fragments.
3. Provide a justification_score (0.0-1.0).
4. Write a brief comment explaining your evaluation.
5. If you think the answer should be corrected, provide a revised_answer.

Return JSON:
{{
  "justification_score": 0.85,
  "uses_external_knowledge": false,
  "comment": "...",
  "revised_answer": null
}}
"""
    return prompt

logger.info(" Промпты для LLM-судьи определены")

# %%
# ВАЛИДАЦИЯ ЧЕРЕЗ LLM-СУДЬЮ

def validate_answer(
    cq: CQFinal,
    final_answer: FinalAnswer,
    evidence_chunks: List[dict],
    openai_client,
    trace_id: str = None
) -> Optional[Validation]:
    """
    Валидирует финальный ответ через LLM-судью.
    
    Args:
        cq: CQ объект
        final_answer: сгенерированный финальный ответ
        evidence_chunks: список чанков
        openai_client: OpenAI клиент
        trace_id: Идентификатор трассировки
        
    Returns:
        Validation объект или None при ошибке
    """
    if not openai_client:
        logger.warning(f"[trace_id={trace_id}] OpenAI клиент не инициализирован, пропускаем валидацию")
        return None
    
    if not evidence_chunks:
        logger.warning(f"[trace_id={trace_id}] Нет evidence chunks для валидации CQ {cq.cq_id}")
        return None
    
    user_prompt = create_validator_user_prompt(cq, final_answer, evidence_chunks)
    max_tokens = LLM_MAX_TOKENS
    retries = LLM_MAX_RETRIES
    
    for attempt in range(1, retries + 1):
        try:
            logger.debug(f"[trace_id={trace_id}] LLM вызов для валидации (CQ {cq.cq_id}, попытка {attempt}/{retries})")
            
            response = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=max_tokens,
                timeout=LLM_TIMEOUT
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Парсим JSON
            try:
                # Убираем markdown code blocks, если есть
                if response_text.startswith("```"):
                    start_idx = response_text.find("{")
                    end_idx = response_text.rfind("}") + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        response_text = response_text[start_idx:end_idx]
                
                result = json.loads(response_text)
                
                # Валидация через Pydantic
                validation = Validation(**result)
                
                logger.debug(f"[trace_id={trace_id}]  Валидация для CQ {cq.cq_id}: justification_score={validation.justification_score:.2f}")
                return validation
                
            except json.JSONDecodeError as e:
                logger.warning(f"[trace_id={trace_id}] Ошибка парсинга JSON валидации (CQ {cq.cq_id}, попытка {attempt}/{retries}): {e}")
                logger.debug(f"[trace_id={trace_id}] Начало ответа LLM (первые 500 символов):")
                logger.debug(response_text[:500])
                
                # Сохраняем полный ответ в файл для отладки
                error_file = OUTPUT_DIR / f"llm_validation_error_{trace_id}_{cq.cq_id}_{int(time.time())}.txt"
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(response_text)
                logger.info(f"[trace_id={trace_id}] Полный ответ сохранен в: {error_file}")
                
                # Увеличиваем max_tokens для следующей попытки
                if attempt < retries:
                    max_tokens = min(max_tokens * 2, 16000)
                    time.sleep(1.0 * attempt)
                    continue
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"[trace_id={trace_id}] Ошибка валидации (CQ {cq.cq_id}, попытка {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(2.0 * attempt)
                continue
            else:
                logger.warning(f"[trace_id={trace_id}] Не удалось валидировать ответ для CQ {cq.cq_id}")
                return None
    
    return None

logger.info(" Функция валидации определена")

# %%
# ОБРАБОТКА ВСЕХ CQs

def process_all_cqs(
    cqs: List[CQFinal],
    weaviate_client,
    openai_client,
    use_validation: bool = USE_VALIDATION,
    trace_id: str = None
) -> List[CQWithFinalAnswer]:
    """
    Обрабатывает все CQs: поиск evidence, генерация ответа, валидация.
    
    Args:
        cqs: список CQ объектов
        weaviate_client: Weaviate клиент
        openai_client: OpenAI клиент
        use_validation: использовать ли LLM-судью для валидации
        trace_id: Идентификатор трассировки
        
    Returns:
        Список CQWithFinalAnswer объектов
    """
    results = []
    
    logger.info(f"[trace_id={trace_id}] Начало обработки {len(cqs)} CQs")
    logger.info(f"[trace_id={trace_id}] Валидация: {'включена' if use_validation else 'отключена'}")
    
    for cq in tqdm(cqs, desc="Обработка CQs", unit="CQ"):
        try:
            # 1. Поиск evidence chunks
            evidence_chunks = search_evidence_for_cq(
                cq, weaviate_client, top_k=RAG_TOP_K, alpha=RAG_ALPHA, trace_id=trace_id
            )
            
            if not evidence_chunks:
                logger.warning(f"[trace_id={trace_id}] Нет evidence chunks для CQ {cq.cq_id}, пропускаем")
                continue
            
            # 2. Генерация финального ответа
            final_answer = generate_final_answer(
                cq, evidence_chunks, openai_client, trace_id=trace_id
            )
            
            # 3. Валидация (опционально)
            validation = None
            if use_validation:
                validation = validate_answer(
                    cq, final_answer, evidence_chunks, openai_client, trace_id=trace_id
                )
            
            # 4. Создание объекта CQWithFinalAnswer
            cq_dict = cq.model_dump()
            cq_with_answer = CQWithFinalAnswer(
                **cq_dict,
                final_answer=final_answer,
                validation=validation
            )
            results.append(cq_with_answer)
            
        except Exception as e:
            logger.error(f"[trace_id={trace_id}] Ошибка обработки CQ {cq.cq_id}: {e}")
            continue
    
    logger.success(f"[trace_id={trace_id}]  Обработано {len(results)} CQs из {len(cqs)}")
    return results

# Обработка всех CQs
if cq_final_list and weaviate_client and openai_client:
    logger.info(f" Начало обработки {len(cq_final_list)} CQs...")
    cq_with_answers = process_all_cqs(
        cq_final_list,
        weaviate_client,
        openai_client,
        use_validation=USE_VALIDATION,
        trace_id=TRACE_ID
    )
else:
    logger.warning(" Нет данных для обработки или клиенты не инициализированы")
    cq_with_answers = []

# %%
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ

if cq_with_answers:
    final_path = OUTPUT_DIR / "cq_with_final_answers.jsonl"
    
    with open(final_path, 'w', encoding='utf-8') as f:
        for cq in cq_with_answers:
            f.write(cq.model_dump_json() + '\n')
    
    logger.success(f" Сохранено {len(cq_with_answers)} CQs с финальными ответами в {final_path}")
    
    # Статистика
    status_counts = Counter(cq.final_answer.status for cq in cq_with_answers)
    answer_type_counts = Counter(cq.final_answer.answer_type for cq in cq_with_answers)
    
    logger.info(f"\n Статистика сохраненных результатов:")
    logger.info(f"  Всего CQs: {len(cq_with_answers)}")
    logger.info(f"  Распределение по status:")
    for status, count in status_counts.items():
        logger.info(f"    {status}: {count}")
    logger.info(f"  Распределение по answer_type:")
    for answer_type, count in answer_type_counts.items():
        logger.info(f"    {answer_type}: {count}")
    
    if USE_VALIDATION:
        validated_count = sum(1 for cq in cq_with_answers if cq.validation is not None)
        logger.info(f"  Валидировано: {validated_count} из {len(cq_with_answers)}")
else:
    logger.warning(" Нет результатов для сохранения")

# %%
# МЕТРИКИ И ВИЗУАЛИЗАЦИЯ

if cq_with_answers:
    # Вычисление метрик
    total_cqs = len(cq_with_answers)
    ok_count = sum(1 for cq in cq_with_answers if cq.final_answer.status == "OK")
    insufficient_count = sum(1 for cq in cq_with_answers if cq.final_answer.status == "INSUFFICIENT_EVIDENCE")
    
    answer_rate = ok_count / total_cqs if total_cqs > 0 else 0.0
    average_confidence = np.mean([cq.final_answer.confidence for cq in cq_with_answers])
    
    logger.info(f"\n Метрики:")
    logger.info(f"  Answer rate: {answer_rate:.2%} ({ok_count}/{total_cqs})")
    logger.info(f"  Average confidence: {average_confidence:.3f}")
    
    if USE_VALIDATION:
        validated_cqs = [cq for cq in cq_with_answers if cq.validation is not None]
        if validated_cqs:
            avg_justification = np.mean([cq.validation.justification_score for cq in validated_cqs])
            logger.info(f"  Average justification score: {avg_justification:.3f}")
    
    # DataFrame для визуализации
    df_results = pd.DataFrame([
        {
            "cq_id": cq.cq_id,
            "super_topic_id": cq.super_topic_id,
            "role": cq.role,
            "text": cq.text[:100] + "..." if len(cq.text) > 100 else cq.text,
            "answer_type": cq.final_answer.answer_type,
            "status": cq.final_answer.status,
            "confidence": cq.final_answer.confidence,
            "justification_score": cq.validation.justification_score if cq.validation else None
        }
        for cq in cq_with_answers
    ])
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Распределение answer_type
    answer_type_counts = df_results['answer_type'].value_counts()
    axes[0, 0].bar(answer_type_counts.index, answer_type_counts.values)
    axes[0, 0].set_xlabel("Answer Type")
    axes[0, 0].set_ylabel("Количество CQs")
    axes[0, 0].set_title("Распределение по answer_type")
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Распределение status
    status_counts = df_results['status'].value_counts()
    axes[0, 1].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title("Распределение по status")
    
    # Распределение confidence
    axes[1, 0].hist(df_results['confidence'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel("Confidence")
    axes[1, 0].set_ylabel("Количество CQs")
    axes[1, 0].set_title("Распределение confidence")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Распределение justification_score (если используется валидация)
    if USE_VALIDATION and df_results['justification_score'].notna().any():
        justification_scores = df_results['justification_score'].dropna()
        axes[1, 1].hist(justification_scores, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel("Justification Score")
        axes[1, 1].set_ylabel("Количество CQs")
        axes[1, 1].set_title("Распределение justification_score")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, "Валидация отключена", ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("Justification Score")
    
    plt.tight_layout()
    plt.show()
    
    # Таблица с примерами
    logger.info("\n Примеры CQs с финальными ответами (первые 10):")
    display(df_results.head(10))
else:
    logger.warning(" Нет данных для визуализации")

# %%
# АНАЛИЗ РЕЗУЛЬТАТОВ

if cq_with_answers:
    # Статистика по супертемам
    super_topic_stats = {}
    for cq in cq_with_answers:
        st_id = cq.super_topic_id
        if st_id not in super_topic_stats:
            super_topic_stats[st_id] = {"total": 0, "ok": 0, "insufficient": 0}
        super_topic_stats[st_id]["total"] += 1
        if cq.final_answer.status == "OK":
            super_topic_stats[st_id]["ok"] += 1
        else:
            super_topic_stats[st_id]["insufficient"] += 1
    
    logger.info(f"\n Статистика по супертемам:")
    for st_id, stats in sorted(super_topic_stats.items()):
        answer_rate = stats["ok"] / stats["total"] if stats["total"] > 0 else 0.0
        logger.info(f"  {st_id}: {stats['ok']}/{stats['total']} ответов ({answer_rate:.2%})")
    
    # Статистика по ролям
    role_stats = {}
    for cq in cq_with_answers:
        role = cq.role
        if role not in role_stats:
            role_stats[role] = {"total": 0, "ok": 0, "insufficient": 0}
        role_stats[role]["total"] += 1
        if cq.final_answer.status == "OK":
            role_stats[role]["ok"] += 1
        else:
            role_stats[role]["insufficient"] += 1
    
    logger.info(f"\n Статистика по ролям:")
    for role, stats in sorted(role_stats.items()):
        answer_rate = stats["ok"] / stats["total"] if stats["total"] > 0 else 0.0
        logger.info(f"  {role}: {stats['ok']}/{stats['total']} ответов ({answer_rate:.2%})")
    
    # Примеры CQs с INSUFFICIENT_EVIDENCE
    insufficient_cqs = [cq for cq in cq_with_answers if cq.final_answer.status == "INSUFFICIENT_EVIDENCE"]
    if insufficient_cqs:
        logger.info(f"\n Примеры CQs с INSUFFICIENT_EVIDENCE (первые 5):")
        for cq in insufficient_cqs[:5]:
            logger.info(f"  {cq.cq_id}: {cq.text[:80]}...")
            logger.info(f"    Ответ: {cq.final_answer.text[:100]}...")
    
    # Примеры CQs с низким justification_score (если используется валидация)
    if USE_VALIDATION:
        low_justification = [
            cq for cq in cq_with_answers
            if cq.validation and cq.validation.justification_score < 0.5
        ]
        if low_justification:
            logger.info(f"\n Примеры CQs с низким justification_score (<0.5, первые 5):")
            for cq in low_justification[:5]:
                logger.info(f"  {cq.cq_id}: {cq.text[:80]}...")
                logger.info(f"    Justification score: {cq.validation.justification_score:.2f}")
                logger.info(f"    Comment: {cq.validation.comment[:100]}...")
else:
    logger.warning(" Нет данных для анализа")
