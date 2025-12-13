# %% [markdown]
"""
# %%
# Демонстрация шага 3: Генерация черновых компетентностных вопросов (CQs)

Этот ноутбук демонстрирует полный процесс генерации черновых компетентностных вопросов (CQs) по темам из шага 1 с использованием:
1. Каталога шаблонов CQs (CLaRO-подобные паттерны)
2. Извлечения доменных сущностей из retrieved chunks
3. Сопоставления сущностей с паттернами
4. Slot filling с эвристиками
5. Инстанцирования паттернов через LLM
6. Генерации answer_hint
7. Фильтрации и дедупликации

Все задачи соответствуют описанию шага 3 из pipeline_desc.md и используют существующие решения проекта (Pydantic, loguru, trace_id, hybrid_search).


### Импорты
"""

# %%
import sys
from pathlib import Path
import json
import os
import time
import re
from typing import Any, Optional, List
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
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
    # Пробуем загрузить из корня проекта
    load_dotenv()

import weaviate
from loguru import logger
from pydantic import BaseModel, Field
from IPython.display import display

from oasis.pipelines.stage2_indexing import hybrid_search
from oasis.pipelines.stage1_extract_topics import _create_openai_client

# %%
# Настройки визуализации
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 200)
pd.set_option('display.width', None)

logger.info("Импорты загружены")

# %%
# Настройки путей и параметров
WEAVIATE_URL = "http://localhost:8081"
TOPICS_JSONL = Path("../outputs/deep_active_learning/canonical_topics.jsonl")
OUTPUT_DIR = Path("../outputs/deep_active_learning/step3")
TOPIC = "deep_active_learning"

# %%
# Параметры RAG поиска
TOPIC_NUMBER = 1  # Порядковый номер темы (1-based)
TOP_K = 20  # Количество чанков для извлечения сущностей (больше чем для финального поиска)
ALPHA = 0.5  # Вес векторного поиска

# %%
# Параметры LLM
LLM_MODEL = os.getenv("OPENAI_CANONICAL_MODEL", "gpt-5.1")
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 4000  # Для обычных запросов
LLM_MAX_TOKENS_RELATIONS = 8000  # Для извлечения отношений (больше из-за списка examples)
LLM_MAX_RETRIES = 3
LLM_TIMEOUT = 300.0  # Таймаут в секундах (5 минут) - увеличен для длинных запросов

# %%
# Параметры генерации CQs
TOP_N_ENTITIES = 10  # Топ-N сущностей для slot filling
DEDUP_SIMILARITY_THRESHOLD = 0.95  # Порог для дедупликации по эмбеддингам

# %%
# Параметры батчевой обработки (для разбиения больших задач на маленькие)
ENTITY_EXTRACTION_BATCH_SIZE = 5  # Количество чанков на батч для извлечения сущностей
RELATION_EXTRACTION_BATCH_SIZE = 5  # Количество чанков на батч для извлечения отношений

# %%
# Trace ID для трассировки
TRACE_ID = "demo_step3_cq_generation"

# %%
# Создаем выходную директорию
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Weaviate URL: {WEAVIATE_URL}")
logger.info(f"Topics file: {TOPICS_JSONL}")
logger.info(f"Output directory: {OUTPUT_DIR}")
logger.info(f"Topic number: {TOPIC_NUMBER}, Top-K: {TOP_K}, Alpha: {ALPHA}")
logger.info(f"LLM model: {LLM_MODEL}, temperature: {LLM_TEMPERATURE}")
logger.info(f"LLM max_tokens: {LLM_MAX_TOKENS} (обычные), {LLM_MAX_TOKENS_RELATIONS} (отношения)")
logger.info(f"LLM timeout: {LLM_TIMEOUT}с")

# %%
# Подключение к Weaviate
try:
    weaviate_client = weaviate.Client(WEAVIATE_URL, startup_period=10)
    logger.success(f" Подключено к Weaviate: {WEAVIATE_URL}")
    
    is_live = weaviate_client.is_live()
    is_ready = weaviate_client.is_ready()
    logger.info(f"  is_live: {is_live}, is_ready: {is_ready}")
except Exception as e:
    logger.error(f" Ошибка подключения к Weaviate: {e}")
    logger.error("Убедитесь, что Weaviate запущен: docker compose up -d weaviate")
    raise

# %%
# Инициализация OpenAI клиента с увеличенным timeout
try:
    openai_client = _create_openai_client(timeout=LLM_TIMEOUT)
    logger.success(f" OpenAI клиент инициализирован (timeout: {LLM_TIMEOUT}с)")
except Exception as e:
    logger.error(f" Ошибка инициализации OpenAI клиента: {e}")
    raise

# %%
# Загрузка тем из canonical_topics.jsonl
def load_topics(jsonl_path: Path) -> list[dict[str, Any]]:
    """Загружает темы из JSONL файла."""
    topics = []
    if not jsonl_path.exists():
        logger.error(f"Файл тем не найден: {jsonl_path}")
        return topics
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                topic = json.loads(line)
                topics.append(topic)
            except json.JSONDecodeError as e:
                logger.warning(f"Ошибка парсинга строки {line_num}: {e}")
    
    return topics

# %%
topics = load_topics(TOPICS_JSONL)
logger.info(f" Загружено тем: {len(topics)}")

# %%
# Выбор темы по номеру
if not topics:
    logger.error("Темы не загружены. Пропускаем.")
    selected_topic = None
else:
    if TOPIC_NUMBER < 1 or TOPIC_NUMBER > len(topics):
        logger.error(f"Неверный номер темы: {TOPIC_NUMBER} (доступно: 1-{len(topics)})")
        selected_topic = None
    else:
        selected_topic = topics[TOPIC_NUMBER - 1]
        logger.info(f" Выбрана тема #{TOPIC_NUMBER}:")
        logger.info(f"  ID: {selected_topic.get('canonical_topic_id', 'N/A')}")
        logger.info(f"  Название: {selected_topic.get('name', 'N/A')}")
        logger.info(f"  Описание: {selected_topic.get('description', 'N/A')[:200]}...")

# %%
# Выполнение RAG поиска для получения retrieved_chunks
if selected_topic:
    # Формируем запрос из названия и описания темы
    topic_name = selected_topic.get('name', '')
    topic_desc = selected_topic.get('description', '')
    query = f"{topic_name}. {topic_desc}"
    
    logger.info(f"\n Выполнение RAG поиска для извлечения контекста...")
    logger.info(f"  Запрос: {query[:150]}...")
    logger.info(f"  Top-K: {TOP_K}, Alpha: {ALPHA}")
    
    try:
        retrieved_chunks = hybrid_search(
            query=query,
            weaviate_client=weaviate_client,
            top_k=TOP_K,
            alpha=ALPHA,
            trace_id=TRACE_ID
        )
        
        logger.success(f" Найдено чанков: {len(retrieved_chunks)}")
        
        # Валидация наличия чанков
        if not retrieved_chunks:
            logger.warning("  Не найдено чанков для темы. Невозможно продолжить генерацию CQs.")
        else:
            logger.info(f"  Примеры chunk_id: {[c.get('chunk_id', 'N/A')[:50] for c in retrieved_chunks[:3]]}")
    except Exception as e:
        logger.error(f" Ошибка поиска: {e}")
        retrieved_chunks = []
else:
    retrieved_chunks = []
    logger.warning("Тема не выбрана, retrieved_chunks пуст")

# %%
# Pydantic модели для шаблонов CQs
from typing import List

class SlotDefinition(BaseModel):
    """Определение слота в шаблоне CQ."""
    name: str = Field(..., description="Имя слота (используется в template как <SlotName>)")
    entity_list: str = Field(..., description="Категория сущностей для заполнения (tasks/models/strategies/datasets/metrics/constraints)")
    description: str = Field(..., description="Описание назначения слота")

class CQPattern(BaseModel):
    """Шаблон компетентностного вопроса."""
    pattern_id: str = Field(..., description="Уникальный идентификатор шаблона")
    role: str = Field(..., description="Онтологическая роль: class/relation/process/constraint/comparison")
    template: str = Field(..., description="Шаблон вопроса с placeholders <SlotName>")
    slots: List[SlotDefinition] = Field(..., description="Список определений слотов")
    
    def get_slot_names(self) -> List[str]:
        """Возвращает список имен слотов."""
        return [slot.name for slot in self.slots]

# %%
# Создание каталога шаблонов
patterns_catalog = [
    CQPattern(
        pattern_id="comparison_methods",
        role="comparison",
        template="How does <Method1> compare to <Method2> in terms of <Metric> for <Task>?",
        slots=[
            SlotDefinition(name="Method1", entity_list="strategies", description="Первый метод/стратегия для сравнения"),
            SlotDefinition(name="Method2", entity_list="strategies", description="Второй метод/стратегия для сравнения"),
            SlotDefinition(name="Metric", entity_list="metrics", description="Метрика для сравнения"),
            SlotDefinition(name="Task", entity_list="tasks", description="Задача, для которой выполняется сравнение"),
        ]
    ),
    CQPattern(
        pattern_id="relation_effect",
        role="relation",
        template="How does <Strategy> affect <Metric> on <Dataset> for <Task>?",
        slots=[
            SlotDefinition(name="Strategy", entity_list="strategies", description="Стратегия активного обучения"),
            SlotDefinition(name="Metric", entity_list="metrics", description="Метрика производительности"),
            SlotDefinition(name="Dataset", entity_list="datasets", description="Набор данных"),
            SlotDefinition(name="Task", entity_list="tasks", description="Тип задачи"),
        ]
    ),
    CQPattern(
        pattern_id="constraint_budget",
        role="constraint",
        template="Under what annotation budget constraints does <Strategy> outperform <Baseline>?",
        slots=[
            SlotDefinition(name="Strategy", entity_list="strategies", description="Стратегия активного обучения"),
            SlotDefinition(name="Baseline", entity_list="strategies", description="Базовый метод для сравнения"),
        ]
    ),
    CQPattern(
        pattern_id="class_definition",
        role="class",
        template="What is <Entity> in the context of <Domain>?",
        slots=[
            SlotDefinition(name="Entity", entity_list="strategies", description="Сущность для определения (метод, модель, стратегия)"),
            SlotDefinition(name="Domain", entity_list="tasks", description="Доменная область применения"),
        ]
    ),
    CQPattern(
        pattern_id="process_description",
        role="process",
        template="How does <Process> work for <Task> using <Method>?",
        slots=[
            SlotDefinition(name="Process", entity_list="strategies", description="Процесс или алгоритм"),
            SlotDefinition(name="Task", entity_list="tasks", description="Тип задачи"),
            SlotDefinition(name="Method", entity_list="strategies", description="Метод реализации"),
        ]
    ),
    CQPattern(
        pattern_id="performance_evaluation",
        role="relation",
        template="What is the performance of <Strategy> on <Dataset> measured by <Metric>?",
        slots=[
            SlotDefinition(name="Strategy", entity_list="strategies", description="Стратегия активного обучения"),
            SlotDefinition(name="Dataset", entity_list="datasets", description="Набор данных"),
            SlotDefinition(name="Metric", entity_list="metrics", description="Метрика оценки"),
        ]
    ),
]

logger.info(f" Создан каталог из {len(patterns_catalog)} шаблонов")
logger.info(f"  Распределение по ролям: {Counter(p.role for p in patterns_catalog)}")

# %%
# Функции сохранения и загрузки каталога
def save_pattern_catalog(path: Path, patterns: List[CQPattern]) -> None:
    """Сохраняет каталог шаблонов в JSON файл."""
    patterns_dict = [p.model_dump() for p in patterns]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(patterns_dict, f, ensure_ascii=False, indent=2)
    logger.info(f" Каталог шаблонов сохранен: {path}")

def load_pattern_catalog(path: Path) -> List[CQPattern]:
    """Загружает каталог шаблонов из JSON файла."""
    if not path.exists():
        logger.warning(f"Файл каталога не найден: {path}")
        return []
    
    with open(path, 'r', encoding='utf-8') as f:
        patterns_dict = json.load(f)
    
    patterns = [CQPattern(**p) for p in patterns_dict]
    logger.info(f" Загружен каталог из {len(patterns)} шаблонов: {path}")
    return patterns

# %%
# Сохранение каталога
pattern_catalog_path = Path(f"../data/{TOPIC}/cq_patterns.json")
pattern_catalog_path.parent.mkdir(parents=True, exist_ok=True)
save_pattern_catalog(pattern_catalog_path, patterns_catalog)

# %%
# Визуализация каталога шаблонов
df_patterns = pd.DataFrame([
    {
        "pattern_id": p.pattern_id,
        "role": p.role,
        "template": p.template,
        "num_slots": len(p.slots),
        "slot_names": ", ".join(p.get_slot_names())
    }
    for p in patterns_catalog
])

logger.info("\n Каталог шаблонов CQs:")
logger.info("=" * 80)
display(df_patterns)

# %%
# Примеры использования каждого шаблона
logger.info("\n Примеры шаблонов:")
for pattern in patterns_catalog[:3]:  # Показываем первые 3
    logger.info(f"\n  {pattern.pattern_id} ({pattern.role}):")
    logger.info(f"    Template: {pattern.template}")
    logger.info(f"    Slots: {', '.join(pattern.get_slot_names())}")

# %%
# Подготовка контекста для LLM из retrieved_chunks
def prepare_entity_extraction_context(chunks: List[dict[str, Any]], top_k: int = 10) -> str:
    """Объединяет top-K чанков в единый контекст для извлечения сущностей."""
    if not chunks:
        return ""
    
    # Берем топ-K чанков по score
    top_chunks = sorted(chunks, key=lambda x: float(x.get('score', 0)), reverse=True)[:top_k]
    
    context_parts = []
    for i, chunk in enumerate(top_chunks, 1):
        chunk_id = chunk.get('chunk_id', 'N/A')
        text = chunk.get('text', '')
        # Преобразуем score в float (может быть строкой из Weaviate)
        score = chunk.get('score', 0)
        try:
            score = float(score) if score is not None else 0.0
        except (ValueError, TypeError):
            score = 0.0
        context_parts.append(f"[Chunk {i} (ID: {chunk_id}, score: {score:.3f})]\n{text}\n")
    
    return "\n".join(context_parts)

if retrieved_chunks:
    context_text = prepare_entity_extraction_context(retrieved_chunks, top_k=TOP_K)
    logger.info(f" Подготовлен контекст из {min(TOP_K, len(retrieved_chunks))} чанков")
    logger.info(f"  Общая длина контекста: {len(context_text)} символов")
    logger.info(f"  Первые 500 символов:\n{context_text[:500]}...")
else:
    context_text = ""
    logger.warning("  Нет retrieved_chunks для подготовки контекста")

# %%
# Промпт для извлечения сущностей
ENTITY_EXTRACTION_SYSTEM_PROMPT = """You are an expert in Deep Active Learning (DAL) and information extraction.
Your task is to extract domain entities from scientific text excerpts.

Extract entities that are EXPLICITLY mentioned in the provided text chunks. Do not invent or hallucinate entities.

Return a JSON object with the following structure:
{
  "tasks": ["list of task types mentioned, e.g., 'text classification', 'image segmentation'"],
  "models": ["list of model architectures mentioned, e.g., 'BERT', 'ResNet'"],
  "strategies": ["list of active learning strategies/methods mentioned, e.g., 'BALD', 'entropy sampling'"],
  "datasets": ["list of dataset names mentioned, e.g., 'CIFAR-10', 'IMDB'"],
  "metrics": ["list of evaluation metrics mentioned, e.g., 'accuracy', 'F1 score'"],
  "constraints": ["list of constraints mentioned, e.g., 'annotation budget', 'computational cost'"]
}

For each entity:
- Use the exact terminology as it appears in the text
- Include variations if they refer to the same concept (e.g., "BALD" and "Bayesian Active Learning by Disagreement")
- Do not include generic terms that are not specific to the domain
- Focus on entities that are relevant to Deep Active Learning

Return ONLY valid JSON, no additional text."""

def create_entity_extraction_user_prompt(topic_name: str, topic_description: str, context_chunks: str) -> dict[str, Any]:
    """Создает user prompt для извлечения сущностей."""
    return {
        "topic_name": topic_name,
        "topic_description": topic_description,
        "context_chunks": context_chunks
    }

logger.info(" Промпты для извлечения сущностей определены")

# %%
def normalize_label(label: str) -> str:
    """Нормализует метку сущности для создания canonical_label."""
    # Приводим к lowercase, удаляем лишние пробелы
    normalized = re.sub(r'\s+', ' ', label.strip().lower())
    return normalized

# %%
# Функция вызова LLM для извлечения сущностей по определенным типам (разбиение задачи)
def extract_entities_llm_by_types(
    topic: dict[str, Any],
    chunks: List[dict[str, Any]],
    entity_types: List[str],
    trace_id: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> dict[str, Any]:
    """Вызывает LLM для извлечения указанных типов сущностей из чанков."""
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    
    if not chunks:
        return {et: [] for et in entity_types}
    
    # Создаем промпт только для указанных типов
    entity_types_str = ", ".join(entity_types)
    system_prompt = f"""You are an expert in Deep Active Learning (DAL) and information extraction.
Your task is to extract domain entities from scientific text excerpts.

Extract ONLY the following entity types: {entity_types_str}

Extract entities that are EXPLICITLY mentioned in the provided text chunks. Do not invent or hallucinate entities.

Return a JSON object with the following structure:
{{
  {', '.join([f'"{et}": ["list of {et} mentioned"]' for et in entity_types])}
}}

For each entity:
- Use the exact terminology as it appears in the text
- Include variations if they refer to the same concept
- Do not include generic terms that are not specific to the domain
- Focus on entities that are relevant to Deep Active Learning

Return ONLY valid JSON, no additional text."""
    
    # Подготовка контекста
    context_text = prepare_entity_extraction_context(chunks, top_k=len(chunks))
    topic_name = topic.get('name', '')
    topic_description = topic.get('description', '')
    
    # Создание промпта
    user_payload = create_entity_extraction_user_prompt(topic_name, topic_description, context_text)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    
    # Вызов LLM с retry логикой
    for attempt in range(max_retries):
        try:
            logger.debug(f"{trace_prefix}LLM вызов для извлечения сущностей типов {entity_types} (попытка {attempt + 1}/{max_retries})")
            response = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Пустой ответ от LLM")
            
            parsed = json.loads(content)
            # Валидация структуры - проверяем только запрошенные типы
            if not all(key in parsed for key in entity_types):
                missing = [k for k in entity_types if k not in parsed]
                raise ValueError(f"Ответ LLM не содержит требуемые ключи: {missing}")
            
            num_entities = sum(len(parsed.get(k, [])) for k in entity_types)
            logger.debug(f"{trace_prefix} Извлечено {num_entities} сущностей типов {entity_types}")
            return parsed
            
        except json.JSONDecodeError as json_exc:
            # Логируем полный ответ LLM для отладки
            content_preview = content[:500] if len(content) > 500 else content
            logger.error(f"{trace_prefix}Ошибка парсинга JSON (типы {entity_types}, попытка {attempt + 1}): {json_exc}")
            logger.error(f"{trace_prefix}Начало ответа LLM (первые 500 символов):\n{content_preview}")
            if len(content) > 500:
                logger.error(f"{trace_prefix}... (всего {len(content)} символов)")
            
            # Сохраняем полный ответ в файл для детального анализа
            error_log_path = OUTPUT_DIR / f"llm_error_response_entities_{'_'.join(entity_types)}_{trace_id}_{int(time.time())}.txt"
            try:
                with open(error_log_path, 'w', encoding='utf-8') as f:
                    f.write(f"Error: {json_exc}\n")
                    f.write(f"Entity types: {entity_types}\n")
                    f.write(f"Attempt: {attempt + 1}/{max_retries}\n")
                    f.write(f"Full LLM response:\n{content}\n")
                logger.error(f"{trace_prefix}Полный ответ сохранен в: {error_log_path}")
            except Exception as save_exc:
                logger.warning(f"{trace_prefix}Не удалось сохранить ответ: {save_exc}")
            
            if attempt < max_retries - 1:
                delay = retry_delay * (2 ** attempt)
                logger.warning(f"{trace_prefix}Повтор через {delay:.1f}с")
                time.sleep(delay)
            else:
                logger.error(f"{trace_prefix}Не удалось распарсить JSON после {max_retries} попыток")
                raise
                
        except Exception as exc:
            if attempt < max_retries - 1:
                delay = retry_delay * (2 ** attempt)
                logger.warning(f"{trace_prefix}Ошибка LLM (типы {entity_types}, попытка {attempt + 1}): {exc}. Повтор через {delay:.1f}с")
                time.sleep(delay)
            else:
                logger.error(f"{trace_prefix}Не удалось извлечь сущности типов {entity_types}: {exc}")
                raise
    
    # Fallback
    return {et: [] for et in entity_types}

# %%
# Функция вызова LLM для извлечения сущностей с разбиением по типам (аналогично stage1_canonical_topics.py)
def extract_entities_llm(
    topic: dict[str, Any],
    chunks: List[dict[str, Any]],
    trace_id: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> dict[str, Any]:
    """Вызывает LLM для извлечения доменных сущностей из чанков с разбиением по типам.
    
    Разбивает задачу на 3 запроса для уменьшения нагрузки и времени ответа:
    - tasks + models
    - strategies + datasets  
    - metrics + constraints
    """
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    
    if not chunks:
        logger.warning(f"{trace_prefix}Нет чанков для извлечения сущностей")
        return {
            "tasks": [], "models": [], "strategies": [],
            "datasets": [], "metrics": [], "constraints": []
        }
    
    # Разбиваем извлечение на 3 группы типов для уменьшения нагрузки
    entity_type_groups = [
        ["tasks", "models"],
        ["strategies", "datasets"],
        ["metrics", "constraints"]
    ]
    
    logger.info(f"{trace_prefix}Извлечение сущностей разбито на {len(entity_type_groups)} запроса по типам")
    
    # Агрегируем результаты из всех групп
    aggregated = {
        "tasks": [], "models": [], "strategies": [],
        "datasets": [], "metrics": [], "constraints": []
    }
    
    # Обрабатываем каждую группу типов
    for group_idx, entity_types in enumerate(entity_type_groups, 1):
        try:
            logger.info(f"{trace_prefix}Запрос {group_idx}/{len(entity_type_groups)}: извлечение типов {entity_types}")
            group_result = extract_entities_llm_by_types(
                topic=topic,
                chunks=chunks,
                entity_types=entity_types,
                trace_id=trace_id,
                max_retries=max_retries,
                retry_delay=retry_delay
            )
            
            # Агрегируем результаты
            for entity_type in entity_types:
                entities = group_result.get(entity_type, [])
                # Дедуплицируем по нормализованному значению
                existing_normalized = {normalize_label(e) for e in aggregated[entity_type]}
                for entity in entities:
                    if entity and entity.strip():
                        normalized = normalize_label(entity)
                        if normalized not in existing_normalized:
                            aggregated[entity_type].append(entity)
                            existing_normalized.add(normalized)
            
            logger.success(f"{trace_prefix} Запрос {group_idx}: извлечено {sum(len(group_result.get(et, [])) for et in entity_types)} сущностей")
            
        except Exception as exc:
            logger.warning(f"{trace_prefix}Ошибка при обработке группы типов {entity_types}: {exc}. Пропускаем группу.")
            continue
    
    # Итоговая статистика
    total_entities = sum(len(aggregated.get(k, [])) for k in aggregated.keys())
    logger.success(f"{trace_prefix} Сущности извлечены из всех групп: {total_entities} уникальных сущностей")
    logger.info(f"{trace_prefix}  Распределение: tasks={len(aggregated['tasks'])}, models={len(aggregated['models'])}, "
                f"strategies={len(aggregated['strategies'])}, datasets={len(aggregated['datasets'])}, "
                f"metrics={len(aggregated['metrics'])}, constraints={len(aggregated['constraints'])}")
    
    return aggregated

# %%
# Вызов LLM для извлечения сущностей
if selected_topic and retrieved_chunks:
    logger.info(f"\n Извлечение сущностей для темы '{selected_topic.get('name', 'N/A')}'...")
    try:
        extracted_entities_raw = extract_entities_llm(
            topic=selected_topic,
            chunks=retrieved_chunks,
            trace_id=TRACE_ID
        )
        logger.success(" Сущности извлечены")
        logger.info(f"  Tasks: {len(extracted_entities_raw.get('tasks', []))}")
        logger.info(f"  Models: {len(extracted_entities_raw.get('models', []))}")
        logger.info(f"  Strategies: {len(extracted_entities_raw.get('strategies', []))}")
        logger.info(f"  Datasets: {len(extracted_entities_raw.get('datasets', []))}")
        logger.info(f"  Metrics: {len(extracted_entities_raw.get('metrics', []))}")
        logger.info(f"  Constraints: {len(extracted_entities_raw.get('constraints', []))}")
    except Exception as e:
        logger.error(f" Ошибка извлечения сущностей: {e}")
        extracted_entities_raw = {
            "tasks": [], "models": [], "strategies": [],
            "datasets": [], "metrics": [], "constraints": []
        }
else:
    extracted_entities_raw = {
        "tasks": [], "models": [], "strategies": [],
        "datasets": [], "metrics": [], "constraints": []
    }
    logger.warning("  Невозможно извлечь сущности: нет темы или чанков")

# %%
# Pydantic модель EntityCandidate
class SourceMention(BaseModel):
    """Упоминание сущности в источнике."""
    chunk_id: str = Field(..., description="ID чанка, где упоминается сущность")
    context_phrase: str = Field(..., description="Контекстная фраза с упоминанием сущности")

class EntityCandidate(BaseModel):
    """Кандидатная сущность, извлеченная из текста."""
    entity_id: str = Field(..., description="Уникальный идентификатор сущности (E_TASK_xxx, E_MODEL_xxx, etc.)")
    label: str = Field(..., description="Метка сущности как она встречается в тексте")
    canonical_label: str = Field(..., description="Каноническая нормализованная метка")
    entity_type: str = Field(..., description="Тип сущности: task/model/strategy/dataset/metric/constraint")
    topic_ids: List[str] = Field(default_factory=list, description="Список ID тем, где встречалась сущность")
    source_mentions: List[SourceMention] = Field(default_factory=list, description="Список упоминаний в источниках")

logger.info(" Pydantic модель EntityCandidate определена")

# %%
# Обработка результатов извлечения: преобразование в EntityCandidate

def find_mentions_in_chunks(entity_label: str, chunks: List[dict[str, Any]], max_context_length: int = 100) -> List[SourceMention]:
    """Находит упоминания сущности в чанках и извлекает контекстные фразы."""
    mentions = []
    entity_lower = entity_label.lower()
    
    for chunk in chunks:
        chunk_id = chunk.get('chunk_id', '')
        text = chunk.get('text', '').lower()
        
        # Ищем упоминание (простой поиск подстроки)
        if entity_lower in text:
            # Находим позицию упоминания
            pos = text.find(entity_lower)
            # Извлекаем контекстную фразу
            start = max(0, pos - max_context_length // 2)
            end = min(len(text), pos + len(entity_lower) + max_context_length // 2)
            context = chunk.get('text', '')[start:end].strip()
            
            mentions.append(SourceMention(
                chunk_id=chunk_id,
                context_phrase=context
            ))
    
    return mentions

def process_extracted_entities(
    extracted_entities: dict[str, Any],
    topic_id: str,
    chunks: List[dict[str, Any]],
    entity_counter: dict[str, int] = None
) -> List[EntityCandidate]:
    """Преобразует сырой ответ LLM в список EntityCandidate."""
    if entity_counter is None:
        entity_counter = defaultdict(int)
    
    all_entities = []
    entity_type_mapping = {
        "tasks": "task",
        "models": "model",
        "strategies": "strategy",
        "datasets": "dataset",
        "metrics": "metric",
        "constraints": "constraint"
    }
    
    for category, entity_type in entity_type_mapping.items():
        entities_list = extracted_entities.get(category, [])
        for entity_label in entities_list:
            if not entity_label or not entity_label.strip():
                continue
            
            entity_counter[entity_type] += 1
            entity_id = f"E_{entity_type.upper()}_{entity_counter[entity_type]:03d}"
            canonical_label = normalize_label(entity_label)
            
            # Находим упоминания в чанках
            mentions = find_mentions_in_chunks(entity_label, chunks)
            
            entity = EntityCandidate(
                entity_id=entity_id,
                label=entity_label,
                canonical_label=canonical_label,
                entity_type=entity_type,
                topic_ids=[topic_id],
                source_mentions=mentions
            )
            all_entities.append(entity)
    
    return all_entities

# %%
# Обработка извлеченных сущностей
if selected_topic and extracted_entities_raw:
    topic_id = selected_topic.get('canonical_topic_id', f'TOPIC_{TOPIC_NUMBER:03d}')
    entity_counter = defaultdict(int)
    entity_candidates = process_extracted_entities(
        extracted_entities_raw,
        topic_id,
        retrieved_chunks,
        entity_counter
    )
    
    logger.success(f" Обработано {len(entity_candidates)} сущностей")
    
    # Группировка по типам
    entities_by_type = defaultdict(list)
    for entity in entity_candidates:
        entities_by_type[entity.entity_type].append(entity)
    
    logger.info(f"  Распределение по типам:")
    for entity_type, entities in entities_by_type.items():
        logger.info(f"    {entity_type}: {len(entities)}")
else:
    entity_candidates = []
    entities_by_type = defaultdict(list)
    logger.warning("  Нет данных для обработки сущностей")

# %%
# Pydantic модель RelationCandidate
class RelationExample(BaseModel):
    """Пример отношения между сущностями."""
    subject: str = Field(..., description="Сущность в позиции субъекта")
    object: str = Field(..., description="Сущность в позиции объекта")
    other: dict[str, str] = Field(default_factory=dict, description="Другие сущности в паттерне")
    chunk_id: str = Field(..., description="ID чанка, где встречается пример")

class RelationCandidate(BaseModel):
    """Кандидатное отношение между сущностями."""
    relation_id: str = Field(..., description="Уникальный идентификатор отношения")
    pattern: str = Field(..., description="Словесный шаблон отношения, например '[Strategy] improves [Metric]'")
    subject_role: str = Field(..., description="Тип сущности в позиции субъекта")
    object_role: str = Field(..., description="Тип сущности в позиции объекта")
    other_roles: List[str] = Field(default_factory=list, description="Типы других сущностей в паттерне")
    examples: List[RelationExample] = Field(default_factory=list, description="Примеры конкретных пар сущностей")

# %%
# Извлечение отношений (опционально)
RELATION_EXTRACTION_SYSTEM_PROMPT = """You are an expert in Deep Active Learning (DAL) and information extraction.
Your task is to identify common relationship patterns between entities mentioned in the provided text chunks.

Extract relationship patterns that are EXPLICITLY mentioned in the text. Each pattern should describe a relationship between entities.

Return a JSON object with a list of relationship patterns:
{
  "relations": [
    {
      "pattern": "verbal template of the relationship, e.g., '[Strategy] improves [Metric] on [Dataset]'",
      "subject_role": "type of entity in subject position (e.g., 'strategy')",
      "object_role": "type of entity in object position (e.g., 'metric')",
      "other_roles": ["list of other entity types in the pattern"],
      "examples": [
        {
          "subject": "entity name in subject",
          "object": "entity name in object",
          "other": {"role": "entity name"},
          "chunk_id": "chunk_id where this example appears"
        }
      ]
    }
  ]
}

Focus on relationships that are relevant to Deep Active Learning, such as:
- Strategy improves/outperforms Metric
- Strategy works for Task
- Strategy requires Constraint
- Model achieves Metric on Dataset

Return ONLY valid JSON, no additional text."""

# %%
# Функция вызова LLM для извлечения отношений из одного батча
def extract_relations_llm_batch(
    topic: dict[str, Any],
    batch_chunks: List[dict[str, Any]],
    entities: List[EntityCandidate],
    trace_id: str | None = None,
    max_retries: int = 3,
    max_tokens: int = None,
    batch_num: int = 0,
    total_batches: int = 1
) -> dict[str, Any]:
    """Вызывает LLM для извлечения паттернов отношений из одного батча чанков."""
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    
    if not batch_chunks:
        return {"relations": []}
    
    # Используем увеличенный max_tokens для отношений, если не указан
    if max_tokens is None:
        max_tokens = LLM_MAX_TOKENS_RELATIONS
    
    context_text = prepare_entity_extraction_context(batch_chunks, top_k=len(batch_chunks))
    topic_name = topic.get('name', '')
    topic_description = topic.get('description', '')
    
    # Создаем список сущностей для контекста
    entities_summary = {}
    for entity_type in ["task", "model", "strategy", "dataset", "metric", "constraint"]:
        entities_summary[entity_type] = [e.label for e in entities if e.entity_type == entity_type]
    
    user_payload = {
        "topic_name": topic_name,
        "topic_description": topic_description,
        "context_chunks": context_text,
        "extracted_entities": entities_summary
    }
    
    messages = [
        {"role": "system", "content": RELATION_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"{trace_prefix}LLM вызов для извлечения отношений (батч {batch_num + 1}/{total_batches}, попытка {attempt + 1}/{max_retries}, max_tokens={max_tokens})")
            response = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            
            if not content:
                raise ValueError("Пустой ответ от LLM")
            
            # Проверяем, не был ли ответ обрезан
            if finish_reason == "length":
                logger.warning(f"{trace_prefix}  Ответ LLM обрезан (батч {batch_num + 1}, finish_reason=length). Увеличьте max_tokens.")
            
            parsed = json.loads(content)
            num_relations = len(parsed.get('relations', []))
            logger.debug(f"{trace_prefix} Батч {batch_num + 1}: извлечено {num_relations} паттернов отношений")
            return parsed
            
        except json.JSONDecodeError as json_exc:
            # Логируем полный ответ LLM для отладки
            content_preview = content[:500] if len(content) > 500 else content
            logger.error(f"{trace_prefix}Ошибка парсинга JSON (батч {batch_num + 1}, попытка {attempt + 1}): {json_exc}")
            logger.error(f"{trace_prefix}Начало ответа LLM (первые 500 символов):\n{content_preview}")
            if len(content) > 500:
                logger.error(f"{trace_prefix}... (всего {len(content)} символов)")
            
            # Сохраняем полный ответ в файл для детального анализа
            error_log_path = OUTPUT_DIR / f"llm_error_response_relations_batch_{batch_num}_{trace_id}_{int(time.time())}.txt"
            try:
                with open(error_log_path, 'w', encoding='utf-8') as f:
                    f.write(f"Error: {json_exc}\n")
                    f.write(f"Batch: {batch_num + 1}/{total_batches}\n")
                    f.write(f"Attempt: {attempt + 1}/{max_retries}\n")
                    f.write(f"Max tokens: {max_tokens}\n")
                    f.write(f"Full LLM response:\n{content}\n")
                logger.error(f"{trace_prefix}Полный ответ сохранен в: {error_log_path}")
            except Exception as save_exc:
                logger.warning(f"{trace_prefix}Не удалось сохранить ответ: {save_exc}")
            
            if attempt < max_retries - 1:
                delay = 1.0 * (2 ** attempt)
                logger.warning(f"{trace_prefix}Повтор через {delay:.1f}с")
                # Увеличиваем max_tokens при повторной попытке
                if max_tokens < 16000:
                    max_tokens = min(max_tokens * 2, 16000)
                    logger.info(f"{trace_prefix}Увеличиваем max_tokens до {max_tokens} для следующей попытки")
                time.sleep(delay)
            else:
                logger.warning(f"{trace_prefix}Не удалось распарсить JSON после {max_retries} попыток: {json_exc}")
                return {"relations": []}
        except Exception as exc:
            if attempt < max_retries - 1:
                delay = 1.0 * (2 ** attempt)
                logger.warning(f"{trace_prefix}Ошибка LLM (батч {batch_num + 1}, попытка {attempt + 1}): {exc}. Повтор через {delay:.1f}с")
                time.sleep(delay)
            else:
                logger.warning(f"{trace_prefix}Не удалось извлечь отношения из батча {batch_num + 1}: {exc}")
                return {"relations": []}
    
    return {"relations": []}

# %%
# Функция вызова LLM для извлечения отношений с батчевой обработкой
def extract_relations_llm(
    topic: dict[str, Any],
    chunks: List[dict[str, Any]],
    entities: List[EntityCandidate],
    trace_id: str | None = None,
    max_retries: int = 3,
    max_tokens: int = None,
    batch_size: int = None
) -> dict[str, Any]:
    """Вызывает LLM для извлечения паттернов отношений батчами."""
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    
    if not chunks:
        return {"relations": []}
    
    if batch_size is None:
        batch_size = RELATION_EXTRACTION_BATCH_SIZE
    
    # Разбиваем чанки на батчи
    total_chunks = len(chunks)
    num_batches = (total_chunks + batch_size - 1) // batch_size  # Округление вверх
    
    logger.info(f"{trace_prefix}Обработка {total_chunks} чанков в {num_batches} батчах для извлечения отношений (размер батча: {batch_size})")
    
    # Агрегируем результаты из всех батчей
    all_relations = []
    seen_patterns = set()  # Для дедупликации паттернов
    
    # Обрабатываем каждый батч
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_chunks)
        batch_chunks = chunks[start_idx:end_idx]
        
        try:
            batch_result = extract_relations_llm_batch(
                topic=topic,
                batch_chunks=batch_chunks,
                entities=entities,
                trace_id=trace_id,
                max_retries=max_retries,
                max_tokens=max_tokens,
                batch_num=batch_idx,
                total_batches=num_batches
            )
            
            # Агрегируем паттерны отношений
            batch_relations = batch_result.get('relations', [])
            for relation in batch_relations:
                pattern = relation.get('pattern', '')
                # Дедуплицируем по паттерну (нормализованному)
                pattern_normalized = normalize_label(pattern)
                if pattern_normalized and pattern_normalized not in seen_patterns:
                    all_relations.append(relation)
                    seen_patterns.add(pattern_normalized)
            
        except Exception as exc:
            logger.warning(f"{trace_prefix}Ошибка при обработке батча {batch_idx + 1}: {exc}. Пропускаем батч.")
            continue
    
    logger.success(f"{trace_prefix} Отношения извлечены из всех батчей: {len(all_relations)} уникальных паттернов")
    
    return {"relations": all_relations}


# %%
# Извлечение отношений (опционально)
if selected_topic and retrieved_chunks and entity_candidates:
    logger.info(f"\n Извлечение паттернов отношений...")
    try:
        relations_raw = extract_relations_llm(
            topic=selected_topic,
            chunks=retrieved_chunks,
            entities=entity_candidates,
            trace_id=TRACE_ID
        )
        
        # Преобразование в RelationCandidate
        relation_candidates = []
        for i, rel in enumerate(relations_raw.get('relations', []), 1):
            relation_id = f"R_{i:03d}"
            examples = [
                RelationExample(**ex) for ex in rel.get('examples', [])
            ]
            relation = RelationCandidate(
                relation_id=relation_id,
                pattern=rel.get('pattern', ''),
                subject_role=rel.get('subject_role', ''),
                object_role=rel.get('object_role', ''),
                other_roles=rel.get('other_roles', []),
                examples=examples
            )
            relation_candidates.append(relation)
        
        logger.success(f" Извлечено {len(relation_candidates)} паттернов отношений")
    except Exception as e:
        logger.warning(f"  Ошибка извлечения отношений: {e}")
        relation_candidates = []
else:
    relation_candidates = []
    logger.info("  Пропущено извлечение отношений (нет данных)")

# %%
# Сохранение EntityCandidate и RelationCandidate в JSONL
def save_entity_candidates(path: Path, entities: List[EntityCandidate]) -> None:
    """Сохраняет список EntityCandidate в JSONL файл."""
    with open(path, 'w', encoding='utf-8') as f:
        for entity in entities:
            f.write(json.dumps(entity.model_dump(), ensure_ascii=False) + '\n')
    logger.info(f" Сохранено {len(entities)} сущностей: {path}")

def save_relation_candidates(path: Path, relations: List[RelationCandidate]) -> None:
    """Сохраняет список RelationCandidate в JSONL файл."""
    with open(path, 'w', encoding='utf-8') as f:
        for relation in relations:
            f.write(json.dumps(relation.model_dump(), ensure_ascii=False) + '\n')
    logger.info(f" Сохранено {len(relations)} отношений: {path}")

# %%
# Сохранение результатов
if entity_candidates:
    entity_path = OUTPUT_DIR / "EntityCandidate.jsonl"
    save_entity_candidates(entity_path, entity_candidates)

if relation_candidates:
    relation_path = OUTPUT_DIR / "RelationCandidate.jsonl"
    save_relation_candidates(relation_path, relation_candidates)

# %%
# Загрузка EntityCandidate и RelationCandidate из JSONL (если файлы существуют)
def load_entity_candidates(path: Path) -> List[EntityCandidate]:
    """Загружает список EntityCandidate из JSONL файла."""
    if not path.exists():
        return []
    
    entities = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                entities.append(EntityCandidate(**data))
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Ошибка при загрузке сущности из {path}: {e}")
                continue
    
    logger.info(f" Загружено {len(entities)} сущностей из {path}")
    return entities

def load_relation_candidates(path: Path) -> List[RelationCandidate]:
    """Загружает список RelationCandidate из JSONL файла."""
    if not path.exists():
        return []
    
    relations = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                relations.append(RelationCandidate(**data))
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Ошибка при загрузке отношения из {path}: {e}")
                continue
    
    logger.info(f" Загружено {len(relations)} отношений из {path}")
    return relations

# %%
# Проверяем наличие сохраненных файлов и загружаем, если они есть
entity_path = OUTPUT_DIR / "EntityCandidate.jsonl"
relation_path = OUTPUT_DIR / "RelationCandidate.jsonl"

# %%
# Загружаем сущности, если файл существует
if entity_path.exists():
    logger.info(f"\n Найден файл с сущностями: {entity_path}")
    logger.info("  Загружаем сохраненные сущности (пропускаем извлечение через LLM)...")
    entity_candidates = load_entity_candidates(entity_path)
    
    # Восстанавливаем entities_by_type для совместимости
    entities_by_type = defaultdict(list)
    for entity in entity_candidates:
        entities_by_type[entity.entity_type].append(entity)
    
    logger.success(f" Загружено {len(entity_candidates)} сущностей")
    logger.info(f"  Распределение по типам:")
    for entity_type, entities in entities_by_type.items():
        logger.info(f"    {entity_type}: {len(entities)}")
else:
    logger.info(f"  Файл с сущностями не найден: {entity_path}")
    logger.info("  Сущности будут извлечены через LLM (см. предыдущие ячейки)")

# %%
# Загружаем отношения, если файл существует
if relation_path.exists():
    logger.info(f"\n Найден файл с отношениями: {relation_path}")
    logger.info("  Загружаем сохраненные отношения (пропускаем извлечение через LLM)...")
    relation_candidates = load_relation_candidates(relation_path)
    logger.success(f" Загружено {len(relation_candidates)} отношений")
else:
    logger.info(f"  Файл с отношениями не найден: {relation_path}")
    logger.info("  Отношения будут извлечены через LLM (см. предыдущие ячейки)")
    relation_candidates = []

# %%
# Визуализация извлеченных сущностей
if entity_candidates:
    # Таблицы по категориям
    for entity_type in ["task", "model", "strategy", "dataset", "metric", "constraint"]:
        entities_of_type = [e for e in entity_candidates if e.entity_type == entity_type]
        if entities_of_type:
            df_type = pd.DataFrame([
                {
                    "entity_id": e.entity_id,
                    "label": e.label,
                    "canonical_label": e.canonical_label,
                    "mentions": len(e.source_mentions)
                }
                for e in entities_of_type
            ])
            logger.info(f"\n Сущности типа '{entity_type}' ({len(entities_of_type)}):")
            display(df_type)
    
    # График распределения по типам
    type_counts = Counter(e.entity_type for e in entity_candidates)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(type_counts.keys(), type_counts.values())
    ax.set_xlabel("Тип сущности")
    ax.set_ylabel("Количество")
    ax.set_title("Распределение извлеченных сущностей по типам")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Топ-10 наиболее частых сущностей (по количеству упоминаний)
    entity_mention_counts = [(e.label, len(e.source_mentions)) for e in entity_candidates]
    entity_mention_counts.sort(key=lambda x: x[1], reverse=True)
    top_entities = entity_mention_counts[:10]
    
    logger.info("\n Топ-10 сущностей по количеству упоминаний:")
    for label, count in top_entities:
        logger.info(f"  {label}: {count} упоминаний")
else:
    logger.warning("  Нет сущностей для визуализации")

# %%
# Конфигурация сопоставления сущностей с паттернами
def map_entities_to_pattern_slots(
    pattern: CQPattern,
    entities: List[EntityCandidate]
) -> dict[str, List[EntityCandidate]]:
    """Сопоставляет сущности со слотами паттерна и возвращает словарь подходящих значений."""
    slot_entities = {}
    
    # Группируем сущности по типам
    entities_by_type = defaultdict(list)
    for entity in entities:
        entities_by_type[entity.entity_type].append(entity)
    
    # Маппинг типов сущностей к категориям entity_list
    type_mapping = {
        "task": "tasks",
        "model": "models",
        "strategy": "strategies",
        "dataset": "datasets",
        "metric": "metrics",
        "constraint": "constraints"
    }
    
    # Для каждого слота находим подходящие сущности
    for slot in pattern.slots:
        entity_list_name = slot.entity_list  # например, "strategies"
        # Находим тип сущности по entity_list_name
        entity_type = None
        for etype, elist in type_mapping.items():
            if elist == entity_list_name:
                entity_type = etype
                break
        
        if entity_type:
            slot_entities[slot.name] = entities_by_type.get(entity_type, [])
        else:
            logger.warning(f"Неизвестный entity_list '{entity_list_name}' для слота '{slot.name}'")
            slot_entities[slot.name] = []
    
    return slot_entities

# %%
# Применяем сопоставление для всех паттернов
pattern_entity_mapping = {}
for pattern in patterns_catalog:
    mapping = map_entities_to_pattern_slots(pattern, entity_candidates)
    pattern_entity_mapping[pattern.pattern_id] = mapping
    
    # Логируем результаты
    logger.info(f"\n Паттерн '{pattern.pattern_id}':")
    for slot_name, entities in mapping.items():
        logger.info(f"  {slot_name}: {len(entities)} доступных сущностей")
        if entities:
            logger.info(f"    Примеры: {', '.join([e.label for e in entities[:3]])}")

logger.success(" Сопоставление сущностей с паттернами завершено")

# %%
# Валидация сопоставления: проверка наличия достаточных сущностей
def validate_pattern_mapping(pattern: CQPattern, slot_entities: dict[str, List[EntityCandidate]]) -> bool:
    """Проверяет, что для каждого слота паттерна есть доступные сущности."""
    for slot in pattern.slots:
        entities = slot_entities.get(slot.name, [])
        if not entities:
            return False
    return True

# %%
# Сохраняем исходный список паттернов для валидации (на случай повторного выполнения ячейки)
if 'pattern_entity_mapping' not in globals() or not pattern_entity_mapping:
    logger.error("  pattern_entity_mapping не найден. Выполните предыдущую ячейку задачи 3.")
    valid_patterns = []
    invalid_patterns = []
else:
    # Фильтрация паттернов без достаточных сущностей
    # Используем исходный patterns_catalog из задачи 1, если он еще не был отфильтрован
    # Проверяем, есть ли в pattern_entity_mapping все паттерны из patterns_catalog
    all_pattern_ids_in_mapping = set(pattern_entity_mapping.keys())
    all_pattern_ids_in_catalog = {p.pattern_id for p in patterns_catalog}
    
    # Если есть рассинхронизация, используем паттерны из mapping
    if not all_pattern_ids_in_catalog.issubset(all_pattern_ids_in_mapping):
        logger.warning("  Обнаружена рассинхронизация между patterns_catalog и pattern_entity_mapping")
        logger.info(f"  Паттерны в catalog: {all_pattern_ids_in_catalog}")
        logger.info(f"  Паттерны в mapping: {all_pattern_ids_in_mapping}")
        # Используем паттерны из mapping для валидации
        patterns_to_validate = [p for p in patterns_catalog if p.pattern_id in all_pattern_ids_in_mapping]
    else:
        patterns_to_validate = patterns_catalog
    
    valid_patterns = []
    invalid_patterns = []

    for pattern in patterns_to_validate:
        if pattern.pattern_id not in pattern_entity_mapping:
            logger.warning(f"  Паттерн {pattern.pattern_id} отсутствует в pattern_entity_mapping, пропускаем")
            invalid_patterns.append(pattern)
            continue
            
        mapping = pattern_entity_mapping[pattern.pattern_id]
        if validate_pattern_mapping(pattern, mapping):
            valid_patterns.append(pattern)
        else:
            invalid_patterns.append(pattern)

logger.info(f"\n Валидация сопоставления:")
logger.info(f"  Валидных паттернов: {len(valid_patterns)}")
logger.info(f"  Невалидных паттернов (недостаточно сущностей): {len(invalid_patterns)}")

if invalid_patterns:
    logger.warning("  Пропущенные паттерны:")
    for pattern in invalid_patterns:
        if pattern.pattern_id in pattern_entity_mapping:
            mapping = pattern_entity_mapping[pattern.pattern_id]
            missing_slots = [slot.name for slot in pattern.slots if not mapping.get(slot.name)]
            logger.warning(f"  {pattern.pattern_id}: отсутствуют сущности для слотов {missing_slots}")
        else:
            logger.warning(f"  {pattern.pattern_id}: отсутствует в pattern_entity_mapping")

# %%
# Используем только валидные паттерны для дальнейшей обработки
patterns_catalog = valid_patterns
logger.info(f" Для генерации CQs будет использовано {len(patterns_catalog)} паттернов")
if patterns_catalog:
    logger.info(f"  Паттерны: {[p.pattern_id for p in patterns_catalog]}")

# %%
# Эвристики для slot filling
import random

def get_top_entities(entities: List[EntityCandidate], top_n: int = 10) -> List[EntityCandidate]:
    """Возвращает топ-N сущностей по количеству упоминаний."""
    sorted_entities = sorted(entities, key=lambda e: len(e.source_mentions), reverse=True)
    return sorted_entities[:top_n]

def find_co_occurring_entities(
    slot_values: dict[str, str],
    relations: List[RelationCandidate]
) -> float:
    """Считает приоритет комбинации на основе совместной встречаемости в RelationCandidate."""
    if not relations:
        return 0.0
    
    score = 0.0
    for relation in relations:
        for example in relation.examples:
            # Проверяем, содержит ли пример все сущности из slot_values
            example_entities = {example.subject, example.object} | set(example.other.values())
            slot_entities = set(slot_values.values())
            
            # Если все сущности из slot_values присутствуют в примере
            if slot_entities.issubset(example_entities):
                score += 1.0
    
    # Нормализуем (максимум 10 примеров = 1.0)
    return min(1.0, score / 10.0)

def generate_slot_candidates(
    pattern: CQPattern,
    entities: List[EntityCandidate],
    relations: List[RelationCandidate],
    top_n: int = 10
) -> List[dict[str, Any]]:
    """Генерирует кандидатов подстановки для паттерна с применением эвристик."""
    mapping = pattern_entity_mapping[pattern.pattern_id]
    candidates = []
    
    # Ограничиваем списки сущностей топ-N
    top_entities_by_slot = {}
    for slot_name, slot_entities in mapping.items():
        top_entities_by_slot[slot_name] = get_top_entities(slot_entities, top_n)
    
    # Генерируем комбинации (декартово произведение для всех слотов)
    from itertools import product
    
    slot_names = [slot.name for slot in pattern.slots]
    slot_entity_lists = [top_entities_by_slot.get(slot_name, []) for slot_name in slot_names]
    
    # Ограничиваем максимальное количество кандидатов
    # Для паттернов с большим количеством слотов декартово произведение может быть огромным
    # Например, для 4 слотов с top_n=10 это 10^4 = 10,000 комбинаций
    # Ограничиваем до разумного максимума (500-1000) и приоритизируем по совместной встречаемости
    max_candidates = 500  # Максимум кандидатов на паттерн
    
    # Генерируем все комбинации, но ограничиваем количество
    all_combinations = []
    for combination in product(*slot_entity_lists):
        slot_values = {slot_names[i]: combination[i].label for i in range(len(slot_names))}
        
        # Специальная обработка для Baseline: если слот Baseline пустой, выбираем случайную стратегию
        if "Baseline" in slot_values and not slot_values["Baseline"]:
            strategies = top_entities_by_slot.get("Strategy", [])
            if strategies:
                # Выбираем случайную стратегию, отличную от Strategy
                other_strategies = [s for s in strategies if s.label != slot_values.get("Strategy", "")]
                if other_strategies:
                    slot_values["Baseline"] = random.choice(other_strategies).label
        
        # Вычисляем priority score на основе совместной встречаемости
        priority_score = find_co_occurring_entities(slot_values, relations)
        
        all_combinations.append({
            "pattern_id": pattern.pattern_id,
            "slot_values": slot_values,
            "priority_score": priority_score
        })
    
    # Сортируем по priority_score и берем топ-N
    all_combinations.sort(key=lambda x: x["priority_score"], reverse=True)
    candidates = all_combinations[:max_candidates]
    
    logger.debug(f"    Сгенерировано {len(all_combinations)} комбинаций, выбрано топ-{len(candidates)} по приоритету")
    
    return candidates

logger.info(" Функции для slot filling определены")

# %%
# Поиск evidence chunks для комбинаций сущностей
def find_evidence_chunks_for_slots(
    slot_values: dict[str, str],
    weaviate_client: weaviate.Client,
    top_k: int = 5
) -> List[dict[str, Any]]:
    """Находит чанки, где упоминаются все сущности из slot_values.
    
    Использует hybrid_search с комбинированным запросом из всех сущностей.
    """
    # Создаем запрос из всех сущностей
    query_parts = list(slot_values.values())
    query = " ".join(query_parts)
    
    try:
        # Увеличиваем top_k для hybrid_search, чтобы после фильтрации осталось достаточно результатов
        search_top_k = top_k * 3  # Ищем в 3 раза больше, чем нужно
        results = hybrid_search(
            query=query,
            weaviate_client=weaviate_client,
            top_k=search_top_k,
            alpha=0.5,
            trace_id=TRACE_ID
        )
        
        # Фильтруем результаты: оставляем только те, где упоминаются все сущности
        filtered_results = []
        for result in results:
            text_lower = result.get('text', '').lower()
            # Проверяем наличие всех сущностей в тексте
            all_present = all(
                entity_value.lower() in text_lower
                for entity_value in slot_values.values()
            )
            if all_present:
                filtered_results.append(result)
        
        # Если после строгой фильтрации ничего не осталось, ослабляем требования:
        # ищем чанки, где присутствует хотя бы 2/3 сущностей
        if not filtered_results and len(slot_values) > 1:
            min_required = max(2, len(slot_values) * 2 // 3)  # Минимум 2/3 сущностей
            for result in results:
                text_lower = result.get('text', '').lower()
                present_count = sum(
                    1 for entity_value in slot_values.values()
                    if entity_value.lower() in text_lower
                )
                if present_count >= min_required:
                    filtered_results.append(result)
        
        return filtered_results[:top_k]
    except Exception as e:
        logger.warning(f"Ошибка поиска evidence chunks: {e}")
        import traceback
        logger.debug(f"  Traceback: {traceback.format_exc()}")
        return []

# %%
# Тестирование поиска evidence chunks
# %%
# Проверяем, что все необходимые переменные определены (задачи 2 и 3 выполнены)
if entity_candidates and patterns_catalog and 'pattern_entity_mapping' in globals() and pattern_entity_mapping:
    test_pattern = patterns_catalog[0]
    test_candidates_raw = generate_slot_candidates(
        test_pattern,
        entity_candidates,
        relation_candidates,
        top_n=3  # Меньше для теста
    )
    
    # Фильтруем кандидатов с дубликатами (как в generate_all_slot_candidates)
    test_candidates = []
    for raw_candidate in test_candidates_raw:
        slot_values = raw_candidate['slot_values']
        # Проверяем уникальность нормализованных значений
        normalized_values = {normalize_label(v) for v in slot_values.values() if v}
        non_empty_values = [v for v in slot_values.values() if v]
        if len(normalized_values) == len(non_empty_values):
            # Все значения уникальны
            test_candidates.append(raw_candidate)
    
    if test_candidates:
        test_candidate = test_candidates[0]
        logger.info(f"\n Тест поиска evidence chunks для кандидата:")
        logger.info(f"  Паттерн: {test_candidate['pattern_id']}")
        logger.info(f"  Slot values: {test_candidate['slot_values']}")
        
        evidence = find_evidence_chunks_for_slots(
            test_candidate['slot_values'],
            weaviate_client,
            top_k=3
        )
        
        logger.info(f"  Найдено evidence chunks: {len(evidence)}")
        if evidence:
            logger.info("  Примеры найденных чанков:")
            for i, ev in enumerate(evidence[:3], 1):
                score = ev.get('score', 0)
                score_float = float(score) if score is not None else 0.0
                text_preview = ev.get('text', '')[:100]
                logger.info(f"    {i}. {ev.get('chunk_id', 'N/A')[:60]} (score: {score_float:.3f})")
                logger.info(f"       Текст: {text_preview}...")
        else:
            logger.warning("    Не найдено ни одного evidence chunk. Возможные причины:")
            logger.warning("     - Слишком строгая фильтрация (все сущности должны быть в одном чанке)")
            logger.warning("     - Сущности не встречаются вместе в корпусе")
            logger.warning("     - Проблема с hybrid_search")
    else:
        logger.warning("  Нет валидных тестовых кандидатов без дубликатов для проверки")
else:
    missing = []
    if not entity_candidates:
        missing.append("entity_candidates (задача 2)")
    if not patterns_catalog:
        missing.append("patterns_catalog (задача 1)")
    if 'pattern_entity_mapping' not in globals() or not pattern_entity_mapping:
        missing.append("pattern_entity_mapping (задача 3)")
    logger.warning(f"  Пропущен тест поиска evidence chunks. Отсутствуют: {', '.join(missing)}")
    logger.info("  Убедитесь, что выполнены ячейки задач 1, 2 и 3")

# %%
# Перезагружаем модуль для применения изменений
import importlib
import oasis.pipelines.stage2_indexing
importlib.reload(oasis.pipelines.stage2_indexing)
# %%
# Переимпортируем функцию после перезагрузки
from oasis.pipelines.stage2_indexing import hybrid_search
from tqdm import tqdm


# %%
# Pydantic модель SlotAssignmentCandidate
class SlotAssignmentCandidate(BaseModel):
    """Кандидат подстановки сущностей в слоты паттерна."""
    pattern_id: str = Field(..., description="ID паттерна")
    slot_values: dict[str, str] = Field(..., description="Заполненные значения слотов {slot_name: entity_label}")
    candidate_evidence_chunks: List[dict[str, Any]] = Field(default_factory=list, description="Список чанков-доказательств")
    priority_score: float = Field(0.0, description="Приоритет комбинации (0.0-1.0) на основе совместной встречаемости")

# %%
# Генерация всех кандидатов для всех паттернов
def generate_all_slot_candidates(
    patterns: List[CQPattern],
    entities: List[EntityCandidate],
    relations: List[RelationCandidate],
    weaviate_client: weaviate.Client,
    top_n: int = 10
) -> List[SlotAssignmentCandidate]:
    """Генерирует все кандидаты подстановки для всех паттернов."""
    all_candidates = []
    
    for pattern in tqdm(patterns, desc="Генерация кандидатов", unit="паттерн"):
        logger.info(f"Генерация кандидатов для паттерна '{pattern.pattern_id}'...")
        
        # Генерируем кандидаты
        raw_candidates = generate_slot_candidates(
            pattern,
            entities,
            relations,
            top_n=top_n
        )
        
        initial_count = len(raw_candidates)
        # Вычисляем теоретическое максимальное количество комбинаций для понимания масштаба
        slot_counts = [len(top_entities_by_slot.get(slot.name, [])) for slot in pattern.slots]
        theoretical_max = 1
        for count in slot_counts:
            theoretical_max *= count if count > 0 else 1
        logger.info(f"  Сгенерировано {initial_count} кандидатов (теоретический максимум: {theoretical_max}, ограничено до 500 по приоритету)")
        
        # Фильтруем кандидатов с дубликатами сущностей (нормализованные значения должны быть уникальны)
        filtered_candidates = []
        skipped_count = 0
        skipped_examples = []  # Сохраняем первые 2 примера для логирования
        
        for raw_candidate in raw_candidates:
            slot_values = raw_candidate['slot_values']
            # Проверяем уникальность нормализованных значений
            normalized_values = {normalize_label(v) for v in slot_values.values() if v}
            non_empty_values = [v for v in slot_values.values() if v]
            if len(normalized_values) == len(non_empty_values):
                # Все значения уникальны
                filtered_candidates.append(raw_candidate)
            else:
                skipped_count += 1
                # Сохраняем первые 2 примера для логирования
                if len(skipped_examples) < 2:
                    # Находим дубликаты для примера
                    duplicates = {}
                    for slot_name, value in slot_values.items():
                        if value:
                            normalized = normalize_label(value)
                            if normalized in duplicates:
                                duplicates[normalized].append(slot_name)
                            else:
                                duplicates[normalized] = [slot_name]
                    duplicate_info = {v: slots for v, slots in duplicates.items() if len(slots) > 1}
                    skipped_examples.append({
                        'slot_values': slot_values,
                        'duplicates': duplicate_info
                    })
        
        # Логируем статистику пропущенных кандидатов
        remaining_count = len(filtered_candidates)
        logger.info(f"  После фильтрации дубликатов: {remaining_count} кандидатов (пропущено {skipped_count})")
        if skipped_count > 0 and skipped_examples:
            for i, example in enumerate(skipped_examples, 1):
                dup_info = ', '.join([f"{v} в слотах {', '.join(slots)}" for v, slots in example['duplicates'].items()])
                logger.info(f"    Пример {i}: дубликаты - {dup_info}")
        
        raw_candidates = filtered_candidates
        
        # Для каждого кандидата ищем evidence chunks с прогресс-баром
        candidates_with_evidence = 0
        for raw_candidate in tqdm(raw_candidates, desc=f"  Поиск evidence для {pattern.pattern_id}", unit="кандидат", leave=False):
            evidence = find_evidence_chunks_for_slots(
                raw_candidate['slot_values'],
                weaviate_client,
                top_k=5
            )
            
            # Фильтруем кандидатов без evidence chunks
            if evidence:
                candidate = SlotAssignmentCandidate(
                    pattern_id=raw_candidate['pattern_id'],
                    slot_values=raw_candidate['slot_values'],
                    candidate_evidence_chunks=evidence,
                    priority_score=raw_candidate['priority_score']
                )
                all_candidates.append(candidate)
                candidates_with_evidence += 1
        
        pattern_candidates_count = len([c for c in all_candidates if c.pattern_id == pattern.pattern_id])
        logger.info(f"  Итого: {pattern_candidates_count} кандидатов с evidence (из {remaining_count} после фильтрации)")
    
    # Сортируем по priority_score
    all_candidates.sort(key=lambda x: x.priority_score, reverse=True)
    
    return all_candidates

# %%
# Генерация всех кандидатов
# %%
# Проверяем, что patterns_catalog содержит валидные паттерны (после задачи 3)
if 'pattern_entity_mapping' in globals() and pattern_entity_mapping:
    # Убеждаемся, что используем валидные паттерны (после валидации в задаче 3)
    if patterns_catalog and entity_candidates:
        logger.info(f"\n Генерация кандидатов подстановки для {len(patterns_catalog)} паттернов...")
        logger.info(f"  Паттерны: {[p.pattern_id for p in patterns_catalog]}")
        
        slot_candidates = generate_all_slot_candidates(
            patterns_catalog,
            entity_candidates,
            relation_candidates,
            weaviate_client,
            top_n=TOP_N_ENTITIES
        )
        
        logger.success(f" Создано {len(slot_candidates)} кандидатов с evidence chunks")
        
        # Статистика по паттернам
        candidates_by_pattern = Counter(c.pattern_id for c in slot_candidates)
        logger.info(f"  Распределение по паттернам:")
        for pattern_id, count in candidates_by_pattern.items():
            logger.info(f"    {pattern_id}: {count} кандидатов")
    else:
        slot_candidates = []
        missing = []
        if not patterns_catalog:
            missing.append("patterns_catalog")
        if not entity_candidates:
            missing.append("entity_candidates")
        logger.warning(f"  Невозможно сгенерировать кандидаты. Отсутствуют: {', '.join(missing)}")
        logger.info("  Убедитесь, что выполнены ячейки задач 1, 2 и 3")
else:
    slot_candidates = []
    logger.warning("  Невозможно сгенерировать кандидаты: не выполнена задача 3 (сопоставление сущностей с паттернами)")
    logger.info("  Выполните ячейку задачи 3 для создания pattern_entity_mapping")

# %%
# Визуализация кандидатов
if slot_candidates:
    # Таблица кандидатов
    df_candidates = pd.DataFrame([
        {
            "pattern_id": c.pattern_id,
            "slot_values": str(c.slot_values),
            "evidence_count": len(c.candidate_evidence_chunks),
            "priority_score": c.priority_score
        }
        for c in slot_candidates[:20]  # Показываем топ-20
    ])
    
    logger.info("\n Топ-20 кандидатов подстановки:")
    display(df_candidates)
    
    # Статистика по паттернам
    candidates_by_pattern = Counter(c.pattern_id for c in slot_candidates)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(candidates_by_pattern.keys(), candidates_by_pattern.values())
    ax.set_xlabel("Pattern ID")
    ax.set_ylabel("Количество кандидатов")
    ax.set_title("Распределение кандидатов по паттернам")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Примеры лучших кандидатов
    logger.info("\n Примеры лучших кандидатов (по priority_score):")
    for i, candidate in enumerate(slot_candidates[:3], 1):
        logger.info(f"\n  {i}. Pattern: {candidate.pattern_id}")
        logger.info(f"     Slots: {candidate.slot_values}")
        logger.info(f"     Priority: {candidate.priority_score:.3f}")
        logger.info(f"     Evidence chunks: {len(candidate.candidate_evidence_chunks)}")
else:
    logger.warning("  Нет кандидатов для визуализации")

# %%
# Промпт для генерации CQ
CQ_GENERATION_SYSTEM_PROMPT = """You are an expert in Deep Active Learning and competency questions (ontology requirements).
You are given a question template, specific slot values (entities), and supporting text excerpts.

Your task:
1. Formulate a competency question EXACTLY following the template
2. Insert the given entities into the template placeholders
3. The question MUST be answerable using the provided text snippets
4. Do NOT introduce any entities not in slot_values
5. Do NOT change the structure of the template (verb tenses, word order, etc.)
6. Return a JSON object with the following fields:
   - text: the formulated question
   - role: the ontological role (same as pattern.role)
   - pattern_id: the pattern ID used
   - slots: the slot values used (same as provided slot_values)
   - expected_answer_type: brief description of expected answer type
   - evidence_chunks: list of chunk_id that were actually used

Return ONLY valid JSON, no additional text."""

def create_cq_generation_user_prompt(
    topic: dict[str, Any],
    pattern: CQPattern,
    slot_values: dict[str, str],
    evidence_chunks: List[dict[str, Any]]
) -> dict[str, Any]:
    """Создает user prompt для генерации CQ."""
    return {
        "topic_id": topic.get('canonical_topic_id', ''),
        "topic_name": topic.get('name', ''),
        "pattern": {
            "pattern_id": pattern.pattern_id,
            "role": pattern.role,
            "template": pattern.template,
            "slots": [{"name": s.name, "entity_list": s.entity_list} for s in pattern.slots]
        },
        "slot_values": slot_values,
        "candidate_evidence_chunks": [
            {
                "chunk_id": ch.get('chunk_id', ''),
                "text": ch.get('text', '')[:500],  # Ограничиваем длину
                "score": ch.get('score', 0)
            }
            for ch in evidence_chunks
        ]
    }

logger.info(" Промпты для генерации CQ определены")

# %%
# Pydantic модель CompetencyQuestion
class CompetencyQuestion(BaseModel):
    """Компетентностный вопрос."""
    cq_id: str = Field(..., description="Уникальный идентификатор вопроса (TOPIC_XXX_CQ_XXX)")
    text: str = Field(..., description="Текст вопроса на английском языке")
    role: str = Field(..., description="Онтологическая роль: class/relation/process/constraint/comparison")
    pattern_id: str = Field(..., description="ID шаблона, по которому построен вопрос")
    slots: dict[str, str] = Field(..., description="Заполненные слоты {slot_name: entity_label}")
    expected_answer_type: str = Field(..., description="Краткое описание типа ожидаемого ответа")
    evidence_chunks: List[str] = Field(..., description="Список chunk_id фрагментов, на основе которых сформулирован вопрос")
    answer_hint: Optional[str] = Field(None, description="Черновой краткий ответ (1-3 предложения)")

logger.info(" Pydantic модель CompetencyQuestion определена")

# %%
# Функция генерации CQ из кандидата
def generate_cq_from_candidate(
    candidate: SlotAssignmentCandidate,
    topic: dict[str, Any],
    pattern: CQPattern,
    trace_id: str | None = None,
    max_retries: int = 3
) -> Optional[CompetencyQuestion]:
    """Генерирует CQ из кандидата подстановки через LLM."""
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    
    user_payload = create_cq_generation_user_prompt(
        topic,
        pattern,
        candidate.slot_values,
        candidate.candidate_evidence_chunks
    )
    
    messages = [
        {"role": "system", "content": CQ_GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"{trace_prefix}LLM вызов для генерации CQ (попытка {attempt + 1}/{max_retries})")
            response = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Пустой ответ от LLM")
            
            parsed = json.loads(content)
            
            # Валидация обязательных полей
            required_fields = ["text", "role", "pattern_id", "slots", "expected_answer_type", "evidence_chunks"]
            if not all(field in parsed for field in required_fields):
                raise ValueError(f"Ответ LLM не содержит все требуемые поля: {required_fields}")
            
            # Создаем CQ (cq_id будет присвоен позже)
            cq = CompetencyQuestion(
                cq_id="",  # Будет присвоен в finalize_cqs_for_topic
                text=parsed["text"],
                role=parsed["role"],
                pattern_id=parsed["pattern_id"],
                slots=parsed["slots"],
                expected_answer_type=parsed["expected_answer_type"],
                evidence_chunks=parsed["evidence_chunks"],
                answer_hint=None  # Будет сгенерирован позже
            )
            
            logger.debug(f"{trace_prefix} CQ сгенерирован: {cq.text[:80]}...")
            return cq
            
        except json.JSONDecodeError as json_exc:
            # Логируем полный ответ LLM для отладки
            content_preview = content[:500] if len(content) > 500 else content
            logger.error(f"{trace_prefix}Ошибка парсинга JSON (попытка {attempt + 1}): {json_exc}")
            logger.error(f"{trace_prefix}Начало ответа LLM (первые 500 символов):\n{content_preview}")
            if len(content) > 500:
                logger.error(f"{trace_prefix}... (всего {len(content)} символов)")
            
            # Сохраняем полный ответ в файл для детального анализа
            error_log_path = OUTPUT_DIR / f"llm_error_response_cq_{trace_id}_{int(time.time())}.txt"
            try:
                with open(error_log_path, 'w', encoding='utf-8') as f:
                    f.write(f"Error: {json_exc}\n")
                    f.write(f"Attempt: {attempt + 1}/{max_retries}\n")
                    f.write(f"Pattern ID: {pattern.pattern_id}\n")
                    f.write(f"Full LLM response:\n{content}\n")
                logger.error(f"{trace_prefix}Полный ответ сохранен в: {error_log_path}")
            except Exception as save_exc:
                logger.warning(f"{trace_prefix}Не удалось сохранить ответ: {save_exc}")
            
            if attempt < max_retries - 1:
                delay = 1.0 * (2 ** attempt)
                logger.warning(f"{trace_prefix}Повтор через {delay:.1f}с")
                time.sleep(delay)
            else:
                logger.error(f"{trace_prefix}Не удалось распарсить JSON после {max_retries} попыток")
                return None
        except Exception as exc:
            if attempt < max_retries - 1:
                delay = 1.0 * (2 ** attempt)
                logger.warning(f"{trace_prefix}Ошибка LLM (попытка {attempt + 1}): {exc}. Повтор через {delay:.1f}с")
                time.sleep(delay)
            else:
                logger.error(f"{trace_prefix}LLM не ответил после {max_retries} попыток: {exc}")
                return None
    
    return None

# %%
# Валидация сгенерированного CQ
def validate_cq(
    cq: CompetencyQuestion,
    pattern: CQPattern,
    slot_values: dict[str, str]
) -> tuple[bool, List[str]]:
    """Проверяет CQ и возвращает признак валидности вместе со списком ошибок."""
    errors = []
    
    # Проверка на дубликаты сущностей в слотах (нормализованные значения должны быть уникальны)
    slot_values_normalized = {k: normalize_label(v) for k, v in slot_values.items() if v}
    unique_values = set(slot_values_normalized.values())
    if len(unique_values) < len(slot_values_normalized):
        # Найдены дубликаты
        duplicates = {}
        for slot_name, normalized_value in slot_values_normalized.items():
            if normalized_value in duplicates:
                duplicates[normalized_value].append(slot_name)
            else:
                duplicates[normalized_value] = [slot_name]
        
        duplicate_slots = {v: slots for v, slots in duplicates.items() if len(slots) > 1}
        for value, slots in duplicate_slots.items():
            errors.append(
                f"Дубликат сущности '{value}' в слотах {', '.join(slots)}. "
                f"Для паттерна '{pattern.pattern_id}' сущности в разных слотах должны быть различными."
            )
    
    # Проверка соответствия формату шаблона
    # Простая проверка: все сущности из slot_values должны присутствовать в тексте
    text_lower = cq.text.lower()
    for slot_name, entity_value in slot_values.items():
        if entity_value.lower() not in text_lower:
            errors.append(f"Сущность '{entity_value}' из слота '{slot_name}' отсутствует в тексте вопроса")
    
    # Проверка наличия всех сущностей из slot_values
    if cq.slots != slot_values:
        errors.append(f"Слоты в CQ не совпадают с предоставленными: {cq.slots} vs {slot_values}")
    
    # Проверка указания evidence_chunks
    if not cq.evidence_chunks:
        errors.append("Не указаны evidence_chunks")
    
    # Проверка соответствия role
    if cq.role != pattern.role:
        errors.append(f"Role в CQ ({cq.role}) не совпадает с role паттерна ({pattern.role})")
    
    # Проверка соответствия pattern_id
    if cq.pattern_id != pattern.pattern_id:
        errors.append(f"Pattern ID в CQ ({cq.pattern_id}) не совпадает с pattern_id паттерна ({pattern.pattern_id})")
    
    # Проверка на наличие лишних сущностей (эвристика: проверяем, что текст не слишком длинный)
    # Это упрощенная проверка, можно улучшить через NER
    
    return len(errors) == 0, errors

# %%
# Тестирование генерации одного CQ
if slot_candidates and selected_topic:
    test_candidate = slot_candidates[0]
    test_pattern = next(p for p in patterns_catalog if p.pattern_id == test_candidate.pattern_id)
    
    logger.info(f"\n Тест генерации CQ для кандидата:")
    logger.info(f"  Pattern: {test_candidate.pattern_id}")
    logger.info(f"  Slot values: {test_candidate.slot_values}")
    
    test_cq = generate_cq_from_candidate(
        test_candidate,
        selected_topic,
        test_pattern,
        trace_id=TRACE_ID
    )
    
    if test_cq:
        is_valid, errors = validate_cq(test_cq, test_pattern, test_candidate.slot_values)
        logger.success(f" CQ сгенерирован")
        logger.info(f"  Text: {test_cq.text}")
        logger.info(f"  Role: {test_cq.role}")
        logger.info(f"  Evidence chunks: {len(test_cq.evidence_chunks)}")
        if is_valid:
            logger.success("   Валидация пройдена")
        else:
            logger.warning(f"    Ошибки валидации: {errors}")
    else:
        logger.error("   Не удалось сгенерировать CQ")

# %%
# Пакетная генерация CQs для всех кандидатов
def generate_cqs_for_topic(
    topic: dict[str, Any],
    patterns: List[CQPattern],
    slot_candidates: List[SlotAssignmentCandidate],
    trace_id: str | None = None,
    max_cqs_per_pattern: int = 5
) -> tuple[List[CompetencyQuestion], dict[str, Any]]:
    """Генерирует CQs для темы и возвращает список вопросов и статистику."""
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    all_cqs = []
    stats = {
        "total_candidates": len(slot_candidates),
        "successful": 0,
        "failed": 0,
        "validation_errors": 0,
        "by_pattern": defaultdict(int)
    }
    
    # Группируем кандидаты по паттернам
    candidates_by_pattern = defaultdict(list)
    for candidate in slot_candidates:
        candidates_by_pattern[candidate.pattern_id].append(candidate)
    
    # Генерируем CQs для каждого паттерна
    for pattern in patterns:
        pattern_candidates = candidates_by_pattern[pattern.pattern_id]
        # Ограничиваем количество кандидатов на паттерн
        pattern_candidates = pattern_candidates[:max_cqs_per_pattern]
        
        logger.info(f"{trace_prefix}Генерация CQs для паттерна '{pattern.pattern_id}' ({len(pattern_candidates)} кандидатов)...")
        
        for candidate in pattern_candidates:
            cq = generate_cq_from_candidate(
                candidate,
                topic,
                pattern,
                trace_id=trace_id
            )
            
            if cq:
                # Валидация
                is_valid, errors = validate_cq(cq, pattern, candidate.slot_values)
                
                if is_valid:
                    all_cqs.append(cq)
                    stats["successful"] += 1
                    stats["by_pattern"][pattern.pattern_id] += 1
                else:
                    stats["validation_errors"] += 1
                    logger.warning(f"{trace_prefix}    CQ не прошел валидацию: {errors}")
            else:
                stats["failed"] += 1
    
    logger.success(f"{trace_prefix} Генерация завершена: {stats['successful']} успешных, {stats['failed']} неудачных")
    return all_cqs, stats

# %%
# Генерация CQs для выбранной темы
if slot_candidates and selected_topic and patterns_catalog:
    logger.info(f"\n Пакетная генерация CQs для темы '{selected_topic.get('name', 'N/A')}'...")
    generated_cqs, generation_stats = generate_cqs_for_topic(
        selected_topic,
        patterns_catalog,
        slot_candidates,
        trace_id=TRACE_ID,
        max_cqs_per_pattern=5
    )
    
    logger.info(f"\n Статистика генерации:")
    logger.info(f"  Всего кандидатов: {generation_stats['total_candidates']}")
    logger.info(f"  Успешных CQs: {generation_stats['successful']}")
    logger.info(f"  Неудачных: {generation_stats['failed']}")
    logger.info(f"  Ошибок валидации: {generation_stats['validation_errors']}")
    logger.info(f"  По паттернам:")
    for pattern_id, count in generation_stats['by_pattern'].items():
        logger.info(f"    {pattern_id}: {count} CQs")
else:
    generated_cqs = []
    generation_stats = {}
    logger.warning("  Невозможно сгенерировать CQs: нет данных")

# %%
# Визуализация результатов генерации
if generated_cqs:
    # Таблица сгенерированных CQs
    df_cqs = pd.DataFrame([
        {
            "cq_id": cq.cq_id if cq.cq_id else f"TEMP_{i}",
            "text": cq.text,
            "role": cq.role,
            "pattern_id": cq.pattern_id,
            "expected_answer_type": cq.expected_answer_type,
            "evidence_count": len(cq.evidence_chunks)
        }
        for i, cq in enumerate(generated_cqs, 1)
    ])
    
    logger.info("\n Сгенерированные CQs:")
    display(df_cqs)
    
    # Распределение по ролям
    role_counts = Counter(cq.role for cq in generated_cqs)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(role_counts.keys(), role_counts.values())
    ax.set_xlabel("Роль")
    ax.set_ylabel("Количество CQs")
    ax.set_title("Распределение CQs по ролям")
    plt.tight_layout()
    plt.show()
    
    # Примеры вопросов с их evidence chunks
    logger.info("\n Примеры CQs с evidence chunks:")
    for i, cq in enumerate(generated_cqs[:3], 1):
        logger.info(f"\n  {i}. {cq.text}")
        logger.info(f"     Role: {cq.role}, Pattern: {cq.pattern_id}")
        logger.info(f"     Evidence chunks ({len(cq.evidence_chunks)}): {cq.evidence_chunks[:3]}")
else:
    logger.warning("  Нет сгенерированных CQs для визуализации")

# %%
# Промпт для генерации answer_hint
ANSWER_HINT_SYSTEM_PROMPT = """You are an expert in Deep Active Learning.
You are given a competency question and supporting text excerpts (evidence chunks).

Your task is to generate a brief answer hint (1-3 sentences) that:
1. Answers the question based STRICTLY on the provided evidence chunks
2. Does not introduce information not present in the evidence
3. Is concise and informative
4. Provides a preliminary answer that can be used for validation

Return a JSON object with a single field:
{
  "answer_hint": "1-3 sentences answering the question based on evidence"
}

Return ONLY valid JSON, no additional text."""

def generate_answer_hint(
    cq: CompetencyQuestion,
    evidence_chunks: List[dict[str, Any]],
    trace_id: str | None = None,
    max_retries: int = 3
) -> Optional[str]:
    """Генерирует краткий answer_hint для CQ на основе evidence chunks."""
    trace_prefix = f"[trace_id={trace_id}] " if trace_id else ""
    
    if not evidence_chunks:
        logger.warning(f"{trace_prefix}Нет evidence chunks для генерации answer_hint")
        return None
    
    # Подготавливаем тексты evidence chunks
    evidence_texts = [
        {
            "chunk_id": ch.get('chunk_id', ''),
            "text": ch.get('text', '')[:500]  # Ограничиваем длину
        }
        for ch in evidence_chunks
    ]
    
    user_payload = {
        "question": cq.text,
        "evidence_chunks": evidence_texts
    }
    
    messages = [
        {"role": "system", "content": ANSWER_HINT_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"{trace_prefix}LLM вызов для генерации answer_hint (попытка {attempt + 1}/{max_retries})")
            response = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=500,  # Меньше токенов для краткого ответа
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Пустой ответ от LLM")
            
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as json_exc:
                # Логируем полный ответ LLM для отладки
                content_preview = content[:500] if len(content) > 500 else content
                logger.error(f"{trace_prefix}Ошибка парсинга JSON (попытка {attempt + 1}): {json_exc}")
                logger.error(f"{trace_prefix}Начало ответа LLM (первые 500 символов):\n{content_preview}")
                if len(content) > 500:
                    logger.error(f"{trace_prefix}... (всего {len(content)} символов)")
                
                # Сохраняем полный ответ в файл для детального анализа
                error_log_path = OUTPUT_DIR / f"llm_error_response_answer_hint_{trace_id}_{int(time.time())}.txt"
                try:
                    with open(error_log_path, 'w', encoding='utf-8') as f:
                        f.write(f"Error: {json_exc}\n")
                        f.write(f"Attempt: {attempt + 1}/{max_retries}\n")
                        f.write(f"CQ ID: {cq.cq_id if cq.cq_id else 'N/A'}\n")
                        f.write(f"Full LLM response:\n{content}\n")
                    logger.error(f"{trace_prefix}Полный ответ сохранен в: {error_log_path}")
                except Exception as save_exc:
                    logger.warning(f"{trace_prefix}Не удалось сохранить ответ: {save_exc}")
                
                if attempt < max_retries - 1:
                    delay = 1.0 * (2 ** attempt)
                    logger.warning(f"{trace_prefix}Повтор через {delay:.1f}с")
                    time.sleep(delay)
                    continue
                else:
                    logger.warning(f"{trace_prefix}Не удалось распарсить JSON после {max_retries} попыток")
                    return None
            
            answer_hint = parsed.get("answer_hint", "")
            
            # Валидация длины (1-3 предложения)
            sentences = answer_hint.split('.')
            num_sentences = len([s for s in sentences if s.strip()])
            if num_sentences > 3:
                logger.warning(f"{trace_prefix}answer_hint слишком длинный ({num_sentences} предложений), обрезаем")
                answer_hint = '. '.join(sentences[:3]) + '.'
            
            logger.debug(f"{trace_prefix} answer_hint сгенерирован: {answer_hint[:80]}...")
            return answer_hint
            
        except Exception as exc:
            if attempt < max_retries - 1:
                delay = 1.0 * (2 ** attempt)
                logger.warning(f"{trace_prefix}Ошибка LLM (попытка {attempt + 1}): {exc}. Повтор через {delay:.1f}с")
                time.sleep(delay)
            else:
                logger.warning(f"{trace_prefix}Не удалось сгенерировать answer_hint: {exc}")
                return None
    
    return None

logger.info(" Функция генерации answer_hint определена")

# %%
# Генерация answer_hint для всех CQs
if generated_cqs:
    logger.info(f"\n Генерация answer_hint для {len(generated_cqs)} CQs...")
    
    # Получаем evidence chunks из Weaviate для каждого CQ
    for i, cq in enumerate(generated_cqs, 1):
        if not cq.evidence_chunks:
            continue
        
        # Получаем тексты evidence chunks из Weaviate
        evidence_chunks_data = []
        for chunk_id in cq.evidence_chunks:
            try:
                result = weaviate_client.query.get(
                    "Chunk",
                    ["chunk_id", "text"]
                ).with_where({
                    "path": ["chunk_id"],
                    "operator": "Equal",
                    "valueText": chunk_id
                }).with_limit(1).do()
                
                chunks = result.get("data", {}).get("Get", {}).get("Chunk", [])
                if chunks:
                    evidence_chunks_data.append({
                        "chunk_id": chunk_id,
                        "text": chunks[0].get("text", "")
                    })
            except Exception as e:
                logger.warning(f"Ошибка получения chunk {chunk_id}: {e}")
        
        if evidence_chunks_data:
            answer_hint = generate_answer_hint(
                cq,
                evidence_chunks_data,
                trace_id=TRACE_ID
            )
            if answer_hint:
                cq.answer_hint = answer_hint
                logger.debug(f"  [{i}/{len(generated_cqs)}] answer_hint сгенерирован")
    
    logger.success(f" answer_hint сгенерирован для {sum(1 for cq in generated_cqs if cq.answer_hint)} CQs")
    
    # Примеры answer_hint
    logger.info("\n Примеры answer_hint:")
    for i, cq in enumerate([cq for cq in generated_cqs if cq.answer_hint][:3], 1):
        logger.info(f"\n  {i}. Question: {cq.text[:80]}...")
        logger.info(f"     Answer hint: {cq.answer_hint}")
else:
    logger.warning("  Нет CQs для генерации answer_hint")

# %%
# Фильтрация некачественных CQs
def filter_low_quality_cqs(cqs: List[CompetencyQuestion]) -> List[CompetencyQuestion]:
    """Удаляет некачественные CQs."""
    filtered = []
    generic_phrases = [
        "what is deep active learning",
        "what is active learning",
        "what are",
        "define",
        "explain what"
    ]
    
    for cq in cqs:
        # Проверка на слишком общие вопросы
        text_lower = cq.text.lower()
        is_generic = any(phrase in text_lower for phrase in generic_phrases)
        if is_generic:
            logger.debug(f"Пропущен общий вопрос: {cq.text[:60]}...")
            continue
        
        # Проверка на пустые слоты
        if not cq.slots or any(not v or not v.strip() for v in cq.slots.values()):
            logger.debug(f"Пропущен вопрос с пустыми слотами: {cq.text[:60]}...")
            continue
        
        # Проверка минимальной длины (минимум 20 символов)
        if len(cq.text) < 20:
            logger.debug(f"Пропущен слишком короткий вопрос: {cq.text[:60]}...")
            continue
        
        # Проверка наличия ключевых слов (должен содержать хотя бы одно существительное/глагол)
        words = text_lower.split()
        if len(words) < 5:  # Минимум 5 слов
            logger.debug(f"Пропущен вопрос с недостаточным количеством слов: {cq.text[:60]}...")
            continue
        
        filtered.append(cq)
    
    return filtered

# %%
# Применяем фильтрацию
if generated_cqs:
    logger.info(f"\n Фильтрация некачественных CQs...")
    filtered_cqs = filter_low_quality_cqs(generated_cqs)
    logger.info(f"  До фильтрации: {len(generated_cqs)}")
    logger.info(f"  После фильтрации: {len(filtered_cqs)}")
    logger.info(f"  Удалено: {len(generated_cqs) - len(filtered_cqs)}")
else:
    filtered_cqs = []
    logger.warning("  Нет CQs для фильтрации")

# %%
# Дедупликация по тексту через эмбеддинги
from sentence_transformers import SentenceTransformer
import torch

def normalize_text_for_dedup(text: str) -> str:
    """Нормализует текст для дедупликации."""
    # Приводим к lowercase, удаляем пунктуацию, нормализуем пробелы
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Удаляем пунктуацию
    text = re.sub(r'\s+', ' ', text)  # Нормализуем пробелы
    return text.strip()

def deduplicate_cqs_by_text(
    cqs: List[CompetencyQuestion],
    threshold: float = 0.95,
    model_path: str = "models/bge-m3"
) -> List[CompetencyQuestion]:
    """Дедуплицирует CQs по тексту через сравнение эмбеддингов."""
    if not cqs:
        return []
    
    logger.info(f"Дедупликация {len(cqs)} CQs (порог similarity: {threshold})...")
    
    # Загружаем модель эмбеддингов
    try:
        # Проверяем доступность GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Загрузка модели эмбеддингов на {device}...")
        
        # Разрешаем путь к модели
        if Path(model_path).exists():
            model = SentenceTransformer(model_path, device=device)
        elif Path(f"../{model_path}").exists():
            model = SentenceTransformer(f"../{model_path}", device=device)
        else:
            # Пробуем загрузить как HuggingFace ID
            model = SentenceTransformer(model_path, device=device)
        
        logger.info(f" Модель загружена")
    except Exception as e:
        logger.warning(f"  Ошибка загрузки модели эмбеддингов: {e}. Используем простую дедупликацию по нормализованному тексту.")
        # Fallback: простая дедупликация по нормализованному тексту
        seen_texts = set()
        unique_cqs = []
        for cq in cqs:
            normalized = normalize_text_for_dedup(cq.text)
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                unique_cqs.append(cq)
        logger.info(f"  Простая дедупликация: {len(unique_cqs)} уникальных из {len(cqs)}")
        return unique_cqs
    
    # Генерируем эмбеддинги для всех вопросов
    texts = [cq.text for cq in cqs]
    try:
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        logger.info(f" Эмбеддинги сгенерированы: {embeddings.shape}")
    except Exception as e:
        logger.error(f" Ошибка генерации эмбеддингов: {e}")
        return cqs  # Возвращаем без дедупликации
    
    # Вычисляем косинусное сходство и удаляем дубликаты
    unique_cqs = []
    seen_indices = set()
    
    for i, cq in enumerate(cqs):
        if i in seen_indices:
            continue
        
        unique_cqs.append(cq)
        
        # Находим похожие вопросы
        for j in range(i + 1, len(cqs)):
            if j in seen_indices:
                continue
            
            # Косинусное сходство
            similarity = np.dot(embeddings[i], embeddings[j])
            
            if similarity >= threshold:
                seen_indices.add(j)
                logger.debug(f"  Дубликат найден: {i} и {j} (similarity: {similarity:.3f})")
    
    logger.info(f" Дедупликация завершена: {len(unique_cqs)} уникальных из {len(cqs)}")
    return unique_cqs

# %%
# Применяем дедупликацию
if filtered_cqs:
    logger.info(f"\n Дедупликация CQs...")
    final_cqs = deduplicate_cqs_by_text(
        filtered_cqs,
        threshold=DEDUP_SIMILARITY_THRESHOLD
    )
    logger.info(f"  До дедупликации: {len(filtered_cqs)}")
    logger.info(f"  После дедупликации: {len(final_cqs)}")
    logger.info(f"  Удалено дубликатов: {len(filtered_cqs) - len(final_cqs)}")
else:
    final_cqs = []
    logger.warning("  Нет CQs для дедупликации")

# %%
# Финальная обработка: присвоение cq_id и группировка по темам
def finalize_cqs_for_topic(
    cqs: List[CompetencyQuestion],
    topic_id: str
) -> dict[str, Any]:
    """Присваивает cq_id и группирует CQs по теме."""
    # Присваиваем cq_id
    for i, cq in enumerate(cqs, 1):
        cq.cq_id = f"{topic_id}_CQ_{i:03d}"
    
    return {
        "topic_id": topic_id,
        "topic_name": selected_topic.get('name', '') if selected_topic else '',
        "cqs": [cq.model_dump() for cq in cqs]
    }

# %%
# Финальная обработка
if final_cqs and selected_topic:
    topic_id = selected_topic.get('canonical_topic_id', f'TOPIC_{TOPIC_NUMBER:03d}')
    draft_cqs = finalize_cqs_for_topic(final_cqs, topic_id)
    
    logger.success(f" Финальная обработка завершена: {len(final_cqs)} CQs для темы {topic_id}")
else:
    draft_cqs = {
        "topic_id": "",
        "topic_name": "",
        "cqs": []
    }
    logger.warning("  Нет CQs для финальной обработки")

# %%
# Сохранение Draft CQs
def save_draft_cqs(path: Path, draft_cqs: dict[str, Any], format: str = "jsonl") -> None:
    """Сохраняет Draft CQs в JSONL или JSON формате."""
    if format == "jsonl":
        # JSONL: по одному CQ на строку
        with open(path, 'w', encoding='utf-8') as f:
            for cq in draft_cqs.get("cqs", []):
                cq_with_topic = {
                    "topic_id": draft_cqs["topic_id"],
                    "topic_name": draft_cqs["topic_name"],
                    **cq
                }
                f.write(json.dumps(cq_with_topic, ensure_ascii=False) + '\n')
        logger.info(f" Сохранено {len(draft_cqs.get('cqs', []))} CQs в JSONL: {path}")
    else:
        # JSON: полная структура
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(draft_cqs, f, ensure_ascii=False, indent=2)
        logger.info(f" Сохранено {len(draft_cqs.get('cqs', []))} CQs в JSON: {path}")

# %%
# Сохранение Draft CQs
if draft_cqs and draft_cqs.get("cqs"):
    draft_cqs_path = OUTPUT_DIR / "draft_cqs.jsonl"
    save_draft_cqs(draft_cqs_path, draft_cqs, format="jsonl")
    
    # Также сохраняем в JSON для удобства просмотра
    draft_cqs_json_path = OUTPUT_DIR / "draft_cqs.json"
    save_draft_cqs(draft_cqs_json_path, draft_cqs, format="json")
    
    # Сохраняем метаданные
    metadata = {
        "trace_id": TRACE_ID,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "topic_id": draft_cqs["topic_id"],
        "topic_name": draft_cqs["topic_name"],
        "total_cqs": len(draft_cqs["cqs"]),
        "statistics": {
            "generation_stats": generation_stats if 'generation_stats' in globals() else {},
            "filtered_count": len(filtered_cqs) if 'filtered_cqs' in globals() else 0,
            "final_count": len(final_cqs) if 'final_cqs' in globals() else 0
        }
    }
    metadata_path = OUTPUT_DIR / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info(f" Метаданные сохранены: {metadata_path}")
else:
    logger.warning("  Нет CQs для сохранения")

# %%
# Вычисление метрик
def compute_cq_metrics(
    final_cqs: List[CompetencyQuestion],
    generation_stats: dict[str, Any],
    slot_candidates: List[SlotAssignmentCandidate]
) -> dict[str, Any]:
    """Вычисляет метрики качества генерации CQs."""
    metrics = {
        "total_cqs": len(final_cqs),
        "cqs_per_topic": len(final_cqs),
        "role_distribution": Counter(cq.role for cq in final_cqs),
        "pattern_distribution": Counter(cq.pattern_id for cq in final_cqs),
        "avg_question_length": 0.0,
        "min_question_length": 0,
        "max_question_length": 0,
        "valid_cqs_ratio": 0.0,
        "candidate_to_cq_ratio": 0.0,
        "coverage": {
            "has_cqs": len(final_cqs) > 0,
            "has_multiple_roles": len(set(cq.role for cq in final_cqs)) > 1
        }
    }
    
    if final_cqs:
        # Длина вопросов (в словах)
        question_lengths = [len(cq.text.split()) for cq in final_cqs]
        metrics["avg_question_length"] = sum(question_lengths) / len(question_lengths)
        metrics["min_question_length"] = min(question_lengths)
        metrics["max_question_length"] = max(question_lengths)
        
        # Доля валидных CQs (прошедших валидацию)
        if generation_stats:
            total_attempted = generation_stats.get("successful", 0) + generation_stats.get("failed", 0) + generation_stats.get("validation_errors", 0)
            if total_attempted > 0:
                metrics["valid_cqs_ratio"] = generation_stats.get("successful", 0) / total_attempted
        
        # Соотношение кандидатов  итоговых CQs
        if slot_candidates:
            metrics["candidate_to_cq_ratio"] = len(final_cqs) / len(slot_candidates)
    
    return metrics

# %%
# Вычисляем метрики
if final_cqs:
    metrics = compute_cq_metrics(
        final_cqs,
        generation_stats if 'generation_stats' in globals() else {},
        slot_candidates if 'slot_candidates' in globals() else []
    )
    
    logger.info("\n Метрики качества генерации CQs:")
    logger.info("=" * 80)
    logger.info(f"  Всего CQs: {metrics['total_cqs']}")
    logger.info(f"  CQs на тему: {metrics['cqs_per_topic']}")
    logger.info(f"  Распределение по ролям: {dict(metrics['role_distribution'])}")
    logger.info(f"  Распределение по паттернам: {dict(metrics['pattern_distribution'])}")
    logger.info(f"  Средняя длина вопроса: {metrics['avg_question_length']:.1f} слов")
    logger.info(f"  Длина вопросов: {metrics['min_question_length']}-{metrics['max_question_length']} слов")
    logger.info(f"  Доля валидных CQs: {metrics['valid_cqs_ratio']:.2%}")
    logger.info(f"  Соотношение кандидатов  CQs: {metrics['candidate_to_cq_ratio']:.3f}")
    logger.info(f"  Coverage: есть CQs={metrics['coverage']['has_cqs']}, несколько ролей={metrics['coverage']['has_multiple_roles']}")
else:
    metrics = {}
    logger.warning("  Нет данных для вычисления метрик")

# %%
# Визуализация метрик
if metrics and final_cqs:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Распределение по ролям
    ax1 = axes[0, 0]
    role_counts = metrics['role_distribution']
    if role_counts:
        ax1.bar(role_counts.keys(), role_counts.values())
        ax1.set_xlabel("Роль")
        ax1.set_ylabel("Количество CQs")
        ax1.set_title("Распределение CQs по ролям")
        ax1.tick_params(axis='x', rotation=45)
    
    # 2. Распределение по паттернам
    ax2 = axes[0, 1]
    pattern_counts = metrics['pattern_distribution']
    if pattern_counts:
        ax2.bar(pattern_counts.keys(), pattern_counts.values())
        ax2.set_xlabel("Pattern ID")
        ax2.set_ylabel("Количество CQs")
        ax2.set_title("Распределение CQs по паттернам")
        ax2.tick_params(axis='x', rotation=45)
    
    # 3. Гистограмма длины вопросов
    ax3 = axes[1, 0]
    question_lengths = [len(cq.text.split()) for cq in final_cqs]
    ax3.hist(question_lengths, bins=20, edgecolor='black', alpha=0.7)
    ax3.axvline(metrics['avg_question_length'], color='red', linestyle='--', label=f'Среднее: {metrics["avg_question_length"]:.1f}')
    ax3.set_xlabel("Длина вопроса (слов)")
    ax3.set_ylabel("Частота")
    ax3.set_title("Распределение длины вопросов")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Статистика эффективности фильтрации
    ax4 = axes[1, 1]
    if 'generation_stats' in globals() and generation_stats:
        stages = ["Кандидаты", "Сгенерировано", "После фильтрации", "Финальные"]
        counts = [
            generation_stats.get('total_candidates', 0),
            generation_stats.get('successful', 0),
            len(filtered_cqs) if 'filtered_cqs' in globals() else 0,
            len(final_cqs)
        ]
        ax4.bar(stages, counts)
        ax4.set_ylabel("Количество")
        ax4.set_title("Эффективность фильтрации")
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    logger.info(" Визуализации метрик созданы")
else:
    logger.warning("  Нет данных для визуализации метрик")

# %%
# Экспорт для экспертной оценки (CSV)
if final_cqs:
    df_expert_review = pd.DataFrame([
        {
            "cq_id": cq.cq_id,
            "text": cq.text,
            "role": cq.role,
            "pattern_id": cq.pattern_id,
            "slots": str(cq.slots),
            "expected_answer_type": cq.expected_answer_type,
            "answer_hint": cq.answer_hint or "",
            "evidence_chunks": ", ".join(cq.evidence_chunks[:3]),  # Первые 3
            "evidence_count": len(cq.evidence_chunks)
        }
        for cq in final_cqs
    ])
    
    expert_review_path = OUTPUT_DIR / "expert_review.csv"
    df_expert_review.to_csv(expert_review_path, index=False, encoding='utf-8')
    logger.info(f" Экспорт для экспертной оценки: {expert_review_path}")
    
    logger.info("\n Примеры CQs для ручной проверки:")
    display(df_expert_review.head(10))
else:
    logger.warning("  Нет CQs для экспорта")
