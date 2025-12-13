# %% [markdown]
"""
# Демонстрация Шага 6: Индукция онтологии (TBox)

Этот ноутбук демонстрирует процесс автоматического построения TBox-части онтологии (классы, иерархия, свойства) на основе финального набора компетентностных вопросов (CQs) и ответов.

**Подход:** "Flexible LLM-Architect with Anchors"
- **Якоря (Anchors):** Фиксированные корневые классы (`Task`, `Method`, `Metric`, `Dataset`, `Constraint`) для обеспечения структурной совместимости с другими научными доменами.
- **Гибкое расширение (Flexible Expansion):** LLM анализирует термины из CQs и проектирует подклассы и свойства, специфичные для Deep Active Learning.
- **Программная генерация:** Итоговый OWL-файл создается через библиотеку `owlready2`.

## Структура пайплайна:
1. **Инициализация**: Настройка окружения и `owlready2`.
2. **Загрузка данных**: Чтение `cq_with_final_answers.jsonl` (вход из Шага 4).
3. **Estimate**: Расчет примерных затрат токенов.
4. **Extraction**: Извлечение ключевых терминов из CQs с помощью LLM.
5. **Consolidation**: Нормализация и группировка синонимов.
6. **Architecture**: Проектирование иерархии классов (LLM-Architect).
7. **Generation**: Генерация OWL-онтологии.
8. **Verification**: Проверка консистентности reasoner'ом.
"""

# %%
import sys
from pathlib import Path
import json
import os
import time
import re
from typing import Any, Optional, List, Dict, Set
from collections import defaultdict, Counter

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm
from owlready2 import *

# Добавляем путь к проекту
sys.path.insert(0, str(Path().absolute().parent))

# Загрузка переменных окружения
env_path = Path().absolute().parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

from oasis.pipelines.stage1_extract_topics import _create_openai_client

# Настройки отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', None)

logger.info(" Импорты и настройки загружены")

# %%
# КОНФИГУРАЦИЯ

TOPIC = "deep_active_learning"
INPUT_FILE = Path(f"../outputs/{TOPIC}/step4/cq_with_final_answers.jsonl")
OUTPUT_DIR = Path(f"../outputs/{TOPIC}/step6")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LLM_MODEL = os.getenv("OPENAI_CANONICAL_MODEL", "gpt-4o")
LLM_TIMEOUT = 120.0
TRACE_ID = "demo_step6_ontology_induction"

# Инициализация OpenAI клиента
try:
    openai_client = _create_openai_client(timeout=LLM_TIMEOUT)
    logger.success(f" OpenAI клиент инициализирован: {LLM_MODEL}")
except Exception as e:
    logger.error(f" Ошибка инициализации OpenAI клиента: {e}")
    openai_client = None

logger.info(f"  Входной файл: {INPUT_FILE}")
logger.info(f"  Выходная директория: {OUTPUT_DIR}")

# %%
# ЗАГРУЗКА ДАННЫХ (CQs)

cqs = []
if INPUT_FILE.exists():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                cqs.append(json.loads(line))
    logger.info(f" Загружено {len(cqs)} CQs")
else:
    logger.warning(f" Файл {INPUT_FILE} не найден. Убедитесь, что Шаг 4 выполнен.")

# Пример данных
if cqs:
    display(pd.DataFrame(cqs[:3])[['cq_id', 'text', 'final_answer']])

# %%
# PYDANTIC МОДЕЛИ

class ExtractedTerm(BaseModel):
    """Термин, извлеченный из CQ."""
    label: str = Field(..., description="Название термина (как в тексте)")
    type_hint: str = Field(..., description="Предполагаемый тип (Task, Method, Metric, Dataset, Constraint, Other)")
    context_cq_id: str = Field(..., description="ID вопроса, откуда извлечен термин")

class ExtractionBatchResult(BaseModel):
    """Результат извлечения для батча CQs."""
    terms: List[ExtractedTerm]

class CanonicalConcept(BaseModel):
    """Нормализованный концепт онтологии."""
    concept_id: str = Field(..., description="Уникальный ID (CamelCase)")
    label: str = Field(..., description="Человекочитаемое название")
    synonyms: List[str] = Field(default_factory=list, description="Список синонимов из сырых терминов")
    base_anchor: str = Field(..., description="К какому якорю относится (Task/Method/Metric/Dataset/Constraint/Other)")
    description_source: Optional[str] = Field(None, description="Текст для rdfs:comment (из ответов)")

class OntologyClassSchema(BaseModel):
    """Схема класса для генерации."""
    class_name: str
    parent_class: str
    comment: str

class OntologyPropertySchema(BaseModel):
    """Схема свойства для генерации."""
    property_name: str
    domain: str
    range: str
    comment: str

class OntologyDesign(BaseModel):
    """Полный дизайн онтологии."""
    classes: List[OntologyClassSchema]
    properties: List[OntologyPropertySchema]

logger.info(" Pydantic модели определены")

# %%
# STEP 1: ИЗВЛЕЧЕНИЕ ТЕРМИНОВ (EXTRACTION)

EXTRACTION_SYSTEM_PROMPT = """
You are an expert ontology engineer in Deep Active Learning.
Your task is to extract domain-specific terms from a list of Competency Questions (CQs) and their answers.

For each CQ, identify key concepts that should become classes or individuals in an ontology.
Categorize them into one of the base anchors:
- Task (e.g., Image Classification, Object Detection)
- Method (e.g., Acquisition Strategy, BALD, CoreSet)
- Metric (e.g., Accuracy, Label Efficiency, F1-score)
- Dataset (e.g., MNIST, CIFAR-10)
- Constraint (e.g., Budget, Noise, Imbalance)
- Other (if it's a crucial concept but fits none of the above)

Ignore generic terms like "paper", "study", "result", "question".
Keep the original phrasing from the text.
"""

def extract_terms_batch(cqs_batch: List[Dict], openai_client) -> List[ExtractedTerm]:
    if not openai_client: return []

    user_content = "Extract terms from these CQs:\n\n"
    for cq in cqs_batch:
        final_ans = cq.get('final_answer', {})
        ans_text = final_ans.get('text', '') if isinstance(final_ans, dict) else str(final_ans)
        user_content += f"ID: {cq['cq_id']}\nQuestion: {cq['text']}\nAnswer: {ans_text}\n---\n"

    try:
        response = openai_client.beta.chat.completions.parse(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            response_format=ExtractionBatchResult,
            temperature=0.0,
            timeout=60.0
        )
        return response.choices[0].message.parsed.terms
    except Exception as e:
        logger.error(f"Extraction error: {e}")
        return []

# Запуск Extraction
all_extracted_terms = []
BATCH_SIZE = 10

if cqs and openai_client:
    logger.info(f" Извлечение терминов из {len(cqs)} CQs...")
    for i in tqdm(range(0, len(cqs), BATCH_SIZE), desc="Extraction"):
        batch = cqs[i:i+BATCH_SIZE]
        terms = extract_terms_batch(batch, openai_client)
        all_extracted_terms.extend(terms)

    logger.success(f" Извлечено {len(all_extracted_terms)} сырых терминов")
    
    # Сохранение промежуточного результата
    with open(OUTPUT_DIR / "raw_terms.jsonl", 'w', encoding='utf-8') as f:
        for t in all_extracted_terms:
            f.write(t.model_dump_json() + '\n')
else:
    logger.warning("Пропуск Extraction (нет данных или клиента)")

# %%
# STEP 2: КОНСОЛИДАЦИЯ (CONSOLIDATION)

CONSOLIDATION_SYSTEM_PROMPT = """
You are an ontology architect.
You are given a list of raw terms extracted from text.
Your task is to consolidate synonyms and normalize them into canonical concepts.

Rules:
1. Group synonyms (e.g., "Active Learning Strategy" and "Acquisition Strategy" -> "AcquisitionStrategy").
2. Create a CamelCase ID for each concept (e.g., "ImageClassification").
3. Assign the correct base anchor (Task, Method, Metric, Dataset, Constraint).
4. Provide a clean label.
"""

class ConsolidationResult(BaseModel):
    concepts: List[CanonicalConcept]

def consolidate_terms(raw_terms: List[ExtractedTerm], openai_client) -> List[CanonicalConcept]:
    if not openai_client: return []
    
    # Уникализируем список сырых терминов для промпта
    unique_labels = sorted(list(set([t.label for t in raw_terms])))
    
    # Разбиваем на чанки, если слишком много (пока попробуем все сразу, если < 200)
    # В реальном проекте лучше делать это итеративно, но для демо ок.
    
    user_content = "Raw terms to consolidate:\n" + ", ".join(unique_labels)
    
    try:
        response = openai_client.beta.chat.completions.parse(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": CONSOLIDATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            response_format=ConsolidationResult,
            temperature=0.0,
            timeout=120.0
        )
        return response.choices[0].message.parsed.concepts
    except Exception as e:
        logger.error(f"Consolidation error: {e}")
        return []

canonical_concepts = []
if all_extracted_terms:
    logger.info(f" Консолидация {len(all_extracted_terms)} терминов...")
    canonical_concepts = consolidate_terms(all_extracted_terms, openai_client)
    logger.success(f" Создано {len(canonical_concepts)} канонических концептов")
    
    # Вывод примеров
    display(pd.DataFrame([c.model_dump() for c in canonical_concepts[:5]]))
    
    # Сохранение
    with open(OUTPUT_DIR / "canonical_concepts.jsonl", 'w', encoding='utf-8') as f:
        for c in canonical_concepts:
            f.write(c.model_dump_json() + '\n')

# %%
# STEP 3: ПРОЕКТИРОВАНИЕ (LLM ARCHITECT)

ARCHITECTURE_SYSTEM_PROMPT = """
You are an OWL Ontology Architect.
Design a class hierarchy and object properties based on the provided list of canonical concepts.

Base Anchors (Root Classes):
- Task
- Method
- Metric
- Dataset
- Constraint

Instructions:
1. Map each concept to a class. Determine its Parent Class (either an Anchor or another concept).
2. Design logical Object Properties (e.g., "methodUsedForTask", "evaluatedOnDataset").
3. Ensure the hierarchy is logical (e.g., "BALD" -> "UncertaintySampling" -> "AcquisitionStrategy" -> "Method").
4. Provide a short rdfs:comment for each class.
"""

def design_ontology(concepts: List[CanonicalConcept], openai_client) -> Optional[OntologyDesign]:
    if not openai_client: return None

    concepts_text = "Concepts to organize:\n"
    for c in concepts:
        concepts_text += f"- {c.concept_id} (Anchor: {c.base_anchor})\n"

    try:
        response = openai_client.beta.chat.completions.parse(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": ARCHITECTURE_SYSTEM_PROMPT},
                {"role": "user", "content": concepts_text}
            ],
            response_format=OntologyDesign,
            temperature=0.0,
            timeout=120.0
        )
        return response.choices[0].message.parsed
    except Exception as e:
        logger.error(f"Architecture error: {e}")
        return None

ontology_design = None
if canonical_concepts:
    logger.info(" Проектирование архитектуры онтологии...")
    ontology_design = design_ontology(canonical_concepts, openai_client)
    
    if ontology_design:
        logger.success(f" Спроектировано {len(ontology_design.classes)} классов и {len(ontology_design.properties)} свойств")
        
        # Сохранение схемы
        with open(OUTPUT_DIR / "ontology_design.json", 'w', encoding='utf-8') as f:
            f.write(ontology_design.model_dump_json(indent=2))
    else:
        logger.error("Не удалось спроектировать онтологию")

# %%
# STEP 4: ГЕНЕРАЦИЯ OWL (OWLREADY2)

def generate_owl_file(design: OntologyDesign, output_path: Path):
    if not design: return

    # Создаем онтологию
    onto_iri = "http://deep_active_learning#"
    onto = get_ontology(onto_iri)

    with onto:
        # 1. Создаем якоря (Anchors)
        class Task(Thing): pass
        class Method(Thing): pass
        class Metric(Thing): pass
        class Dataset(Thing): pass
        class Constraint(Thing): pass
        
        anchor_map = {
            "Task": Task,
            "Method": Method,
            "Metric": Metric,
            "Dataset": Dataset,
            "Constraint": Constraint
        }

        # 2. Создаем классы из дизайна
        # Сначала создадим мапу всех классов (имя -> класс), чтобы разрешить зависимости
        # Будем создавать в два прохода: сначала пустые классы, потом иерархию
        created_classes = anchor_map.copy()

        # Проход 1: создание типов (динамически)
        for cls_schema in design.classes:
            if cls_schema.class_name not in created_classes:
                # Создаем как потомок Thing временно
                NewClass = types.new_class(cls_schema.class_name, (Thing,))
                created_classes[cls_schema.class_name] = NewClass

        # Проход 2: выстраивание иерархии и добавление комментариев
        for cls_schema in design.classes:
            cls = created_classes.get(cls_schema.class_name)
            parent_name = cls_schema.parent_class
            
            # Определяем родителя
            parent = created_classes.get(parent_name)
            if not parent:
                # Если родитель не найден, пробуем найти ближайший якорь по имени
                # Или оставляем Thing (но лучше залогировать)
                parent = Thing
            
            # Устанавливаем родителя
            cls.is_a = [parent]
            
            # Добавляем комментарий
            cls.comment = [cls_schema.comment]
            cls.label = [cls_schema.class_name]  # Можно добавить пробелы в label

        # 3. Создаем свойства
        for prop_schema in design.properties:
            # ObjectProperty
            NewProp = types.new_class(prop_schema.property_name, (ObjectProperty,))
            
            # Domain & Range
            dom_cls = created_classes.get(prop_schema.domain)
            rng_cls = created_classes.get(prop_schema.range)
            
            if dom_cls: NewProp.domain = [dom_cls]
            if rng_cls: NewProp.range = [rng_cls]
            
            NewProp.comment = [prop_schema.comment]
            NewProp.label = [prop_schema.property_name]

    # Сохранение
    onto.save(file=str(output_path), format="rdfxml")
    logger.success(f" Онтология сохранена в {output_path}")
    return onto

if ontology_design:
    owl_path = OUTPUT_DIR / "deep_active_learning.owl"
    logger.info(" Генерация OWL файла...")
    onto = generate_owl_file(ontology_design, owl_path)

# %%
# STEP 5: ВЕРИФИКАЦИЯ (REASONER)

if 'onto' in locals() and onto:
    logger.info(" Запуск Reasoner (HermiT)...")
    try:
        with onto:
            sync_reasoner(infer_property_values=True)
        logger.success(" Reasoning завершен успешно. Онтология консистентна.")
        
        # Проверка на несогласованные классы
        inconsistent = list(onto.inconsistent_classes())
        if inconsistent:
            logger.error(f" Обнаружены несогласованные классы: {inconsistent}")
        else:
            logger.success(" Несогласованных классов не найдено.")
            
    except Exception as e:
        logger.error(f"Ошибка Reasoner: {e}")

# %%
# МЕТРИКИ И АНАЛИЗ

if 'onto' in locals() and onto:
    classes = list(onto.classes())
    properties = list(onto.object_properties())
    
    logger.info("\n Статистика онтологии:")
    logger.info(f"  Всего классов: {len(classes)}")
    logger.info(f"  Всего свойств: {len(properties)}")
    
    # Оценка покрытия (Coverage)
    # Проверим, сколько канонических концептов стали классами
    class_names = set(c.name for c in classes)
    concept_names = set(c.concept_id for c in canonical_concepts)
    
    covered = concept_names.intersection(class_names)
    coverage = len(covered) / len(concept_names) if concept_names else 0
    
    logger.info(f"  Покрытие концептов: {coverage:.1%} ({len(covered)}/{len(concept_names)})")
    
    # Сохранение метрик
    metrics = {
        "total_classes": len(classes),
        "total_properties": len(properties),
        "coverage": coverage,
        "covered_concepts": list(covered)
    }
    
    with open(OUTPUT_DIR / "ontology_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f" Метрики сохранены в {OUTPUT_DIR / 'ontology_metrics.json'}")
