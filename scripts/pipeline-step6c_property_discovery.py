# %% [markdown]
"""
# Демонстрация Шага 6C: Поиск дополнительных объектных свойств (Property Discovery)

Этот ноутбук реализует процесс автоматического поиска семантических связей между классами онтологии на основе их совместной встречаемости в текстах вопросов (CQs) и ответов.

**Входные данные (после Canonization 6B):**
*   `deep_active_learning_canonical.owl` - каноническая онтология (без дубликатов классов).
*   `cq_canonical.jsonl` - CQs с обновленными ссылками на канонические классы.

**Задачи (по Step-6+.md):**
1.  **Entity Linking & Co-occurrence:** Поиск пар классов, встречающихся в одном контексте.
2.  **Profile Aggregation:** Сбор статистики и примеров для каждой пары.
3.  **Property Discovery Agent:** LLM-агент, предлагающий новые ObjectProperties.
4.  **Ontology Update:** Добавление утвержденных свойств в онтологию.

**Выход:** `deep_active_learning_enriched.owl`
"""

# %%
import sys
from pathlib import Path
import json
import os
import re
from typing import Any, Optional, List, Dict, Set, Tuple
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm
from owlready2 import *
from sentence_transformers import SentenceTransformer, util

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
BASE_DIR = Path(f"../outputs/{TOPIC}")

# Входные данные (Результат Step 6B - Canonization)
INPUT_OWL = BASE_DIR / "step6b/deep_active_learning_canonical.owl"
INPUT_CQS = BASE_DIR / "step6b/cq_canonical.jsonl"

# Выходные данные
OUTPUT_DIR = BASE_DIR / "step6c"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_OWL = OUTPUT_DIR / "deep_active_learning_enriched.owl"

# Настройки модели LLM
LLM_MODEL = os.getenv("OPENAI_CANONICAL_MODEL", "gpt-5.1")
LLM_TIMEOUT = 300.0

# Модель эмбеддингов (для дедупликации свойств)
EMBEDDING_MODEL = "models/bge-m3"
project_root = Path().absolute().parent
model_path = project_root / EMBEDDING_MODEL

if model_path.exists():
    EMBEDDING_MODEL_PATH = str(model_path)
    logger.info(f" Используется локальная модель эмбеддингов: {EMBEDDING_MODEL_PATH}")
else:
    logger.warning(f" Локальная модель {model_path} не найдена, используется HuggingFace")
    EMBEDDING_MODEL_PATH = "BAAI/bge-m3"

# Параметры фильтрации
MIN_COOCCURRENCE = 3  # Минимальное число совместных появлений для анализа пары
CONFIDENCE_THRESHOLD = 0.7 # Порог уверенности LLM для создания свойства

# Инициализация клиентов
try:
    openai_client = _create_openai_client(timeout=LLM_TIMEOUT)
    logger.success(f" OpenAI клиент инициализирован: {LLM_MODEL}")
except Exception as e:
    logger.error(f" Ошибка инициализации OpenAI: {e}")
    openai_client = None

logger.info(f"  Вход OWL: {INPUT_OWL}")
logger.info(f"  Вход CQs: {INPUT_CQS}")
logger.info(f"  Выход: {OUTPUT_DIR}")

# %%
# PYDANTIC МОДЕЛИ

class ClassCooccurrenceEvent(BaseModel):
    cq_id: str
    classes: List[str]  # List of class IRIs
    text_snippet: str
    role: str = "relation"

class ClassPairProfile(BaseModel):
    pair_id: str
    class_a_iri: str
    class_b_iri: str
    class_a_label: str
    class_b_label: str
    cooccurrence_count: int
    example_snippets: List[str]
    roles_distribution: Dict[str, int]

class PropertySuggestion(BaseModel):
    pair_id: str
    suggest_decision: str = Field(..., pattern="^(create|skip|map_existing)$")
    property_iri_suggestion: Optional[str] = None
    property_label: Optional[str] = None
    property_description: Optional[str] = None
    domain_class_iri: Optional[str] = None
    range_class_iri: Optional[str] = None
    is_functional: bool = False
    is_symmetric: bool = False
    mapped_existing_property_iri: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    comment: str

logger.info(" Pydantic модели определены")

# %%
# STEP 6C.1: ENTITY LINKING & CO-OCCURRENCE EXTRACTION

def build_class_lookup(onto_path: Path) -> Dict[str, str]:
    """Builds a map of normalized labels -> canonical IRI"""
    onto = get_ontology(str(onto_path)).load()
    lookup = {}
    
    for cls in onto.classes():
        if cls == Thing: continue
        
        # Основной label
        lbl = cls.label[0] if cls.label else cls.name
        lookup[lbl.lower()] = cls.iri
        lookup[cls.name.lower()] = cls.iri
        
        # Синонимы (если есть в аннотациях, но пока берем label)
        # Можно добавить нормализацию (удаление спецсимволов)
    
    logger.info(f" Индекс классов построен: {len(lookup)} ключей")
    return lookup, onto

def extract_cooccurrences(cqs_path: Path, lookup: Dict[str, str]) -> List[ClassCooccurrenceEvent]:
    events = []
    
    if not cqs_path.exists():
        return events
        
    with open(cqs_path, 'r') as f:
        for line in f:
            try:
                cq_data = json.loads(line)
                text = cq_data.get('text', '') + " " + cq_data.get('final_answer', {}).get('text', '')
                text_lower = text.lower()
                
                found_irises = set()
                
                # Простой поиск подстрок (можно улучшить через FlashText или Spacy)
                # Сортируем ключи по длине, чтобы находить самые длинные совпадения
                sorted_keys = sorted(lookup.keys(), key=len, reverse=True)
                
                for key in sorted_keys:
                    if key in text_lower:
                        # Проверка на границы слов (чтобы 'task' не находилось внутри 'multitask')
                        pattern = r'(^|\W)' + re.escape(key) + r'(\W|$)'
                        if re.search(pattern, text_lower):
                            found_irises.add(lookup[key])
                            
                # Если найдено >= 2 классов, это событие
                if len(found_irises) >= 2:
                    events.append(ClassCooccurrenceEvent(
                        cq_id=cq_data.get('cq_id', 'unknown'),
                        classes=list(found_irises),
                        text_snippet=text[:500] # Берем начало или контекст
                    ))
            except Exception as e:
                continue
                
    return events

lookup_map, current_onto = build_class_lookup(INPUT_OWL)
events = extract_cooccurrences(INPUT_CQS, lookup_map)

logger.success(f" Найдено {len(events)} событий совместного появления классов")

# Сохранение событий (опционально)
events_file = OUTPUT_DIR / "class_cooccurrence_events.jsonl"
with open(events_file, 'w') as f:
    for e in events:
        f.write(e.model_dump_json() + '\n')

# %%
# STEP 6C.2: PROFILE AGGREGATION

def aggregate_profiles(events: List[ClassCooccurrenceEvent], onto) -> List[ClassPairProfile]:
    pairs_data = defaultdict(lambda: {
        "count": 0, 
        "snippets": [],
        "class_a_label": "",
        "class_b_label": ""
    })
    
    for ev in events:
        # Генерируем все пары
        sorted_classes = sorted(ev.classes)
        for i in range(len(sorted_classes)):
            for j in range(i + 1, len(sorted_classes)):
                pair_key = (sorted_classes[i], sorted_classes[j])
                
                pairs_data[pair_key]["count"] += 1
                if len(pairs_data[pair_key]["snippets"]) < 5: # Храним до 5 примеров
                    pairs_data[pair_key]["snippets"].append(ev.text_snippet)
                    
    profiles = []
    idx = 1
    
    for (iri_a, iri_b), data in pairs_data.items():
        if data["count"] < MIN_COOCCURRENCE:
            continue
            
        # Получаем метки классов из онтологии
        cls_a = onto.search_one(iri=iri_a)
        cls_b = onto.search_one(iri=iri_b)
        
        label_a = cls_a.label[0] if cls_a and cls_a.label else Path(iri_a).name
        label_b = cls_b.label[0] if cls_b and cls_b.label else Path(iri_b).name
        
        profile = ClassPairProfile(
            pair_id=f"CPP_{idx:03d}",
            class_a_iri=iri_a,
            class_b_iri=iri_b,
            class_a_label=label_a,
            class_b_label=label_b,
            cooccurrence_count=data["count"],
            example_snippets=data["snippets"],
            roles_distribution={"relation": data["count"]} # Пока заглушка
        )
        profiles.append(profile)
        idx += 1
        
    return profiles

profiles = aggregate_profiles(events, current_onto)
profiles.sort(key=lambda p: p.cooccurrence_count, reverse=True)

logger.success(f" Сформировано {len(profiles)} профилей пар классов (min_count={MIN_COOCCURRENCE})")

# Сохранение профилей
profiles_file = OUTPUT_DIR / "class_pair_profiles.jsonl"
with open(profiles_file, 'w') as f:
    for p in profiles:
        f.write(p.model_dump_json() + '\n')

# Показать топ профилей
if profiles:
    display(pd.DataFrame([
        {
            "Class A": p.class_a_label,
            "Class B": p.class_b_label,
            "Count": p.cooccurrence_count,
            "Snippet": p.example_snippets[0][:100] + "..."
        } for p in profiles[:5]
    ]))

# %%
# STEP 6C.3: PROPERTY DISCOVERY AGENT (LLM)

SYSTEM_PROMPT_DISCOVERY = """You are an ontology engineering assistant for a Deep Active Learning domain.

You receive a profile of two OWL classes that frequently co-occur in competency questions and answers.
Your task is to decide whether there should be an explicit object property between these classes, and if so, propose its design.

Guidelines:
- Base your reasoning ONLY on the provided labels and text snippets.
- Aim for domain-specific, reusable properties (e.g., "evaluatedWithMetric", "appliedToDataset", "usesAcquisitionFunction").
- Prefer a single clear property over many overlapping ones.
- If an existing property label from the input would already fit this pair, prefer mapping to it instead of creating a new one.
- Output MUST be valid JSON.
"""

def suggest_property(profile: ClassPairProfile, openai_client) -> Optional[PropertySuggestion]:
    if not openai_client: return None
    
    # Формируем контекст
    profile_json = json.dumps({
        "pair_id": profile.pair_id,
        "class_a": {"iri": profile.class_a_iri, "label": profile.class_a_label},
        "class_b": {"iri": profile.class_b_iri, "label": profile.class_b_label},
        "contexts": profile.example_snippets
    }, indent=2)
    
    user_prompt = f"""Here is a profile of two ontology classes that frequently co-occur in Deep Active Learning texts:

Class pair profile (JSON):
{profile_json}

Decide:

1. Should an explicit object property be defined between these classes?
   Options:
   - "create": propose a new property.
   - "map_existing": reuse a property that already exists in the ontology (if mentioned in the profile).
   - "skip": do nothing (contexts are too vague or inconsistent).

2. If you choose "create":
   - propose a property label (CamelCase, suitable as OWL object property name);
   - write a 1-3 sentence description of its meaning;
   - choose the domain and range classes from the given pair (which one is subject, which one is object);
   - assess whether the property is likely functional (one-to-one) or not, and whether it is symmetric.

3. If you choose "map_existing":
   - specify which existing property label from the input should be used.

Return JSON matching PropertySuggestion schema.
"""

    try:
        response = openai_client.beta.chat.completions.parse(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_DISCOVERY},
                {"role": "user", "content": user_prompt}
            ],
            response_format=PropertySuggestion,
            temperature=0.0,
            timeout=60.0
        )
        suggestion = response.choices[0].message.parsed
        suggestion.pair_id = profile.pair_id # Ensure ID match
        return suggestion
    except Exception as e:
        logger.error(f"Error suggesting for {profile.pair_id}: {e}")
        return None

suggestions_file = OUTPUT_DIR / "property_suggestions.jsonl"
suggestions = []

if suggestions_file.exists():
    logger.info(f" Загрузка предложений из {suggestions_file}")
    with open(suggestions_file, 'r') as f:
        for line in f:
            suggestions.append(PropertySuggestion.model_validate_json(line))
else:
    if profiles and openai_client:
        logger.info(f" Запуск Property Discovery Agent для {len(profiles)} пар...")
        for profile in tqdm(profiles, desc="Discovery"):
            s = suggest_property(profile, openai_client)
            if s:
                suggestions.append(s)
                
        with open(suggestions_file, 'w') as f:
            for s in suggestions:
                f.write(s.model_dump_json() + '\n')
        logger.success(f" Получено {len(suggestions)} предложений")

# Статистика
stats = Counter([s.suggest_decision for s in suggestions])
logger.info(f"Decisions: {dict(stats)}")

# %%
# STEP 6C.4: ONTOLOGY UPDATE

def apply_properties(owl_path: Path, suggestions: List[PropertySuggestion], output_path: Path):
    if not owl_path.exists(): return
    
    onto = get_ontology(str(owl_path)).load()
    
    added_count = 0
    
    with onto:
        for s in suggestions:
            if s.suggest_decision != "create" or s.confidence < CONFIDENCE_THRESHOLD:
                continue
                
            if not s.property_label or not s.domain_class_iri or not s.range_class_iri:
                continue
                
            # Проверка существования классов
            domain_cls = onto.search_one(iri=s.domain_class_iri)
            range_cls = onto.search_one(iri=s.range_class_iri)
            
            if not domain_cls or not range_cls:
                continue
                
            # Создание свойства
            # Проверяем, есть ли уже такое свойство по имени
            prop_name = s.property_label
            existing_prop = onto.search_one(iri=f"*{prop_name}")
            
            if existing_prop:
                # Если свойство есть, расширяем домен/рейндж
                existing_prop.domain.append(domain_cls)
                existing_prop.range.append(range_cls)
                logger.info(f"Updated property {prop_name}: added domain/range")
            else:
                # Создаем новое
                # Динамическое создание класса свойства
                NewProp = type(prop_name, (ObjectProperty,), {})
                NewProp.domain = [domain_cls]
                NewProp.range = [range_cls]
                NewProp.label = [s.property_label]
                NewProp.comment = [s.property_description]
                
                if s.is_functional:
                    NewProp.is_a.append(FunctionalProperty)
                if s.is_symmetric:
                    NewProp.is_a.append(SymmetricProperty)
                    
                added_count += 1
                logger.info(f"Created property {prop_name}: {domain_cls.name} -> {range_cls.name}")

    onto.save(file=str(output_path), format="rdfxml")
    logger.success(f" Онтология обновлена. Добавлено/обновлено {added_count} свойств.")
    return onto

if suggestions:
    logger.info(" Применение новых свойств к онтологии...")
    enriched_onto = apply_properties(INPUT_OWL, suggestions, OUTPUT_OWL)
    
    # Metrics
    if enriched_onto:
        props_count = len(list(enriched_onto.object_properties()))
        logger.info(f" Всего Object Properties в новой онтологии: {props_count}")
