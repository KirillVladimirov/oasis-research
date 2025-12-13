# %% [markdown]
"""
# Demo Step 6B: Ontology Canonization

Этот ноутбук завершает рефакторинг онтологии, выполняя физическое слияние эквивалентных классов.
Предыдущий шаг (6A) определил эквивалентность через `owl:equivalentClass`, но сохранил все URI.
Этот шаг выбирает **канонический URI** для каждой группы синонимов и переписывает все ссылки в онтологии и артефактах.

**План действий (Task 6A.5):**
1. **Build Equivalence Graph:** Построение графа эквивалентностей на основе `owl:equivalentClass`.
2. **Identify Families:** Выделение компонент связности (семейств синонимов).
3. **Select Canonical:** Выбор главного класса в семье (по частоте использования и длине имени).
4. **Rewrite Ontology:** Замена всех URI на канонические (через `rdflib`).
5. **Update Artifacts:** Обновление ссылок в CQs.
6. **Validation:** Проверка консистентности через `owlready2`.
"""

# %%
import sys
from pathlib import Path
import json
import re
from typing import List, Dict, Set, Tuple, Any
from collections import defaultdict, Counter

import pandas as pd
import networkx as nx
import rdflib
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD
from owlready2 import *
from loguru import logger
from tqdm import tqdm

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path().absolute().parent))

# Настройки логирования
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

# %%
# КОНФИГУРАЦИЯ

TOPIC = "deep_active_learning"
BASE_DIR = Path(f"../outputs/{TOPIC}")

# Входные данные от шага 6A
INPUT_OWL = BASE_DIR / "step6a/deep_active_learning_refined_v2.owl"
INPUT_PROFILES = BASE_DIR / "step6a/class_profiles_v2.jsonl"

# CQs для обновления
INPUT_CQS = BASE_DIR / "step4/cq_with_final_answers.jsonl"

# Выходные данные шага 6B
OUTPUT_DIR = BASE_DIR / "step6b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_OWL = OUTPUT_DIR / "deep_active_learning_canonical.owl"
OUTPUT_MAPPING = OUTPUT_DIR / "canonical_mapping.json"
OUTPUT_CQS = OUTPUT_DIR / "cq_canonical.jsonl"

logger.info(f" Рабочая директория: {OUTPUT_DIR}")
logger.info(f" Входная онтология: {INPUT_OWL}")

# %%
# 1. ЗАГРУЗКА RDF И ПРОФИЛЕЙ

def load_profiles(path: Path) -> Dict[str, Any]:
    profiles = {}
    if not path.exists():
        logger.warning(f"Файл профилей не найден: {path}")
        return profiles
    
    with open(path, 'r') as f:
        for line in f:
            p = json.loads(line)
            profiles[p['class_iri']] = p
    return profiles

logger.info(" Загрузка RDF графа (может занять время)...")
g = Graph()
g.parse(INPUT_OWL)
logger.success(f" Граф загружен. Всего триплетов: {len(g)}")

profiles_map = load_profiles(INPUT_PROFILES)
logger.info(f" Загружено {len(profiles_map)} профилей классов")

# %%
# 2. ПОСТРОЕНИЕ ГРАФА ЭКВИВАЛЕНТНОСТЕЙ

def build_equivalence_families(graph: Graph) -> List[Set[str]]:
    eq_graph = nx.Graph()
    
    # Добавляем все классы как узлы
    for s in graph.subjects(RDF.type, OWL.Class):
        if isinstance(s, URIRef):
            eq_graph.add_node(str(s))
            
    # Добавляем ребра эквивалентности
    edge_count = 0
    for s, o in graph.subject_objects(OWL.equivalentClass):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            eq_graph.add_edge(str(s), str(o))
            edge_count += 1
            
    logger.info(f"Найдено {edge_count} связей owl:equivalentClass")
    
    # Ищем компоненты связности (семьи)
    families = list(nx.connected_components(eq_graph))
    # Оставляем только семьи > 1 элемента
    families = [f for f in families if len(f) > 1]
    
    return families

families = build_equivalence_families(g)
logger.info(f" Найдено {len(families)} семейств эквивалентных классов (кандидаты на слияние)")

# Пример семьи
if families:
    logger.info(f"Пример семьи: {list(families[0])[:3]}...")

# %%
# 3. ВЫБОР КАНОНИЧЕСКИХ КЛАССОВ

def select_canonical_iri(family: Set[str], profiles: Dict[str, Any]) -> str:
    candidates = []
    
    for iri in family:
        p = profiles.get(iri)
        
        if p:
            # Эвристики:
            # 1. usage_stats.num_cqs (чем больше, тем лучше) -> берем с минусом для asc sort
            # 2. длина label (чем короче, тем лучше)
            # 3. сам IRI (лексикографически, для стабильности)
            usage = p.get('usage_stats', {}).get('num_cqs', 0)
            label = p.get('label', '')
            candidates.append((-usage, len(label), iri))
        else:
            # Если профиля нет, считаем usage=0, label=iri
            candidates.append((0, len(iri), iri))
            
    # Сортировка
    candidates.sort()
    
    # Победитель - первый
    return candidates[0][2]

mapping = {} # old_iri -> canonical_iri
stats_merge = {"merged_classes": 0, "families_processed": 0}

for family in families:
    canonical = select_canonical_iri(family, profiles_map)
    stats_merge["families_processed"] += 1
    
    for iri in family:
        if iri != canonical:
            mapping[iri] = canonical
            stats_merge["merged_classes"] += 1

with open(OUTPUT_MAPPING, 'w') as f:
    json.dump(mapping, f, indent=2)
    
logger.success(f" Маппинг создан. Схлопывается {stats_merge['merged_classes']} классов в {stats_merge['families_processed']} канонических.")

# %%
# 4. ПЕРЕЗАПИСЬ ОНТОЛОГИИ (CANONIZATION)

def rewrite_ontology(graph: Graph, mapping: Dict[str, str]) -> Graph:
    new_g = Graph()
    
    # Копируем неймспейсы
    for prefix, uri in graph.namespaces():
        new_g.bind(prefix, uri)
        
    # Счетчики для отладки
    triples_kept = 0
    triples_modified = 0
    reflexive_skipped = 0
    
    for s, p, o in tqdm(graph, desc="Rewriting triples"):
        # Mapping Subject
        new_s = s
        s_str = str(s)
        if isinstance(s, URIRef) and s_str in mapping:
            new_s = URIRef(mapping[s_str])
            
        # Mapping Predicate
        new_p = p
        p_str = str(p)
        if isinstance(p, URIRef) and p_str in mapping:
            new_p = URIRef(mapping[p_str])
            
        # Mapping Object
        new_o = o
        o_str = str(o)
        if isinstance(o, URIRef) and o_str in mapping:
            new_o = URIRef(mapping[o_str])
            
        # Проверка на рефлексивность (A equivalentClass A)
        if new_p == OWL.equivalentClass and new_s == new_o:
            reflexive_skipped += 1
            continue
            
        # Добавляем в новый граф
        new_g.add((new_s, new_p, new_o))
        
        if new_s != s or new_p != p or new_o != o:
            triples_modified += 1
        else:
            triples_kept += 1
            
    logger.info(f"Stats: Modified={triples_modified}, Kept={triples_kept}, Skipped (reflexive)={reflexive_skipped}")
    return new_g

logger.info(" Начало канонизации графа...")
canonical_g = rewrite_ontology(g, mapping)
logger.info(f"Сохранение в {OUTPUT_OWL}...")
canonical_g.serialize(destination=str(OUTPUT_OWL), format="xml")
logger.success(" Онтология перезаписана и сохранена.")

# %%
# 5. ОБНОВЛЕНИЕ АРТЕФАКТОВ (CQs)

def update_cqs(input_path: Path, output_path: Path, mapping: Dict[str, str]):
    if not input_path.exists():
        logger.warning(f"CQ файл не найден: {input_path}")
        return
        
    updates_count = 0
    
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            # Простая текстовая замена URI
            # Это безопасно, так как IRI уникальны и длинны
            # Но лучше парсить JSON, чтобы не сломать ключи
            try:
                data = json.loads(line)
                json_str = json.dumps(data)
                
                # Заменяем IRI в строковом представлении JSON
                # Сортируем ключи маппинга по длине (desc), чтобы избежать частичных замен
                # хотя IRI полные, так что риск минимален.
                # Но эффективнее просто заменить значения, если мы знаем структуру.
                # Однако IRI могут быть в тексте ответов, в полях context и т.д.
                # Поэтому replace по всей строке JSON допустим для IRI.
                
                original_str = json_str
                for old_iri, new_iri in mapping.items():
                    if old_iri in json_str:
                        json_str = json_str.replace(old_iri, new_iri)
                
                if json_str != original_str:
                    updates_count += 1
                    
                fout.write(json_str + '\n')
            except Exception as e:
                logger.error(f"Error processing CQ line: {e}")
                
    logger.success(f" Обновлен файл CQs. Изменено записей: {updates_count}")

update_cqs(INPUT_CQS, OUTPUT_CQS, mapping)

# %%
# 6. VALIDATION & METRICS

logger.info(" Запуск Reasoner для проверки консистентности...")

try:
    onto = get_ontology(str(OUTPUT_OWL)).load()
    with onto:
        sync_reasoner(infer_property_values=True)
    logger.success(" Reasoner успешно отработал. Онтология консистентна.")
    
    classes_count = len(list(onto.classes()))
    logger.info(f" Итоговое количество классов: {classes_count}")
    
    # Сравнение с исходным
    original_count = len(profiles_map)
    diff = original_count - classes_count
    logger.info(f" Уменьшение количества классов: {original_count} -> {classes_count} (Схлопнуто {diff})")
    
except Exception as e:
    logger.error(f" Ошибка валидации: {e}")
