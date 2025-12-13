# %% [markdown]
"""
# Демонстрация Шага 6A: Агрессивная очистка и уточнение онтологии

Этот ноутбук реализует процесс автоматического поиска и слияния дублирующих классов в созданной OWL-онтологии (Step 6).
Реализована **агрессивная стратегия** слияния для минимизации количества классов.

**Задачи:**
1. **Export Profiles:** Извлечение профилей классов (+ нормализация корней).
2. **Clustering (Siblings):** Поиск похожих классов среди сиблингов (пониженные пороги).
3. **Sibling Merge Agent:** Агрессивное слияние синонимов и мелких вариаций.
4. **Parent Collapse:** Поиск и слияние избыточных детей в родителя.
5. **Apply Patch:** Применение изменений к OWL-файлу.

**Вход:** `deep_active_learning.owl` (Step 6)  
**Выход:** `deep_active_learning_refined.owl`
"""

# %%
import sys
from pathlib import Path
import json
import os
import time
import re
from typing import Any, Optional, List, Dict, Set, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm
from owlready2 import *
import networkx as nx
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
INPUT_OWL = BASE_DIR / "step6/deep_active_learning.owl"
INPUT_CQS = BASE_DIR / "step4/cq_with_final_answers.jsonl"
OUTPUT_DIR = BASE_DIR / "step6a"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Настройки модели LLM
LLM_MODEL = os.getenv("OPENAI_CANONICAL_MODEL", "gpt-5.1")
LLM_TIMEOUT = 300.0

# Модель эмбеддингов (ADR-0007: используем локальную bge-m3)
EMBEDDING_MODEL = "models/bge-m3"
project_root = Path().absolute().parent
model_path = project_root / EMBEDDING_MODEL

if model_path.exists():
    EMBEDDING_MODEL_PATH = str(model_path)
    logger.info(f" Используется локальная модель эмбеддингов: {EMBEDDING_MODEL_PATH}")
else:
    logger.warning(f" Локальная модель {model_path} не найдена, используется HuggingFace")
    EMBEDDING_MODEL_PATH = "BAAI/bge-m3"

# АГРЕССИВНЫЕ ПОРОГИ СХОЖЕСТИ
SIMILARITY_THRESHOLD_SEMANTIC = 0.75  # Снижено с 0.85
SIMILARITY_THRESHOLD_LEXICAL = 0.60   # Снижено с 0.80

# Стоп-слова для нормализации корней (чтобы сблизить ComputationalCost и ComputationalEfficiency)
STOP_SUFFIXES = [
    "constraint", "constraints", 
    "metric", "metrics", 
    "loss", "function", 
    "strategy", "strategies", 
    "method", "methods", 
    "algorithm", "algorithms",
    "task", "tasks", 
    "problem", "problems",
    "learning", "approach"
]

# Инициализация клиентов
try:
    openai_client = _create_openai_client(timeout=LLM_TIMEOUT)
    logger.success(f" OpenAI клиент инициализирован: {LLM_MODEL}")
except Exception as e:
    logger.error(f" Ошибка инициализации OpenAI: {e}")
    openai_client = None

# %%
# PYDANTIC МОДЕЛИ

class ClassProfile(BaseModel):
    class_iri: str
    label: str
    normalized_label: str
    tokens: List[str]
    root_tokens: List[str]  # New: tokens without common suffixes
    parent_iris: List[str]
    sibling_iris: List[str]
    text_description: str
    example_cq_ids: List[str] = []
    usage_stats: Dict[str, int] = {"num_cqs": 0, "num_answers": 0}

class ClassMergeCluster(BaseModel):
    cluster_id: str
    class_iris: List[str]
    class_profiles: List[ClassProfile]
    similarity_stats: Dict[str, Any]

class MergeAction(BaseModel):
    action: str = Field(..., pattern="^(alias|keep_subclass|deprecate)$")
    from_class_iri: str
    to_class_iri: Optional[str] = None
    reason: str

class CanonicalClassInfo(BaseModel):
    class_iri: Optional[str]
    preferred_label: Optional[str]

class ClassMergeDecision(BaseModel):
    cluster_id: str
    decision: str = Field(..., pattern="^(merge|keep_hierarchy|keep_distinct|drop_some)$")
    canonical_class: CanonicalClassInfo
    actions: List[MergeAction]
    comment: str

logger.info(" Pydantic модели определены")

# %%
# STEP 6A.1: EXPORT CLASS PROFILES

def normalize_label(label: str) -> str:
    s = label.lower()
    s = re.sub(r'[^\w\s]', '', s)
    return s.strip()

def get_tokens(normalized_label: str) -> List[str]:
    return normalized_label.split()

def normalize_root(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOP_SUFFIXES]

def export_class_profiles(owl_path: Path, cqs_path: Path) -> List[ClassProfile]:
    if not owl_path.exists():
        logger.error(f"OWL файл не найден: {owl_path}")
        return []

    onto = get_ontology(str(owl_path)).load()
    logger.info(f" Онтология загружена. Классов: {len(list(onto.classes()))}")

    cqs_text = []
    if cqs_path.exists():
        with open(cqs_path, 'r') as f:
            for line in f:
                if line.strip():
                    cqs_text.append(json.loads(line))
    
    profiles = []
    
    for cls in onto.classes():
        if cls == Thing: continue
        
        lbl = cls.label[0] if cls.label else cls.name
        norm_lbl = normalize_label(lbl)
        tokens = get_tokens(norm_lbl)
        
        parents = []
        for p in cls.is_a:
            if isinstance(p, Thing.__class__) and p != Thing:
                try: parents.append(p.iri)
                except: pass
        
        desc = cls.comment[0] if cls.comment else ""
        
        found_cqs = []
        for cq in cqs_text:
            if norm_lbl in cq.get('text', '').lower():
                found_cqs.append(cq['cq_id'])
        
        profile = ClassProfile(
            class_iri=cls.iri,
            label=lbl,
            normalized_label=norm_lbl,
            tokens=tokens,
            root_tokens=normalize_root(tokens), # New field
            parent_iris=parents,
            sibling_iris=[],
            text_description=desc,
            example_cq_ids=found_cqs[:5],
            usage_stats={"num_cqs": len(found_cqs), "num_answers": 0}
        )
        profiles.append(profile)
        
    return profiles

profiles_file = OUTPUT_DIR / "class_profiles_v2.jsonl"
profiles = []

if profiles_file.exists():
    logger.info(f" Загрузка профилей из {profiles_file}")
    with open(profiles_file, 'r') as f:
        for line in f:
            profiles.append(ClassProfile.model_validate_json(line))
else:
    logger.info(" Генерация профилей классов (v2)...")
    profiles = export_class_profiles(INPUT_OWL, INPUT_CQS)
    with open(profiles_file, 'w') as f:
        for p in profiles:
            f.write(p.model_dump_json() + '\n')
    logger.success(f" Профили сохранены ({len(profiles)} классов)")

if profiles:
    # Show example of root tokens
    display(pd.DataFrame([{
        "label": p.label, 
        "tokens": p.tokens, 
        "root_tokens": p.root_tokens
    } for p in profiles[:5]]))

# %%
# STEP 6A.2: CLUSTERING (AGGRESSIVE SIBLING MATCHING)

def compute_jaccard(tokens1: List[str], tokens2: List[str]) -> float:
    s1, s2 = set(tokens1), set(tokens2)
    if not s1 or not s2: return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))

def find_clusters(profiles: List[ClassProfile], embedding_model_path: str) -> List[ClassMergeCluster]:
    if not profiles: return []
    
    logger.info(f" Загрузка модели {embedding_model_path}...")
    model = SentenceTransformer(embedding_model_path)
    
    texts = [f"{p.label}. {p.text_description}" for p in profiles]
    embeddings = model.encode(texts, convert_to_tensor=True)
    
    G = nx.Graph()
    for p in profiles:
        G.add_node(p.class_iri)
        
    logger.info(" Вычисление попарной схожести...")
    cos_scores = util.cos_sim(embeddings, embeddings)
    
    edges_added = 0
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            p1 = profiles[i]
            p2 = profiles[j]
            
            sem_sim = float(cos_scores[i][j])
            lex_sim = compute_jaccard(p1.tokens, p2.tokens)
            root_sim = compute_jaccard(p1.root_tokens, p2.root_tokens)
            
            # Shared Parents: Aggressive check
            # Если есть хотя бы один общий родитель - считаем сиблингами
            shared_parents = bool(set(p1.parent_iris).intersection(set(p2.parent_iris)))
            
            # Условие схожести (более мягкое)
            # Достаточно высокой семантики, лексики ИЛИ корневой схожести
            is_similar = (
                (sem_sim >= SIMILARITY_THRESHOLD_SEMANTIC) or 
                (lex_sim >= SIMILARITY_THRESHOLD_LEXICAL) or 
                (root_sim >= 0.5)
            )
            
            # Агрессивное условие группировки:
            # ТОЛЬКО внутри сиблингов (общий родитель), но зато берем почти всё похожее
            # Либо очень высокая лексическая схожесть (на случай ошибок в иерархии)
            condition = shared_parents or (lex_sim > 0.9)
            
            if is_similar and condition:
                G.add_edge(p1.class_iri, p2.class_iri, weight=sem_sim)
                edges_added += 1
                
    logger.info(f" Добавлено {edges_added} ребер схожести")
    
    clusters = []
    cluster_idx = 1
    for component in nx.connected_components(G):
        if len(component) < 2: continue
        
        comp_profiles = [p for p in profiles if p.class_iri in component]
        
        cluster = ClassMergeCluster(
            cluster_id=f"CMC_{cluster_idx:03d}",
            class_iris=list(component),
            class_profiles=comp_profiles,
            similarity_stats={"size": len(component)}
        )
        clusters.append(cluster)
        cluster_idx += 1
        
    return clusters

clusters_file = OUTPUT_DIR / "class_merge_clusters_v2.jsonl"
clusters = []

if clusters_file.exists():
    logger.info(f" Загрузка кластеров из {clusters_file}")
    with open(clusters_file, 'r') as f:
        for line in f:
            clusters.append(ClassMergeCluster.model_validate_json(line))
else:
    logger.info(" Запуск кластеризации (v2)...")
    clusters = find_clusters(profiles, EMBEDDING_MODEL_PATH)
    with open(clusters_file, 'w') as f:
        for c in clusters:
            f.write(c.model_dump_json() + '\n')
    logger.success(f" Найдено {len(clusters)} кластеров кандидатов")
    
if clusters:
    print(f"Пример кластера: {clusters[0].cluster_id}")
    for p in clusters[0].class_profiles:
        print(f" - {p.label} (Root: {p.root_tokens})")

# %%
# STEP 6A.3: SIBLING MERGE AGENT (AGGRESSIVE)

# Aggressive System Prompt
SYSTEM_PROMPT_AGENT = """You are an ontology engineering expert optimizing a Deep Active Learning ontology.

Your goal is to MINIMIZE the number of distinct classes while preserving essential distinctions.
You receive a cluster of 'sibling' classes (classes sharing a parent or very similar names).

Analysis Rules:
1. If classes differ only by synonymy or minor phrasing (e.g., "Cost" vs "ComputationalCost"), you MUST MERGE them (action: "alias").
2. If one class is clearly a special case of another but shares the same parent in the input, propose "keep_subclass" ONLY IF the distinction is critical for the domain. Otherwise, merge.
3. If a class is redundant, unused (num_cqs=0), or vague, propose "deprecate".
4. "keep_distinct" should be the LAST RESORT, used only for clearly orthogonal concepts.

Prioritize compact ontologies. Do not be afraid to merge.
"""

def decide_cluster(cluster: ClassMergeCluster, openai_client) -> Optional[ClassMergeDecision]:
    if not openai_client: return None
    
    # Pass extra info: usage_stats, root_tokens, parents
    simple_profiles = []
    for p in cluster.class_profiles:
        simple_profiles.append({
            "iri": p.class_iri,
            "label": p.label,
            "root_tokens": p.root_tokens,
            "parents": p.parent_iris,
            "usage_stats": p.usage_stats
        })
    
    user_prompt = f"""Cluster of candidate siblings (JSON):
{json.dumps({"cluster_id": cluster.cluster_id, "class_profiles": simple_profiles}, indent=2)}

Task: Minimize redundancy.
1. Choose a canonical class (best label/most used).
2. Mark others as "alias" (merge), "keep_subclass", or "deprecate".
3. Only use "keep_distinct" if absolutely necessary.

Return valid JSON matching ClassMergeDecision schema.
"""
    
    try:
        response = openai_client.beta.chat.completions.parse(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_AGENT},
                {"role": "user", "content": user_prompt}
            ],
            response_format=ClassMergeDecision,
            temperature=0.0,
            timeout=60.0
        )
        decision = response.choices[0].message.parsed
        decision.cluster_id = cluster.cluster_id 
        return decision
    except Exception as e:
        logger.error(f"Error deciding cluster {cluster.cluster_id}: {e}")
        return None

decisions_file = OUTPUT_DIR / "class_merge_decisions_v2.jsonl"
decisions = []

if decisions_file.exists():
    logger.info(f" Загрузка решений из {decisions_file}")
    with open(decisions_file, 'r') as f:
        for line in f:
            decisions.append(ClassMergeDecision.model_validate_json(line))
else:
    if clusters and openai_client:
        logger.info(f" Запуск Sibling Merge Agent для {len(clusters)} кластеров...")
        for cluster in tqdm(clusters, desc="Sibling Merge"):
            d = decide_cluster(cluster, openai_client)
            if d:
                decisions.append(d)
        with open(decisions_file, 'w') as f:
            for d in decisions:
                f.write(d.model_dump_json() + '\n')
        logger.success(f" Получено {len(decisions)} решений")

# %%
# STEP 6A.4: PARENT COLLAPSE (NEW STEP)
# Цель: найти родителей, чьи дети являются "листьями" и, возможно, избыточны.
# Схлопнуть детей в родителя, если они не несут уникальной смысловой нагрузки.

def find_parent_child_candidates(profiles: List[ClassProfile]) -> List[ClassMergeCluster]:
    # 1. Построим индекс children
    children_map = defaultdict(list)
    iri_to_profile = {p.class_iri: p for p in profiles}
    
    for p in profiles:
        for parent in p.parent_iris:
            children_map[parent].append(p.class_iri)
            
    # 2. Найдем родителей, у которых есть дети-листья (у которых нет своих детей)
    # Либо просто берем всех родителей и их детей
    candidates = []
    pc_idx = 1
    
    for parent_iri, children_iris in children_map.items():
        if len(children_iris) == 0: continue
        
        # Проверим, есть ли родитель в наших профилях
        if parent_iri not in iri_to_profile: continue
        
        parent_profile = iri_to_profile[parent_iri]
        child_profiles = [iri_to_profile[child] for child in children_iris if child in iri_to_profile]
        
        if not child_profiles: continue

        # Формируем кластер "Parent + Children"
        all_profiles = [parent_profile] + child_profiles
        
        cluster = ClassMergeCluster(
            cluster_id=f"PC_{pc_idx:03d}",
            class_iris=[p.class_iri for p in all_profiles],
            class_profiles=all_profiles,
            similarity_stats={"type": "parent_collapse", "parent": parent_profile.label}
        )
        candidates.append(cluster)
        pc_idx += 1
        
    return candidates

# System Prompt for Parent Collapse
SYSTEM_PROMPT_PARENT = """You are optimizing an ontology hierarchy.
You are given a Parent class and its Subclasses.
Your goal: Simplify the hierarchy.

Decide for each Subclass:
1. "alias": The subclass is just a synonym or minor variant of the Parent. Merge it into Parent.
2. "deprecate": The subclass is too specific, unused, or vague. Remove it.
3. "keep_subclass": The subclass represents a DISTINCT, IMPORTANT concept that specializes the Parent significantly.

If in doubt, prefer "alias" or "deprecate" to keep the ontology flat and clean.
"""

def decide_parent_collapse(cluster: ClassMergeCluster, openai_client) -> Optional[ClassMergeDecision]:
    if not openai_client: return None
    
    # Identify parent (it's stored in stats or usually the one being pointed to)
    # В нашем коде parent - это первый профиль или тот, который в stats
    parent_label = cluster.similarity_stats.get("parent")
    
    simple_profiles = []
    for p in cluster.class_profiles:
        is_parent = (p.label == parent_label)
        simple_profiles.append({
            "role": "PARENT" if is_parent else "CHILD",
            "iri": p.class_iri,
            "label": p.label,
            "description": p.text_description,
            "usage_stats": p.usage_stats
        })
        
    user_prompt = f"""Parent-Child Group (JSON):
{json.dumps({"group_id": cluster.cluster_id, "hierarchy": simple_profiles}, indent=2)}

Task: Collapse unnecessary children into the Parent.
1. Identify the Parent class.
2. For each CHILD: decide 'alias' (merge to parent), 'deprecate', or 'keep_subclass'.
"""

    try:
        response = openai_client.beta.chat.completions.parse(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_PARENT},
                {"role": "user", "content": user_prompt}
            ],
            response_format=ClassMergeDecision,
            temperature=0.0,
            timeout=60.0
        )
        decision = response.choices[0].message.parsed
        decision.cluster_id = cluster.cluster_id 
        return decision
    except Exception as e:
        logger.error(f"Error deciding PC {cluster.cluster_id}: {e}")
        return None

pc_decisions_file = OUTPUT_DIR / "parent_collapse_decisions.jsonl"
pc_decisions = []

if pc_decisions_file.exists():
    logger.info(f" Загрузка решений Parent Collapse из {pc_decisions_file}")
    with open(pc_decisions_file, 'r') as f:
        for line in f:
            pc_decisions.append(ClassMergeDecision.model_validate_json(line))
else:
    pc_candidates = find_parent_child_candidates(profiles)
    logger.info(f" Найдено {len(pc_candidates)} групп Parent-Child")
    
    if pc_candidates and openai_client:
        logger.info(" Запуск Parent Collapse Agent...")
        for cluster in tqdm(pc_candidates, desc="Parent Collapse"):
            d = decide_parent_collapse(cluster, openai_client)
            if d:
                pc_decisions.append(d)
                
        with open(pc_decisions_file, 'w') as f:
            for d in pc_decisions:
                f.write(d.model_dump_json() + '\n')
        logger.success(f" Получено {len(pc_decisions)} решений по схлопыванию иерархии")

# %%
# STEP 6A.5: APPLY ALL PATCHES

def apply_decisions(owl_path: Path, all_decisions: List[ClassMergeDecision], output_path: Path):
    if not owl_path.exists(): return
    
    onto = get_ontology(str(owl_path)).load()
    
    actions_count = {"alias": 0, "keep_subclass": 0, "deprecate": 0}
    
    # Чтобы не применять противоречивые действия, можно сначала собрать все actions
    # Но для простоты применяем последовательно.
    
    with onto:
        for d in all_decisions:
            if d.decision == "keep_distinct": continue
            
            for act in d.actions:
                from_cls = onto.search_one(iri=act.from_class_iri)
                # Если to_class_iri пустой (например при deprecate), to_cls = None
                to_cls = onto.search_one(iri=act.to_class_iri) if act.to_class_iri else None
                
                if not from_cls: continue
                
                if act.action == "alias" and to_cls:
                    # Merge
                    if to_cls != from_cls:
                        from_cls.equivalent_to.append(to_cls)
                        from_cls.deprecated.append(True)
                        from_cls.comment.append(f"Merged into {to_cls.name}")
                        actions_count["alias"] += 1
                        
                elif act.action == "keep_subclass" and to_cls:
                    # Ensure hierarchy
                    if to_cls not in from_cls.is_a:
                        from_cls.is_a.append(to_cls)
                    actions_count["keep_subclass"] += 1
                    
                elif act.action == "deprecate":
                    from_cls.deprecated.append(True)
                    from_cls.comment.append(f"Deprecated: {act.reason}")
                    actions_count["deprecate"] += 1

    onto.save(file=str(output_path), format="rdfxml")
    logger.success(f" Онтология обновлена и сохранена в {output_path}")
    logger.info(f"Статистика действий: {actions_count}")
    return onto

refined_owl_path = OUTPUT_DIR / "deep_active_learning_refined_v2.owl"

all_decisions = decisions + pc_decisions

if all_decisions:
    logger.info(f" Применение {len(all_decisions)} пакетов решений...")
    refined_onto = apply_decisions(INPUT_OWL, all_decisions, refined_owl_path)

# %%
# METRICS

if 'refined_onto' in locals() and refined_onto:
    logger.info(" Запуск Reasoner...")
    try:
        with refined_onto:
            sync_reasoner(infer_property_values=True)
        logger.success(" Онтология консистентна")
    except Exception as e:
        logger.error(f" Ошибка Reasoner: {e}")

    classes_total = len(list(refined_onto.classes()))
    active_classes = 0
    for c in refined_onto.classes():
        if not c.deprecated:
            active_classes += 1
            
    logger.info(f" Итоги Refinement V2 (Aggressive):")
    logger.info(f"  Всего классов: {classes_total}")
    logger.info(f"  Активных классов: {active_classes}")
    logger.info(f"  Deprecated/Merged: {classes_total - active_classes}")
