# Метрики оценки качества автоматизированной генерации (Step 9 и Step 11)

Два контура оценки: **Step 9** — компетентностные вопросы (CQ), онтология и выравнивание CQ–онтология; **Step 11** — граф знаний (KG). Ниже перечислены метрики по каждому контуру и выходные артефакты.

---

## 1. Step 9: оценка CQ, онтологии и выравнивания

**Скрипт:** `scripts/pipeline/step9_eval.py`  
**Оркестрация:** `oasis.pipelines.eval_run.run_all`  
**Входы:** по каждому датасету — `artifacts/<dataset>/cqs_grounded.jsonl`, `artifacts/<dataset>/ontology.owl`  
**Выход:** `artifacts/<dataset>/eval/` (метрики, summary, отчёт).

По каждому датасету по очереди запускаются:

- `scripts/eval/evaluate_cqs.py` → **cqs_metrics.json**
- `scripts/eval/evaluate_ontology.py` → **ontology_metrics.json**
- `scripts/eval/evaluate_alignment.py` → **alignment_metrics.json**, **cq_grounding.jsonl**

Затем формируются **summary.csv** и **report.md**.

### 1.1. Метрики компетентностных вопросов (CQ)

| Метрика | Описание | Источник |
|--------|----------|----------|
| **total** | Число CQ в наборе | cqs_metrics.json |
| **missing_text** | Число записей без текста вопроса | cqs_metrics.json |
| **ends_with_qmark_ratio** | Доля вопросов, оканчивающихся на «?» | well_formedness |
| **starts_with_interrogative_ratio** | Доля вопросов, начинающихся с вопросительного слова (what, how, which, …) | well_formedness |
| **contains_placeholder_ratio** | Доля с плейсхолдерами (xxx, tbd, [mask] и т.п.) | well_formedness |
| **multiple_questions_ratio** | Доля формулировок с несколькими вопросами в одном тексте | well_formedness |
| **trailing_period_without_qmark_ratio** | Доля с точкой в конце без «?» | well_formedness |
| **bad_charset_ratio** | Доля символов из «плохого» набора (управляющие и т.п.) | well_formedness |
| **has_conjunction_ratio** | Доля с союзами and/or | atomicity |
| **has_semicolon_or_colon_ratio** | Доля с «;» или «:» | atomicity |
| **avg_commas** | Среднее число запятых на вопрос | atomicity |
| **non_atomic_ratio** | Доля неатомарных (сложных) вопросов | atomicity |
| **atomicity_proxy_rate** | 1 − non_atomic_ratio | atomicity |
| **length_tokens: min / avg / p95 / max** | Длина в токенах (min, среднее, 95‑й перцентиль, max) | length_tokens |
| **exact_duplicates_count** | Число точных дубликатов | duplicates |
| **unique_ratio** | Доля уникальных по тексту вопросов | duplicates |
| **near_duplicates_jaccard** | Пары near-duplicate по Jaccard (порог, число пар, ratio, примеры) | duplicates |
| **semantic_duplicates** | (опционально) семантическая дедупликация и кластеры | semantic_duplicates |
| **llm_judge** | (опционально) сводка LLM-оценки качества/дублей | llm_judge |

В **summary.csv** для CQ попадают: **cqs_total**, **cqs_missing_text**, **cqs_ends_with_qmark_ratio**, **cqs_starts_with_interrogative_ratio**, **cqs_avg_len_tokens**, **cqs_p95_len_tokens**, **cqs_exact_duplicates_count**, **cqs_unique_ratio**.

### 1.2. Метрики онтологии

| Метрика | Описание | Источник |
|--------|----------|----------|
| **size: triples** | Число триплетов в графе | size |
| **size: classes** | Число OWL/RDFS классов | size |
| **size: object_properties** | Число object properties | size |
| **size: data_properties** | Число datatype properties | size |
| **size: annotation_properties** | Число annotation properties | size |
| **size: individuals** | Число индивидов | size |
| **size: annotations** | Число аннотаций | size |
| **structure: subclass_edges_count** | Число рёбер rdfs:subClassOf | structure |
| **structure: max_depth** | Максимальная глубина иерархии подклассов | structure |
| **structure: avg_depth** | Средняя глубина | structure |
| **structure: avg_branching_factor** | Средний коэффициент ветвления | structure |
| **structure: roots_count** | Число корневых классов | structure |
| **richness: relationship_richness** | object_properties / (object + data properties) | richness |
| **richness: attribute_richness** | data_properties / classes | richness |
| **richness: inheritance_richness** | subclass_edges / classes | richness |
| **richness: annotation_richness** | Доля сущностей с аннотациями | richness |
| **labels** | По секциям (classes, object_properties, …): missing_ratio, примеры без label | labels |
| **sanity_checks** | dangling classes/properties, коллизии меток, bad_naming_ratio | sanity_checks |
| **consistency** | Результат reasoner: is_consistent, unsatisfiable_classes_count, список несовместных классов | consistency |

В **summary.csv** для онтологии: **ontology_triples**, **ontology_classes**, **ontology_object_properties**, **ontology_data_properties**, **ontology_annotation_properties**, **ontology_max_depth**, **ontology_avg_depth**, **ontology_*_label_missing_ratio** по секциям.

### 1.3. Метрики выравнивания CQ и онтологии

| Метрика | Описание | Источник |
|--------|----------|----------|
| **grounded_cq_ratio** | Доля CQ, для которых найдено выравнивание с классами/свойствами онтологии | alignment_metrics.json |
| **classes_covered_by_cqs_ratio** | Доля классов онтологии, упомянутых в CQ | alignment_metrics.json |
| **properties_covered_by_cqs_ratio** | Доля свойств онтологии, упомянутых в CQ | alignment_metrics.json |
| **answerability_proxy_rate** | Доля CQ, для которых по токенам/паттернам есть «ответимость» (прокси) | alignment_metrics.json |

Все четыре попадают в **summary.csv** и в **report.md**.

**Артефакты Step 9 (на датасет):**

- `artifacts/<dataset>/eval/cqs_metrics.json`
- `artifacts/<dataset>/eval/ontology_metrics.json`
- `artifacts/<dataset>/eval/alignment_metrics.json`
- `artifacts/<dataset>/eval/cq_grounding.jsonl`
- `artifacts/<dataset>/eval/summary.csv`
- `artifacts/<dataset>/eval/report.md`
- `artifacts/<dataset>/eval/run.log`

---

## 2. Step 11: оценка графа знаний (KG)

**Скрипт:** `scripts/pipeline/step11_kg_eval.py`  
**Логика:** `oasis.pipelines.kg_eval`  
**Входы:** по датасету — `artifacts/<dataset>/kg/kg.ttl`, опционально `ontology.owl`, `kg_provenance.csv` / `kg_provenance.parquet`  
**Выход:** по датасету — `artifacts/<dataset>/kg_eval/`; сводно — `artifacts/kg_eval/kg_metrics_all.csv`, `kg_metrics_all_sorted.csv`.

### 2.1. Quality gates (валидность)

| Метрика | Описание |
|--------|----------|
| **parse_ok** | 1 — kg.ttl успешно распарсен, 0 — ошибка (остальные метрики NA) |
| **shacl_ran** | 1 — SHACL проверка выполнялась, 0 — не выполнялась / не доступна |
| **shacl_violations** | Число нарушений SHACL или пусто (NA), если shacl_ran=0 |
| **error_msg** | Сообщение об ошибке при parse_ok=0 |

### 2.2. Масштаб и наполнение (ABox)

| Метрика | Описание |
|--------|----------|
| **n_triples** | Общее число триплетов в графе |
| **n_statements** | Число реифицированных утверждений (Support) |
| **n_entities** | Число сущностей (Paper + Chunk + CQ) |
| **typed_ratio** | Доля узлов с rdf:type |
| **n_papers** | Число документов (Paper) |
| **n_chunks** | Число чанков (Chunk) |
| **n_cqs** | Число компетентностных вопросов (CQ) в графе |

### 2.3. Provenance / grounding

| Метрика | Описание |
|--------|----------|
| **evidence_coverage** | Доля statement’ов с хотя бы одним prov:wasDerivedFrom (evidence) |
| **avg_evidence_per_statement** | Среднее число evidence на statement (у нас обычно 1) |
| **doc_diversity** | Число уникальных doc_id в evidence |
| **doc_top1_share** | Доля statement’ов, поддержанных самым частым документом |
| **doc_entropy** | Энтропия Шеннона по распределению statement’ов по doc_id |

### 2.4. Структура графа (Entity-граф)

| Метрика | Описание |
|--------|----------|
| **n_components** | Число слабо связных компонент |
| **giant_component_share** | Доля узлов в крупнейшей компоненте |
| **avg_degree** | Средняя степень узла (2× рёбра / узлы) |

### 2.5. Распределение предикатов

| Метрика | Описание |
|--------|----------|
| **predicate_top1_share** | Доля рёбер с самым частым предикатом |
| **predicate_top3_share** | Суммарная доля топ‑3 предикатов |
| **predicate_entropy** | Энтропия Шеннона по предикатам (разнообразие отношений) |

### 2.6. Служебные и статус

| Метрика | Описание |
|--------|----------|
| **dataset_id** | Идентификатор датасета |
| **kg_path** | Путь к kg.ttl |
| **run_ts** | Время запуска (ISO UTC) |
| **runtime_sec** | Время выполнения оценки (сек) |
| **status** | **OK** / **WARN** / **FAIL** (stoplight) |

**Правила status (11.5):**

- **FAIL:** parse_ok=0 или (shacl_ran=1 и shacl_violations>0) или evidence_coverage<0.99  
- **WARN:** doc_top1_share>0.4 или predicate_top1_share>0.5  
- **OK:** иначе  

**Артефакты Step 11**

- По датасету в `artifacts/<dataset>/kg_eval/`:  
  **schema_metrics.json**, **kb_metrics.json**, **graph_metrics.json**, **provenance_metrics.json**, **rdf_stats.json**, **shacl_report.txt**, **shacl_violations_count.txt**, **stoplight.json**, **all_metrics.json**
- Сводно в `artifacts/kg_eval/`:  
  **kg_metrics_all.csv** (все колонки контракта), **kg_metrics_all_sorted.csv** (сортировка по evidence_coverage и shacl_violations).

---

## 3. Запуск

- **Step 9 (все датасеты из конфига):**  
  `python scripts/pipeline/step9_eval.py --config configs/pipeline.yaml`  
  Опционально: `--datasets A B`, `--eval-config path`, `--reasoner pellet|hermit`.

- **Step 11 (один датасет):**  
  `python scripts/pipeline/step11_kg_eval.py --config configs/pipeline.yaml [--dataset NAME]`

- **Step 11 (все датасеты, единый CSV):**  
  `python scripts/pipeline/step11_kg_eval.py --config configs/pipeline.yaml --all`  
  Опционально: `--no-shacl`, `--manifest path/to/datasets_manifest.csv`.
