# Метрики оценки (Step 9 и Step 11)

Два контура: **Step 9** — CQ, онтология и выравнивание; **Step 11** — граф знаний (KG).

---

## Step 9: CQ, онтология, выравнивание

**Скрипт:** `scripts/pipeline/step9_eval.py`  
**Входы:** `artifacts/<dataset>/cqs_grounded.jsonl`, `ontology.owl`  
**Выход:** `artifacts/<dataset>/eval/` (cqs_metrics.json, ontology_metrics.json, alignment_metrics.json, summary.csv, report.md).

**Что замеряется:**

- **CQ:** число вопросов, доля с «?» и с вопросительным словом в начале, длина в токенах, точные и near-дубликаты, атомарность (союзы, запятые).
- **Онтология:** число триплетов, классов, свойств, глубина иерархии, полнота меток, консистентность (reasoner).
- **Выравнивание:** доля CQ с grounding в онтологию, покрытие классов и свойств CQ, прокси ответимости.

---

## Step 11: граф знаний (KG)

**Скрипт:** `scripts/pipeline/step11_kg_eval.py`  
**Входы:** `artifacts/<dataset>/kg/kg.ttl`, опционально ontology.owl, kg_provenance.csv  
**Выход:** `artifacts/<dataset>/kg_eval/`; сводно — `artifacts/kg_eval/kg_metrics_all.csv`.

**Что замеряется:**

- **Валидность:** parse_ok, shacl_ran, shacl_violations.
- **Масштаб:** n_triples, n_statements, n_entities, n_papers, n_chunks, n_cqs, typed_ratio.
- **Provenance:** evidence_coverage, doc_diversity, doc_top1_share, doc_entropy.
- **Граф:** n_components, giant_component_share, avg_degree.
- **Предикаты:** predicate_top1_share, predicate_top3_share, predicate_entropy.
- **Статус:** OK / WARN / FAIL (FAIL при parse_ok=0, нарушениях SHACL или evidence_coverage&lt;0.99; WARN при высокой концентрации по одному документу или одному предикату).

---

## Запуск

- **Step 9:** `python scripts/pipeline/step9_eval.py --config configs/pipeline.yaml` (опционально `--datasets`, `--reasoner`).
- **Step 11 (один датасет):** `python scripts/pipeline/step11_kg_eval.py --config configs/pipeline.yaml [--dataset NAME]`
- **Step 11 (все датасеты):** `python scripts/pipeline/step11_kg_eval.py --config configs/pipeline.yaml --all`
