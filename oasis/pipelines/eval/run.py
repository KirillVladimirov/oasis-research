# Оркестрация оценки: CQ → онтология → выравнивание, затем summary.csv и report.md.

from __future__ import annotations

import csv
import json
from pathlib import Path

from loguru import logger

from oasis.pipelines.eval.eval_alignment import run as run_alignment
from oasis.pipelines.eval.eval_cqs import run as run_cqs
from oasis.pipelines.eval.eval_ontology import run as run_ontology


def _extract_summary(
    cqs_metrics: dict,
    ontology_metrics: dict,
    alignment_metrics: dict,
    exp_id: str,
    run_id: str,
) -> dict:
    cqs_wf = cqs_metrics.get("well_formedness", {})
    cqs_len = cqs_metrics.get("length_tokens", {})
    cqs_dup = cqs_metrics.get("duplicates", {})
    ont_size = ontology_metrics.get("size", {})
    ont_depth = ontology_metrics.get("subclass_depth") or ontology_metrics.get("structure", {})
    labels = ontology_metrics.get("labels", {})

    def label_missing_ratio(section: str) -> float:
        return labels.get(section, {}).get("missing_ratio", 0.0)

    return {
        "exp_id": exp_id,
        "run_id": run_id,
        "cqs_total": cqs_metrics.get("total", 0),
        "cqs_missing_text": cqs_metrics.get("missing_text", 0),
        "cqs_ends_with_qmark_ratio": cqs_wf.get("ends_with_qmark_ratio", 0.0),
        "cqs_starts_with_interrogative_ratio": cqs_wf.get("starts_with_interrogative_ratio", 0.0),
        "cqs_avg_len_tokens": cqs_len.get("avg", 0.0),
        "cqs_p95_len_tokens": cqs_len.get("p95", 0.0),
        "cqs_exact_duplicates_count": cqs_dup.get("exact_duplicates_count", 0),
        "cqs_unique_ratio": cqs_dup.get("unique_ratio", 0.0),
        "ontology_triples": ont_size.get("triples", 0),
        "ontology_classes": ont_size.get("classes", 0),
        "ontology_object_properties": ont_size.get("object_properties", 0),
        "ontology_data_properties": ont_size.get("data_properties", 0),
        "ontology_annotation_properties": ont_size.get("annotation_properties", 0),
        "ontology_max_depth": ont_depth.get("max_depth", 0),
        "ontology_avg_depth": ont_depth.get("avg_depth", 0.0),
        "ontology_classes_label_missing_ratio": label_missing_ratio("classes"),
        "ontology_object_properties_label_missing_ratio": label_missing_ratio("object_properties"),
        "ontology_data_properties_label_missing_ratio": label_missing_ratio("data_properties"),
        "ontology_annotation_properties_label_missing_ratio": label_missing_ratio(
            "annotation_properties"
        ),
        "alignment_grounded_cq_ratio": alignment_metrics.get("grounded_cq_ratio", 0.0),
        "alignment_classes_covered_ratio": alignment_metrics.get(
            "classes_covered_by_cqs_ratio", 0.0
        ),
        "alignment_properties_covered_ratio": alignment_metrics.get(
            "properties_covered_by_cqs_ratio", 0.0
        ),
        "alignment_answerability_proxy_rate": alignment_metrics.get(
            "answerability_proxy_rate", 0.0
        ),
    }


def run_eval(
    cqs_path: Path,
    ontology_path: Path,
    out_dir: Path,
    config_path: Path | None = None,
    reasoner: str = "pellet",
    semantic_dedup: bool = False,
    semantic_thresholds: str = "0.92,0.95",
    log_level: str = "INFO",
) -> int:
    """Запуск полной оценки для одного датасета: CQ, онтология, выравнивание, summary и report. Возвращает 0 при успехе."""
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = out_dir / "run.log"
    logger.remove()
    logger.add(run_log_path, level=log_level)

    cqs_metrics_path = out_dir / "cqs_metrics.json"
    ontology_metrics_path = out_dir / "ontology_metrics.json"
    alignment_metrics_path = out_dir / "alignment_metrics.json"
    grounding_path = out_dir / "cq_grounding.jsonl"
    summary_path = out_dir / "summary.csv"
    report_path = out_dir / "report.md"

    try:
        run_cqs(
            input_path=cqs_path,
            output_path=cqs_metrics_path,
            config_path=config_path,
            log_level=log_level,
            semantic_dedup=semantic_dedup,
            semantic_thresholds=semantic_thresholds,
        )
    except Exception as e:
        logger.error("eval_cqs failed: {}", e)
        return 1

    try:
        run_ontology(
            input_path=ontology_path,
            output_path=ontology_metrics_path,
            config_path=config_path,
            reasoner=reasoner,
            log_level=log_level,
        )
    except Exception as e:
        logger.error("eval_ontology failed: {}", e)
        return 1

    try:
        run_alignment(
            cqs_path=cqs_path,
            ontology_path=ontology_path,
            output_path=alignment_metrics_path,
            grounding_path=grounding_path,
            config_path=config_path,
            log_level=log_level,
        )
    except Exception as e:
        logger.error("eval_alignment failed: {}", e)
        return 1

    cqs_metrics = json.loads(cqs_metrics_path.read_text(encoding="utf-8"))
    ontology_metrics = json.loads(ontology_metrics_path.read_text(encoding="utf-8"))
    alignment_metrics = json.loads(alignment_metrics_path.read_text(encoding="utf-8"))

    run_id = out_dir.name
    exp_id = out_dir.parent.name
    summary = _extract_summary(
        cqs_metrics, ontology_metrics, alignment_metrics, exp_id, run_id
    )
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    logger.info("Wrote summary to {}", summary_path)

    report_lines = [
        "# Отчет по оценке",
        "",
        f"- exp_id: {exp_id}",
        f"- run_id: {run_id}",
        "",
        "## Компетентностные вопросы",
        f"- total: {cqs_metrics.get('total', 0)}",
        f"- ends_with_qmark_ratio: {cqs_metrics.get('well_formedness', {}).get('ends_with_qmark_ratio', 0.0):.3f}",
        f"- starts_with_interrogative_ratio: {cqs_metrics.get('well_formedness', {}).get('starts_with_interrogative_ratio', 0.0):.3f}",
        f"- avg_len_tokens: {cqs_metrics.get('length_tokens', {}).get('avg', 0.0):.2f}",
        f"- exact_duplicates_count: {cqs_metrics.get('duplicates', {}).get('exact_duplicates_count', 0)}",
        "",
        "## Онтология",
        f"- triples: {ontology_metrics.get('size', {}).get('triples', 0)}",
        f"- classes: {ontology_metrics.get('size', {}).get('classes', 0)}",
        f"- object_properties: {ontology_metrics.get('size', {}).get('object_properties', 0)}",
        f"- data_properties: {ontology_metrics.get('size', {}).get('data_properties', 0)}",
        f"- max_depth: {ontology_metrics.get('structure', {}).get('max_depth', 0)}",
        "",
        "## Выравнивание",
        f"- grounded_cq_ratio: {alignment_metrics.get('grounded_cq_ratio', 0.0):.3f}",
        f"- classes_covered_by_cqs_ratio: {alignment_metrics.get('classes_covered_by_cqs_ratio', 0.0):.3f}",
        f"- properties_covered_by_cqs_ratio: {alignment_metrics.get('properties_covered_by_cqs_ratio', 0.0):.3f}",
        f"- answerability_proxy_rate: {alignment_metrics.get('answerability_proxy_rate', 0.0):.3f}",
        "",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Wrote report to {}", report_path)
    return 0
